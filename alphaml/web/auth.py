"""
Accounts for the multi-user mode: users and login sessions in a SQLite database in the data folder.

Passwords are stored as salted scrypt hashes and session tokens only as SHA-256 hashes, so a copy of the
database reveals neither.
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
MIN_PASSWORD_LENGTH = 8
SESSION_DAYS = 30
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


class AuthError(ValueError):
    pass


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, **SCRYPT)
    return f'scrypt${salt.hex()}${digest.hex()}'


def verify_password(password, stored):
    try:
        _, salt, digest = stored.split('$')
        candidate = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt), **SCRYPT)
    except ValueError:
        return False
    return hmac.compare_digest(candidate.hex(), digest)


_DUMMY_HASH = hash_password(secrets.token_hex(8))  # equalises timing for unknown accounts


def _token_hash(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def normalize_email(email):
    return (email or '').strip().lower()


def check_password(password):
    if len(password or '') < MIN_PASSWORD_LENGTH:
        raise AuthError(f'The password must have at least {MIN_PASSWORD_LENGTH} characters.')


class UserStore:
    def __init__(self, path):
        self.path = str(path)
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS password_resets (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires REAL NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0);
            ''')
            # Consent to the privacy terms (GDPR): which version of the terms, and when (added later: migrate)
            columns = {row['name'] for row in db.execute('PRAGMA table_info(users)')}
            for column in ('consent_version', 'consent_at'):
                if column not in columns:
                    db.execute(f'ALTER TABLE users ADD COLUMN {column} TEXT')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys = ON')
        try:
            with db:  # commit, or roll back on error
                yield db
        finally:
            db.close()

    @staticmethod
    def _user(row):
        if row is None:
            return None
        return {'id': row['id'], 'email': row['email'], 'created': row['created'],
                'consent_version': row['consent_version'], 'consent_at': row['consent_at']}

    # -------------------------------- users -------------------------------- #

    def create_user(self, email, password, consent_version=None):
        """A new account; consent_version: the privacy terms the user agreed to when signing up."""
        email = normalize_email(email)
        if not EMAIL_RE.match(email) or len(email) > 254:
            raise AuthError('Please enter a valid email address.')
        check_password(password)
        created = datetime.now(timezone.utc).isoformat(timespec='seconds')
        consent_at = created if consent_version else None
        try:
            with self._db() as db:
                cursor = db.execute('INSERT INTO users (email, password, created, consent_version, consent_at) '
                                    'VALUES (?, ?, ?, ?, ?)',
                                    (email, hash_password(password), created, consent_version, consent_at))
        except sqlite3.IntegrityError:
            raise AuthError('An account with this email address already exists.') from None
        return {'id': cursor.lastrowid, 'email': email, 'created': created, 'consent_version': consent_version,
                'consent_at': consent_at}

    def set_consent(self, user_id, consent_version):
        """Record that the user agreed to this version of the privacy terms, now."""
        with self._db() as db:
            db.execute('UPDATE users SET consent_version = ?, consent_at = ? WHERE id = ?',
                       (consent_version, datetime.now(timezone.utc).isoformat(timespec='seconds'), user_id))

    def authenticate(self, email, password):
        """The user with this email and password, or None."""
        with self._db() as db:
            row = db.execute('SELECT * FROM users WHERE email = ?', (normalize_email(email),)).fetchone()
        if row is None:
            verify_password(password, _DUMMY_HASH)
            return None
        return self._user(row) if verify_password(password, row['password']) else None

    def get(self, email):
        with self._db() as db:
            return self._user(db.execute('SELECT * FROM users WHERE email = ?', (normalize_email(email),)).fetchone())

    def list(self):
        with self._db() as db:
            return [self._user(row) for row in db.execute('SELECT * FROM users ORDER BY id')]

    def set_password(self, user_id, password):
        """Change a password and end all of the user's sessions."""
        check_password(password)
        with self._db() as db:
            db.execute('UPDATE users SET password = ? WHERE id = ?', (hash_password(password), user_id))
            db.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))

    def delete(self, user_id):
        with self._db() as db:
            db.execute('DELETE FROM users WHERE id = ?', (user_id,))

    # ------------------------------- sessions ------------------------------ #

    def create_session(self, user_id):
        token = secrets.token_urlsafe(32)
        with self._db() as db:
            db.execute('INSERT INTO sessions VALUES (?, ?, ?)',
                       (_token_hash(token), user_id, time.time() + SESSION_DAYS * 86400))
        return token

    def session_user(self, token):
        with self._db() as db:
            row = db.execute('SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id '
                             'WHERE token_hash = ? AND expires > ?', (_token_hash(token), time.time())).fetchone()
        return self._user(row)

    def delete_session(self, token):
        with self._db() as db:
            db.execute('DELETE FROM sessions WHERE token_hash = ?', (_token_hash(token),))

    def purge_sessions(self):
        """Delete expired sessions and reset links."""
        with self._db() as db:
            db.execute('DELETE FROM sessions WHERE expires <= ?', (time.time(),))
            db.execute('DELETE FROM password_resets WHERE expires <= ?', (time.time(),))

    # ---------------------------- password reset --------------------------- #

    def create_reset(self, user_id, minutes):
        """A single-use password-reset token, valid for `minutes`. Earlier unused tokens stop working."""
        token = secrets.token_urlsafe(32)
        with self._db() as db:
            db.execute('UPDATE password_resets SET used = 1 WHERE user_id = ? AND used = 0', (user_id,))
            db.execute('INSERT INTO password_resets VALUES (?, ?, ?, 0)',
                       (_token_hash(token), user_id, time.time() + minutes * 60))
        return token

    def reset_password(self, token, password):
        """
        Set a new password with a reset token and end the user's sessions. Returns the user, or None if the
        token is invalid, used or expired.
        """
        check_password(password)  # before the token is used up
        with self._db() as db:
            row = db.execute('SELECT users.* FROM password_resets JOIN users ON users.id = password_resets.user_id '
                             'WHERE token_hash = ? AND used = 0 AND expires > ?',
                             (_token_hash(token), time.time())).fetchone()
            if row is None:
                return None
            db.execute('UPDATE password_resets SET used = 1 WHERE token_hash = ?', (_token_hash(token),))
            db.execute('UPDATE users SET password = ? WHERE id = ?', (hash_password(password), row['id']))
            db.execute('DELETE FROM sessions WHERE user_id = ?', (row['id'],))
        return self._user(row)


class RateLimiter:
    """Allow at most `limit` events per key (e.g. failed logins of a client and email) within `window` seconds."""

    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self._events = {}
        self._lock = threading.Lock()

    def _recent(self, key):
        cutoff = time.time() - self.window
        recent = [t for t in self._events.get(key, []) if t > cutoff]
        if recent:
            self._events[key] = recent
        else:
            self._events.pop(key, None)
        return recent

    def exceeded(self, key):
        with self._lock:
            return len(self._recent(key)) >= self.limit

    def add(self, key):
        with self._lock:
            self._events[key] = self._recent(key) + [time.time()]

    def clear(self, key):
        with self._lock:
            self._events.pop(key, None)

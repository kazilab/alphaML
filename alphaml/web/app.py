"""alphaML 2.0 web application: REST API and single-page UI."""
import hashlib
import hmac
import io
import logging
import os
import re
import shutil
import threading
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, Response,
                     UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..config import (ALGORITHMS, API_FEATURE_METHODS, FEATURE_METHODS, LOG_TYPES, MODEL_NAMES,
                      NORMALIZATION_METHODS, PARAM_SEARCH_METHODS, PREDICT_DEFAULTS, RANK_METRICS, SAMPLING_METHODS,
                      TRAIN_DEFAULTS)
from ..utils.io import EXAMPLE_TRAIN_URLS, TABLE_SUFFIXES, TRAIN_FILES, download_and_extract, find_file
from .auth import SESSION_DAYS, AuthError, RateLimiter, UserStore, normalize_email
from .datasets import DatasetStore
from .jobs import ACTIVE, JobManager, read_json
from .mail import MailSettings, send_mail
from .uploads import UploadError, UploadStore

logger = logging.getLogger('alphaml.web')

STATIC = Path(__file__).parent / 'static'
CHUNK = 1 << 20
IMAGE_TYPES = {'.png': 'image', '.pdf': 'pdf', '.csv': 'table', '.xlsx': 'table', '.json': 'data',
               '.pkl': 'model', '.log': 'log'}
SESSION_COOKIE = 'alphaml_session'
# Version of the privacy terms users agree to. Change it when the terms change: everyone is asked to agree again.
PRIVACY_TERMS = '2026-10-18'
GDPR_URL = 'https://eur-lex.europa.eu/eli/reg/2016/679/oj'
RESET_MINUTES = 60
RESET_SENT = ('If an account uses this email, a reset link is on its way. It works once, '
              f'for {RESET_MINUTES} minutes.')
EXAMPLE_NAME = 'Example data (figshare)'

FeatureMethod = Literal[tuple(API_FEATURE_METHODS)]
Algorithm = Literal[tuple(ALGORITHMS)]


class TrainRequest(BaseModel):
    dataset_id: str
    model_name: Literal[tuple(MODEL_NAMES)] = TRAIN_DEFAULTS['model_name']
    automl_models: List[Algorithm] = Field(TRAIN_DEFAULTS['automl_models'], min_length=1)
    automl_metric: Literal[tuple(RANK_METRICS)] = TRAIN_DEFAULTS['automl_metric']
    automl_top_k: int = Field(TRAIN_DEFAULTS['automl_top_k'], ge=1, le=len(ALGORITHMS))
    automl_time_budget: float = Field(TRAIN_DEFAULTS['automl_time_budget'], ge=0, le=7 * 24 * 60)
    automl_ensemble: bool = TRAIN_DEFAULTS['automl_ensemble']
    sampling_method: Literal[tuple(SAMPLING_METHODS)] = TRAIN_DEFAULTS['sampling_method']
    param_search: Literal[tuple(PARAM_SEARCH_METHODS)] = TRAIN_DEFAULTS['param_search']
    control_fitting: bool = TRAIN_DEFAULTS['control_fitting']
    col_label: str = Field(min_length=1)
    pos_class: str = Field(min_length=1)
    neg_class: str = Field(min_length=1)
    test_size: float = Field(TRAIN_DEFAULTS['test_size'], gt=0, lt=1)
    random_state: int = Field(TRAIN_DEFAULTS['random_state'], ge=0)
    n_trials: int = Field(TRAIN_DEFAULTS['n_trials'], ge=1, le=5000)
    cv: int = Field(TRAIN_DEFAULTS['cv'], ge=2, le=50)
    features: FeatureMethod = TRAIN_DEFAULTS['features']
    features2: FeatureMethod = TRAIN_DEFAULTS['features2']
    features3: FeatureMethod = TRAIN_DEFAULTS['features3']
    features4: FeatureMethod = TRAIN_DEFAULTS['features4']
    n_top_features: int = Field(TRAIN_DEFAULTS['n_top_features'], ge=0)
    min_disp: float = TRAIN_DEFAULTS['min_disp']
    log_type: Literal[tuple(LOG_TYPES)] = TRAIN_DEFAULTS['log_type']
    top_n_genes_pca: int = Field(TRAIN_DEFAULTS['top_n_genes_pca'], ge=1)
    top_n_genes_rp: int = Field(TRAIN_DEFAULTS['top_n_genes_rp'], ge=1)
    n_clusters: int = Field(TRAIN_DEFAULTS['n_clusters'], ge=1)
    n_components: int = Field(TRAIN_DEFAULTS['n_components'], ge=1)
    max_features: int = Field(TRAIN_DEFAULTS['max_features'], ge=0)
    threshold: str = TRAIN_DEFAULTS['threshold']
    min_sel_features_rfe: int = Field(TRAIN_DEFAULTS['min_sel_features_rfe'], ge=1)
    min_sel_features_sfs: int = Field(TRAIN_DEFAULTS['min_sel_features_sfs'], ge=1)
    normalization: Literal[tuple(NORMALIZATION_METHODS)] = TRAIN_DEFAULTS['normalization']
    run_feat_imp: bool = TRAIN_DEFAULTS['run_feat_imp']
    run_shap: bool = TRAIN_DEFAULTS['run_shap']
    run_lime: bool = TRAIN_DEFAULTS['run_lime']
    run_roc: bool = TRAIN_DEFAULTS['run_roc']

    @field_validator('threshold')
    @classmethod
    def _threshold(cls, value):
        # Same forms as scikit-learn SelectFromModel: a number, 'mean', 'median' or e.g. '1.5*mean'
        value = value.strip()
        if not re.fullmatch(r'(\d+(\.\d+)?\*)?(mean|median)|\d+(\.\d+)?(e-?\d+)?', value):
            raise ValueError("use a number, 'mean', 'median' or a scaled form such as '1.5*mean'")
        return value


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    signup_code: Optional[str] = Field(None, max_length=256)
    accept_terms: bool = False  # the privacy terms (GDPR consent), required to create an account


class Consent(BaseModel):
    accept_terms: bool


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class UploadStart(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=1)


class DatasetFromUploads(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    files: Dict[str, str]  # file stem (labeled_data, data_labels, ...) -> upload id


class ResetRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ResetConfirm(BaseModel):
    token: str = Field(min_length=16, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class AccountDeletion(BaseModel):
    password: str = Field(min_length=1, max_length=256)


def _env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ('0', 'false', 'no', 'off')


async def _save_upload(upload, dest, max_bytes):
    size = 0
    with open(dest, 'wb') as fh:
        while chunk := await upload.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f'{upload.filename} is larger than {max_bytes >> 20} MB.')
            fh.write(chunk)


def _table_suffix(upload, field):
    suffix = Path(upload.filename or '').suffix.lower()
    if suffix not in TABLE_SUFFIXES:
        raise HTTPException(400, f'{field}: please upload a .csv, .tsv or .txt file.')
    return suffix


def _file_kind(path):
    return IMAGE_TYPES.get(path.suffix.lower(), 'other')


def data_home(home=None):
    """
    Folder for uploaded data sets and job results: home, else ALPHAML_HOME, else <Documents>/alphaML.
    (The Docker image sets ALPHAML_HOME=/data, which docker-compose.yml maps to the project's tmp folder.)
    """
    if home or os.environ.get('ALPHAML_HOME'):
        return Path(home or os.environ['ALPHAML_HOME']).expanduser().resolve()
    documents = Path.home() / 'Documents'
    return (documents if documents.is_dir() else Path.home()) / 'alphaML'


def create_app(home=None, max_jobs=None, auth=None, retention_days=None, allowed_hosts=None):
    """
    The alphaML web application.

    auth: require accounts; each user sees only their own data sets and jobs (default: ALPHAML_AUTH, off).
    retention_days: delete data sets and jobs this many days after they were created, 0 to keep them
    (default: ALPHAML_RETENTION_DAYS, 14 with accounts and 0 without).
    allowed_hosts: host names the server answers to (default: ALPHAML_ALLOWED_HOSTS, localhost and 127.0.0.1).
    """
    home = data_home(home)
    home.mkdir(parents=True, exist_ok=True)
    max_jobs = int(max_jobs or os.environ.get('ALPHAML_MAX_JOBS', 1))
    max_upload = int(os.environ.get('ALPHAML_MAX_UPLOAD_MB', 2048)) << 20
    # Uploads are sent in chunks: one request of a public proxy such as Cloudflare may carry at most 100 MB
    chunk_bytes = min(int(os.environ.get('ALPHAML_UPLOAD_CHUNK_MB', 8)), 64) << 20
    auth = _env_flag('ALPHAML_AUTH', '0') if auth is None else bool(auth)
    if retention_days is None:
        retention_days = float(os.environ.get('ALPHAML_RETENTION_DAYS') or (14 if auth else 0))
    # Model bundles are pickles: with accounts, uploading one would let any user run code on the server
    allow_model_upload = _env_flag('ALPHAML_ALLOW_MODEL_UPLOAD', '0' if auth else '1')
    allow_signup = _env_flag('ALPHAML_ALLOW_SIGNUP', '1')
    signup_code = os.environ.get('ALPHAML_SIGNUP_CODE') or None
    mail = MailSettings.from_env()
    # Address users reach the server at, for links in emails (default: the address of the request)
    public_url = os.environ.get('ALPHAML_PUBLIC_URL', '').strip().rstrip('/')
    # Behind a proxy or tunnel every connection comes from the proxy; it passes the visitor's address in this
    # header (e.g. CF-Connecting-IP for a Cloudflare Tunnel). Only safe while the port is reachable through
    # that proxy alone, as with the localhost-only port in docker-compose.yml.
    client_ip_header = os.environ.get('ALPHAML_CLIENT_IP_HEADER', '').strip()
    # Shown in the privacy notice: who runs this server, and how to contact them about personal data
    operator = os.environ.get('ALPHAML_OPERATOR', '').strip()
    privacy_contact = os.environ.get('ALPHAML_PRIVACY_CONTACT', '').strip()
    max_user_jobs = int(os.environ.get('ALPHAML_MAX_JOBS_PER_USER', 3))
    cookie_secure = _env_flag('ALPHAML_COOKIE_SECURE', '0')
    # The terms name the retention period, so a different period is a different version of the terms
    terms_version = f'{PRIVACY_TERMS}/{retention_days:g}d'
    if allowed_hosts is None:
        allowed_hosts = [h.strip() for h in os.environ.get('ALPHAML_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
                         if h.strip()]

    datasets = DatasetStore(home / 'datasets')
    jobs = JobManager(home / 'jobs', max_running=max_jobs)
    uploads = UploadStore(home / 'uploads', max_bytes=max_upload, chunk_bytes=chunk_bytes)
    users = UserStore(home / 'users.db') if auth else None
    failed_logins = RateLimiter(limit=5, window=900)
    reset_requests = RateLimiter(limit=3, window=3600)    # reset emails per address
    client_requests = RateLimiter(limit=20, window=3600)  # reset requests and failed reset links per client
    example_lock = threading.Lock()
    stop = threading.Event()

    def purge_expired():
        """Delete data sets and finished jobs older than the retention period, and expired sessions."""
        if users:
            users.purge_sessions()
        uploads.purge(max_age_hours=24)  # uploads that were started but never used
        if not retention_days:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        for job in jobs.list():
            if job['status'] not in ACTIVE and datetime.fromisoformat(job['created']) < cutoff:
                jobs.delete(job['id'])
                logger.info(f"Retention: deleted job {job['id']}")
        used = jobs.active_inputs()
        for record in datasets.list():
            in_use = any(p in used for p in datasets.inputs(record['id']).values() if p)
            if not record.get('public') and not in_use and datetime.fromisoformat(record['created']) < cutoff:
                datasets.delete(record['id'])
                logger.info(f"Retention: deleted data set {record['id']}")

    def purge_loop():
        while True:
            try:
                purge_expired()
            except Exception:
                logger.exception('Retention clean-up failed')
            if stop.wait(3600):
                return

    @asynccontextmanager
    async def lifespan(_):
        threading.Thread(target=purge_loop, name='alphaml-retention', daemon=True).start()
        yield
        stop.set()
        jobs.shutdown()

    app = FastAPI(title='alphaML 2.0', version=__version__, lifespan=lifespan)
    app.state.datasets = datasets
    app.state.jobs = jobs
    app.state.users = users
    app.state.retention_days = retention_days
    app.state.purge_expired = purge_expired
    if '*' not in allowed_hosts:
        # Only answer to known host names: stops DNS-rebinding pages from reaching a local server
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware('http')
    async def require_api_header(request, call_next):
        # Browsers only send custom headers from this app's own pages, so other websites cannot make a
        # visitor's browser upload data or start jobs here (cross-site request forgery).
        if (request.url.path.startswith('/api/') and request.method not in ('GET', 'HEAD', 'OPTIONS')
                and request.headers.get('x-alphaml') != '1'):
            return JSONResponse({'detail': 'Missing X-AlphaML request header.'}, status_code=403)
        return await call_next(request)

    # ----------------------------- users ----------------------------------- #

    def current_user(request: Request):
        """The logged-in user, None without accounts; 401 if accounts are on and nobody is logged in."""
        if not auth:
            return None
        token = request.cookies.get(SESSION_COOKIE)
        user = users.session_user(token) if token else None
        if user is None:
            raise HTTPException(401, 'Please log in.')
        return user

    def consented_user(request: Request):
        """current_user, who must also have agreed to the current privacy terms to use data and jobs."""
        user = current_user(request)
        if user is not None and user['consent_version'] != terms_version:
            raise HTTPException(403, 'Please agree to the privacy terms first.')
        return user

    def client_ip(request):
        """The visitor's address, for rate limits."""
        if client_ip_header:
            forwarded = request.headers.get(client_ip_header, '').split(',')[0].strip()
            if forwarded:
                return forwarded
        return request.client.host if request.client else ''

    def owner_of(user):
        return user['id'] if user else None

    def visible(record, user):
        return not auth or record.get('owner') == user['id'] or bool(record.get('public'))

    def start_session(response, user):
        response.set_cookie(SESSION_COOKIE, users.create_session(user['id']), max_age=SESSION_DAYS * 86400,
                            httponly=True, samesite='lax', secure=cookie_secure, path='/')
        return {'email': user['email']}

    def require_accounts():
        if not auth:
            raise HTTPException(404, 'Accounts are not enabled on this server.')

    def expires(record):
        if not retention_days or record.get('public'):
            return None
        created = datetime.fromisoformat(record['created'])
        return (created + timedelta(days=retention_days)).isoformat(timespec='seconds')

    # ------------------------------ helpers -------------------------------- #

    def dataset_or_404(dataset_id, user):
        record = datasets.get(dataset_id)
        if record is None or not visible(record, user):
            raise HTTPException(404, 'Data set not found.')
        return record

    def job_or_404(job_id, user):
        job = jobs.get(job_id)
        if job is None or not visible(job, user):
            raise HTTPException(404, 'Job not found.')
        return job

    def upload_or_404(upload_id, user):
        record = uploads.get(upload_id) if upload_id else None
        if record is None or (auth and record.get('owner') != user['id']):
            raise HTTPException(404, 'Upload not found.')
        return record

    def take_upload(record, dest):
        try:
            return uploads.take(record['id'], dest)
        except UploadError as exc:
            raise HTTPException(exc.status, str(exc)) from None

    def dataset_view(record):
        return {**record, 'expires': expires(record)}

    def check_job_quota(user):
        if auth and sum(1 for j in jobs.list() if j.get('owner') == user['id'] and j['status'] in ACTIVE) \
                >= max_user_jobs:
            raise HTTPException(429, f'You already have {max_user_jobs} queued or running jobs. '
                                     f'Wait for one to finish or cancel one.')

    def result_path(job_id, name):
        root = (jobs.job_dir(job_id) / 'results').resolve()
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, 'File not found.')
        return path

    def job_view(job, detail=False):
        folder = jobs.job_dir(job['id'])
        view = {k: v for k, v in job.items() if k != 'inputs'}
        view['expires'] = expires(job)
        if job['status'] in ACTIVE:
            view['progress'] = read_json(folder / 'progress.json')
        if detail:
            results = folder / 'results'
            view['files'] = [{'name': p.relative_to(results).as_posix(), 'size': p.stat().st_size,
                              'kind': _file_kind(p)}
                             for p in sorted(results.rglob('*')) if p.is_file()]
        else:
            view.pop('summary', None)
            view['pscore'] = (job.get('summary') or {}).get('pscore')
        return view

    def delete_user_data(user_id):
        for record in uploads.list():
            if record.get('owner') == user_id:
                uploads.delete(record['id'])
        for job in jobs.list():
            if job.get('owner') == user_id:
                jobs.delete(job['id'])
        for record in datasets.list():
            if record.get('owner') == user_id:
                datasets.delete(record['id'])

    # ------------------------------- options ------------------------------- #

    @app.get('/api/options')
    def options():
        return {'version': __version__, 'models': MODEL_NAMES, 'algorithms': ALGORITHMS,
                'rank_metrics': RANK_METRICS, 'sampling_methods': SAMPLING_METHODS,
                'param_search_methods': PARAM_SEARCH_METHODS, 'normalization_methods': NORMALIZATION_METHODS,
                'log_types': LOG_TYPES, 'feature_methods': FEATURE_METHODS, 'train_defaults': TRAIN_DEFAULTS,
                'predict_defaults': PREDICT_DEFAULTS, 'allow_model_upload': allow_model_upload,
                'max_running_jobs': jobs.max_running, 'auth': auth, 'allow_signup': allow_signup,
                'signup_code_required': bool(signup_code), 'retention_days': retention_days,
                'password_reset': auth and mail.configured, 'upload_chunk_bytes': chunk_bytes,
                'max_upload_bytes': max_upload,
                'privacy': {'operator': operator, 'contact': privacy_contact, 'gdpr_url': GDPR_URL,
                            'terms_version': terms_version}}

    # ------------------------------ accounts ------------------------------- #

    @app.post('/api/auth/register', status_code=201)
    def register(body: Credentials, response: Response):
        require_accounts()
        if not allow_signup:
            raise HTTPException(403, 'New accounts are created by the administrator of this server.')
        if signup_code and not hmac.compare_digest((body.signup_code or '').strip(), signup_code):
            raise HTTPException(403, 'Wrong sign-up code.')
        if not body.accept_terms:
            raise HTTPException(400, 'Please agree to the privacy terms to create an account.')
        try:
            user = users.create_user(body.email, body.password, consent_version=terms_version)
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from None
        return start_session(response, user)

    @app.post('/api/auth/login')
    def login(body: Credentials, request: Request, response: Response):
        require_accounts()
        key = (client_ip(request), body.email.strip().lower())
        if failed_logins.exceeded(key):
            raise HTTPException(429, 'Too many failed logins. Try again in 15 minutes.')
        user = users.authenticate(body.email, body.password)
        if user is None:
            failed_logins.add(key)
            raise HTTPException(401, 'Wrong email or password.')
        failed_logins.clear(key)
        return start_session(response, user)

    @app.post('/api/auth/password-reset')
    def request_password_reset(body: ResetRequest, request: Request, background: BackgroundTasks):
        """Email a single-use reset link. The reply is the same whether or not the address has an account, and
        the email is sent after the reply."""
        require_accounts()
        if not mail.configured:
            raise HTTPException(503, 'Password reset by email is not set up on this server. '
                                     'Ask the administrator to reset your password.')
        client = client_ip(request)
        if client_requests.exceeded(client):
            raise HTTPException(429, 'Too many requests. Try again later.')
        client_requests.add(client)
        email = normalize_email(body.email)
        if not reset_requests.exceeded(email):
            reset_requests.add(email)
            user = users.get(email)
            if user is not None:
                token = users.create_reset(user['id'], RESET_MINUTES)
                base = public_url or str(request.base_url).rstrip('/')
                text = ('Someone asked to reset the password of your alphaML 2.0 account.\n\n'
                        f'Set a new password here (the link works once, for {RESET_MINUTES} minutes):\n'
                        f'{base}/#/reset/{token}\n\n'
                        'If it was not you, ignore this email; your password stays as it is.\n')
                background.add_task(send_mail, mail, user['email'], 'Reset your alphaML password', text)
        return {'sent': True, 'detail': RESET_SENT}

    @app.post('/api/auth/password-reset/confirm')
    def confirm_password_reset(body: ResetConfirm, request: Request, response: Response):
        """Set a new password with a reset link; other devices are logged out."""
        require_accounts()
        client = client_ip(request)
        if client_requests.exceeded(client):
            raise HTTPException(429, 'Too many requests. Try again later.')
        try:
            user = users.reset_password(body.token, body.password)
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from None
        if user is None:
            client_requests.add(client)
            raise HTTPException(400, 'This reset link is invalid, used or expired. Ask for a new one.')
        return start_session(response, user)

    @app.post('/api/auth/logout')
    def logout(request: Request, response: Response):
        token = request.cookies.get(SESSION_COOKIE)
        if users and token:
            users.delete_session(token)
        response.delete_cookie(SESSION_COOKIE, path='/')
        return {'ok': True}

    @app.get('/api/me')
    def me(user=Depends(current_user)):
        return {'auth': auth, 'email': user['email'] if user else None,
                'consent': user is None or user['consent_version'] == terms_version}

    @app.post('/api/account/consent')
    def give_consent(body: Consent, user=Depends(current_user)):
        """Agree to the current privacy terms (accounts created by the administrator, or terms that changed)."""
        require_accounts()
        if not body.accept_terms:
            raise HTTPException(400, 'Please agree to the privacy terms to use alphaML.')
        users.set_consent(user['id'], terms_version)
        return {'consent': True}

    @app.post('/api/account/password')
    def change_password(body: PasswordChange, response: Response, user=Depends(current_user)):
        require_accounts()
        if users.authenticate(user['email'], body.current_password) is None:
            raise HTTPException(403, 'The current password is wrong.')
        try:
            users.set_password(user['id'], body.new_password)
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from None
        return start_session(response, user)  # other sessions were ended

    @app.post('/api/account/delete')
    def delete_account(body: AccountDeletion, response: Response, user=Depends(current_user)):
        require_accounts()
        if users.authenticate(user['email'], body.password) is None:
            raise HTTPException(403, 'The password is wrong.')
        delete_user_data(user['id'])
        users.delete(user['id'])
        response.delete_cookie(SESSION_COOKIE, path='/')
        return {'ok': True}

    # ------------------------------- uploads ------------------------------- #

    @app.post('/api/uploads', status_code=201)
    def start_upload(body: UploadStart, user=Depends(consented_user)):
        """Start a chunked upload; the file is then sent with PUT /api/uploads/{id}?offset=... in chunks."""
        suffix = Path(body.filename).suffix.lower()
        if suffix not in TABLE_SUFFIXES and not (suffix == '.pkl' and allow_model_upload):
            raise HTTPException(400, f'{body.filename}: please upload a .csv, .tsv or .txt file.')
        try:
            record = uploads.start(owner_of(user), body.filename, body.size)
        except UploadError as exc:
            raise HTTPException(exc.status, str(exc)) from None
        return {'id': record['id'], 'chunk_size': chunk_bytes}

    @app.put('/api/uploads/{upload_id}')
    async def upload_chunk(upload_id: str, request: Request, offset: int = Query(..., ge=0),
                           user=Depends(consented_user)):
        upload_or_404(upload_id, user)
        if int(request.headers.get('content-length') or 0) > chunk_bytes:
            raise HTTPException(413, f'A chunk can carry at most {chunk_bytes >> 20} MB.')
        data = bytearray()
        async for part in request.stream():
            data += part
            if len(data) > chunk_bytes:
                raise HTTPException(413, f'A chunk can carry at most {chunk_bytes >> 20} MB.')
        try:
            received = await run_in_threadpool(uploads.write, upload_id, offset, bytes(data))
        except UploadError as exc:
            raise HTTPException(exc.status, str(exc)) from None
        return {'received': received}

    @app.get('/api/uploads/{upload_id}')
    def upload_status(upload_id: str, user=Depends(consented_user)):
        record = upload_or_404(upload_id, user)
        return {'received': record['received'], 'size': record['size']}

    @app.delete('/api/uploads/{upload_id}', status_code=204)
    def cancel_upload(upload_id: str, user=Depends(consented_user)):
        upload_or_404(upload_id, user)
        uploads.delete(upload_id)

    # ------------------------------ data sets ------------------------------ #

    @app.get('/api/datasets')
    def list_datasets(user=Depends(consented_user)):
        return [dataset_view(r) for r in datasets.list() if visible(r, user)]

    @app.post('/api/datasets', status_code=201)
    async def upload_dataset(name: str = Form(..., min_length=1, max_length=120),
                             labeled_data: UploadFile = File(...),
                             data_labels: UploadFile = File(...),
                             data_for_feature_selection: Optional[UploadFile] = File(None),
                             unlabeled_data: Optional[UploadFile] = File(None),
                             suggested_features: Optional[UploadFile] = File(None),
                             user=Depends(consented_user)):
        uploads = {'labeled_data': labeled_data, 'data_labels': data_labels,
                   'data_for_feature_selection': data_for_feature_selection, 'unlabeled_data': unlabeled_data,
                   'suggested_features': suggested_features}
        uploads = {stem: f for stem, f in uploads.items() if f is not None and f.filename}
        suffixes = {stem: _table_suffix(f, stem) for stem, f in uploads.items()}
        dataset_id, folder = datasets.new_folder()
        try:
            for stem, upload in uploads.items():
                await _save_upload(upload, folder / f'{stem}{suffixes[stem]}', max_upload)
            record = await run_in_threadpool(datasets.finalize, dataset_id, name.strip(), owner_of(user))
            return dataset_view(record)
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        except Exception as exc:
            raise HTTPException(400, f'Could not read the data set: {exc}') from exc

    @app.post('/api/datasets/from-uploads', status_code=201)
    def create_dataset_from_uploads(body: DatasetFromUploads, user=Depends(consented_user)):
        """Create a data set from finished chunked uploads."""
        unknown = sorted(set(body.files) - set(TRAIN_FILES))
        if unknown:
            raise HTTPException(422, f"Unknown data set files: {', '.join(unknown)}")
        records = {stem: upload_or_404(upload_id, user) for stem, upload_id in body.files.items()}
        for stem, record in records.items():
            if record['suffix'] not in TABLE_SUFFIXES:
                raise HTTPException(400, f'{stem}: please upload a .csv, .tsv or .txt file.')
        dataset_id, folder = datasets.new_folder()
        try:
            stats = {stem: take_upload(record, folder / f"{stem}{record['suffix']}")
                     for stem, record in records.items()}
            stats = {stem: counted for stem, counted in stats.items() if counted}
            return dataset_view(datasets.finalize(dataset_id, body.name.strip(), owner_of(user), stats=stats))
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        except Exception as exc:
            raise HTTPException(400, f'Could not read the data set: {exc}') from exc

    @app.post('/api/datasets/example', status_code=201)
    async def example_dataset(user=Depends(consented_user)):
        def fetch():
            # One shared, read-only copy for everyone
            with example_lock:
                existing = next((r for r in datasets.list() if r.get('public')), None)
                if existing:
                    return existing
                dataset_id, folder = datasets.new_folder()
                try:
                    download_and_extract(EXAMPLE_TRAIN_URLS, folder)
                    # keep only the recognised input files
                    keep = {find_file(folder, stem) for stem in TRAIN_FILES} - {None}
                    for path in folder.iterdir():
                        if path not in keep:
                            shutil.rmtree(path) if path.is_dir() else path.unlink()
                    return datasets.finalize(dataset_id, EXAMPLE_NAME, public=True)
                except Exception:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise

        try:
            return dataset_view(await run_in_threadpool(fetch))
        except Exception as exc:
            raise HTTPException(502, f'Could not download the example data: {exc}') from exc

    @app.get('/api/datasets/{dataset_id}')
    def get_dataset(dataset_id: str, user=Depends(consented_user)):
        return dataset_view(dataset_or_404(dataset_id, user))

    @app.delete('/api/datasets/{dataset_id}', status_code=204)
    def delete_dataset(dataset_id: str, user=Depends(consented_user)):
        record = dataset_or_404(dataset_id, user)
        if auth and record.get('public'):
            raise HTTPException(403, 'The shared example data set cannot be deleted.')
        used = jobs.active_inputs()
        if any(p in used for p in datasets.inputs(dataset_id).values() if p):
            raise HTTPException(409, 'A queued or running job uses this data set.')
        datasets.delete(dataset_id)

    # -------------------------------- jobs --------------------------------- #

    @app.get('/api/jobs')
    def list_jobs(user=Depends(consented_user)):
        return [job_view(job) for job in jobs.list() if visible(job, user)]

    @app.post('/api/jobs/train', status_code=201)
    def create_train_job(request: TrainRequest, user=Depends(consented_user)):
        dataset = dataset_or_404(request.dataset_id, user)
        check_job_quota(user)
        params = request.model_dump(exclude={'dataset_id'})
        title = f"{request.model_name} · {request.col_label}"
        if request.model_name != 'AutoML':  # AutoML settings do not apply
            params = {k: v for k, v in params.items() if not k.startswith('automl_')}
        job_id = jobs.create_dir()
        return job_view(jobs.submit(job_id, 'train', title,
                                    {'dataset_id': dataset['id'], 'dataset_name': dataset['name'], 'train': params},
                                    datasets.inputs(dataset['id']), owner=owner_of(user)))

    @app.post('/api/jobs/predict', status_code=201)
    async def create_predict_job(test_data: Optional[UploadFile] = File(None),
                                 test_data_upload: Optional[str] = Form(None),
                                 model_job_id: Optional[str] = Form(None),
                                 model_file: Optional[UploadFile] = File(None),
                                 model_upload: Optional[str] = Form(None),
                                 normalization: Optional[str] = Form(None),
                                 col_label: Optional[str] = Form(None),
                                 pos_class: Optional[str] = Form(None),
                                 neg_class: Optional[str] = Form(None),
                                 run_shap: bool = Form(False),
                                 run_lime: bool = Form(False),
                                 user=Depends(consented_user)):
        if normalization and normalization not in NORMALIZATION_METHODS:
            raise HTTPException(422, f'normalization must be one of {NORMALIZATION_METHODS}')
        # Files come either in this request or as chunked uploads (their ids)
        has_test_file = test_data is not None and bool(test_data.filename)
        if has_test_file == bool(test_data_upload):
            raise HTTPException(422, 'Upload a test data file.')
        has_upload = (model_file is not None and bool(model_file.filename)) or bool(model_upload)
        if bool(model_job_id) == has_upload:
            raise HTTPException(422, 'Choose either a training job or a model file.')
        test_record = upload_or_404(test_data_upload, user) if test_data_upload else None
        if test_record and test_record['suffix'] not in TABLE_SUFFIXES:
            raise HTTPException(400, 'test_data: please upload a .csv, .tsv or .txt file.')
        model_record = upload_or_404(model_upload, user) if model_upload else None
        if model_record and model_record['suffix'] != '.pkl':
            raise HTTPException(400, 'The model file must be a .pkl model bundle.')

        source = None
        if model_job_id:
            source_job = job_or_404(model_job_id, user)
            results = jobs.job_dir(model_job_id) / 'results'
            bundles = sorted(results.glob('*_bundle.pkl'))
            if source_job['kind'] != 'train' or source_job['status'] != 'succeeded' or not bundles:
                raise HTTPException(422, 'The selected job has no trained model.')
            source = source_job

        if has_upload and not allow_model_upload:
            raise HTTPException(403, 'Model uploads are disabled on this server.')
        check_job_quota(user)
        job_id = jobs.create_dir()
        inputs_dir = jobs.job_dir(job_id) / 'inputs'
        inputs_dir.mkdir()
        try:
            if test_record:
                test_path = inputs_dir / f"test_data{test_record['suffix']}"
                take_upload(test_record, test_path)
            else:
                test_path = inputs_dir / f'test_data{_table_suffix(test_data, "test_data")}'
                await _save_upload(test_data, test_path, max_upload)
            if source is not None:
                model_path = bundles[0]
                title = f"Predict · {source['title']}"
            else:
                model_path = inputs_dir / 'model_bundle.pkl'
                if model_record:
                    take_upload(model_record, model_path)
                else:
                    await _save_upload(model_file, model_path, max_upload)
                title = f"Predict · {model_record['filename'] if model_record else model_file.filename}"
        except Exception:
            shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
            raise

        params = {'model_job_id': model_job_id,
                  'predict': {'normalization': normalization or None, 'col_label': col_label or None,
                              'pos_class': pos_class or None, 'neg_class': neg_class or None,
                              'run_shap': run_shap, 'run_lime': run_lime}}
        inputs = {'test_data': str(test_path), 'model': str(model_path)}
        return job_view(jobs.submit(job_id, 'predict', title, params, inputs, owner=owner_of(user)))

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str, user=Depends(consented_user)):
        return job_view(job_or_404(job_id, user), detail=True)

    @app.get('/api/jobs/{job_id}/log')
    def job_log(job_id: str, offset: int = 0, user=Depends(consented_user)):
        job_or_404(job_id, user)
        path = jobs.job_dir(job_id) / 'run.log'
        if not path.is_file():
            return {'offset': 0, 'text': ''}
        with open(path, 'rb') as fh:
            fh.seek(max(0, offset))
            data = fh.read(CHUNK)
        return {'offset': max(0, offset) + len(data), 'text': data.decode('utf-8', errors='replace')}

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str, user=Depends(consented_user)):
        job_or_404(job_id, user)
        return job_view(jobs.cancel(job_id))

    @app.delete('/api/jobs/{job_id}', status_code=204)
    def delete_job(job_id: str, user=Depends(consented_user)):
        job_or_404(job_id, user)
        jobs.delete(job_id)

    @app.get('/api/jobs/{job_id}/files/{name:path}')
    def job_file(job_id: str, name: str, download: bool = False, user=Depends(consented_user)):
        job_or_404(job_id, user)
        path = result_path(job_id, name)
        disposition = 'attachment' if download or _file_kind(path) not in ('image', 'pdf') else 'inline'
        return FileResponse(path, filename=path.name, content_disposition_type=disposition)

    @app.get('/api/jobs/{job_id}/preview/{name:path}')
    def preview_file(job_id: str, name: str, rows: int = 200, user=Depends(consented_user)):
        import pandas as pd

        job_or_404(job_id, user)
        path = result_path(job_id, name)
        rows = max(1, min(rows, 2000))
        if path.suffix.lower() == '.csv':
            frame = pd.read_csv(path, nrows=rows)
        elif path.suffix.lower() == '.xlsx':
            frame = pd.read_excel(path, nrows=rows)
        else:
            raise HTTPException(400, 'Only CSV and Excel files can be previewed.')
        frame = frame.astype(object).where(frame.notna(), None)
        return {'columns': [str(c) for c in frame.columns], 'rows': frame.values.tolist()}

    @app.get('/api/jobs/{job_id}/archive')
    def job_archive(job_id: str, user=Depends(consented_user)):
        job = job_or_404(job_id, user)
        folder = jobs.job_dir(job_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in sorted((folder / 'results').rglob('*')):
                if path.is_file():
                    zf.write(path, f'results/{path.relative_to(folder / "results").as_posix()}')
            if (folder / 'run.log').is_file():
                zf.write(folder / 'run.log', 'run.log')
        buffer.seek(0)
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', f"alphaML_{job['title']}_{job_id}") + '.zip'
        return StreamingResponse(buffer, media_type='application/zip',
                                 headers={'Content-Disposition': f'attachment; filename="{name}"'})

    # -------------------------------- UI ----------------------------------- #

    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    # Caches (browsers, Cloudflare) keep scripts and styles for hours: every version gets its own address, and
    # the page that names them is revalidated on each visit
    index_html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for asset in ('app.js', 'styles.css'):
        version = hashlib.sha256((STATIC / asset).read_bytes()).hexdigest()[:12]
        index_html = index_html.replace(f'/static/{asset}"', f'/static/{asset}?v={version}"')

    @app.get('/', include_in_schema=False)
    def index():
        return HTMLResponse(index_html, headers={'Cache-Control': 'no-cache'})

    return app

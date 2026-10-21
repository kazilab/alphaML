import json
import re
import shutil
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from alphaml.web import create_app
from alphaml.web.__main__ import main as cli

HEADERS = {'X-AlphaML': '1'}
TRAIN = {'model_name': 'LASSO', 'param_search': 'Predefined', 'col_label': 'Trametinib', 'pos_class': 'resistant',
         'neg_class': 'sensitive', 'features': 'None'}
SLOW_TRAIN = {**TRAIN, 'model_name': 'SVC', 'param_search': 'Optuna', 'n_trials': 5000}


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv('ALPHAML_MAX_JOBS_PER_USER', '2')
    home = tmp_path_factory.mktemp('auth_home')
    app = create_app(home=home, max_jobs=2, auth=True, allowed_hosts=['testserver'])
    mp.undo()
    with TestClient(app, headers=HEADERS) as owner_client:  # runs the app's start-up and shutdown
        yield app, home, owner_client


def new_client(app):
    return TestClient(app, headers=HEADERS)


def register(client, email, password='correct horse battery'):
    response = client.post('/api/auth/register', json={'email': email, 'password': password, 'accept_terms': True})
    assert response.status_code == 201, response.text
    return response.json()


def upload(client, dataset, name='data'):
    files = {stem: (f'{stem}.csv', (dataset / f'{stem}.csv').read_bytes(), 'text/csv')
             for stem in ('labeled_data', 'data_labels')}
    response = client.post('/api/datasets', data={'name': name}, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def wait(client, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(0.5)
    raise AssertionError(f'job {job_id} did not finish')


# ------------------------------ request checks ------------------------------ #

def test_state_changes_need_the_api_header(tmp_path):
    app = create_app(home=tmp_path, auth=False, allowed_hosts=['testserver'])
    client = TestClient(app)
    assert client.post('/api/jobs/train', json={}).status_code == 403  # e.g. a form posted by another website
    assert client.post('/api/jobs/train', json={}, headers=HEADERS).status_code == 422
    assert client.get('/api/options').status_code == 200


def test_unknown_host_names_are_refused(tmp_path):
    app = create_app(home=tmp_path, auth=False)  # default: localhost and 127.0.0.1
    assert TestClient(app, base_url='http://attacker.example').get('/api/options').status_code == 400
    assert TestClient(app, base_url='http://localhost:8765').get('/api/options').status_code == 200


# --------------------------------- accounts --------------------------------- #

def test_login_is_required(server):
    app, _, _ = server
    anonymous = new_client(app)
    options = anonymous.get('/api/options').json()
    assert options['auth'] and options['retention_days'] == 14 and not options['allow_model_upload']
    for path in ('/api/datasets', '/api/jobs', '/api/me'):
        assert anonymous.get(path).status_code == 401
    assert 'alphaML 2.0' in anonymous.get('/').text  # the app shell itself is public


def test_registration_rules(server):
    client = new_client(server[0])
    assert client.post('/api/auth/register', json={'email': 'not-an-email', 'password': 'long enough 1', 'accept_terms': True}) \
        .status_code == 400
    assert client.post('/api/auth/register', json={'email': 'short@example.org', 'password': 'short', 'accept_terms': True}) \
        .status_code == 400
    register(client, 'Dup@Example.org')
    again = client.post('/api/auth/register', json={'email': 'dup@example.org', 'password': 'long enough 1', 'accept_terms': True})
    assert again.status_code == 400 and 'already exists' in again.text
    assert client.get('/api/me').json()['email'] == 'dup@example.org'


def test_users_only_see_their_own_data(server, dataset):
    app, _, alice = server
    register(alice, 'alice@example.org')
    data = upload(alice, dataset, 'Alice data')
    job = wait(alice, alice.post('/api/jobs/train', json={'dataset_id': data['id'], **TRAIN}).json()['id'])
    assert job['status'] == 'succeeded', job['error']
    image = next(f['name'] for f in job['files'] if f['kind'] == 'image')

    bob = new_client(app)
    register(bob, 'bob@example.org')
    assert bob.get('/api/datasets').json() == [] and bob.get('/api/jobs').json() == []
    for path in (f"/api/datasets/{data['id']}", f"/api/jobs/{job['id']}", f"/api/jobs/{job['id']}/log",
                 f"/api/jobs/{job['id']}/files/{image}", f"/api/jobs/{job['id']}/archive"):
        assert bob.get(path).status_code == 404, path
    assert bob.delete(f"/api/datasets/{data['id']}").status_code == 404
    assert bob.post(f"/api/jobs/{job['id']}/cancel").status_code == 404
    assert bob.post('/api/jobs/train', json={'dataset_id': data['id'], **TRAIN}).status_code == 404
    test_file = {'test_data': ('t.csv', (dataset / 'test_data.csv').read_bytes(), 'text/csv')}
    assert bob.post('/api/jobs/predict', data={'model_job_id': job['id']}, files=test_file).status_code == 404

    assert [d['id'] for d in alice.get('/api/datasets').json()] == [data['id']]
    assert alice.get(f"/api/jobs/{job['id']}/files/{image}").status_code == 200
    # Model uploads would let any account run code on the server
    upload_model = {**test_file, 'model_file': ('m.pkl', b'not a pickle', 'application/octet-stream')}
    assert alice.post('/api/jobs/predict', files=upload_model).status_code == 403
    # Expiry is shown with every item
    created = datetime.fromisoformat(data['created'])
    assert datetime.fromisoformat(data['expires']) == created + timedelta(days=14)


def test_registration_needs_consent_to_the_privacy_terms(server):
    app = server[0]
    client = new_client(app)
    options = client.get('/api/options').json()['privacy']
    assert options['gdpr_url'] == 'https://eur-lex.europa.eu/eli/reg/2016/679/oj'
    refused = client.post('/api/auth/register', json={'email': 'noconsent@example.org', 'password': 'long enough 1'})
    assert refused.status_code == 400 and 'privacy terms' in refused.text
    assert app.state.users.get('noconsent@example.org') is None  # no account without consent
    register(client, 'consent@example.org')
    user = app.state.users.get('consent@example.org')
    assert user['consent_version'] == options['terms_version'] and user['consent_at']
    assert client.get('/api/me').json()['consent'] is True


def test_accounts_without_consent_cannot_use_data(server, dataset):
    app = server[0]
    app.state.users.create_user('admin.made@example.org', 'made by admin')  # e.g. `alphaml-web users add`
    client = new_client(app)
    assert client.post('/api/auth/login', json={'email': 'admin.made@example.org', 'password': 'made by admin'}) \
        .status_code == 200
    assert client.get('/api/me').json()['consent'] is False
    for path in ('/api/datasets', '/api/jobs'):
        assert client.get(path).status_code == 403
    assert client.post('/api/uploads', json={'filename': 'a.csv', 'size': 1}).status_code == 403
    assert client.post('/api/account/consent', json={'accept_terms': False}).status_code == 400
    assert client.post('/api/account/consent', json={'accept_terms': True}).status_code == 200
    assert client.get('/api/datasets').status_code == 200

    # Deleting the account (withdrawing consent) never needs consent
    app.state.users.create_user('leaving@example.org', 'leaving password')
    leaving = new_client(app)
    leaving.post('/api/auth/login', json={'email': 'leaving@example.org', 'password': 'leaving password'})
    assert leaving.post('/api/account/delete', json={'password': 'leaving password'}).status_code == 200


def test_new_retention_period_needs_new_consent(tmp_path):
    first = create_app(home=tmp_path, auth=True, retention_days=14, allowed_hosts=['testserver'])
    with TestClient(first, headers=HEADERS) as client:
        register(client, 'terms@example.org')
        assert client.get('/api/me').json()['consent'] is True
    second = create_app(home=tmp_path, auth=True, retention_days=30, allowed_hosts=['testserver'])
    with TestClient(second, headers=HEADERS) as client:
        client.post('/api/auth/login', json={'email': 'terms@example.org', 'password': 'correct horse battery'})
        assert client.get('/api/me').json()['consent'] is False  # the terms now say 30 days


def test_logout_and_login(server):
    app = server[0]
    client = new_client(app)
    register(client, 'carl@example.org')
    assert client.post('/api/auth/logout').status_code == 200
    assert client.get('/api/datasets').status_code == 401
    assert client.post('/api/auth/login', json={'email': 'carl@example.org', 'password': 'wrong password'}) \
        .status_code == 401
    assert client.post('/api/auth/login', json={'email': 'CARL@example.org', 'password': 'correct horse battery'}) \
        .status_code == 200
    assert client.get('/api/me').json()['email'] == 'carl@example.org'


def test_failed_logins_are_throttled(server):
    client = new_client(server[0])
    register(client, 'mallory@example.org')
    for _ in range(5):
        assert client.post('/api/auth/login', json={'email': 'mallory@example.org', 'password': 'guess'}) \
            .status_code == 401
    blocked = client.post('/api/auth/login', json={'email': 'mallory@example.org', 'password': 'correct horse battery'})
    assert blocked.status_code == 429


def test_rate_limits_use_the_proxy_client_header(tmp_path, monkeypatch):
    # Behind a tunnel every connection comes from the tunnel; limits must apply per visitor
    monkeypatch.setenv('ALPHAML_CLIENT_IP_HEADER', 'CF-Connecting-IP')
    app = create_app(home=tmp_path, auth=True, allowed_hosts=['testserver'])
    with TestClient(app, headers=HEADERS) as client:
        register(client, 'proxy@example.org')
        wrong = {'email': 'proxy@example.org', 'password': 'wrong password'}
        for _ in range(5):
            client.post('/api/auth/login', json=wrong, headers={'CF-Connecting-IP': '203.0.113.7'})
        right = {'email': 'proxy@example.org', 'password': 'correct horse battery'}
        assert client.post('/api/auth/login', json=right, headers={'CF-Connecting-IP': '203.0.113.7'}).status_code == 429
        assert client.post('/api/auth/login', json=right, headers={'CF-Connecting-IP': '198.51.100.2'}).status_code == 200


def test_job_quota_per_user(server, dataset):
    client = new_client(server[0])
    register(client, 'quota@example.org')
    data = upload(client, dataset)
    ids = [client.post('/api/jobs/train', json={'dataset_id': data['id'], **SLOW_TRAIN}).json()['id'] for _ in range(2)]
    third = client.post('/api/jobs/train', json={'dataset_id': data['id'], **SLOW_TRAIN})
    assert third.status_code == 429
    for job_id in ids:
        assert client.post(f'/api/jobs/{job_id}/cancel').json()['status'] == 'cancelled'
    assert client.post('/api/jobs/train', json={'dataset_id': data['id'], **TRAIN}).status_code == 201


def test_retention_deletes_old_data_but_keeps_the_shared_example(server, dataset):
    app, home, _ = server
    client = new_client(app)
    register(client, 'old@example.org')
    old = upload(client, dataset, 'old')
    recent = upload(client, dataset, 'recent')
    job = wait(client, client.post('/api/jobs/train', json={'dataset_id': old['id'], **TRAIN}).json()['id'])

    # A shared example data set, as "Load example data" creates it
    store = app.state.datasets
    example_id, folder = store.new_folder()
    for stem in ('labeled_data', 'data_labels'):
        shutil.copy(dataset / f'{stem}.csv', folder / f'{stem}.csv')
    store.finalize(example_id, 'Example', public=True)

    long_ago = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat(timespec='seconds')
    for path in (home / 'datasets' / old['id'] / 'dataset.json', home / 'jobs' / job['id'] / 'job.json',
                 home / 'datasets' / example_id / 'dataset.json'):
        record = json.loads(path.read_text())
        record['created'] = long_ago
        path.write_text(json.dumps(record))
    app.state.purge_expired()

    assert client.get(f"/api/datasets/{old['id']}").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert not (home / 'jobs' / job['id']).exists() and not (home / 'datasets' / old['id']).exists()
    assert client.get(f"/api/datasets/{recent['id']}").status_code == 200
    shared = client.get(f'/api/datasets/{example_id}').json()
    assert shared['public'] and shared['expires'] is None
    assert client.delete(f'/api/datasets/{example_id}').status_code == 403  # users cannot delete it


def test_change_password_and_delete_account(server, dataset):
    app, home, _ = server
    client = new_client(app)
    register(client, 'dana@example.org', 'first password')
    assert client.post('/api/account/password', json={'current_password': 'nope', 'new_password': 'second password'}) \
        .status_code == 403
    assert client.post('/api/account/password',
                       json={'current_password': 'first password', 'new_password': 'second password'}).status_code == 200
    assert client.get('/api/me').status_code == 200  # still logged in on this device
    other = new_client(app)
    assert other.post('/api/auth/login', json={'email': 'dana@example.org', 'password': 'first password'}) \
        .status_code == 401
    data = upload(client, dataset)
    assert client.post('/api/account/delete', json={'password': 'nope'}).status_code == 403
    assert client.post('/api/account/delete', json={'password': 'second password'}).status_code == 200
    assert not (home / 'datasets' / data['id']).exists()
    assert other.post('/api/auth/login', json={'email': 'dana@example.org', 'password': 'second password'}) \
        .status_code == 401


def test_password_reset_needs_email_set_up(server):
    client = new_client(server[0])
    assert not client.get('/api/options').json()['password_reset']
    assert client.post('/api/auth/password-reset', json={'email': 'alice@example.org'}).status_code == 503


def test_password_reset_by_email(tmp_path, monkeypatch):
    monkeypatch.setenv('ALPHAML_SMTP_HOST', 'smtp.example.org')
    monkeypatch.setenv('ALPHAML_SMTP_FROM', 'alphaML <noreply@example.org>')
    sent = []
    monkeypatch.setattr('alphaml.web.app.send_mail', lambda settings, to, subject, body: sent.append((to, body)) or True)
    app = create_app(home=tmp_path, auth=True, allowed_hosts=['testserver'])
    with TestClient(app, headers=HEADERS) as client:
        assert client.get('/api/options').json()['password_reset']
        register(client, 'fay@example.org', 'old password 1')
        client.post('/api/auth/logout')

        # Same reply whether or not the address has an account; only the account gets an email
        unknown = client.post('/api/auth/password-reset', json={'email': 'nobody@example.org'})
        known = client.post('/api/auth/password-reset', json={'email': 'FAY@example.org'})
        assert unknown.status_code == known.status_code == 200 and unknown.json() == known.json()
        assert [to for to, _ in sent] == ['fay@example.org']
        token = re.search(r'http://testserver/#/reset/([\w-]+)', sent[0][1]).group(1)

        def confirm(token, password):
            return client.post('/api/auth/password-reset/confirm', json={'token': token, 'password': password})

        assert confirm(token, 'short').status_code == 400  # rejected without using up the link
        assert confirm(token, 'new password 1').status_code == 200
        assert client.get('/api/me').json()['email'] == 'fay@example.org'
        assert confirm(token, 'another password 1').status_code == 400  # single use
        assert client.post('/api/auth/login', json={'email': 'fay@example.org', 'password': 'old password 1'}) \
            .status_code == 401

        # A newer link retires older ones
        client.post('/api/auth/password-reset', json={'email': 'fay@example.org'})
        client.post('/api/auth/password-reset', json={'email': 'fay@example.org'})
        older, newer = (re.search(r'#/reset/([\w-]+)', body).group(1) for _, body in sent[1:3])
        assert confirm(older, 'third password 1').status_code == 400
        assert confirm(newer, 'third password 1').status_code == 200

        # At most 3 emails per address and hour
        client.post('/api/auth/password-reset', json={'email': 'fay@example.org'})
        assert len(sent) == 3

        # Links expire
        users = app.state.users
        expired = users.create_reset(users.get('fay@example.org')['id'], minutes=-1)
        assert confirm(expired, 'fourth password 1').status_code == 400


def test_admin_commands(server, dataset, monkeypatch, capsys):
    app, home, _ = server
    monkeypatch.setattr('getpass.getpass', lambda prompt='': 'admin set password')
    cli(['--home', str(home), 'users', 'add', 'erin@example.org'])
    client = new_client(app)
    assert client.post('/api/auth/login', json={'email': 'erin@example.org', 'password': 'admin set password'}) \
        .status_code == 200
    assert client.post('/api/account/consent', json={'accept_terms': True}).status_code == 200
    data = upload(client, dataset)

    # Data from before accounts were switched on belongs to nobody until it is claimed
    unowned = home / 'datasets' / data['id'] / 'dataset.json'
    record = json.loads(unowned.read_text())
    record['owner'] = None
    unowned.write_text(json.dumps(record))

    def own_datasets():
        return [d['id'] for d in client.get('/api/datasets').json() if not d['public']]

    assert own_datasets() == []
    cli(['--home', str(home), 'users', 'claim', 'erin@example.org'])
    assert own_datasets() == [data['id']]

    cli(['--home', str(home), 'users', 'list'])
    assert 'erin@example.org' in capsys.readouterr().out
    monkeypatch.setattr('getpass.getpass', lambda prompt='': 'reset by admin')
    cli(['--home', str(home), 'users', 'reset-password', 'erin@example.org'])
    assert client.get('/api/me').status_code == 401  # sessions were ended
    cli(['--home', str(home), 'users', 'delete', 'erin@example.org', '--yes'])
    assert not (home / 'datasets' / data['id']).exists()
    assert client.post('/api/auth/login', json={'email': 'erin@example.org', 'password': 'reset by admin'}) \
        .status_code == 401

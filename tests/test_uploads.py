import json
import random
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from alphaml.web import create_app
from alphaml.web.datasets import _columns, _count_rows
from alphaml.web.uploads import TableStats

from conftest import make_dataset

HEADERS = {'X-AlphaML': '1'}
MB = 1 << 20


# ------------------------- statistics while uploading ------------------------- #

@pytest.mark.parametrize('text', [
    'id,a,b\nS1,1,2\nS2,3,4\n',                       # plain
    'id,a,b\r\nS1,1,2\r\n"S 2",3,4',                  # Windows line ends, quotes, no final newline
    'id\tgene1\tgene2\nx1\t0.5\t1\nx2\t2\t3\n\n',     # tabs and a trailing blank line
])
def test_table_stats_match_reading_the_file(tmp_path, text):
    sep = '\t' if '\t' in text else ','
    path = tmp_path / ('table.tsv' if sep == '\t' else 'table.csv')
    path.write_bytes(text.encode())
    expected = {'rows': _count_rows(path), 'columns': len(_columns(path)) - 1,
                'ids': [i for i in pd.read_csv(path, sep=sep, usecols=[0]).iloc[:, 0].astype(str) if i != 'nan']}
    data = text.encode()
    for _ in range(20):  # cut into random chunks, down to one byte
        stats, pos = TableStats(sep), 0
        while pos < len(data):
            size = random.randint(1, 5)
            stats.feed(data[pos:pos + size])
            pos += size
        assert stats.result() == expected


def test_table_stats_on_a_real_data_file(dataset):
    path = dataset / 'labeled_data.csv'
    data = path.read_bytes()
    stats = TableStats(',')
    for pos in range(0, len(data), 4099):
        stats.feed(data[pos:pos + 4099])
    result = stats.result()
    assert result['rows'] == _count_rows(path) == 90
    assert result['columns'] == len(_columns(path)) - 1 == 120
    assert result['ids'] == list(pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str))


# ------------------------------- chunked uploads ------------------------------ #

@pytest.fixture(scope='module')
def big_dataset(tmp_path_factory):
    # A few MB, so that 1 MB chunks split each data file several times
    return make_dataset(tmp_path_factory.mktemp('big'), n_samples=160, n_features=1500)


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv('ALPHAML_UPLOAD_CHUNK_MB', '1')
    mp.setenv('ALPHAML_MAX_UPLOAD_MB', '20')
    app = create_app(home=tmp_path_factory.mktemp('home'), auth=False, allowed_hosts=['testserver'])
    mp.undo()
    with TestClient(app, headers=HEADERS) as c:
        yield c


def chunked_upload(client, path):
    data = path.read_bytes()
    start = client.post('/api/uploads', json={'filename': path.name, 'size': len(data)})
    assert start.status_code == 201, start.text
    upload_id, chunk = start.json()['id'], start.json()['chunk_size']
    for offset in range(0, len(data), chunk):
        reply = client.put(f'/api/uploads/{upload_id}?offset={offset}', content=data[offset:offset + chunk])
        assert reply.status_code == 200, reply.text
    assert client.get(f'/api/uploads/{upload_id}').json() == {'received': len(data), 'size': len(data)}
    return upload_id


def test_chunked_data_set_matches_a_single_request_upload(client, big_dataset):
    stems = ('labeled_data', 'data_labels', 'unlabeled_data')
    assert (big_dataset / 'labeled_data.csv').stat().st_size > 3 * MB
    ids = {stem: chunked_upload(client, big_dataset / f'{stem}.csv') for stem in stems}
    chunked = client.post('/api/datasets/from-uploads', json={'name': 'chunked', 'files': ids})
    assert chunked.status_code == 201, chunked.text

    files = {stem: (f'{stem}.csv', (big_dataset / f'{stem}.csv').read_bytes(), 'text/csv') for stem in stems}
    whole = client.post('/api/datasets', data={'name': 'whole'}, files=files)
    assert whole.status_code == 201, whole.text
    for key in ('n_samples', 'n_features', 'n_matched_labels', 'label_columns'):
        assert chunked.json()[key] == whole.json()[key], key
    for stem in stems:
        assert chunked.json()['files'][stem]['rows'] == whole.json()['files'][stem]['rows']
        assert chunked.json()['files'][stem].get('columns') == whole.json()['files'][stem].get('columns')
    # A used upload is gone
    assert client.get(f"/api/uploads/{ids['labeled_data']}").status_code == 404


def test_chunk_order_retries_and_limits(client, big_dataset):
    data = (big_dataset / 'labeled_data.csv').read_bytes()
    upload_id = client.post('/api/uploads', json={'filename': 'x.csv', 'size': len(data)}).json()['id']
    first = data[:MB]
    assert client.put(f'/api/uploads/{upload_id}?offset=0', content=first).json()['received'] == MB
    # Sending the same chunk again (a retry after a lost reply) is accepted once
    assert client.put(f'/api/uploads/{upload_id}?offset=0', content=first).json()['received'] == MB
    gap = client.put(f'/api/uploads/{upload_id}?offset={2 * MB}', content=data[2 * MB:3 * MB])
    assert gap.status_code == 409 and str(MB) in gap.text
    assert client.put(f'/api/uploads/{upload_id}?offset={MB}', content=b'x' * (MB + 1)).status_code == 413
    # An unfinished upload cannot be used
    unfinished = client.post('/api/datasets/from-uploads', json={'name': 'u', 'files': {'labeled_data': upload_id}})
    assert unfinished.status_code == 400 and 'completely' in unfinished.text
    assert client.delete(f'/api/uploads/{upload_id}').status_code == 204
    assert client.get(f'/api/uploads/{upload_id}').status_code == 404

    assert client.post('/api/uploads', json={'filename': 'big.csv', 'size': 21 * MB}).status_code == 413
    assert client.post('/api/uploads', json={'filename': 'x.exe', 'size': 10}).status_code == 400
    other = client.post('/api/uploads', json={'filename': 'y.csv', 'size': 5}).json()['id']
    assert client.post('/api/datasets/from-uploads', json={'name': 'n', 'files': {'secret': other}}) \
        .status_code == 422


def test_prediction_with_chunked_test_data(client, dataset):
    files = {s: (f'{s}.csv', (dataset / f'{s}.csv').read_bytes(), 'text/csv') for s in ('labeled_data', 'data_labels')}
    data = client.post('/api/datasets', data={'name': 'p'}, files=files).json()
    train = client.post('/api/jobs/train', json={
        'dataset_id': data['id'], 'model_name': 'LASSO', 'param_search': 'Predefined', 'col_label': 'Trametinib',
        'pos_class': 'resistant', 'neg_class': 'sensitive', 'features': 'None'}).json()
    while client.get(f"/api/jobs/{train['id']}").json()['status'] in ('queued', 'running'):
        time.sleep(0.5)
    test_id = chunked_upload(client, dataset / 'test_data.csv')
    job = client.post('/api/jobs/predict', data={'model_job_id': train['id'], 'test_data_upload': test_id})
    assert job.status_code == 201, job.text
    while (result := client.get(f"/api/jobs/{job.json()['id']}").json())['status'] in ('queued', 'running'):
        time.sleep(0.5)
    assert result['status'] == 'succeeded', result['error']
    assert result['summary']['n_samples'] == 15


def test_uploads_are_private_and_abandoned_ones_are_removed(tmp_path):
    app = create_app(home=tmp_path, auth=True, allowed_hosts=['testserver'])
    with TestClient(app, headers=HEADERS) as owner:
        other = TestClient(app, headers=HEADERS)
        for client, email in ((owner, 'up1@example.org'), (other, 'up2@example.org')):
            client.post('/api/auth/register', json={'email': email, 'password': 'upload password', 'accept_terms': True})
        upload_id = owner.post('/api/uploads', json={'filename': 'a.csv', 'size': 3}).json()['id']
        assert other.get(f'/api/uploads/{upload_id}').status_code == 404
        assert other.put(f'/api/uploads/{upload_id}?offset=0', content=b'abc').status_code == 404
        assert other.delete(f'/api/uploads/{upload_id}').status_code == 404
        assert owner.put(f'/api/uploads/{upload_id}?offset=0', content=b'abc').status_code == 200
        assert other.post('/api/datasets/from-uploads', json={'name': 'n', 'files': {'labeled_data': upload_id}}) \
            .status_code == 404

        record_path = tmp_path / 'uploads' / upload_id / 'upload.json'
        record = json.loads(record_path.read_text())
        record['created'] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(timespec='seconds')
        record_path.write_text(json.dumps(record))
        app.state.purge_expired()
        assert owner.get(f'/api/uploads/{upload_id}').status_code == 404
        assert not (tmp_path / 'uploads' / upload_id).exists()

import re
import time

import pytest
from fastapi.testclient import TestClient

from alphaml.web import create_app


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    app = create_app(home=tmp_path_factory.mktemp('home'), max_jobs=2, auth=False, allowed_hosts=['testserver'])
    with TestClient(app, headers={'X-AlphaML': '1'}) as c:
        yield c


def wait(client, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(0.5)
    log = client.get(f'/api/jobs/{job_id}/log').json()['text'][-2000:]
    raise AssertionError(f"job {job_id} did not finish: {job['status']} {job.get('progress')}\n{log}")


def upload(client, dataset):
    files = {stem: (f'{stem}.csv', (dataset / f'{stem}.csv').read_bytes(), 'text/csv')
             for stem in ('labeled_data', 'data_labels', 'unlabeled_data')}
    response = client.post('/api/datasets', data={'name': 'Synthetic'}, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_ui_and_options(client):
    page = client.get('/')
    assert 'alphaML 2.0' in page.text and page.headers['cache-control'] == 'no-cache'
    # Every version of the scripts gets its own address, so caches never serve an old one
    script = re.search(r'/static/app\.js\?v=([0-9a-f]{12})"', page.text)
    assert script and client.get(f'/static/app.js?v={script.group(1)}').status_code == 200
    assert re.search(r'/static/styles\.css\?v=[0-9a-f]{12}"', page.text)
    options = client.get('/api/options').json()
    assert options['version'] == '0.2.0' and options['train_defaults']['model_name'] == 'AutoML'
    assert options['models'][0] == 'AutoML' and 'Test_Briefly' not in options['models']
    assert 'TabNet' not in options['models'] and 'ModelX' not in options['feature_methods']


def test_dataset_upload_summary(client, dataset):
    record = upload(client, dataset)
    assert record['n_samples'] == 90 and record['n_features'] == 120 and record['n_matched_labels'] == 90
    column = next(c for c in record['label_columns'] if c['name'] == 'Trametinib')
    assert {v['value'] for v in column['values']} == {'resistant', 'sensitive', 'unknown'}
    assert client.get('/api/datasets').json()[0]['id'] == record['id']


def test_dataset_upload_rejects_bad_files(client):
    files = {'labeled_data': ('x.xlsx', b'1', 'application/octet-stream'),
             'data_labels': ('y.csv', b'a,b\n1,2\n', 'text/csv')}
    assert client.post('/api/datasets', data={'name': 'bad'}, files=files).status_code == 400


def test_train_then_predict(client, dataset):
    record = upload(client, dataset)
    response = client.post('/api/jobs/train', json={
        'dataset_id': record['id'], 'model_name': 'AutoML', 'automl_models': ['LASSO', 'SVC', 'RandomForest'],
        'automl_top_k': 2, 'automl_metric': 'roc_auc', 'n_trials': 2, 'cv': 3, 'col_label': 'Trametinib',
        'pos_class': 'resistant', 'neg_class': 'sensitive', 'features': 'MedianMAD', 'run_feat_imp': True})
    assert response.status_code == 201, response.text
    job = wait(client, response.json()['id'])
    assert job['status'] == 'succeeded', job['error']
    names = {f['name'] for f in job['files']}
    assert any(n.endswith('_bundle.pkl') for n in names) and 'summary.json' in names
    assert 'Trametinib_no_AutoML_leaderboard.csv' in names
    assert job['summary']['scores'][0]['Metric'] == 'Accuracy'
    assert job['summary']['selected_model'] == job['summary']['leaderboard'][0]['Model']

    image = next(f['name'] for f in job['files'] if f['kind'] == 'image')
    assert client.get(f"/api/jobs/{job['id']}/files/{image}").headers['content-type'] == 'image/png'
    scores = next(f['name'] for f in job['files'] if f['name'].endswith('_scores.csv'))
    assert client.get(f"/api/jobs/{job['id']}/preview/{scores}").json()['columns'][1] == 'Train'
    assert client.get(f"/api/jobs/{job['id']}/files/..%2Fjob.json").status_code == 404
    archive = client.get(f"/api/jobs/{job['id']}/archive")
    assert archive.headers['content-type'] == 'application/zip' and len(archive.content) > 1000
    assert 'AutoML selected' in client.get(f"/api/jobs/{job['id']}/log").json()['text']

    test_file = {'test_data': ('test.csv', (dataset / 'test_data.csv').read_bytes(), 'text/csv')}
    response = client.post('/api/jobs/predict', data={'model_job_id': job['id'], 'run_lime': 'true'},
                           files=test_file)
    assert response.status_code == 201, response.text
    prediction = wait(client, response.json()['id'])
    assert prediction['status'] == 'succeeded', prediction['error']
    assert prediction['summary']['n_samples'] == 15
    assert any(f['name'].endswith('_test_LIME.pdf') for f in prediction['files'])


def test_cancel_and_delete(client, dataset):
    record = upload(client, dataset)
    response = client.post('/api/jobs/train', json={
        'dataset_id': record['id'], 'model_name': 'SVC', 'param_search': 'Optuna', 'n_trials': 5000,
        'col_label': 'Trametinib', 'pos_class': 'resistant', 'neg_class': 'sensitive', 'features': 'None'})
    job_id = response.json()['id']
    deadline = time.time() + 60
    while client.get(f'/api/jobs/{job_id}').json()['status'] == 'queued' and time.time() < deadline:
        time.sleep(0.3)
    assert client.delete(f"/api/datasets/{record['id']}").status_code == 409  # in use
    assert client.post(f'/api/jobs/{job_id}/cancel').json()['status'] == 'cancelled'
    assert client.delete(f'/api/jobs/{job_id}').status_code == 204
    assert client.get(f'/api/jobs/{job_id}').status_code == 404
    assert client.delete(f"/api/datasets/{record['id']}").status_code == 204


def test_validation_errors(client):
    bad = client.post('/api/jobs/train', json={'dataset_id': 'abc', 'model_name': 'TabNet', 'col_label': 'x',
                                               'pos_class': 'a', 'neg_class': 'b'})
    assert bad.status_code == 422
    assert client.post('/api/jobs/train', json={'dataset_id': 'ffffffffffff', 'col_label': 'x', 'pos_class': 'a',
                                                'neg_class': 'b', 'threshold': 'rm -rf'}).status_code == 422
    for automl in ({'automl_models': []}, {'automl_models': ['TabNet']}, {'automl_metric': 'accuracy'}):
        assert client.post('/api/jobs/train', json={'dataset_id': 'ffffffffffff', 'col_label': 'x', 'pos_class': 'a',
                                                    'neg_class': 'b', **automl}).status_code == 422

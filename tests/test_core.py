import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef
from sklearn.random_projection import GaussianRandomProjection

from alphaml import predict, train
from alphaml.utils import feature_selection as fs
from alphaml.config import ALGORITHMS
from alphaml.models import MODELS
from alphaml.utils.metrics import binary_metrics, fold_error, kappa_mcc_error
from alphaml.utils.prep import encode_labels, normalize_samples


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.0, size=(40, 30))
    x[:, 5] = x[:, 4] * 2 + rng.normal(0, 0.01, 40)  # highly correlated pair
    x[:, 7] = 3.0  # constant feature
    return pd.DataFrame(x, columns=[f'g{i}' for i in range(30)])


# ------------------- vectorized methods match the originals ------------------- #

def test_remove_highly_correlated_matches_original(frame):
    corr = frame.corr()
    expected = []
    for gene in corr.columns:
        if not any(abs(corr.loc[gene, kept]) >= 0.9 for kept in expected):
            expected.append(gene)
    assert fs.remove_highly_correlated_feat(frame, block_size=4) == expected
    assert 'g5' not in expected and 'g7' in expected


def test_select_by_clustering_matches_original(frame):
    df_t = frame.T
    kmeans = KMeans(n_clusters=8, random_state=0).fit(df_t)
    expected = []
    for cluster_id in range(8):
        members = np.where(kmeans.labels_ == cluster_id)[0]
        distances = [np.linalg.norm(df_t.iloc[i] - kmeans.cluster_centers_[cluster_id]) for i in members]
        expected.append(df_t.index[members[int(np.argmin(distances))]])
    assert fs.select_by_clustering(frame, n_clusters=8) == expected


def test_random_projection_matches_original(frame):
    data = frame.drop(columns='g7')  # np.corrcoef gives NaN for constant columns
    n_components = math.floor(4 * np.log10(data.shape[1]) / (0.5 ** 2 / 2 - 0.5 ** 3 / 3))
    projected = GaussianRandomProjection(n_components=n_components, random_state=3).fit_transform(data)
    corr = np.corrcoef(data.T, projected.T)[:data.shape[1], data.shape[1]:]
    expected = pd.Series(np.abs(corr).sum(axis=1), index=data.columns).sort_values(ascending=False).head(10)
    assert fs.select_by_random_projection(data, top_n_genes_rp=10, random_state=3) == list(expected.index)


def test_select_by_mad_matches_original(frame):
    mads = frame.apply(lambda c: np.median(np.abs(c - np.median(c))), axis=0)
    assert fs.select_by_mad(frame) == list(frame.columns[mads > np.median(mads)])


def test_hvf_returns_feature_names(frame):
    selected = fs.hvf(np.log2(frame + 1), n_top_features=10)
    assert 10 <= len(selected) <= 11 and set(selected) <= set(frame.columns)


def test_iterative_selector_refuses_large_inputs(frame):
    with pytest.raises(ValueError, match='limited to'):
        fs.iterative_feature_selector(frame, np.arange(40) % 2, max_input_features=20)


def test_config_lists_every_model():
    assert ALGORITHMS == list(MODELS)


# ---------------------------------- metrics ---------------------------------- #

def test_mcc_matches_sklearn():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    assert binary_metrics(y_true, y_pred)['MCC'] == pytest.approx(matthews_corrcoef(y_true, y_pred))


def test_kappa_mcc_error_treats_worse_than_chance_as_chance():
    y = np.array([1, 0, 1, 0, 1, 0])
    assert kappa_mcc_error(y, y) == pytest.approx(0)
    assert kappa_mcc_error(y, 1 - y) == pytest.approx(1)


# ---------------------------------- labels ----------------------------------- #

def test_encode_labels_aligns_and_drops_unknown():
    data = pd.DataFrame({'g': [1.0, 2.0, 3.0, 4.0]}, index=['a', 'b', 'c', 'd'])
    labels = pd.DataFrame({'drug': [' Resistant', 'sensitive', 'other', 'resistant']}, index=['d', 'c', 'b', 'a'])
    data_out, y = encode_labels(data, labels, 'drug', 'resistant', 'sensitive')
    assert list(y.index) == ['a', 'c', 'd'] and list(y) == [1, 0, 1]
    assert list(data_out.index) == ['a', 'c', 'd']


def test_normalize_samples_is_row_wise():
    df = pd.DataFrame([[1.0, 3.0, 5.0], [2.0, 2.0, 2.0]])
    assert normalize_samples(df, 'min_max').values.tolist() == [[0, 0.5, 1], [0, 0, 0]]
    assert np.allclose(normalize_samples(df, 'standardization').iloc[0].std(ddof=0), 1)


# ------------------------------ end to end ----------------------------------- #

def test_train_and_predict(dataset, tmp_path):
    out = tmp_path / 'train'
    summary = train(dataset / 'labeled_data.csv', dataset / 'data_labels.csv', out,
                    data_for_feature_selection=dataset / 'data_for_feature_selection.csv',
                    unlabeled_data=dataset / 'unlabeled_data.csv', model_name='LASSO', param_search='Predefined',
                    features='MedianMAD', features2='RemoveHighCorrFeat', n_trials=2, cv=3,
                    run_feat_imp=True, run_roc=True)
    assert summary['n_samples'] == 89  # one sample had an unknown label
    assert np.isfinite(summary['pscore']) and 0 <= summary['roc_auc'] <= 1
    prefix = 'Trametinib_no_Predefined_LASSO_fit_No'
    for suffix in ('bundle.pkl', 'scores.csv', 'test_prediction.xlsx', 'confusion_matrix.png',
                   'valid_accuracy_curve.png', 'AUC_curve.png', 'global_feat_imp.csv', 'unlabeled_prediction.xlsx',
                   'parameters.json'):
        assert (out / f'{prefix}_{suffix}').is_file(), suffix
    assert json.loads((out / 'summary.json').read_text())['pscore'] == pytest.approx(summary['pscore'])

    # Prediction with the bundle: features, normalization and class names come from the model
    result = predict(dataset / 'test_data.csv', tmp_path / 'pred', model=out / f'{prefix}_bundle.pkl')
    assert result['n_samples'] == 15 and set(result['class_counts']) <= {'resistant', 'sensitive'}
    with pytest.raises(ValueError, match='not an alphaML model bundle'):
        predict(dataset / 'test_data.csv', tmp_path / 'pred', model={'model': None})


def test_xgboost_search_then_roc(dataset, tmp_path):
    # Regression: parallel XGBoost folds with changing worker counts used to livelock in OpenMP
    summary = train(dataset / 'labeled_data.csv', dataset / 'data_labels.csv', tmp_path, model_name='XGBoost',
                    param_search='Optuna', n_trials=5, cv=5, features='MedianMAD', run_roc=True)
    assert 0 <= summary['roc_auc'] <= 1


def test_predict_reports_missing_features(dataset, tmp_path):
    out = tmp_path / 'train'
    train(dataset / 'labeled_data.csv', dataset / 'data_labels.csv', out, model_name='Ridge',
          param_search='Predefined', features='None')
    test = pd.read_csv(dataset / 'test_data.csv', index_col=0).drop(columns=['GENE1', 'GENE2'])
    with pytest.raises(ValueError, match='2 of the 120 model features are missing'):
        predict(test, tmp_path / 'pred', model=out / 'Trametinib_no_Predefined_Ridge_fit_No_bundle.pkl')


def test_automl(dataset, tmp_path):
    summary = train(dataset / 'labeled_data.csv', dataset / 'data_labels.csv', tmp_path / 'out', model_name='AutoML',
                    automl_models=['LASSO', 'RandomForest', 'SVC'], automl_top_k=2, n_trials=3, cv=3,
                    automl_time_budget=0, automl_ensemble=True, features='MedianMAD')
    board = summary['leaderboard']
    stages = [row['Stage'].split(' ')[0] for row in board]
    assert stages.count('screened') == 3 and stages.count('tuned') == 2 and stages.count('ensemble') == 1
    assert [row['Rank'] for row in board] == list(range(1, len(board) + 1))
    assert [row['Objective'] for row in board] == sorted(row['Objective'] for row in board)
    assert summary['selected_model'] == board[0]['Model']
    assert summary['scores'][0]['Test'] > 0.6  # the synthetic data has real signal
    bundle = next((tmp_path / 'out').glob('*_bundle.pkl'))
    assert bundle.name.startswith('Trametinib_no_AutoML_')
    assert (tmp_path / 'out' / 'Trametinib_no_AutoML_leaderboard.png').is_file()
    assert predict(dataset / 'test_data.csv', tmp_path / 'pred', model=bundle)['n_samples'] == 15


def test_ensemble_recipe(dataset, tmp_path):
    from alphaml.models.engine import roc_auc
    from alphaml.models.folds import build

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 10))
    y = (x[:, 0] + rng.normal(0, 0.5, 80) > 0).astype(int)
    recipe = [('LASSO', {'alpha': 0.001, 'max_iter': 1000, 'random_state': 1}),
              ('RandomForest', {'criterion': 'gini', 'max_depth': 4, 'n_estimators': 50, 'random_state': 1})]
    model = build(recipe, x, y, final=True)
    members = [m.predict_proba(x)[:, 1] for m in model.estimators_]
    assert np.allclose(model.predict_proba(x)[:, 1], np.mean(members, axis=0))
    assert pickle.loads(pickle.dumps(model)).predict(x).tolist() == model.predict(x).tolist()
    assert roc_auc(recipe, x, y, n_trials=2, cv=3, random_state=1, prefix=str(tmp_path / 'ens')) > 0.8


def test_fold_error_matches_original_control_fitting():
    # The original custom_score: sqrt(|train error - test error| * test error), or test error when equal
    assert fold_error(0.9, 0.7, control_fitting=True) == pytest.approx((abs(0.1 - 0.3) * 0.3) ** 0.5)
    assert fold_error(0.7, 0.7, control_fitting=True) == pytest.approx(0.3)
    assert fold_error(0.9, 0.7, control_fitting=False) == pytest.approx(0.3)

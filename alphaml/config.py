"""
Option lists and defaults shared by the Python API, the web API and the web UI.
"""

# Algorithms (names match models/specs.py). 'AutoML' chooses among them automatically.
ALGORITHMS = ['AdaBoost', 'CatBoost', 'ElasticNet', 'ExtraTrees', 'GradientBoosting', 'HistGradientBoosting',
              'LASSO', 'LightGBM', 'MLPC', 'RandomForest', 'Ridge', 'SVC', 'NuSVC', 'XGBoost']
MODEL_NAMES = ['AutoML'] + ALGORITHMS
SAMPLING_METHODS = ['no', 'over', 'under']
PARAM_SEARCH_METHODS = ['Optuna', 'Bayes', 'Grid', 'Predefined']
NORMALIZATION_METHODS = ['min_max', 'standardization', 'None']
LOG_TYPES = ['log2', 'log10', 'ln', 'None']

# Cross-validated scores AutoML ranks models by (all in [0, 1], higher is better)
RANK_METRICS = {
    'kappa_mcc': "Cohen's kappa × MCC",
    'roc_auc': 'ROC AUC',
    'balanced_accuracy': 'Balanced accuracy',
    'f1': 'F1 score',
}

# Feature selection methods offered in the UI. 'IterativeFeatSel' is also accepted by the API but it is
# combinatorial (all 1-3 feature subsets), so it is only practical on a handful of pre-selected features.
FEATURE_METHODS = ['HVF', 'MedianMAD', 'SelByPCA', 'RandomProjection', 'SelByClustering', 'SelByNMF',
                   'SelectByRF', 'RecursiveFeatElim', 'SeqFeatSel', 'RemoveHighCorrFeat', 'Suggested', 'None']
API_FEATURE_METHODS = FEATURE_METHODS + ['IterativeFeatSel']

TRAIN_DEFAULTS = dict(
    model_name='AutoML',
    automl_models=list(ALGORITHMS),
    automl_metric='kappa_mcc',
    automl_top_k=3,
    automl_time_budget=10.0,
    automl_ensemble=True,
    sampling_method='no',
    param_search='Optuna',
    control_fitting=False,
    col_label='Trametinib',
    pos_class='resistant',
    neg_class='sensitive',
    test_size=0.2,
    random_state=12,
    n_trials=50,
    cv=5,
    features='HVF',
    features2='None',
    features3='None',
    features4='None',
    n_top_features=0,
    min_disp=1.5,
    log_type='log2',
    top_n_genes_pca=500,
    top_n_genes_rp=500,
    n_clusters=500,
    n_components=500,
    max_features=0,
    threshold='1.5*mean',
    min_sel_features_rfe=10,
    min_sel_features_sfs=10,
    normalization='min_max',
    run_feat_imp=False,
    run_shap=False,
    run_lime=False,
    run_roc=False,
)

PREDICT_DEFAULTS = dict(
    normalization=None,  # None: as stored in the model
    run_shap=False,
    run_lime=False,
)


def yes_no(flag):
    return 'Yes' if flag else 'No'

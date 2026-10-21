"""
Model specifications.

Each model is described once: how to build the estimator, its Optuna / Bayes / Grid search spaces, its
predefined parameters, and how to draw its training curve. The shared engine (engine.py) does the rest.
"""
import importlib
from dataclasses import dataclass
from typing import Callable, Optional

from ..utils.sampling import scale_pos_weight_


def _cls(module, name):
    """Import model libraries on first use, so worker processes only load the library they need."""
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class ModelSpec:
    name: str                   # option name, e.g. 'XGBoost'
    algorithm: str              # used in result file names, e.g. 'XGBClassifier'
    make: Callable              # make(params, y_train, final=False) -> estimator
    optuna: Callable            # optuna(trial) -> params
    bayes: Callable             # bayes() -> scikit-optimize search space
    grid: dict
    predefined: dict
    curve: str                  # 'evals' (booster eval sets), 'staged', 'trees' or 'iter' (refit per iteration)
    iter_param: str = 'n_estimators'
    # Run CV folds in parallel processes. Off for libraries with their own OpenMP thread pools (XGBoost, LightGBM,
    # CatBoost): they use many cores per fit, and their pools can livelock in reused workers.
    parallel_folds: bool = True
    threads_param: Optional[str] = None    # estimator argument setting its thread count, if it has one
    # > 0: at most one thread per this many training samples, for models whose threads cost more than they
    # save on small data (HistGradientBoosting runs several times slower with 16 threads than with 1 below ~5000)
    min_samples_per_thread: int = 0
    sample_weight: bool = False     # pass balanced sample weights to fit()
    eval_fit: Optional[Callable] = None    # (x, y, x_test, y_test) -> extra fit() kwargs for curve == 'evals'
    eval_curve: Optional[Callable] = None  # (model) -> (xlabel, {label: values}) for curve == 'evals'


def _space():
    from skopt.space import Categorical, Integer, Real
    return Categorical, Integer, Real


LEARNING_RATES = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]
FRACTIONS = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
HALF_STEPS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
N_ESTIMATORS_GRID = [100, 125, 150, 175, 200, 225, 250, 275]


def _eval_set(x, y, x_test, y_test):
    return dict(eval_set=[(x, y), (x_test, y_test)])


def _four_curves(labels, train_a, test_a, train_b, test_b):
    return {f'Train {labels[0]}': train_a, f'Test {labels[0]}': test_a,
            f'Train {labels[1]}': train_b, f'Test {labels[1]}': test_b}


# ------------------------------- AdaBoost -------------------------------- #

def _ada(params, y, final=False):
    base = _cls('sklearn.tree', 'ExtraTreeClassifier')(criterion='log_loss', max_depth=4, class_weight='balanced')
    return _cls('sklearn.ensemble', 'AdaBoostClassifier')(**params, estimator=base)


def _ada_optuna(trial):
    return dict(learning_rate=trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.05, 0.1, 0.5]),
                n_estimators=trial.suggest_int('n_estimators', 100, 300, step=10))


def _ada_bayes():
    Categorical, _, _ = _space()
    return {'learning_rate': Categorical([0.005, 0.01, 0.05, 0.1, 0.5]),
            'n_estimators': Categorical([100, 300])}


# ------------------------------- CatBoost -------------------------------- #

def _cat(params, y, final=False):
    extra = dict(eval_metric='BalancedAccuracy', custom_metric=['BrierScore']) if final else {}
    return _cls('catboost', 'CatBoostClassifier')(**params, verbose=False, auto_class_weights='Balanced',
                                                  allow_writing_files=False, **extra)


def _cat_optuna(trial):
    return dict(learning_rate=trial.suggest_float('learning_rate', 0.005, 0.05, step=0.005),
                depth=trial.suggest_int('depth', 5, 8, step=1),
                subsample=trial.suggest_float('subsample', 0.65, 0.9, step=0.05),
                colsample_bylevel=trial.suggest_float('colsample_bylevel', 0.65, 0.9, step=0.05),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0, 4, step=0.5),
                random_strength=trial.suggest_int('random_strength', 1, 4, step=1),
                iterations=trial.suggest_int('iterations', 100, 300, step=10))


def _cat_bayes():
    Categorical, Integer, _ = _space()
    return {'learning_rate': Categorical(LEARNING_RATES),
            'depth': Integer(5, 8),
            'subsample': Categorical(FRACTIONS),
            'colsample_bylevel': Categorical(FRACTIONS),
            'l2_leaf_reg': Categorical(HALF_STEPS),
            'random_strength': Integer(1, 4),
            'iterations': Integer(100, 300)}


def _cat_eval_fit(x, y, x_test, y_test):
    # Keep every iteration: the test set is only monitored, never used to pick the model
    return dict(eval_set=[(x, y), (x_test, y_test)], use_best_model=False)


def _cat_curve(model):
    results = model.get_evals_result()
    train, test = results['validation_0'], results['validation_1']

    def pick(metrics, prefix):
        return next(v for k, v in metrics.items() if k.startswith(prefix))
    return 'iteration', _four_curves(('Balanced Accuracy', 'Brier Score'),
                                     pick(train, 'BalancedAccuracy'), pick(test, 'BalancedAccuracy'),
                                     pick(train, 'BrierScore'), pick(test, 'BrierScore'))


# --------------------- SGD: ElasticNet, LASSO, Ridge --------------------- #

def _sgd(penalty):
    def make(params, y, final=False):
        return _cls('sklearn.linear_model', 'SGDClassifier')(**params, loss='log_loss', class_weight='balanced',
                                                             penalty=penalty)
    return make


def _sgd_optuna(trial):
    return dict(alpha=trial.suggest_float('alpha', 0.0001, 0.01, step=0.0005),
                max_iter=trial.suggest_int('max_iter', 500, 1500, step=100))


def _sgd_bayes():
    _, Integer, Real = _space()
    return {'alpha': Real(0.0001, 0.01), 'max_iter': Integer(500, 1500)}


def _eln_optuna(trial):
    return dict(alpha=trial.suggest_float('alpha', 0.0001, 0.01, step=0.0005),
                l1_ratio=trial.suggest_float('l1_ratio', 0.1, 0.9, step=0.05),
                max_iter=trial.suggest_int('max_iter', 500, 1500, step=100))


def _eln_bayes():
    _, Integer, Real = _space()
    return {'alpha': Real(0.0001, 0.01), 'l1_ratio': Real(0.1, 0.9), 'max_iter': Integer(500, 1500)}


SGD_GRID = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], 'max_iter': [500, 750, 1000, 1250, 1500]}


# ---------------------- RandomForest and ExtraTrees ---------------------- #

def _forest(name):
    def make(params, y, final=False):
        return _cls('sklearn.ensemble', name)(**params, class_weight='balanced')
    return make


def _forest_optuna(trial):
    return dict(criterion=trial.suggest_categorical('criterion', ['gini', 'log_loss']),
                max_depth=trial.suggest_int('max_depth', 5, 8, step=1),
                n_estimators=trial.suggest_int('n_estimators', 100, 300, step=10))


def _forest_bayes():
    Categorical, Integer, _ = _space()
    return {'criterion': Categorical(['gini', 'log_loss']), 'max_depth': Integer(5, 8),
            'n_estimators': Integer(100, 300)}


FOREST_GRID = {'criterion': ['gini', 'log_loss'], 'max_depth': [6, 7, 8], 'n_estimators': N_ESTIMATORS_GRID}
FOREST_PREDEFINED = {'criterion': 'log_loss', 'max_depth': 6, 'n_estimators': 250}


# --------------------------- GradientBoosting ---------------------------- #

def _gbc(params, y, final=False):
    return _cls('sklearn.ensemble', 'GradientBoostingClassifier')(**params)


def _gbc_optuna(trial):
    return dict(learning_rate=trial.suggest_float('learning_rate', 0.005, 0.05, step=0.005),
                max_depth=trial.suggest_int('max_depth', 5, 8, step=1),
                subsample=trial.suggest_float('subsample', 0.65, 0.9, step=0.05),
                n_estimators=trial.suggest_int('n_estimators', 100, 300, step=10))


def _gbc_bayes():
    Categorical, Integer, _ = _space()
    return {'learning_rate': Categorical(LEARNING_RATES), 'max_depth': Integer(5, 8),
            'subsample': Categorical(FRACTIONS), 'n_estimators': Integer(100, 300)}


# ------------------------- HistGradientBoosting -------------------------- #

def _hgb(params, y, final=False):
    return _cls('sklearn.ensemble', 'HistGradientBoostingClassifier')(**params, class_weight='balanced')


def _hgb_optuna(trial):
    return dict(learning_rate=trial.suggest_float('learning_rate', 0.005, 0.05, step=0.005),
                max_depth=trial.suggest_int('max_depth', 5, 8, step=1),
                l2_regularization=trial.suggest_float('l2_regularization', 0, 4, step=0.5),
                max_iter=trial.suggest_int('max_iter', 100, 300, step=10))


def _hgb_bayes():
    Categorical, Integer, _ = _space()
    return {'learning_rate': Categorical(LEARNING_RATES), 'max_depth': Integer(5, 8),
            'l2_regularization': Categorical(HALF_STEPS), 'max_iter': Integer(100, 300)}


# ------------------------------- LightGBM -------------------------------- #

def _lgb(params, y, final=False):
    extra = dict(metric=['average_precision', 'binary_error']) if final else {}
    return _cls('lightgbm', 'LGBMClassifier')(**params, verbose=-1, class_weight='balanced', **extra)


def _lgb_curve(model):
    train, test = list(model.evals_result_.values())[:2]
    return 'n_estimators', _four_curves(('Average Precision', 'Binary error'),
                                        train['average_precision'], test['average_precision'],
                                        train['binary_error'], test['binary_error'])


# --------------------------------- XGBoost -------------------------------- #

def _xgb(params, y, final=False):
    extra = dict(eval_metric=['map', 'error']) if final else {}
    return _cls('xgboost', 'XGBClassifier')(**params, objective='binary:logistic', verbosity=0,
                                            scale_pos_weight=scale_pos_weight_(y), **extra)


def _xgb_eval_fit(x, y, x_test, y_test):
    return dict(eval_set=[(x, y), (x_test, y_test)], verbose=False)


def _xgb_curve(model):
    results = model.evals_result()
    train, test = results['validation_0'], results['validation_1']
    return 'n_estimators', _four_curves(('Mean Average Precision', 'Binary classification error'),
                                        train['map'], test['map'], train['error'], test['error'])


def _boost_optuna(trial, gamma):
    params = dict(learning_rate=trial.suggest_float('learning_rate', 0.005, 0.05, step=0.005),
                  max_depth=trial.suggest_int('max_depth', 5, 8, step=1),
                  subsample=trial.suggest_float('subsample', 0.65, 0.9, step=0.05),
                  colsample_bytree=trial.suggest_float('colsample_bytree', 0.65, 0.9, step=0.05))
    if gamma:
        params['gamma'] = trial.suggest_float('gamma', 0, 4, step=0.5)
    params.update(reg_alpha=trial.suggest_float('reg_alpha', 0, 4, step=0.5),
                  reg_lambda=trial.suggest_float('reg_lambda', 0, 4, step=0.5),
                  min_child_weight=trial.suggest_int('min_child_weight', 1, 4, step=1),
                  n_estimators=trial.suggest_int('n_estimators', 100, 300, step=10))
    return params


def _boost_bayes(gamma):
    def space():
        Categorical, Integer, _ = _space()
        params = {'learning_rate': Categorical(LEARNING_RATES), 'max_depth': Integer(5, 8),
                  'subsample': Categorical(FRACTIONS), 'colsample_bytree': Categorical(FRACTIONS)}
        if gamma:
            params['gamma'] = Categorical(HALF_STEPS)
        params.update({'reg_alpha': Categorical(HALF_STEPS), 'reg_lambda': Categorical(HALF_STEPS),
                       'min_child_weight': Integer(1, 4), 'n_estimators': Integer(100, 300)})
        return params
    return space


def _boost_grid(gamma):
    grid = {'learning_rate': [0.005, 0.01, 0.05], 'max_depth': [6, 7, 8], 'subsample': [0.75],
            'colsample_bytree': [0.75]}
    if gamma:
        grid['gamma'] = [1, 2]
    grid.update({'reg_alpha': [1, 2], 'reg_lambda': [1, 2], 'min_child_weight': [2, 3],
                 'n_estimators': [150, 175, 200, 225, 250, 275]})
    return grid


def _boost_predefined(gamma):
    params = {'learning_rate': 0.0075, 'max_depth': 6, 'subsample': 0.75, 'colsample_bytree': 0.75}
    if gamma:
        params['gamma'] = 2
    params.update({'reg_alpha': 2.5, 'reg_lambda': 2, 'min_child_weight': 2.5, 'n_estimators': 250})
    return params


# --------------------------------- MLP ----------------------------------- #

HIDDEN_LAYER_SIZES = [16, 25, 32, 50, 64, 100, 128, 200, 256, 400, 512]
MLP_MAX_ITER = [100, 150, 200, 250, 300, 350, 400, 450, 500]


def _mlp(params, y, final=False):
    return _cls('sklearn.neural_network', 'MLPClassifier')(**params)


def _mlp_optuna(trial):
    return dict(hidden_layer_sizes=trial.suggest_categorical('hidden_layer_sizes', HIDDEN_LAYER_SIZES),
                max_iter=trial.suggest_int('max_iter', 100, 500, step=50))


def _mlp_bayes():
    Categorical, _, _ = _space()
    return {'hidden_layer_sizes': Categorical(HIDDEN_LAYER_SIZES), 'max_iter': Categorical(MLP_MAX_ITER)}


# ------------------------------ SVC / NuSVC ------------------------------ #
# Probability calibration (an internal 5-fold CV) only affects predict_proba, not predict. Searches and the
# ROC cross-validation score with predict/decision_function, so only the final model enables it.

def _svm(name):
    def make(params, y, final=False):
        return _cls('sklearn.svm', name)(**params, probability=final, class_weight='balanced')
    return make


def _svc_optuna(trial):
    return dict(C=trial.suggest_float('C', 0.1, 1, step=0.05),
                max_iter=trial.suggest_int('max_iter', 100, 2000, step=100))


def _nusvc_optuna(trial):
    return dict(nu=trial.suggest_float('nu', 0.1, 1, step=0.05),
                max_iter=trial.suggest_int('max_iter', 100, 2000, step=100))


def _svm_bayes(key):
    def space():
        _, Integer, Real = _space()
        return {key: Real(0.1, 1), 'max_iter': Integer(100, 2000)}
    return space


SVM_GRID_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
SVM_GRID_MAX_ITER = [500, 750, 1000, 1025, 1500, 1570, 2000]


SPECS = [
    ModelSpec('AdaBoost', 'AdaBoostClassifier', _ada, _ada_optuna, _ada_bayes,
              grid={'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5], 'n_estimators': N_ESTIMATORS_GRID},
              predefined={'learning_rate': 0.01, 'n_estimators': 200},
              curve='staged'),
    ModelSpec('CatBoost', 'CatBoostClassifier', _cat, _cat_optuna, _cat_bayes,
              grid={'learning_rate': [0.005, 0.01, 0.05], 'depth': [6, 7, 8], 'subsample': [0.75],
                    'colsample_bylevel': [0.75], 'l2_leaf_reg': [1, 2], 'random_strength': [2, 3],
                    'iterations': [150, 175, 200, 225, 250, 275]},
              predefined={'learning_rate': 0.0075, 'depth': 6, 'subsample': 0.75, 'colsample_bylevel': 0.75,
                          'l2_leaf_reg': 2, 'random_strength': 2.5, 'iterations': 250},
              curve='evals', iter_param='iterations', parallel_folds=False, threads_param='thread_count',
              eval_fit=_cat_eval_fit, eval_curve=_cat_curve),
    ModelSpec('ElasticNet', 'ElasticNet', _sgd('elasticnet'), _eln_optuna, _eln_bayes,
              grid={**SGD_GRID, 'l1_ratio': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6]},
              predefined={'alpha': 0.0001, 'max_iter': 1250, 'l1_ratio': 0.2},
              curve='iter', iter_param='max_iter'),
    ModelSpec('ExtraTrees', 'ExtraTreesClassifier', _forest('ExtraTreesClassifier'), _forest_optuna, _forest_bayes,
              grid=FOREST_GRID, predefined=FOREST_PREDEFINED, curve='trees', threads_param='n_jobs'),
    ModelSpec('GradientBoosting', 'GradientBoostingClassifier', _gbc, _gbc_optuna, _gbc_bayes,
              grid={'learning_rate': [0.005, 0.01, 0.05], 'max_depth': [6, 7, 8], 'subsample': [0.75, 0.8, 0.85],
                    'n_estimators': N_ESTIMATORS_GRID},
              predefined={'learning_rate': 0.0075, 'max_depth': 6, 'subsample': 0.75, 'n_estimators': 250},
              curve='staged', sample_weight=True),
    ModelSpec('HistGradientBoosting', 'HistGradientBoostingClassifier', _hgb, _hgb_optuna, _hgb_bayes,
              grid={'learning_rate': [0.005, 0.01, 0.05], 'max_depth': [6, 7, 8], 'l2_regularization': [1, 2],
                    'max_iter': N_ESTIMATORS_GRID},
              predefined={'learning_rate': 0.0075, 'max_depth': 6, 'l2_regularization': 2.5, 'max_iter': 250},
              curve='staged', iter_param='max_iter', min_samples_per_thread=5000),
    ModelSpec('LASSO', 'LASSO', _sgd('l1'), _sgd_optuna, _sgd_bayes,
              grid=SGD_GRID, predefined={'alpha': 0.0001, 'max_iter': 1250},
              curve='iter', iter_param='max_iter'),
    ModelSpec('LightGBM', 'LGBMClassifier', _lgb, lambda trial: _boost_optuna(trial, gamma=False),
              _boost_bayes(gamma=False), grid=_boost_grid(gamma=False), predefined=_boost_predefined(gamma=False),
              curve='evals', parallel_folds=False, threads_param='n_jobs', eval_fit=_eval_set,
              eval_curve=_lgb_curve),
    ModelSpec('MLPC', 'MLPClassifier', _mlp, _mlp_optuna, _mlp_bayes,
              grid={'hidden_layer_sizes': HIDDEN_LAYER_SIZES, 'max_iter': MLP_MAX_ITER},
              predefined={'hidden_layer_sizes': (512, 256, 128), 'max_iter': 500},
              curve='iter', iter_param='max_iter'),
    ModelSpec('RandomForest', 'RandomForestClassifier', _forest('RandomForestClassifier'), _forest_optuna,
              _forest_bayes, grid=FOREST_GRID, predefined=FOREST_PREDEFINED, curve='trees', threads_param='n_jobs'),
    ModelSpec('Ridge', 'Ridge', _sgd('l2'), _sgd_optuna, _sgd_bayes,
              grid=SGD_GRID, predefined={'alpha': 0.0001, 'max_iter': 1250},
              curve='iter', iter_param='max_iter'),
    ModelSpec('SVC', 'SVC', _svm('SVC'), _svc_optuna, _svm_bayes('C'),
              grid={'C': SVM_GRID_VALUES, 'max_iter': SVM_GRID_MAX_ITER},
              predefined={'C': 0.9, 'max_iter': 1500},
              curve='iter', iter_param='max_iter'),
    ModelSpec('NuSVC', 'NuSVC', _svm('NuSVC'), _nusvc_optuna, _svm_bayes('nu'),
              grid={'nu': SVM_GRID_VALUES, 'max_iter': SVM_GRID_MAX_ITER},
              predefined={'nu': 0.5, 'max_iter': 1500},
              curve='iter', iter_param='max_iter'),
    ModelSpec('XGBoost', 'XGBClassifier', _xgb, lambda trial: _boost_optuna(trial, gamma=True),
              _boost_bayes(gamma=True), grid=_boost_grid(gamma=True), predefined=_boost_predefined(gamma=True),
              curve='evals', parallel_folds=False, threads_param='n_jobs', eval_fit=_xgb_eval_fit,
              eval_curve=_xgb_curve),
]

MODELS = {spec.name: spec for spec in SPECS}

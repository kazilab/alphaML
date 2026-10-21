"""
Shared training engine: cross-validation, hyperparameter search, final model, training curves, ROC
cross-validation and feature importance, for any model described in specs.py.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

from ..utils.metrics import KAPPA_MCC_SCORER, fold_error, rank_score
from ..utils.plots import plot_curves, plot_optuna_importance, plot_roc_auc_acc
from .ensemble import SoftVotingEnsemble
from .folds import MAX_THREADS, fit, fit_kwargs, fold_predictions, members, roc_fold
from .specs import MODELS

logger = logging.getLogger('alphaml')

WORST_ERROR = 1.0  # error of a chance-level model


def _plan(recipe, n_tasks):
    """
    (worker processes, threads per fit) for n_tasks folds of a recipe. Parallel folds share the cores between
    them; sequential folds (threads None) give each fit up to folds.MAX_THREADS threads.
    """
    if not all(MODELS[name].parallel_folds for name, _ in members(recipe)):
        return 1, None
    cpus = cpu_count()  # respects container CPU limits
    workers = max(1, min(n_tasks, cpus))
    return workers, max(1, cpus // workers)


def cross_validate(recipe, x, y, splits, final=False):
    """Fold predictions of a recipe (None for infeasible folds), folds in parallel where that is safe."""
    workers, threads = _plan(recipe, len(splits))
    return Parallel(n_jobs=workers)(delayed(fold_predictions)(recipe, x, y, train, valid, final, threads)
                                    for train, valid in splits)


def cv_summary(folds, splits, y, metric, control_fitting):
    """
    Mean and SD of the validation score, train-validation gap, objective (mean fold error, lower is better),
    and mean AUC, accuracy and MCC over folds. Infeasible folds count as errors of 1.
    """
    valid, train, errors, auc, accuracy, mcc = [], [], [], [], [], []
    for fold, (tr, va) in zip(folds, splits):
        if fold is None:
            errors.append(WORST_ERROR)
            continue
        valid.append(rank_score(metric, y[va], fold['pred'], fold['score']))
        train.append(rank_score(metric, y[tr], fold['train_pred'], fold['train_score']))
        errors.append(fold_error(train[-1], valid[-1], control_fitting))
        accuracy.append(accuracy_score(y[va], fold['pred']))
        mcc.append(matthews_corrcoef(y[va], fold['pred']))
        auc.append(roc_auc_score(y[va], fold['score']) if len(np.unique(y[va])) > 1 else np.nan)

    def mean(values):
        return float(np.nanmean(values)) if values and not np.all(np.isnan(values)) else np.nan

    return {'score': mean(valid), 'sd': float(np.nanstd(valid)) if valid else np.nan,
            'gap': mean(train) - mean(valid), 'objective': mean(errors), 'auc': mean(auc),
            'accuracy': mean(accuracy), 'mcc': mean(mcc)}


def predict_single_threaded(model):
    """
    Make a fitted model predict on one thread. Explanations and permutation importance make thousands of small
    predictions, where starting threads costs more than it saves. Returns the model.
    """
    for estimator in model.estimators_ if isinstance(model, SoftVotingEnsemble) else [model]:
        # Only scikit-learn forests are switched: boosters reject or mishandle set_params once fitted
        if type(estimator).__module__.startswith('sklearn') and 'n_jobs' in estimator.get_params():
            estimator.set_params(n_jobs=1)
    return model


# ------------------------- Hyperparameter search ------------------------- #


def optuna_search(spec, x, y, n_trials, cv, random_state, control_fitting, importance_base, progress):
    import optuna
    from optuna.samplers import TPESampler

    splits = list(KFold(n_splits=cv, random_state=random_state, shuffle=True).split(x))

    def objective(trial):
        params = {**spec.optuna(trial), 'random_state': random_state}
        folds = cross_validate((spec.name, params), x, y, splits)
        return cv_summary(folds, splits, y, 'kappa_mcc', control_fitting)['objective']

    def report(study, trial):
        progress('Hyperparameter search', f'Optuna trial {trial.number + 1}/{n_trials}, '
                                          f'best error {study.best_value:.4f}')

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name='Optuna optimization',
                                sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=36000, callbacks=[report])
    plot_optuna_importance(study, importance_base)
    return dict(study.best_params)


def _search_estimator(spec, y, random_state, threads):
    params = {'random_state': random_state}
    if spec.threads_param:
        params[spec.threads_param] = min(threads or MAX_THREADS, MAX_THREADS)
    return spec.make(params, y)


def bayes_search(spec, x, y, n_trials, cv, random_state, progress):
    from skopt import BayesSearchCV

    done = []

    def report(_):
        done.append(1)
        progress('Hyperparameter search', f'Bayes iteration {len(done)}/{n_trials}')

    workers, threads = _plan((spec.name, {}), cv)
    opt = BayesSearchCV(_search_estimator(spec, y, random_state, threads),
                        search_spaces=spec.bayes(),
                        scoring=KAPPA_MCC_SCORER,
                        n_iter=n_trials,
                        cv=cv,
                        n_jobs=workers,
                        random_state=random_state,
                        error_score=-WORST_ERROR,
                        verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt.fit(x, y, callback=report, **fit_kwargs(spec, y))
    return dict(opt.best_params_)


def grid_search(spec, x, y, cv, random_state):
    threads = 1 if spec.parallel_folds else None
    opt = GridSearchCV(_search_estimator(spec, y, random_state, threads),
                       param_grid=spec.grid,
                       scoring=KAPPA_MCC_SCORER,
                       cv=cv,
                       n_jobs=-1 if spec.parallel_folds else 1,
                       error_score=-WORST_ERROR,
                       verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt.fit(x, y, **fit_kwargs(spec, y))
    return dict(opt.best_params_)


def search_parameters(spec, x, y, param_search, n_trials, cv, random_state, control_fitting, prefix, progress):
    logger.info(f'Searching parameters for {spec.algorithm} using {param_search}')
    if param_search == 'Optuna':
        params = optuna_search(spec, x, y, n_trials, cv, random_state, control_fitting,
                               f'{prefix}_optuna_param_importance', progress)
    elif param_search == 'Bayes':
        params = bayes_search(spec, x, y, n_trials, cv, random_state, progress)
    elif param_search == 'Grid':
        params = grid_search(spec, x, y, cv, random_state)
    elif param_search == 'Predefined':
        # Hardcoded parameters from prior experiments, domain knowledge, or literature
        params = dict(spec.predefined)
    else:
        raise ValueError('Please choose a parameter search method from Optuna, Bayes, Grid, and Predefined')
    params['random_state'] = random_state
    return params


# ----------------------------- Final model ------------------------------- #


def fit_final(spec, params, x, y, x_test, y_test):
    extra = spec.eval_fit(x, y, x_test, y_test) if spec.curve == 'evals' else {}
    return fit(spec, params, x, y, final=True, **extra)


def _curves_from_proba(p_train, y, p_test, y_test):
    return {'Train Accuracy': [np.mean((p > 0.5) == y) for p in p_train],
            'Test Accuracy': [np.mean((p > 0.5) == y_test) for p in p_test],
            'Train loss': [np.mean((p - y) ** 2) for p in p_train],  # Brier score
            'Test loss': [np.mean((p - y_test) ** 2) for p in p_test]}


def _cumulative_tree_proba(model, x):
    """Forest probabilities after 1, 2, ... trees (a forest averages its trees' probabilities)."""
    total = np.zeros(len(x))
    out = []
    for k, tree in enumerate(model.estimators_, start=1):
        total += tree.predict_proba(x)[:, 1]
        out.append(total / k)
    return out


def _iteration_curves(spec, params, model, x, y, x_test, y_test, max_points=30):
    """
    Scores after 1..max_iter iterations for estimators without staged predictions.

    Refits at up to max_points iteration budgets up to the point where the final model converged; any larger
    budget gives the same model, so the curve is flat from there on.
    """
    max_iter = int(params[spec.iter_param])
    converged = int(np.max(getattr(model, 'n_iter_', max_iter)))
    last = max(1, min(max_iter, converged))
    steps = np.unique(np.rint(np.geomspace(1, last, num=min(max_points, last))).astype(int))
    curves = {'Train Accuracy': [], 'Test Accuracy': [], 'Train loss': [], 'Test loss': []}

    def record(m):
        curves['Train Accuracy'].append(accuracy_score(y, m.predict(x)))
        curves['Test Accuracy'].append(accuracy_score(y_test, m.predict(x_test)))
        curves['Train loss'].append(np.mean((m.predict_proba(x)[:, 1] - y) ** 2))
        curves['Test loss'].append(np.mean((m.predict_proba(x_test)[:, 1] - y_test) ** 2))

    for step in steps[:-1]:
        record(fit(spec, {**params, spec.iter_param: int(step)}, x, y, final=True))
    record(model)
    xs = list(steps)
    if last < max_iter:
        record(model)
        xs.append(max_iter)
    return xs, curves


def training_curve(spec, model, params, x, y, x_test, y_test, base):
    """Train/test scores against boosting rounds, trees or iterations, from a single training run."""
    if spec.curve == 'evals':
        xlabel, curves = spec.eval_curve(model)
        xs = np.arange(1, len(next(iter(curves.values()))) + 1)
    elif spec.curve == 'staged':
        xlabel = spec.iter_param
        curves = _curves_from_proba([p[:, 1] for p in model.staged_predict_proba(x)], y,
                                    [p[:, 1] for p in model.staged_predict_proba(x_test)], y_test)
        xs = np.arange(1, len(curves['Train loss']) + 1)
    elif spec.curve == 'trees':
        xlabel = spec.iter_param
        curves = _curves_from_proba(_cumulative_tree_proba(model, x), y,
                                    _cumulative_tree_proba(model, x_test), y_test)
        xs = np.arange(1, len(curves['Train loss']) + 1)
    else:
        xlabel = spec.iter_param
        xs, curves = _iteration_curves(spec, params, model, x, y, x_test, y_test)
    plot_curves(np.asarray(xs), curves, xlabel, f'Scores vs {xlabel}', base)


# ----------------------- Repeated cross-validation ----------------------- #


def roc_auc(recipe, x, y, n_trials, cv, random_state, prefix):
    """ROC curve and accuracy from cv-fold cross-validation repeated n_trials times."""
    splits = list(RepeatedKFold(n_splits=cv, n_repeats=n_trials, random_state=random_state).split(x))
    workers, threads = _plan(recipe, len(splits))
    folds = Parallel(n_jobs=workers)(
        delayed(roc_fold)(recipe, x, y, train, test, threads) for train, test in splits)
    results = {'fpr': [f[0] for f in folds], 'tpr': [f[1] for f in folds],
               'accuracy': [f[2] for f in folds], 'auc': [f[3] for f in folds]}
    pd.DataFrame({'accuracy': results['accuracy'], 'auc': results['auc']},
                 index=pd.RangeIndex(1, len(folds) + 1, name='fold')).to_csv(f'{prefix}_repeated_accuracy.csv')
    return plot_roc_auc_acc(results, f'{prefix}_AUC_curve', f'{prefix}_accuracy_plot_k_fold_cv_roc')


# --------------------------- Feature importance -------------------------- #


def feature_importance(model, x_test, y_test, names, random_state):
    if hasattr(model, 'feature_importances_'):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, 'coef_'):
        importance = np.ravel(model.coef_)
    else:
        logger.info(f'{type(model).__name__} does not have built-in feature importances.')
        importance = np.full(len(names), np.nan)
    # Worker processes only pay off when there is enough work: features x test samples
    n_jobs = min(8, cpu_count()) if x_test.size >= 50_000 else 1
    permutation = permutation_importance(estimator=model, X=x_test, y=y_test, scoring='accuracy', n_repeats=5,
                                         random_state=random_state, n_jobs=n_jobs)
    return pd.DataFrame({'Importance score': importance, 'Permutation importance': permutation.importances_mean},
                        index=pd.Index(names, name='Gene name'))

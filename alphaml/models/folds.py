"""
Fitting and per-fold work. Cross-validation folds run in worker processes, so this module only imports what a
fold needs (model libraries are imported lazily by specs.py).

A recipe describes a model to fit: (model name, params), or a list of them for a soft-voting ensemble.
"""
import os
import warnings

import numpy as np
from joblib import cpu_count
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from ..utils.metrics import positive_scores
from ..utils.sampling import sample_weight_
from .ensemble import SoftVotingEnsemble
from .specs import MODELS

# Threads per model fit. More threads than this rarely speed up a fit on omics-sized data, and OpenMP thread
# pools that busy-wait (Linux) slow down badly when a fit gets many more threads than it can use.
MAX_THREADS = int(os.environ.get('ALPHAML_THREADS') or 0) or min(cpu_count(), 16)


def members(recipe):
    return recipe if isinstance(recipe, list) else [recipe]


def fit_kwargs(spec, y):
    return {'sample_weight': sample_weight_(y)} if spec.sample_weight else {}


def fit(spec, params, x, y, final=False, threads=None, **extra):
    """Fit one model. threads: thread count for models that take one (None: MAX_THREADS)."""
    limit = min(threads or MAX_THREADS, MAX_THREADS)
    if spec.min_samples_per_thread:
        limit = max(1, min(limit, len(x) // spec.min_samples_per_thread))
    if spec.threads_param:
        params = {**params, spec.threads_param: limit}
    model = spec.make(params, y, final=final)
    # OpenMP models without a thread argument (HistGradientBoosting) follow the same limit
    with warnings.catch_warnings(), threadpool_limits(limits=limit, user_api='openmp'):
        warnings.simplefilter('ignore')  # convergence warnings for short iteration budgets
        model.fit(x, y, **fit_kwargs(spec, y), **extra)
    return model


def build(recipe, x, y, final=False, threads=None):
    """Fit the model a recipe describes."""
    if isinstance(recipe, list):
        return SoftVotingEnsemble(recipe, threads=threads).fit(x, y)
    name, params = recipe
    return fit(MODELS[name], params, x, y, final=final, threads=threads)


def fold_predictions(recipe, x, y, train, valid, final, threads):
    """Train and validation predictions of one fold, or None if the parameters are infeasible for it."""
    try:
        model = build(recipe, x[train], y[train], final=final, threads=threads)
    except ValueError as exc:
        if 'infeasible' in str(exc):  # NuSVC: nu too large for the class balance of this fold
            return None
        raise
    return {'pred': model.predict(x[valid]), 'score': positive_scores(model, x[valid]),
            'train_pred': model.predict(x[train]), 'train_score': positive_scores(model, x[train])}


def roc_fold(recipe, x, y, train, test, threads):
    """ROC curve, accuracy and AUC of one cross-validation fold."""
    model = build(recipe, x[train], y[train], threads=threads)
    accuracy = accuracy_score(y[test], model.predict(x[test]))
    if len(np.unique(y[test])) < 2:
        return None, None, accuracy, np.nan
    scores = positive_scores(model, x[test])
    fpr, tpr, _ = roc_curve(y[test], scores)
    return fpr, tpr, accuracy, roc_auc_score(y[test], scores)

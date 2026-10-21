"""
AutoML: choose the algorithm, its hyperparameters and, optionally, a soft-voting ensemble automatically.

1. Screen:  cross-validate every candidate algorithm with its predefined parameters.
2. Tune:    run an Optuna search for each of the top_k algorithms; they share the time budget.
3. Combine: cross-validate the average of the top algorithms' probabilities as an ensemble.

Every candidate is scored on the same stratified folds of the training data (the test set is never used).
The candidate with the lowest objective (mean fold error, see utils.metrics.fold_error) is selected.
"""
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import RANK_METRICS
from ..utils.io import to_jsonable
from ..utils.plots import plot_leaderboard
from .engine import WORST_ERROR, cross_validate, cv_summary
from .specs import MODELS

logger = logging.getLogger('alphaml')


@dataclass
class AutoMLResult:
    recipe: object              # (model name, params), or a list of them for an ensemble
    label: str                  # the selected model: an algorithm name or 'Ensemble'
    leaderboard: pd.DataFrame


def _row(model, stage, recipe, summary, seconds):
    if isinstance(recipe, list):
        parameters = ' + '.join(name for name, _ in recipe)
    else:
        parameters = json.dumps(to_jsonable(recipe[1]))
    return {'Model': model, 'Stage': stage, 'Score': summary['score'], 'SD': summary['sd'],
            'Train-CV gap': summary['gap'], 'AUC': summary['auc'], 'Accuracy': summary['accuracy'],
            'MCC': summary['mcc'], 'Objective': summary['objective'], 'Seconds': round(seconds, 1),
            'Parameters': parameters, 'recipe': recipe}


def _objective(row):
    return WORST_ERROR if np.isnan(row['Objective']) else row['Objective']


def _average_folds(member_folds):
    """Fold predictions of the soft-voting ensemble from its members' fold probabilities."""
    out = []
    for folds in zip(*member_folds):
        if any(f is None for f in folds):
            out.append(None)
            continue
        score = np.mean([f['score'] for f in folds], axis=0)
        train_score = np.mean([f['train_score'] for f in folds], axis=0)
        out.append({'pred': (score > 0.5).astype(int), 'score': score,
                    'train_pred': (train_score > 0.5).astype(int), 'train_score': train_score})
    return out


def _tune(name, x, y, splits, metric, n_trials, timeout, control_fitting, random_state, report):
    import optuna
    from optuna.samplers import TPESampler

    spec = MODELS[name]

    def objective(trial):
        params = {**spec.optuna(trial), 'random_state': random_state}
        summary = cv_summary(cross_validate((name, params), x, y, splits), splits, y, metric, control_fitting)
        trial.set_user_attr('summary', summary)
        return WORST_ERROR if np.isnan(summary['objective']) else summary['objective']

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=random_state))
    started = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[lambda s, t: report(t.number + 1, s.best_trial.user_attrs['summary']['score'])])
    best = study.best_trial
    params = {**best.params, 'random_state': random_state}
    logger.info(f'{name}: {len(study.trials)} trials, best {metric} {best.user_attrs["summary"]["score"]:.4f}')
    return _row(name, f'tuned ({len(study.trials)} trials)', (name, params), best.user_attrs['summary'],
                time.time() - started)


def run_automl(x, y, models, metric='kappa_mcc', top_k=3, n_trials=50, time_budget=10.0, cv=5, ensemble=True,
               control_fitting=False, random_state=12, base=None, progress=None):
    """
    Select a model for training data x, y.

    models: candidate algorithm names. metric: key of config.RANK_METRICS. top_k: algorithms to tune.
    n_trials: Optuna trials per tuned algorithm. time_budget: minutes shared by the tuning searches
    (0: no limit). base: path prefix for the leaderboard files.
    """
    progress = progress or (lambda stage, detail='': None)
    unknown = [m for m in models if m not in MODELS]
    if unknown or not models:
        raise ValueError(f"Choose AutoML algorithms from {', '.join(MODELS)} (got {unknown or 'none'}).")
    if metric not in RANK_METRICS:
        raise ValueError(f"metric must be one of {', '.join(RANK_METRICS)}")
    splits = list(StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state).split(x, y))
    logger.info(f'AutoML: {len(models)} algorithms, ranked by {RANK_METRICS[metric]} in {cv}-fold cross-validation')

    # 1. Screen every algorithm with its predefined parameters
    rows = []
    for i, name in enumerate(models, start=1):
        progress('AutoML: screening', f'{i}/{len(models)}: {name}')
        recipe = (name, {**MODELS[name].predefined, 'random_state': random_state})
        started = time.time()
        summary = cv_summary(cross_validate(recipe, x, y, splits), splits, y, metric, control_fitting)
        rows.append(_row(name, 'screened', recipe, summary, time.time() - started))
        logger.info(f'{name}: {metric} {summary["score"]:.4f} ± {summary["sd"]:.4f}')

    # 2. Tune the best algorithms, sharing the time budget
    top = [r['Model'] for r in sorted(rows, key=_objective)[:max(1, top_k)]]
    deadline = time.time() + time_budget * 60 if time_budget else None
    for j, name in enumerate(top, start=1):
        timeout = None
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.info(f'Time budget used up: {name} and later algorithms keep their predefined parameters')
                break
            timeout = remaining / (len(top) - j + 1)

        def report(trial, score, name=name, j=j):
            progress('AutoML: tuning', f'{name} ({j}/{len(top)}): trial {trial}/{n_trials}, best {score:.4f}')

        rows.append(_tune(name, x, y, splits, metric, n_trials, timeout, control_fitting, random_state, report))

    best = {}
    for row in rows:
        if row['Model'] not in best or _objective(row) < _objective(best[row['Model']]):
            best[row['Model']] = row

    # 3. Soft-voting ensemble of the top algorithms (probability outputs, so fitted as final models)
    if ensemble and len(top) >= 2:
        progress('AutoML: ensemble', ' + '.join(top))
        recipe = [best[name]['recipe'] for name in top]
        started = time.time()
        folds = _average_folds([cross_validate(r, x, y, splits, final=True) for r in recipe])
        summary = cv_summary(folds, splits, y, metric, control_fitting)
        rows.append(_row('Ensemble', 'ensemble', recipe, summary, time.time() - started))
        logger.info(f'Ensemble of {" + ".join(top)}: {metric} {summary["score"]:.4f}')

    # Rank: ties go to the earlier (simpler) candidate
    ranked = sorted(rows, key=_objective)
    winner = ranked[0]
    leaderboard = pd.DataFrame([{k: v for k, v in r.items() if k != 'recipe'} for r in ranked])
    leaderboard.insert(0, 'Rank', range(1, len(ranked) + 1))
    if base:
        leaderboard.to_csv(f'{base}_leaderboard.csv', index=False)
        plot_leaderboard(leaderboard, RANK_METRICS[metric], f'{base}_leaderboard')
    logger.info(f'AutoML selected {winner["Model"]} ({winner["Stage"]}), {metric} {winner["Score"]:.4f}')
    return AutoMLResult(recipe=winner['recipe'], label=winner['Model'], leaderboard=leaderboard)

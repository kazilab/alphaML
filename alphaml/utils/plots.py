import logging

import matplotlib
matplotlib.use('Agg')  # all figures are written to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

logger = logging.getLogger('alphaml')


def save_figure(fig, base, png=True):
    """Save a figure as <base>.pdf (and <base>.png for the web viewer) and close it."""
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    if png:
        fig.savefig(f'{base}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_curves(x, curves, xlabel, title, base):
    """Line plot of several score/loss series against iterations."""
    colors = ['red', 'green', 'orange', 'blue']
    fig, ax = plt.subplots(figsize=(6, 4))
    for (label, values), color in zip(curves.items(), colors):
        ax.plot(x[:len(values)], values, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.legend()
    save_figure(fig, base)


def plot_confusion_matrix(y_true, y_pred, axis_labels, base):
    # Rows/columns ordered positive class first:
    #                        Predicted
    #                   Positive  Negative
    # Actual Positive       TP      FN
    #        Negative       FP      TN
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(axis_labels)).plot(ax=ax)
    ax.set_title('Confusion matrix', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    n_pos = int(y_true.sum())
    ax.set_ylabel(f'True label, ({axis_labels[0][:3]}: {n_pos}, {axis_labels[1][:3]}: {len(y_true) - n_pos})',
                  fontsize=12)
    save_figure(fig, base)


def plot_parameters(parameters, base):
    params_df = pd.DataFrame([{k: str(v) for k, v in parameters.items()}]).T
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(params_df)))
    ax.axis('off')
    ax.table(cellText=params_df.values, rowLabels=params_df.index, loc='center')
    save_figure(fig, base, png=False)


def plot_roc_auc_acc(results, base_roc, base_acc):
    """Mean ROC curve (± 2 SD) and per-fold accuracy from repeated cross-validation."""
    fpr_mean = np.linspace(0, 1, 101)
    interp_tprs = []
    for fpr, tpr in zip(results['fpr'], results['tpr']):
        if fpr is None:
            continue
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2 * np.std(interp_tprs, axis=0)
    auc = np.nanmean(results['auc'])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(fpr_mean, np.clip(tpr_mean - tpr_std, 0, 1), np.clip(tpr_mean + tpr_std, 0, 1),
                    color=(52 / 255, 152 / 255, 219 / 255, 0.2), linewidth=0)
    ax.plot(fpr_mean, tpr_mean, color=(41 / 255, 128 / 255, 185 / 255), linewidth=2, label=f'AUC: {auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.grid(color=(189 / 255, 195 / 255, 199 / 255, 0.5))
    ax.legend(loc='lower right')
    save_figure(fig, base_roc)

    accuracy = np.asarray(results['accuracy'])
    sem = pd.Series(accuracy).sem()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(np.arange(1, len(accuracy) + 1), accuracy, color='lightblue', edgecolor='steelblue', s=18)
    ax.set_xlim(0, len(accuracy) + 0.5)
    ax.set_title('Accuracy plot from cross-validation')
    ax.set_xlabel('Repeated cross-validation number')
    ax.set_ylabel('Accuracy')
    ax.text(0.99, 0.02, f'Average accuracy: {accuracy.mean():.3f}, SEM: {sem:.5f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9)
    save_figure(fig, base_acc)
    return auc


def plot_leaderboard(leaderboard, metric_label, base):
    """Cross-validated score (± SD) of every AutoML candidate, best on top."""
    colors = {'screened': '#9aa5b1', 'tuned': '#2456c9', 'ensemble': '#c2410c'}
    df = leaderboard.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(df) + 1.2))
    bar_colors = [colors.get(stage.split(' ')[0], '#9aa5b1') for stage in df['Stage']]
    ax.barh(np.arange(len(df)), df['Score'].fillna(0), xerr=df['SD'].fillna(0), color=bar_colors,
            error_kw={'elinewidth': 1, 'ecolor': '#444'})
    ax.set_yticks(np.arange(len(df)), [f"{m}  ({s.split(' ')[0]})" for m, s in zip(df['Model'], df['Stage'])])
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlim(0, 1)
    ax.set_xlabel(f'{metric_label} (cross-validation mean ± SD)')
    ax.set_title('AutoML leaderboard')
    ax.grid(axis='x', color='#ddd')
    ax.set_axisbelow(True)
    save_figure(fig, base)


def plot_optuna_importance(study, base):
    try:
        import optuna
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ax = optuna.visualization.matplotlib.plot_param_importances(study)
        save_figure(ax.figure, base)
    except Exception as exc:  # needs at least two parameters and enough completed trials
        logger.info(f'Optuna parameter importance plot skipped: {exc}')

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, f1_score, jaccard_score, make_scorer, matthews_corrcoef, roc_auc_score)

# Metrics excluded from the NegLog2-RMSL score
DROP_METRICS = ['AUC', "Cohen's Kappa", 'F1 Score', 'Jaccard', 'MCC', 'Average Precision Score',
                'AUC (probability)']
# Metrics reported as-is in the loss table; every other loss is 1 - metric
RAW_LOSSES = ('Negative Likelihood Ratio', 'Brier Score Loss')


def positive_scores(model, x):
    """Positive-class probability, or the decision function for models without probabilities."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    return None


def _safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:  # only one class present
        return np.nan


def binary_metrics(y_true, y_pred, y_proba=None):
    """All binary classification metrics from one set of predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = tp / (tp + fn)  # Recall, True positive rate (TPR)
        specificity = tn / (tn + fp)
        metrics = {
            'Accuracy': (tp + tn) / (tp + fn + fp + tn),
            'AUC': _safe_auc(y_true, y_pred),  # from predicted labels, equals balanced accuracy
            "Cohen's Kappa": 2 * (tp * tn - fp * fn) / ((tp + fp) * (tn + fp) + (tp + fn) * (tn + fn)),
            'F1 Score': 2 * tp / (2 * tp + fp + fn),
            'Jaccard': jaccard_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Negative Likelihood Ratio': (1 - sensitivity) / specificity,  # FNR/TNR
            'NPV': tn / (fn + tn),  # Negative predictive value
            'Precision': tp / (tp + fp),  # Positive predictive value (PPV)
            'Sensitivity': sensitivity,
            'Specificity': specificity,
        }
    if y_proba is not None:
        metrics.update({
            'Average Precision Score': average_precision_score(y_true, y_proba),
            'Brier Score Loss': brier_score_loss(y_true, y_proba),
            'AUC (probability)': _safe_auc(y_true, y_proba),
        })
    return metrics


def binary_losses(metrics):
    """Loss counterpart of every metric: 1 - metric (FDR, FNR, FPR, ...), or the raw value for losses."""
    return {name: (value if name in RAW_LOSSES else 1 - value) for name, value in metrics.items()}


def model_metrics(model, x, y):
    proba = model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else None
    return binary_metrics(y, model.predict(x), proba)


# Combined Cohen's kappa and MCC scorer


def kappa_mcc_error(y_true, y_pred):
    k = cohen_kappa_score(y_true, y_pred)
    m = matthews_corrcoef(y_true, y_pred)
    # Worse-than-chance predictions (negative kappa/MCC) count as chance level
    return 1 - np.sqrt(max(k, 0) * max(m, 0))


KAPPA_MCC_SCORER = make_scorer(kappa_mcc_error, greater_is_better=False)


def rank_score(metric, y_true, y_pred, y_score=None):
    """Score used to compare models (see config.RANK_METRICS): in [0, 1], higher is better."""
    if metric == 'kappa_mcc':
        return 1 - kappa_mcc_error(y_true, y_pred)
    if metric == 'roc_auc':
        return _safe_auc(y_true, y_score)
    if metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    if metric == 'f1':
        return f1_score(y_true, y_pred, zero_division=0)
    raise ValueError(f'Unknown metric {metric!r}')


def fold_error(train_score, valid_score, control_fitting):
    """
    Error of one cross-validation fold: 1 - validation score. With control_fitting, the geometric mean of the
    validation error and the train/validation gap, so overfitted models rank lower.
    """
    train_error, valid_error = 1 - train_score, 1 - valid_score
    if not control_fitting or train_error == valid_error:
        return valid_error
    return (abs(train_error - valid_error) * valid_error) ** 0.5


# Calculate Negative log2 RMSL


def neglog2rmsl(model, x_train, y_train, x_test, y_test):
    train = model_metrics(model, x_train, y_train)
    test = model_metrics(model, x_test, y_test)
    test_df = pd.DataFrame.from_dict(test, orient='index', columns=['Test'])
    train_df = pd.DataFrame.from_dict(train, orient='index', columns=['Train'])
    losses_df = pd.DataFrame.from_dict(binary_losses(test), orient='index', columns=['Test loss'])

    keep = [m for m in test_df.index if m not in DROP_METRICS]
    # RMS of differences between train and test scores
    diff = np.mean(np.square(train_df.loc[keep, 'Train'] - test_df.loc[keep, 'Test'])) ** 0.5
    # RMS of test scores losses
    loss_ = np.mean(np.square(losses_df.loc[keep, 'Test loss'])) ** 0.5
    # Final score: geometric mean, then negative log2
    factor = -1 * np.log2(((diff + 0.05) * loss_) ** 0.5)
    all_score_df = pd.concat([train_df, test_df, losses_df], axis=1).round(decimals=3)
    all_score_df.loc['NegLog2-RMSL'] = round(float(factor), 3)  # Negative log2 Root Mean Squared Loss
    return all_score_df, float(factor)


def prediction_table(model, x, y_true=None, index=None, pos_class=None, neg_class=None):
    """Predictions (and probabilities) per sample, optionally with true labels and class names."""
    table = pd.DataFrame(index=index if index is not None else range(len(x)))
    if y_true is not None:
        table['true_labels'] = np.asarray(y_true)
    table['predictions'] = model.predict(x)
    if hasattr(model, 'predict_proba'):
        table['pred_proba'] = model.predict_proba(x)[:, 1]
    if pos_class is not None and neg_class is not None:
        table['predicted_class'] = table['predictions'].map({1: pos_class, 0: neg_class})
    return table

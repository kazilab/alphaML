"""
Feature selection methods.

Unsupervised methods take a DataFrame (samples x features) and return a list of feature names.
Supervised methods also take the labels.
"""
import logging
import math
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import comb
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import LinearSVC

from .sampling import scale_pos_weight_

logger = logging.getLogger('alphaml')


def _check_frame(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input should be a pandas DataFrame')


# Highly variable features adapted from scanpy.preprocessing._highly_variable_genes._highly_variable_genes_single_batch


def hvf(data, log_type='log2', n_top_features=0, min_disp=1, max_disp=np.inf, n_bins=20, min_mean=0.0125, max_mean=3):
    # Undo the log(x+1) transform before computing dispersions
    data = data.fillna(0)
    if log_type == 'log2':
        data = np.exp2(data) - 1
    elif log_type == 'log10':
        data = np.power(10.0, data) - 1
    elif log_type == 'ln':
        data = np.expm1(data)

    mean = data.mean(axis=0)
    mean[mean == 0] = 1e-6  # set entries equal to zero to small value
    variance = data.var(axis=0, ddof=1)
    dispersion = variance / mean
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)
    df = pd.DataFrame({'means': mean, 'dispersions': dispersion})

    df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
    disp_grouped = df.groupby('mean_bin', observed=False)['dispersions']
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)

    one_gene_per_bin = disp_std_bin.isnull()
    disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[one_gene_per_bin.values].values
    disp_mean_bin[one_gene_per_bin.values] = 0
    # actually do the normalization
    df['dispersions_norm'] = (df['dispersions'].values - disp_mean_bin[df['mean_bin'].values].values
                              ) / disp_std_bin[df['mean_bin'].values].values

    dispersion_norm = np.array(df['dispersions_norm'].values, dtype=float)  # writable copy

    if n_top_features >= 2:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
        if n_top_features > data.shape[1]:
            logger.info('`n_top_features` > number of features, returning all features.')
            selected_features = data.columns
        elif n_top_features > dispersion_norm.size:
            logger.info('`n_top_features` > number of normalized dispersions, '
                        'returning all features with normalized dispersions.')
            selected_features = df.index
        else:
            disp_cut_off = dispersion_norm[n_top_features - 1]
            feat_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
            selected_features = df.index[feat_subset]
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0
        feat_subset = np.logical_and.reduce((mean.values > min_mean,
                                             mean.values < max_mean,
                                             dispersion_norm > min_disp,
                                             dispersion_norm < max_disp
                                             ))
        selected_features = df.index[feat_subset]

    return list(selected_features)


# 1. Select from model using Random Forest


def select_from_rf(x_train, y_train, threshold, max_features):
    max_features = max_features if max_features > 1 else None
    selector = SelectFromModel(RandomForestClassifier(n_estimators=250,
                                                      max_depth=6,
                                                      random_state=18,
                                                      criterion='log_loss',
                                                      class_weight='balanced',
                                                      n_jobs=-1),
                               threshold=threshold,
                               norm_order=1,
                               max_features=max_features,
                               importance_getter='auto'
                               ).fit(x_train.to_numpy(), np.asarray(y_train))
    return list(x_train.columns[selector.get_support()])


# 2. Recursive feature elimination with cross-validation to select features


def recursive_feature_elimination(x_train, y_train, min_sel_features_rfe):
    rfe = RFECV(LinearSVC(C=0.7,
                          penalty='l1',
                          dual=False,
                          random_state=18,
                          class_weight='balanced'
                          ),
                step=1,
                min_features_to_select=min_sel_features_rfe,
                scoring='average_precision',
                n_jobs=-1,
                ).fit(x_train.to_numpy(), np.asarray(y_train))
    return list(x_train.columns[rfe.support_])


# 3. Sequential Feature Selection.


def seq_feat_sel(x_train, y_train, min_sel_features_sfs):
    from xgboost import XGBClassifier

    y = np.asarray(y_train)
    # Candidate features are evaluated in parallel, so each XGBoost model runs single-threaded.
    sfs = SequentialFeatureSelector(XGBClassifier(learning_rate=0.025,
                                                  max_depth=6,
                                                  subsample=0.75,
                                                  colsample_bytree=0.75,
                                                  gamma=2,
                                                  reg_alpha=2.5,
                                                  reg_lambda=2,
                                                  min_child_weight=2.5,
                                                  n_estimators=75,
                                                  random_state=18,
                                                  objective='binary:logistic',
                                                  verbosity=0,
                                                  n_jobs=1,
                                                  scale_pos_weight=scale_pos_weight_(y)
                                                  ),
                                    n_features_to_select=min_sel_features_sfs,
                                    scoring='average_precision',
                                    n_jobs=-1,
                                    ).fit(x_train.to_numpy(), y)
    return list(x_train.columns[sfs.get_support()])


# 4. Select by median MAD


def select_by_mad(df):
    _check_frame(df)
    x = df.to_numpy(dtype=float)
    mads = np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    # Select features that pass the median value of MAD
    return list(df.columns[mads > np.median(mads)])


# 5 Select by PCA


def select_by_pca(df, top_n_genes_pca=500, explained_variance_ratio=0.8):
    _check_frame(df)
    pca = PCA().fit(df.to_numpy(dtype=float))
    # Number of components that explain the desired portion of the variance
    cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_explained_variance_ratio >= explained_variance_ratio) + 1
    # Aggregated loading scores from the selected components
    loading_scores = pd.Series(np.abs(pca.components_[:n_components]).sum(axis=0), index=df.columns)
    return list(loading_scores.sort_values(ascending=False).head(top_n_genes_pca).index)


# 6 Random Projection


def select_by_random_projection(df, top_n_genes_rp=500, epsilon=0.5, random_state=None):
    _check_frame(df)
    # Johnson-Lindenstrauss lemma, https://doi.org/10.1002/rsa.10073
    # k ≥ (4 * log(n)) / (ε^2 / 2 - ε^3 / 3), ε = 0.1 and 0.5
    x = df.to_numpy(dtype=float)
    n_components = max(1, math.floor(4 * np.log10(x.shape[1]) / (epsilon ** 2 / 2 - epsilon ** 3 / 3)))
    projected = GaussianRandomProjection(n_components=n_components, random_state=random_state).fit_transform(x)

    # Correlation between each original feature and each projected feature. Only the features x
    # components block is computed (the full (p+k) x (p+k) matrix needs gigabytes for whole-genome data).
    def _unit_columns(a):
        a = a - a.mean(axis=0)
        norm = np.linalg.norm(a, axis=0)
        norm[norm == 0] = np.inf  # constant columns get zero correlation
        return a / norm

    corr = _unit_columns(x).T @ _unit_columns(projected)
    feature_importance = np.abs(corr).sum(axis=1)
    ranked = pd.Series(feature_importance, index=df.columns).sort_values(ascending=False)
    return list(ranked.head(top_n_genes_rp).index)


# 7 Correlation-based method


def remove_highly_correlated_feat(df, correlation_threshold=0.9, block_size=512):
    """
    Greedily keep features (in column order) that are not highly correlated with an already kept feature.

    Correlation rows are computed in blocks with a matrix product instead of materialising the full
    features x features correlation matrix.
    """
    _check_frame(df)
    x = df.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = np.inf  # constant features are never removed and never remove others
    x = x / norm

    n_features = x.shape[1]
    removed = np.zeros(n_features, dtype=bool)
    keep = []
    start = 0
    while start < n_features:
        candidates = start + np.flatnonzero(~removed[start:])[:block_size]
        if candidates.size == 0:
            break
        rows = np.abs(x[:, candidates].T @ x) >= correlation_threshold
        for row, feature in zip(rows, candidates):
            if removed[feature]:
                continue
            keep.append(feature)
            row[:feature + 1] = False
            removed |= row
        start = candidates[-1] + 1
    return list(df.columns[keep])


# 8 Clustering-based method


def select_by_clustering(df, n_clusters=500):
    _check_frame(df)
    # Rows represent features and columns represent samples
    x = df.to_numpy(dtype=float).T
    n_clusters = min(n_clusters, x.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
    labels = kmeans.labels_
    # Representative feature per cluster: the one closest to its cluster center
    distance = kmeans.transform(x)[np.arange(x.shape[0]), labels]
    representative = pd.Series(distance).groupby(labels).idxmin()
    return list(df.columns[representative.to_numpy()])


# 9 Non-negative Matrix Factorization (NMF)


def select_by_nmf(df, n_components=500, top_n_genes_per_component=1):
    _check_frame(df)
    x = df.to_numpy(dtype=float)
    if (x < 0).any():
        raise ValueError('Input data should be non-negative')
    h = NMF(n_components=n_components, random_state=0).fit(x).components_
    # Top N features with the highest coefficients in each NMF component
    top = np.unique(np.argsort(h, axis=1)[:, -top_n_genes_per_component:])
    return list(df.columns[top])


# 10 Iterative Feature Selection


def _subset_score(x, y, subset, cv):
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    return np.mean(cross_val_score(clf, x[:, subset], y, cv=cv))


def iterative_feature_selector(x_train, y_train, min_features=1, max_features=3, cv=5, max_input_features=40):
    _check_frame(x_train)
    n_features = x_train.shape[1]
    if n_features > max_input_features:
        raise ValueError(f'IterativeFeatSel tests every subset of {min_features}-{max_features} features and '
                         f'is limited to {max_input_features} input features (got {n_features}). '
                         f'Reduce the features with an earlier selection step.')
    x = x_train.to_numpy()
    y = np.asarray(y_train)
    subsets = [s for k in range(min_features, max_features + 1) for s in combinations(range(n_features), k)]
    logger.info(f'Total iterations: {sum(comb(n_features, k) for k in range(min_features, max_features + 1)):.0f}')
    scores = Parallel(n_jobs=-1)(delayed(_subset_score)(x, y, list(s), cv) for s in subsets)
    best = subsets[int(np.argmax(scores))]
    return [x_train.columns[i] for i in best]

import logging
import os

import numpy as np
import pandas as pd

from .feature_selection import (hvf, iterative_feature_selector, recursive_feature_elimination,
                                remove_highly_correlated_feat, select_by_clustering, select_by_mad, select_by_nmf,
                                select_by_pca, select_by_random_projection, select_from_rf, seq_feat_sel)
from .io import as_frame, read_feature_list

logger = logging.getLogger('alphaml')


# ------------------------------ Normalization ----------------------------- #


def normalize_samples(df, normalization):
    """Normalize each sample (row) over its features, so a sample's values never depend on other samples."""
    if normalization == 'None':
        return df
    values = df.to_numpy(dtype=float)
    if normalization == 'min_max':
        # Min-max normalization between 0 and 1
        low = values.min(axis=1, keepdims=True)
        spread = values.max(axis=1, keepdims=True) - low
        scaled = (values - low) / np.where(spread == 0, 1, spread)
    elif normalization == 'standardization':
        # Mean 0 and standard deviation 1
        std = values.std(axis=1, keepdims=True)
        scaled = (values - values.mean(axis=1, keepdims=True)) / np.where(std == 0, 1, std)
    else:
        raise ValueError('Please provide a valid normalization method!')
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


# ---------------------------------- Labels -------------------------------- #


def encode_labels(labeled_data, data_labels, column_name, pos_class, neg_class):
    """
    Encode the class column as 1 (positive) / 0 (negative) and align it with the labeled data.

    Samples with any other label are dropped. Labels are matched to data by sample name, or by position when
    the two files use different sample names but have the same number of rows.
    """
    if column_name not in data_labels.columns:
        raise ValueError(f"Column '{column_name}' not found in the labels file. "
                         f"Available columns: {', '.join(map(str, data_labels.columns[:20]))}")
    raw = data_labels[column_name].astype(str).str.strip().str.lower()
    pos, neg = pos_class.strip().lower(), neg_class.strip().lower()
    labels = pd.Series(np.select([raw == pos, raw == neg], [1, 0], default=-1), index=data_labels.index)

    common = labeled_data.index.intersection(labels.index)
    if len(common) == 0:
        if len(labels) != len(labeled_data):
            raise ValueError('Sample names in the labels file do not match the labeled data, '
                             'and the two files have different numbers of rows.')
        logger.info('Sample names differ between labels and data; matching them by row order.')
        labels.index = labeled_data.index
        common = labeled_data.index
    elif len(common) < len(labeled_data):
        logger.info(f'{len(labeled_data) - len(common)} labeled samples have no label and were dropped.')

    labels = labels.loc[common]
    known = labels >= 0
    if (~known).any():
        logger.info(f"{int((~known).sum())} samples are neither '{pos_class}' nor '{neg_class}' and were dropped.")
    labels = labels[known].astype(int)
    if labels.nunique() < 2:
        raise ValueError(f"Both classes '{pos_class}' and '{neg_class}' must be present in column '{column_name}'.")
    labels.name = column_name
    return labeled_data.loc[labels.index], labels


# ---------------------------- Feature selection --------------------------- #


def _select(method, df_fs, df_labeled, labels, options, suggested_features, random_state):
    """Run one feature selection method. Unsupervised methods use df_fs, supervised ones use df_labeled."""
    o = options
    methods = {
        'HVF': lambda: hvf(data=df_fs, log_type=o['log_type'], n_top_features=o['n_top_features'],
                           min_disp=o['min_disp']),
        'MedianMAD': lambda: select_by_mad(df=df_fs),
        'SelByPCA': lambda: select_by_pca(df=df_fs, top_n_genes_pca=o['top_n_genes_pca']),
        'RandomProjection': lambda: select_by_random_projection(df=df_fs, top_n_genes_rp=o['top_n_genes_rp'],
                                                                random_state=random_state),
        'SelByClustering': lambda: select_by_clustering(df=df_fs, n_clusters=o['n_clusters']),
        'SelByNMF': lambda: select_by_nmf(df=df_fs, n_components=o['n_components']),
        'RemoveHighCorrFeat': lambda: remove_highly_correlated_feat(df=df_fs),
        'SelectByRF': lambda: select_from_rf(x_train=df_labeled, y_train=labels, threshold=o['threshold'],
                                             max_features=o['max_features']),
        'RecursiveFeatElim': lambda: recursive_feature_elimination(x_train=df_labeled, y_train=labels,
                                                                   min_sel_features_rfe=o['min_sel_features_rfe']),
        'SeqFeatSel': lambda: seq_feat_sel(x_train=df_labeled, y_train=labels,
                                           min_sel_features_sfs=o['min_sel_features_sfs']),
        'IterativeFeatSel': lambda: iterative_feature_selector(x_train=df_labeled, y_train=labels),
    }
    if method == 'None':
        return list(df_labeled.columns)
    if method == 'Suggested':
        if suggested_features is None:
            raise ValueError('Feature selection "Suggested" needs a suggested features file.')
        available = set(df_labeled.columns)
        selected = [f for f in suggested_features if f in available]
        if len(selected) < len(suggested_features):
            logger.info(f'{len(suggested_features) - len(selected)} suggested features are not in the data.')
        logger.info('Suggested features are being used')
        return selected
    if method not in methods:
        raise ValueError(f"Please provide a valid feature selection method! Got '{method}'.")
    logger.info(f'Selecting features by {method}')
    try:
        selected = methods[method]()
    except Exception as exc:
        raise ValueError(f'Feature selection by {method} failed: {exc}. '
                         f'Please consider an alternative method.') from exc
    if not selected:
        logger.info(f'{method} failed to generate a feature list, we reverted the list to the current features')
        selected = list(df_labeled.columns)
    return selected


def select_features(labeled_data, labels, data_for_feature_selection, methods, options, result_path, column_name,
                    suggested_features=None, random_state=None, progress=None):
    """Apply up to four feature selection methods in sequence and return the final feature list."""
    df_fs, df_labeled = data_for_feature_selection, labeled_data
    for step, method in enumerate(methods, start=1):
        if progress:
            progress('Feature selection', f'Step {step}/{len(methods)}: {method} on {df_labeled.shape[1]} features')
        selected = _select(method, df_fs, df_labeled, labels, options, suggested_features, random_state)
        if method != 'None':
            names = [methods[i] if i < step else 'None' for i in range(4)]
            fs_name = f"{column_name}_{'_'.join(names)}_Feat_M{step}"
            pd.DataFrame(selected, columns=['Selected features']).to_csv(
                os.path.join(result_path, f'{fs_name}.csv'), index=False)
            logger.info(f'{method} selected {len(selected)} features, saved as {fs_name}.csv')
        df_fs, df_labeled = df_fs[selected], df_labeled[selected]
    return list(df_labeled.columns)


def preprocessing(labeled_data, data_labels, result_path, column_name, pos_class, neg_class,
                  features, features2, features3, features4, normalization='min_max',
                  data_for_feature_selection=None, unlabeled_data=None, suggested_features=None,
                  random_state=None, progress=None, **options):
    """
    Load data, encode labels, select features and normalize.

    Args:
        labeled_data: samples x features (DataFrame or file path)
        data_labels: samples x label columns (DataFrame or file path)
        result_path: folder to save results
        column_name: label column to predict, for example 'Trametinib'
        pos_class: positive class label, for example 'resistant'
        neg_class: negative class label, for example 'sensitive'
        features, features2, features3, features4: feature selection methods applied in sequence
        normalization: data normalization method
        data_for_feature_selection: optional data for unsupervised feature selection (default: labeled_data)
        unlabeled_data: optional samples without labels, predicted with the final model
        suggested_features: optional feature list (or file path) for the 'Suggested' method
        options: method parameters (n_top_features, min_disp, log_type, top_n_genes_pca, top_n_genes_rp,
                 n_clusters, n_components, max_features, threshold, min_sel_features_rfe, min_sel_features_sfs)

    Returns:
        labels, normalized labeled data, normalized unlabeled data (or None)
    """
    if progress:
        progress('Loading data')
    else:
        logger.info('Loading data')
    labeled = as_frame(labeled_data)
    label_table = as_frame(data_labels, numeric=False)
    fs_data = as_frame(data_for_feature_selection) if data_for_feature_selection is not None else None
    unlabeled = as_frame(unlabeled_data) if unlabeled_data is not None else None
    if isinstance(suggested_features, (str, os.PathLike)):
        suggested_features = read_feature_list(suggested_features)

    labeled, labels = encode_labels(labeled, label_table, column_name, pos_class, neg_class)
    logger.info(f'{len(labels)} labeled samples ({int(labels.sum())} {pos_class}, '
                f'{int(len(labels) - labels.sum())} {neg_class}), {labeled.shape[1]} features')

    # Use only features present in every data matrix
    common = labeled.columns
    for other in (fs_data, unlabeled):
        if other is not None:
            common = common.intersection(other.columns, sort=False)
    if len(common) < labeled.shape[1]:
        logger.info(f'{labeled.shape[1] - len(common)} features are missing from the other data files and were '
                    f'dropped.')
    labeled = labeled[common]
    fs_data = fs_data[common] if fs_data is not None else labeled

    methods = [features, features2, features3, features4]
    selected = select_features(labeled, labels, fs_data, methods, options, result_path, column_name,
                               suggested_features=suggested_features, random_state=random_state,
                               progress=progress)
    logger.info(f'{len(selected)} features selected')

    if progress:
        progress('Normalizing data')
    sel_labeled = normalize_samples(labeled[selected], normalization)
    sel_labeled.to_csv(os.path.join(result_path, f'{column_name}_{features}_normalized_labeled_data.csv'))
    sel_unlabeled = None
    if unlabeled is not None:
        sel_unlabeled = normalize_samples(unlabeled[selected], normalization)
        sel_unlabeled.to_csv(os.path.join(result_path, f'{column_name}_{features}_normalized_unlabeled_data.csv'))
    return labels, sel_labeled, sel_unlabeled

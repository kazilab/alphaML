import warnings
import logging
import pandas as pd
import pickle
from .deeplink import modelx
from .utils import (hvf, select_from_rf, min_max, standardize_dataframe, get_data,
                    recursive_feature_elimination, seq_feat_sel, get_pred_data,
                    select_by_mad, select_by_pca, select_by_random_projection,
                    select_by_clustering, select_by_nmf, iterative_feature_selector,
                    remove_highly_correlated_feat, preprocessing_df)


warnings.filterwarnings("ignore")


def preprocessing(data_path,
                  result_path,
                  column_name,
                  pos_class,
                  neg_class,
                  features,
                  features2,
                  features3,
                  features4,
                  n_top_features,
                  min_disp,
                  log_type,
                  top_n_genes_pca,
                  top_n_genes_rp,
                  n_clusters,
                  n_components,
                  max_features,
                  threshold,
                  min_sel_features_rfe,
                  min_sel_features_sfs,
                  latent_dim,
                  min_sel_features,
                  fdr,
                  normalization='min_max'):

    """
    Feature selection using ModelX knockoffs.

    Args:
        data_path: Path to the data files
        result_path: path to save results
        column_name: name of the data file column from where data labels will be collected
        pos_class: positive class label, for example 'resistant'
        neg_class: negative class label, for example 'sensitive'
        features: feature selection method to be used, get from dropdown list
        features2: feature selection method to be used, get from dropdown list
        features3: feature selection method to be used, get from dropdown list
        features4: feature selection method to be used, get from dropdown list
        n_top_features: HVF parameter
        top_n_genes_pca: PCA parameter,
        top_n_genes_rp: Random projection parameter,
        n_clusters:  Clustering parameter,
        n_components: NMF parameter,
        min_disp: HVF parameter
        log_type: HVF parameter
        threshold: select_from_model parameter
        min_sel_features_rfe: RFE parameter
        min_sel_features_sfs: SFS parameter
        max_features: select_from_model parameter
        latent_dim: DeepLink parameter
        min_sel_features: DeepLink parameter
        fdr: DeepLink parameter
        normalization: data normalization method

    Returns:
        Selected data. labels, sel_labeled_data, sel_unlabeled_data
    """

    # Check the availability of data else download
    get_data(data_path)

    logging.info('Loading data')
    data_labels = pd.read_csv(data_path+'data_labels.csv',
                              index_col=0,
                              header=0,
                              sep=',',
                              low_memory=False
                              )
    labeled_data = pd.read_csv(data_path+'labeled_data.csv',
                               index_col=0,
                               header=0,
                               sep=',',
                               low_memory=False
                               )
    labeled_data = preprocessing_df(labeled_data)
    unlabeled_data = pd.read_csv(data_path+'unlabeled_data.csv',
                                 index_col=0,
                                 header=0,
                                 sep=',',
                                 low_memory=False
                                 )
    unlabeled_data = preprocessing_df(unlabeled_data)
    data_for_feature_selection = pd.read_csv(data_path+'data_for_feature_selection.csv',
                                             index_col=0,
                                             header=0,
                                             sep=',',
                                             low_memory=False
                                             )
    data_for_feature_selection = preprocessing_df(data_for_feature_selection)
    # Encode labels
    pos_class = pos_class.strip().lower()
    neg_class = neg_class.strip().lower()
    data_labels[column_name] = data_labels[column_name].apply(
        lambda x: int(1) if x.strip().lower() == pos_class else int(0) if x.strip().lower() == neg_class else None)
    labels = data_labels[column_name]

    def apply_(feature, df1, df2, fs_name):

        def execute_feature_selection_method(method, kwargs):
            return method(**kwargs)

        def save_and_log_selected_features(selected_features_):
            if not selected_features_:
                raise ValueError(
                    'Method returned empty feature. Please consider an alternative method.')
            elif isinstance(selected_features_, Exception):
                raise ValueError(
                    f'Method raised an error. Please consider an alternative method.')

            df = pd.DataFrame(selected_features_, columns=['Selected features'])
            df.to_csv(f"{result_path}{fs_name}.csv", index=False)
            logging.info(f"select by {fs_name}, and saved as features_{fs_name}.csv in alphaML_results folder")

        feature_selection_methods = {
            'HVF': {
                'function': hvf,
                'kwargs': {'data': df1, 'log_type': log_type, 'n_top_features': n_top_features, 'min_disp': min_disp}
            },
            'MedianMAD': {
                'function': select_by_mad,
                'kwargs': {'df': df1}
            },
            'SelByPCA': {
                'function': select_by_pca,
                'kwargs': {'df': df1, 'top_n_genes_pca': top_n_genes_pca},
            },
            'RandomProjection': {
                'function': select_by_random_projection,
                'kwargs': {'df': df1, 'top_n_genes_rp': top_n_genes_rp},
            },
            'SelByClustering': {
                'function': select_by_clustering,
                'kwargs': {'df': df1, 'n_clusters': n_clusters},
            },
            'SelByNMF': {
                'function': select_by_nmf,
                'kwargs': {'df': df1, 'n_components': n_components},
            },
            'RemoveHighCorrFeat': {
                'function': remove_highly_correlated_feat,
                'kwargs': {'df': df1},
            },
            'SelectByRF': {
                'function': select_from_rf,
                'kwargs': {'x_train': df2, 'y_train': labels, 'threshold': threshold, 'max_features': max_features},
            },
            'RecursiveFeatElim': {
                'function': recursive_feature_elimination,
                'kwargs': {'x_train': df2, 'y_train': labels, 'min_sel_features_rfe': min_sel_features_rfe},
            },
            'SeqFeatSel': {
                'function': seq_feat_sel,
                'kwargs': {'x_train': df2, 'y_train': labels, 'min_sel_features_sfs': min_sel_features_sfs},
            },
            'ModelX': {
                'function': modelx,
                'kwargs': {'data': df2, 'dataY': labels, 'latent_dim': latent_dim,
                           'min_sel_features': min_sel_features,
                           'fdr': fdr, 'result_path': result_path},
            },
            'IterativeFeatSel': {
                'function': iterative_feature_selector,
                'kwargs': {'x_train': df2, 'y_train': labels},
            }
        }
        if feature == 'Suggested':
            selected_features = pd.read_csv(data_path+'suggested_features.csv',
                                            index_col=None,
                                            header=0,
                                            sep=',',
                                            low_memory=False
                                            ).values.ravel().tolist()
            logging.info("Suggested features is being used")
        elif feature == 'None':
            selected_features = df2.columns.tolist()
        elif feature in feature_selection_methods:
            print(f'Selecting features by {feature}')
            selected_features = execute_feature_selection_method(feature_selection_methods[feature]['function'],
                                                                 feature_selection_methods[feature]['kwargs'])
            if not selected_features:
                selected_features = df2.columns.tolist()
                error_txt = f'{feature} failed to generate a feature list, '
                logging.info(error_txt+f'we reverted the list to the current dataframe index')
                print(error_txt+f'we reverted the list to the current dataframe index')
            else:
                selected_features = selected_features
            save_and_log_selected_features(selected_features)
        else:
            raise ValueError('Please provide a valid feature selection method!')

        return df1[selected_features], df2[selected_features]

    df11, df21 = apply_(features,
                        data_for_feature_selection,
                        labeled_data,
                        column_name + '_' + features + '_' + 'None' + '_' + 'None' + '_' + 'None' + '_Feat_M1'
                        )
    df12, df22 = apply_(features2,
                        df11,
                        df21,
                        column_name + '_' + features + '_' + features2 + '_' + 'None' + '_' + 'None' + '_Feat_M2'
                        )
    df13, df23 = apply_(features3,
                        df12,
                        df22,
                        column_name + '_' + features + '_' + features2 + '_' + features3 + '_' + 'None' + '_Feat_M3'
                        )
    df14, df24 = apply_(features4,
                        df13,
                        df23,
                        column_name + '_' + features + '_' + features2 + '_' + features3 + '_' + features4 + '_Feat_M4'
                        )

    features_ = df14.columns
    df_labeled = labeled_data[features_]
    df_unlabeled = unlabeled_data[features_]

    if normalization == 'min_max':
        # Normalize data using min-max normalization between 0 and 1.
        sel_labeled_data4 = min_max(df_labeled.T).T
        sel_unlabeled_data4 = min_max(df_unlabeled.T).T
    elif normalization == 'standardization':
        # Normalize data using min-max normalization between 0 and 1.
        sel_labeled_data4 = standardize_dataframe(df_labeled.T).T
        sel_unlabeled_data4 = standardize_dataframe(df_unlabeled.T).T
    elif normalization == 'None':
        sel_labeled_data4 = df_labeled
        sel_unlabeled_data4 = df_unlabeled
    else:
        raise ValueError('Please provide a valid normalization method!')

    sel_labeled_data4.to_csv(result_path + column_name + '_' + features + '_normalized_labeled_data.csv')
    sel_unlabeled_data4.to_csv(result_path + column_name + '_' + features + '_normalized_unlabeled_data.csv')
    return labels, sel_labeled_data4, sel_unlabeled_data4


def test_data_preprocessing(data_path,
                            normalization='min_max'):

    """
    Feature selection using ModelX knockoffs.

    Args:
        data_path: Path to the data files
        normalization: data normalization method

    Returns:
        Selected data. labels, sel_labeled_data, sel_unlabeled_data
    """

    # Check the availability of data else download
    get_pred_data(data_path)

    logging.info('Loading data')

    test_data = pd.read_csv(data_path+'test_data.csv', index_col=0, header=0, sep=',', low_memory=False)
    features = pd.read_csv(data_path + 'selected_features.csv', index_col=None, header=0, sep=',', low_memory=False)
    features = features.values.ravel()
    selected_test_data = test_data[features]

    model_path = data_path + 'model.pkl'
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    if normalization == 'min_max':
        # Normalize data using min-max normalization between 0 and 1.
        normalized_test_data = min_max(selected_test_data.T).T
    elif normalization == 'standardization':
        # Normalize data using min-max normalization between 0 and 1.
        normalized_test_data = standardize_dataframe(selected_test_data.T).T
    elif normalization == 'None':
        normalized_test_data = selected_test_data
    else:
        raise ValueError('Please provide a valid normalization method!')

    return model, normalized_test_data

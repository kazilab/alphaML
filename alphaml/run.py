import warnings
import time
import os
import logging
from .utils.prep import preprocessing
from .models import *


def aml(model_name='XGBoost',
        sampling_method="no",
        param_search="Optuna",
        col_label='Trametinib',
        pos_class='resistant',
        neg_class='sensitive',
        test_size=0.2,
        random_state=12,
        n_trials=50,
        cv=5,
        max_epochs=500,
        patience=50,
        features='HVF',
        features2='None',
        features3='None',
        features4='None',
        n_top_features=1000,
        min_disp=1.0,
        log_type='log2',
        top_n_genes_pca=500,
        top_n_genes_rp=500,
        n_clusters=500,
        n_components=500,
        max_features=500,
        threshold='1.0*mean',
        min_sel_features_rfe=10,
        min_sel_features_sfs=10,
        latent_dim=32,
        min_sel_features=10,
        fdr=0.25,
        normalization='min_max',
        run_feat_imp='No',
        run_shap='No',
        run_lime='No',
        run_roc='No',
        control_fitting='Yes'
        ):

    # Check the availability of result and log folder otherwise create
    user_documents = os.path.expanduser("~/Documents")
    data_path = os.path.join(user_documents, "alphaML_data/")
    result_path = os.path.join(user_documents, "alphaML_results/")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    f_name = col_label + '_' + sampling_method + '_' + param_search + '_' + model_name + '_fit_' + control_fitting
    # Set the log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
    log_level = logging.INFO
    log_file_path = os.path.abspath(result_path + f"{f_name}_alphaML_run.log")
    # Configure logging settings
    logging.basicConfig(filename=log_file_path,
                        level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    start = time.strftime("%Y-%m-%d %H:%M:%S")
    warnings.filterwarnings("ignore")
    logging.info(f'* Analysis started for {model_name}, sampling method {sampling_method}, hyperparam {param_search}*')
    print(f'* Analysis started for {model_name}, sampling method {sampling_method}, hyperparam {param_search}*')
    logging.info('Start time : %s', start)
    print('*** Analysis started *** @ ', start)

    axis_labels = (pos_class, neg_class)

    labels, labeled_data, unlabeled_data = preprocessing(data_path=data_path,
                                                         result_path=result_path,
                                                         column_name=col_label,
                                                         pos_class=pos_class,
                                                         neg_class=neg_class,
                                                         features=features,
                                                         features2=features2,
                                                         features3=features3,
                                                         features4=features4,
                                                         n_top_features=n_top_features,
                                                         min_disp=min_disp,
                                                         log_type=log_type,
                                                         top_n_genes_pca=top_n_genes_pca,
                                                         top_n_genes_rp=top_n_genes_rp,
                                                         n_clusters=n_clusters,
                                                         n_components=n_components,
                                                         max_features=max_features,
                                                         threshold=threshold,
                                                         min_sel_features_rfe=min_sel_features_rfe,
                                                         min_sel_features_sfs=min_sel_features_sfs,
                                                         latent_dim=latent_dim,
                                                         min_sel_features=min_sel_features,
                                                         fdr=fdr,
                                                         normalization=normalization)

    models = [(abc, 'AdaBoost'),
              (cbc, 'CatBoost'),
              (eln, 'ElasticNet'),
              (etc, 'ExtraTrees'),
              (gbc, 'GradientBoosting'),
              (hgb, 'HistGradientBoosting'),
              (las, 'LASSO'),
              (lgb, 'LightGBM'),
              (mlp, 'MLPC'),
              (rfc, 'RandomForest'),
              (rid, 'Ridge'),
              (svc, 'SVC'),
              (svn, 'NuSVC'),
              (tab, 'TabNet'),
              (xgb, 'XGBoost'),
              (all, 'Test_Briefly')
              ]

    if param_search == 'Optuna':
        control_fitting = control_fitting
    else:
        control_fitting = 'No'
    # Run single model #
    sel_model = None
    for model, name in models:
        if name == model_name:
            sel_model = model
            print('data loaded for', name)
            pscore = model(labeled_data=labeled_data,
                           labels=labels,
                           col_label=col_label,
                           axis_labels=axis_labels,
                           path=result_path,
                           test_size=test_size,
                           random_state=random_state,
                           n_trials=n_trials,
                           cv=cv,
                           sampling_method=sampling_method,
                           param_search=param_search,
                           run_feat_imp=run_feat_imp,
                           run_shap=run_shap,
                           run_lime=run_lime,
                           run_roc=run_roc,
                           unlabeled_data=unlabeled_data,
                           patience=patience,
                           max_epochs=max_epochs,
                           control_fitting=control_fitting
                           )
            print('NegLog2RMSL: ', pscore)
            break

    if sel_model is None:
        print('No prediction model will be built')
    else:
        pass
    print('**** Analysis finished ****')
    finished = time.strftime("%Y-%m-%d %H:%M:%S")
    print('Finish time :', finished)
    del (sampling_method, param_search, result_path, labels, labeled_data, unlabeled_data)
    logging.info('**** Analysis finished ****')
    finished = time.strftime("%Y-%m-%d %H:%M:%S")
    logging.info('Finish time : %s', finished)
    print('\n **** Analysis finished **** @ ', finished)

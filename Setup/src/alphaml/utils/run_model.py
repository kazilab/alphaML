import pandas as pd
import numpy as np
import logging
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .utils import sampling, shap_, lime_
from .metrics import confusion_matrix_
from .all_ml import run_ml_models
# ****************** Run model ******************************


def run_model(algorithm,
              labeled_data,
              labels,
              col_label,
              axis_labels,
              path,
              test_size,
              random_state,
              n_trials,
              cv,
              sampling_method,
              param_search,
              optim_,
              roc_auc,
              run_feat_imp,
              run_shap,
              run_lime,
              run_roc,
              unlabeled_data,
              patience,
              max_epochs,
              control_fitting
              ):
    """
    Build and use a TabNet model and calculate scores.
    
    Args:
        algorithm: the name of algorithm
        labeled_data: labeled data matrix.
        labels: Labels for data.
        col_label: Label of class column
        axis_labels: Axis labels for data confusion metrics, put positive class first and then negative class.
        path: Path to store model and log files.
        test_size: Test size to divide the data
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        optim_: optimization function
        roc_auc: roc_auc function
        run_feat_imp: Calculate feature importance
        run_shap: Calculate SHAP plots
        run_lime: Calculate lime plots
        run_roc: Calculate AUC-ROC curve using 5-fold cv
        unlabeled_data: unlabeled data matrix.
        patience: Number of epochs with no improvement before early stopping.
        max_epochs: Maximum number of epochs to train for.
        control_fitting: Whether NegLog2RMSL will be used.
    
    Returns:
        Save results in the alphaML_Results folder.
    """
    logging.info(f'**** Building model with  ****')
    print(f'**** Building model with {algorithm} ****')
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    # Sampling methods
    sampling_methods = ['no', 'over', 'under']
    if sampling_method not in sampling_methods:
        raise ValueError(f"Invalid sampling method '{sampling_method}'."
                         f" Available options are: {', '.join(sampling_methods[:-1])}, {sampling_methods[-1]}")

    # Separate 20% test data for testing the model before using sampling methods
    train_data, test_data, train_labels, test_data_labels = train_test_split(labeled_data,
                                                                             labels,
                                                                             test_size=float(test_size),
                                                                             random_state=int(random_state)
                                                                             )
    # Use defined sampling methods, under sampling, overs sampling or no sampling.

    # Data for model optimization and building
    data, data_labels = sampling(x_train=train_data,
                                 y_train=train_labels,
                                 sampling_method=sampling_method
                                 )
    # Data for only repeated model building and auc_roc calculation
    labeled_data_s, labels_s = sampling(x_train=labeled_data,
                                        y_train=labels,
                                        sampling_method=sampling_method
                                        )

    if algorithm == 'all_ml':
        logging.info(f'Running machine learning models for a brief test')
        print(f'Running machine learning models for a brief test')
        run_ml_models(x_train=data,
                      x_test=test_data,
                      y_train=data_labels,
                      y_test=test_data_labels,
                      path=path,
                      random_state=random_state,
                      sampling_method=sampling_method,
                      col_label=col_label
                      )
        pscore = np.nan

    elif algorithm == 'TabNetClassifier':
        # use _optim method to calculate parameters, and to build supervised and unsupervised models.

        parameters, unsupervised_model, model, name_, pscore = optim_(data=data,
                                                                      test_data=test_data,
                                                                      data_labels=data_labels,
                                                                      test_data_labels=test_data_labels,
                                                                      path=path,
                                                                      random_state=random_state,
                                                                      sampling_method=sampling_method,
                                                                      param_search=param_search,
                                                                      max_epochs=max_epochs,
                                                                      patience=patience,
                                                                      unlabeled_data=unlabeled_data,
                                                                      test_size=test_size,
                                                                      cv=cv,
                                                                      n_trials=n_trials,
                                                                      col_label=col_label,
                                                                      control_fitting=control_fitting
                                                                      )

        params_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['values']).T
        params_df['n_steps'] = params_df['n_steps'].convert_dtypes(False, False, True, False, False)
        params_df['n_d'] = params_df['n_d'].convert_dtypes(False, False, True, False, False)
        params_df['n_a'] = params_df['n_a'].convert_dtypes(False, False, True, False, False)
        params_df['n_independent'] = params_df['n_independent'].convert_dtypes(False, False, True, False, False)
        params_df['n_shared'] = params_df['n_shared'].convert_dtypes(False, False, True, False, False)
        params_df = params_df.T
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=params_df.values, rowLabels=params_df.index, loc='center')
        fig.savefig(f"{p_name}_parameters.pdf")
        logging.info(f"{algorithm} model parameters have been saved in alphaML_results folder in a pdf file")
        plt.close()
        del params_df, fig

        if run_feat_imp == 'Yes':
            logging.info(f'Calculating feature importance for {algorithm}')
            print(f'Calculating feature importance for {algorithm}')
            feat_imp = pd.DataFrame(model.feature_importances_)
            perm_imp1 = permutation_importance(model,
                                               test_data.to_numpy(),
                                               test_data_labels.to_numpy(),
                                               scoring='accuracy'
                                               )
            perm_imp = pd.DataFrame(perm_imp1.importances_mean)
            combined = pd.concat([feat_imp, perm_imp], axis=1).T
            combined.columns = test_data.T.reset_index().iloc[:, 0].values.ravel()  # use gene name from the index
            combined_ = combined.T
            combined_.columns = ['Importance score', 'Permutation importance']
            combined_.index.names = ['Gene name']
            file_= f"{p_name}_global_feat_imp.csv"
            combined_.to_csv(file_)
            logging.info(f"{algorithm} global feature importance file has been saved in alphaML_results folder")
            del feat_imp, perm_imp1, perm_imp, combined, combined_
        else:
            logging.info(f'Feature importance has not been calculated for {algorithm}! ')
            logging.info(f'Please pass run_feat_imp="Yes" to calculate feature importance.')

        # *** Drawing confusion matrix ***
        logging.info(f'Drawing confusion matrix for {algorithm}')
        print(f'Drawing confusion matrix for {algorithm}')
        confusion_matrix_(model=model,
                          x_test=test_data.to_numpy(),
                          y_true=test_data_labels.to_numpy(),
                          path=path,
                          col_label=col_label,
                          sampling_method=sampling_method,
                          param_search=param_search,
                          axis_labels=axis_labels,
                          algorithm=algorithm,
                          control_fitting=control_fitting
                          )

        # *** Generating SHAP plots ***
        if run_shap == 'Yes':
            logging.info(f'Feature importance using SHAP for {algorithm}')
            print(f'Feature importance using SHAP for {algorithm}')
            shap_(model=model,
                  x_test=test_data,
                  path=path,
                  sampling_method=sampling_method,
                  param_search=param_search,
                  algorithm=algorithm,
                  col_label=col_label,
                  control_fitting=control_fitting
                  )
        else:
            logging.info('SHAP plot has not been generated! Please pass run_shap="Yes" to generate SHAP plot.')

        # *** Generating LIME plots ***
        if run_lime == 'Yes':
            logging.info(f'Feature importance using LIME for {algorithm}')
            print(f'Feature importance using LIME for {algorithm}')
            lime_(model=model,
                  x_test=test_data,
                  path=path,
                  sampling_method=sampling_method,
                  param_search=param_search,
                  algorithm=algorithm,
                  col_label=col_label,
                  control_fitting=control_fitting
                  )
        else:
            logging.info(f'LIME plot has not been generated for {algorithm}! '
                         f'Please pass run_lime="Yes" to generate LIME plot for {algorithm}.')

        # *** Generating ROC plot ***
        if run_roc == 'Yes':
            logging.info('Drawing ROC curve')
            print('Drawing ROC curve')
            roc_auc(labeled_data_s=labeled_data_s,
                    labels_s=labels_s,
                    path=path,
                    parameters=parameters,
                    random_state=random_state,
                    n_trials=n_trials,
                    cv=cv,
                    sampling_method=sampling_method,
                    param_search=param_search,
                    max_epochs=max_epochs,
                    patience=patience,
                    data=data,
                    unsupervised_model=unsupervised_model,
                    col_label=col_label,
                    control_fitting=control_fitting
                    )
        else:
            logging.info('AUC-ROC curve has not been generated! Please pass run_roc=True to generate AUC-ROC curve.')
        pscore = pscore

    else:
        # use _optim method to calculate parameters, and to build supervised and unsupervised models.
        # Parameter search methods
        p_search_methods = ['Optuna', 'Bayes', 'Grid', 'Predefined']
        if param_search not in p_search_methods:
            raise ValueError(f"Invalid parameter search method '{param_search}'. "
                             f"Available options are: {', '.join(p_search_methods[:-1])}, {p_search_methods[-1]}")
        logging.info(f'Using {sampling_method} sampling method and {param_search} for this analysis!')
        print(f'Using {sampling_method} sampling method and {param_search} for this analysis!')

        parameters, model, optim_model, pscore = optim_(data=data,
                                                        test_data=test_data,
                                                        data_labels=data_labels,
                                                        test_data_labels=test_data_labels,
                                                        path=path,
                                                        random_state=random_state,
                                                        sampling_method=sampling_method,
                                                        param_search=param_search,
                                                        cv=cv,
                                                        n_trials=n_trials,
                                                        col_label=col_label,
                                                        control_fitting=control_fitting
                                                        )
        params_df = pd.DataFrame([parameters]).T
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=params_df.values, rowLabels=params_df.index, loc='center')
        fig.savefig(f"{p_name}_parameters.pdf")
        logging.info(f"{algorithm} model parameters have been saved in venDx_results folder in a pdf file")
        plt.close()
        del params_df, fig

        if run_feat_imp == 'Yes':
            logging.info(f'Calculating feature importance for {algorithm}')
            print(f'Calculating feature importance for {algorithm}')

            if hasattr(model, 'feature_importances_'):
                feat_imp = pd.DataFrame(model.feature_importances_)
            else:
                logging.info(f"{algorithm} does not have built-in feature importances. Skipping.")
                print(f"{algorithm} does not have built-in feature importances. Skipping.")
                feat_imp = pd.DataFrame(np.nan, index=[0], columns=range(len(test_data.columns))).T
            perm_imp1 = permutation_importance(estimator=model,
                                               X=test_data.to_numpy(),
                                               y=test_data_labels.to_numpy(),
                                               scoring='accuracy',
                                               n_repeats=5,
                                               random_state=random_state
                                               )
            perm_imp = pd.DataFrame(perm_imp1.importances_mean)
            combined = pd.concat([feat_imp, perm_imp], axis=1).T
            combined.columns = test_data.T.reset_index().iloc[:, 0].values.ravel()  # use gene name from the index
            combined_ = combined.T
            combined_.columns = ['Importance score', 'Permutation importance']
            combined_.index.names = ['Gene name']
            combined_.to_csv(f"{p_name}_global_feat_imp.csv")
            logging.info(f"{algorithm} global feature importance file has been saved in venDx_results folder")
            del feat_imp, perm_imp1, perm_imp, combined, combined_
        else:
            logging.info(f'Feature importance have not been calculated for {algorithm}! ')
            logging.info(f'Please pass run_feat_imp="Yes" to calculate feature importance for {algorithm}.')

        # *** Drawing confusion matrix ***
        logging.info(f'Drawing confusion matrix for {algorithm}')
        print(f'Drawing confusion matrix for {algorithm}')
        confusion_matrix_(model=model,
                          x_test=test_data,
                          y_true=test_data_labels,
                          path=path,
                          col_label=col_label,
                          sampling_method=sampling_method,
                          param_search=param_search,
                          axis_labels=axis_labels,
                          algorithm=algorithm,
                          control_fitting=control_fitting
                          )

        # *** Generating SHAP plots ***
        if run_shap == 'Yes':
            logging.info(f'Feature importance using SHAP for {algorithm}')
            print(f'Feature importance using SHAP for {algorithm}')
            shap_(model=model,
                  x_test=test_data,
                  path=path,
                  sampling_method=sampling_method,
                  param_search=param_search,
                  algorithm=algorithm,
                  col_label=col_label,
                  control_fitting=control_fitting
                  )
        else:
            logging.info(f'SHAP plot has not been generated for {algorithm}! '
                         f'Please pass run_shap="Yes" to generate SHAP plot for {algorithm}.')

        # *** Generating LIME plots ***
        if run_lime == 'Yes':
            logging.info(f'Feature importance using LIME for {algorithm}')
            print(f'Feature importance using LIME for {algorithm}')
            lime_(model=model,
                  x_test=test_data,
                  path=path,
                  sampling_method=sampling_method,
                  param_search=param_search,
                  algorithm=algorithm,
                  col_label=col_label,
                  control_fitting=control_fitting
                  )
        else:
            logging.info(f'LIME plot has not been generated for {algorithm}! '
                         f'Please pass run_lime="Yes" to generate LIME plot for {algorithm}.')

        # *** Generating ROC plot ***
        if run_roc == 'Yes':
            logging.info(f'Drawing ROC curve for {algorithm}')
            print(f'Drawing ROC curve for {algorithm}')
            roc_auc(labeled_data_s=labeled_data_s,
                    labels_s=labels_s,
                    path=path,
                    parameters=parameters,
                    random_state=random_state,
                    n_trials=n_trials,
                    cv=cv,
                    sampling_method=sampling_method,
                    param_search=param_search,
                    col_label=col_label,
                    control_fitting=control_fitting
                    )
        else:
            logging.info(f'AUC-ROC curve has not been generated for {algorithm}! '
                         f'Please pass run_roc="Yes" to generate AUC-ROC curve for {algorithm}.')
        pscore = pscore

    del (data, test_data, data_labels, test_data_labels, path, random_state, sampling_method, param_search,
         labeled_data, labels, test_size, n_trials, cv, run_feat_imp, run_shap, run_roc, labeled_data_s, labels_s)
    return pscore

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, make_scorer
import optuna
from optuna import visualization
from optuna.samplers import TPESampler
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer

import matplotlib.pyplot as plt
import plotly.io as pio
import pickle
from ..utils.run_model import run_model
from ..utils.metrics import test_prediction, neglog2rmsl, custom_score, kappa_mcc_error
from ..utils.utils import plot_roc_auc_acc
from lightgbm import LGBMClassifier as Classifier
algorithm = 'LGBMClassifier'

# **** Build a model and calculate scores ****


def model_(data,
           test_data,
           data_labels,
           test_data_labels,
           path,
           parameters,
           sampling_method,
           param_search,
           col_label,
           control_fitting
           ):
    """
    Build a model and calculate scores.
    
    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        path: Path to store model and log files.
        parameters: model parameter's
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores.
    
    Returns:
        A tuple containing the unsupervised model, the supervised model, and a factor for performance.
    """
    logging.info(f'Building Model using {algorithm}')
    print(f'Building Model using {algorithm}')
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    model = Classifier(**parameters,
                       metric=['average_precision', 'binary_error'],
                       verbose=-1,
                       class_weight='balanced'
                       )
    model.fit(data, data_labels,
              eval_set=[(data, data_labels), (test_data, test_data_labels)],
              verbose=-1
              )

    filename = f"{p_name}_model.pkl"
    pickle.dump(model, open(filename, 'wb'))
    logging.info(f"{algorithm} model saved in alphaML_result folder")
    # Retrieve the evaluation results
    eval_results = model.evals_result_
    # Calculate accuracies
    train_map = eval_results['training']['average_precision']
    train_error = eval_results['training']['binary_error']
    valid_map = eval_results['valid_1']['average_precision']
    valid_error = eval_results['valid_1']['binary_error']

    # Plot the accuracies against n_estimator
    plt.plot(train_map, label="Train Average Precision", color="red")
    plt.plot(valid_map, label="Test Average Precision", color="green")
    plt.plot(train_error, label="Train Binary error", color="orange")
    plt.plot(valid_error, label="Test Binary error", color="blue")
    plt.xlabel("n_estimator")
    plt.ylabel("Scores")
    plt.title("Scores vs n_estimator")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    filename1 = f"{p_name}_valid_accuracy_curve.pdf"
    fig.savefig(filename1)
    plt.close()
    logging.info(f"Validation accuracy curve for {algorithm} has been saved in alphaML_result folder")
    del fig

    # Export test result with probabilities
    combined_df = test_prediction(model, test_data, test_data_labels)
    filename2 = f"{p_name}_test_prediction.xlsx"
    combined_df.to_excel(filename2)
    logging.info(f"Test predictions for {algorithm} has been saved in alphaML_result folder")
    del combined_df

    # Calculate scores for test and train data, and negative log2 RMSL
    all_score_df, factor = neglog2rmsl(model, data, data_labels, test_data, test_data_labels)
    all_score_df.to_csv(f"{p_name}_scores.csv")
    logging.info(f"Scores for {algorithm} have been saved in alphaML_result folder in a csv file")
    return model, factor

# ****** ROC AUC Curve ******


def roc_auc(labeled_data_s,
            labels_s,
            path,
            parameters,
            random_state,
            n_trials,
            cv,
            sampling_method,
            param_search,
            col_label,
            control_fitting
            ):
    """
    Draw AUC-ROC curve using 5-fold cv.
    
    Args:

        labeled_data_s: Labeled data after sampling
        labels_s: Labels after sampling
        parameters: model parameter's
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train score.
    
    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Building RepeatedKFold {algorithm} Model')
    print(f'Building RepeatedKFold {algorithm} Model')
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    kf = RepeatedKFold(n_splits=cv,
                       n_repeats=n_trials,
                       random_state=random_state
                       )
    metrics = ['auc', 'fpr', 'tpr', 'accuracy', 'thresholds']
    results = {m: [] for m in metrics}
    xx = labeled_data_s.values
    yy = labels_s.values
    for train, test in kf.split(xx):
        x_train, x_test = xx[train], xx[test]
        y_train, y_test = yy[train], yy[test]
        clf_roc = Classifier(**parameters,
                             verbose=-1,
                             class_weight='balanced'
                             )
        # Wrap the custom evaluation metric to pass additional parameters
        clf_roc.fit(x_train, y_train)
        y_preds = clf_roc.predict(x_test)
        labels = y_test
        fpr, tpr, thresholds = roc_curve(labels, y_preds)
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['thresholds'].append(thresholds)
        results['accuracy'].append(accuracy_score(labels, y_preds))
        results['auc'].append(roc_auc_score(labels, y_preds))
    
    file_auc = f"{p_name}_repeated_accuracy.csv"
    pd.DataFrame(results).to_csv(file_auc)
    logging.info(f"Repeated accuracy scores for {algorithm} have been saved in alphaML_result folder in a csv file")
    # Draw the curve and plots
    fig, fig1 = plot_roc_auc_acc(n_trials, cv, results)
    pio.write_image(fig, f"{p_name}_AUC_curve.pdf")
    logging.info(f"Repeated AUC curve for {algorithm} has been saved in alphaML_result folder")
    f1_f = f"{p_name}_accuracy_plot_k_fold_cv_roc.pdf"
    pio.write_image(fig1, f1_f)
    logging.info(f"Repeated accuracy plot for {algorithm} has been saved in alphaML_result folder")
    del results, fig1, fig, clf_roc, xx, yy, kf, metrics, labeled_data_s, labels_s


# ***** Parameter optimization *****
# ***** Parameter search by OPTUNA *****


def optuna_(data,
            test_data,
            data_labels,
            test_data_labels,
            path,
            random_state,
            sampling_method,
            param_search,
            cv,
            n_trials,
            col_label,
            control_fitting
            ):
    """
    Perform Optuna hyperparameter search for a given model.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores.

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Searching parameter for {algorithm} using Optuna')
    print(f'Searching parameter for {algorithm} using Optuna')
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.05, step=0.005)
        max_depth = trial.suggest_int("max_depth", 5, 8, step=1)
        subsample = trial.suggest_float("subsample", 0.65, 0.9, step=0.05)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.65, 0.9, step=0.05)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 4, step=0.5)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 4, step=0.5)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 4, step=1)
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=10)

        params_ = dict(learning_rate=learning_rate,
                       max_depth=max_depth,
                       subsample=subsample,
                       colsample_bytree=colsample_bytree,
                       reg_alpha=reg_alpha,
                       reg_lambda=reg_lambda,
                       min_child_weight=min_child_weight,
                       random_state=random_state,
                       n_estimators=n_estimators
                       )
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        cv_score_array = []
        xx = data.to_numpy()
        yy = data_labels.to_numpy()
        for train_index, test_index in kf.split(xx):
            x_train, x_valid = xx[train_index], xx[test_index]
            y_train, y_valid = yy[train_index], yy[test_index]
            clf_opti = Classifier(**params_, verbose=-1,
                                  class_weight='balanced')
            clf_opti.fit(x_train, y_train)
            if control_fitting == 'Yes':
                cs = custom_score(clf_opti, x_train, y_train, x_valid, y_valid)
            else:
                cs = kappa_mcc_error(y_valid, clf_opti.predict(x_valid))
            cv_score_array.append(cs)
            # print('cv_score: ', cs)
        avg = np.mean(cv_score_array)
        # print('Average cv_score: ', avg)
        return avg

    study = optuna.create_study(direction="minimize",
                                study_name='Optuna optimization',
                                sampler=TPESampler()
                                )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=36000)
    fig = visualization.plot_param_importances(study)
    filename = f"{p_name}_optuna_param_importance.pdf"
    fig.write_image(filename)
    logging.info(f"Optuna parameter importance for {algorithm} has been saved in alphaML_result folder")

    additional_parameters = {
        'random_state': random_state
    }
    parameters = study.best_params
    parameters.update(additional_parameters)

    model, performance = model_(data=data,
                                test_data=test_data,
                                data_labels=data_labels,
                                test_data_labels=test_data_labels,
                                path=path,
                                parameters=parameters,
                                sampling_method=sampling_method,
                                param_search=param_search,
                                col_label=col_label,
                                control_fitting=control_fitting
                                )
    opti_model = 'Optuna'
    return parameters, model, opti_model, performance


# ***** BayesSearchCV *****


def bayes_(data,
           test_data,
           data_labels,
           test_data_labels,
           path,
           random_state,
           sampling_method,
           param_search,
           cv,
           n_trials,
           col_label,
           control_fitting
           ):
    """
    Perform Bayesian optimization using BayesSearchCV for a given model.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores.

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Searching parameter for {algorithm} using BayesSearchCV')
    print(f'Searching parameter for {algorithm} using BayesSearchCV')

    space = {'learning_rate': Categorical([0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]),
             'max_depth': Integer(5, 8),
             'subsample': Categorical([0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
             'colsample_bytree': Categorical([0.65, 0.7, 0.75, 0.8, 0.85, 0.9]),
             'reg_alpha': Categorical([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
             'reg_lambda': Categorical([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
             'min_child_weight': Integer(1, 4),
             'n_estimators': Integer(100, 300),
             }
    opt = BayesSearchCV(Classifier(verbose=-1,
                                   random_state=random_state,
                                   class_weight='balanced'
                                   ),
                        search_spaces=space,
                        scoring=make_scorer(kappa_mcc_error, greater_is_better=False),
                        n_iter=n_trials,
                        cv=cv,
                        verbose=0,
                        random_state=random_state
                        )
    opt.fit(data, data_labels)

    additional_parameters = {
        'random_state': random_state
    }
    parameters = opt.best_params_
    parameters.update(additional_parameters)

    model, performance = model_(data=data,
                                test_data=test_data,
                                data_labels=data_labels,
                                test_data_labels=test_data_labels,
                                path=path,
                                parameters=parameters,
                                sampling_method=sampling_method,
                                param_search=param_search,
                                col_label=col_label,
                                control_fitting=control_fitting
                                )
    opti_model = 'Bayes'
    return parameters, model, opti_model, performance


# ****** GridSearchCV *******


def grid_(data,
          test_data,
          data_labels,
          test_data_labels,
          path,
          random_state,
          sampling_method,
          param_search,
          cv,
          n_trials,
          col_label,
          control_fitting
          ):
    """
    Perform optimization using GridSearchCV for a given model.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores.

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Searching parameter for {algorithm} using GridSearchCV')
    print(f'Searching parameter for {algorithm} using GridSearchCV')

    space = {'learning_rate': [0.005, 0.01, 0.05],
             'max_depth': [6, 7, 8],
             'subsample': [0.75],
             'colsample_bytree': [0.75],
             'reg_alpha': [1, 2],
             'reg_lambda': [1, 2],
             'min_child_weight': [2, 3],
             'n_estimators': [150, 175, 200, 225, 250, 275]
             }

    opt = GridSearchCV(Classifier(verbose=-1,
                                  random_state=random_state,
                                  class_weight='balanced'
                                  ),
                       param_grid=space,
                       scoring=make_scorer(kappa_mcc_error, greater_is_better=False),
                       cv=cv,
                       verbose=0
                       )

    opt.fit(data, data_labels)

    additional_parameters = {
        'random_state': random_state
    }
    parameters = opt.best_params_
    parameters.update(additional_parameters)

    model, performance = model_(data=data,
                                test_data=test_data,
                                data_labels=data_labels,
                                test_data_labels=test_data_labels,
                                path=path,
                                parameters=parameters,
                                sampling_method=sampling_method,
                                param_search=param_search,
                                col_label=col_label,
                                control_fitting=control_fitting
                                )
    opti_model = 'Grid'
    return parameters, model, opti_model, performance


# ***** Predefined parameter *****


def predefined_(data,
                test_data,
                data_labels,
                test_data_labels,
                path,
                random_state,
                sampling_method,
                param_search,
                cv,
                n_trials,
                col_label,
                control_fitting
                ):
    """
    Build a model using Predefined parameters.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores.

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    # Hardcoded Predefined parameters
    # These parameters can be obtained from prior experiments, domain knowledge, or literature
    logging.info(f'{algorithm} Model will be built using Predefined parameters')
    print(f'{algorithm} Model will be built using Predefined parameters')

    parameters = {'learning_rate': 0.0075,
                  'max_depth': 6,
                  'subsample': 0.75,
                  'colsample_bytree': 0.75,
                  'reg_alpha': 2.5,
                  'reg_lambda': 2,
                  'min_child_weight': 2.5,
                  'n_estimators': 250,
                  'random_state': random_state
                  }
    model, performance = model_(data=data,
                                test_data=test_data,
                                data_labels=data_labels,
                                test_data_labels=test_data_labels,
                                path=path,
                                parameters=parameters,
                                sampling_method=sampling_method,
                                param_search=param_search,
                                col_label=col_label,
                                control_fitting=control_fitting
                                )
    opti_model = 'Predefined'
    return parameters, model, opti_model, performance


def optim_(data,
           test_data,
           data_labels,
           test_data_labels,
           path,
           random_state,
           sampling_method,
           param_search,
           cv,
           n_trials,
           col_label,
           control_fitting
           ):
    """
    Trains a model using a specified hyperparameter search method.

    Args:
        data: data matrix.
        test_data: Test data matrix.
        data_labels: Labels for data.
        test_data_labels: Labels for test data.
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores.

    Returns:
        A dictionary containing the best hyperparameter, the unsupervised model,
        the supervised model, and the training history.
    """
    logging.info(f'Running hyperparameter optimization for {algorithm}')
    print(f'Running hyperparameter optimization for {algorithm}')
    search_methods = {
        'Optuna': optuna_,
        'Bayes': bayes_,
        'Grid': grid_,
        'Predefined': predefined_
    }

    if param_search not in search_methods:
        error_message = "Please suggest a search parameter from the Best, Optuna, Bayes, Grid, and Predefined"
        logging.error(error_message)
        raise ValueError(error_message)

    search_method = search_methods[param_search]

    parameters, model, optim_model, pscore = search_method(data,
                                                           test_data,
                                                           data_labels,
                                                           test_data_labels,
                                                           path,
                                                           random_state,
                                                           sampling_method,
                                                           param_search,
                                                           cv,
                                                           n_trials,
                                                           col_label,
                                                           control_fitting
                                                           )

    return parameters, model, optim_model, pscore

# run model


def building(labeled_data,
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
             run_feat_imp,
             run_shap,
             run_lime,
             run_roc,
             unlabeled_data,
             patience,
             max_epochs,
             control_fitting
             ):
    pscore = run_model(algorithm=algorithm,
                       labeled_data=labeled_data,
                       labels=labels,
                       col_label=col_label,
                       axis_labels=axis_labels,
                       path=path,
                       test_size=test_size,
                       random_state=random_state,
                       n_trials=n_trials,
                       cv=cv,
                       sampling_method=sampling_method,
                       param_search=param_search,
                       optim_=optim_,
                       roc_auc=roc_auc,
                       run_feat_imp=run_feat_imp,
                       run_shap=run_shap,
                       run_lime=run_lime,
                       run_roc=run_roc,
                       unlabeled_data=unlabeled_data,
                       patience=patience,
                       max_epochs=max_epochs,
                       control_fitting=control_fitting
                       )
    return pscore

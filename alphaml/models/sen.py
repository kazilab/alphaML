import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, make_scorer, brier_score_loss
import optuna
from optuna import visualization
from optuna.samplers import TPESampler
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import matplotlib.pyplot as plt
import plotly.io as pio
import pickle
from ..utils.run_model import run_model
from ..utils.metrics import test_prediction, neglog2rmsl, custom_score, kappa_mcc_error
from ..utils.utils import plot_roc_auc_acc
from sklearn.linear_model import SGDClassifier as Classifier
algorithm = 'ElasticNet'

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
                       loss='log_loss',
                       class_weight='balanced', penalty='elasticnet'
                       )
    model.fit(data, data_labels)
    filename = f"{p_name}_model.pkl"
    pickle.dump(model, open(filename, 'wb'))
    logging.info(f"{algorithm} model saved in alphaML_result folder")

    # To compare train and test accuracy and loss
    max_iter = int(parameters['max_iter'])
    parm_del_max_iter = parameters.copy()
    del parm_del_max_iter['max_iter']
    max_iter_spaces = np.linspace(1, max_iter, max_iter, endpoint=True, dtype=int)

    results = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for max_iter_space in max_iter_spaces:
        rf = Classifier(max_iter=max_iter_space,
                        **parm_del_max_iter,
                        loss='log_loss',
                        class_weight='balanced', penalty='elasticnet'
                        )
        rf.fit(data, data_labels)
        train_acc = accuracy_score(data_labels, rf.predict(data))
        test_acc = accuracy_score(test_data_labels, rf.predict(test_data))
        train_loss = brier_score_loss(data_labels, rf.predict_proba(data)[:, 1])
        test_loss = brier_score_loss(test_data_labels, rf.predict_proba(test_data)[:, 1])
        for key, value in zip(results.keys(), [train_acc, test_acc, train_loss, test_loss]):
            results[key].append(value)

    plt.figure(figsize=(6, 4))
    colors = {'train_acc': 'r', 'test_acc': 'g', 'train_loss': 'b', 'test_loss': 'c'}
    labels = {'train_acc': 'Train Accuracy', 'test_acc': 'Test Accuracy', 'train_loss': 'Train loss',
              'test_loss': 'Test loss'}
    for key, color in colors.items():
        plt.plot(max_iter_spaces, results[key], color, label=labels[key])
    plt.ylabel('Score')
    plt.xlabel('n_estimators')
    plt.title("Scores vs n_estimators")
    plt.legend()
    fig = plt.gcf()
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
                             loss='log_loss',
                             class_weight='balanced', penalty='elasticnet'
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
        alpha = trial.suggest_float("alpha", 0.0001, 0.01, step=0.0005)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9, step=0.05)
        max_iter = trial.suggest_int("max_iter", 500, 1500, step=100)

        params_ = dict(alpha=alpha,
                       random_state=random_state,
                       max_iter=max_iter,
                       l1_ratio=l1_ratio
                       )
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        cv_score_array = []
        xx = data.to_numpy()
        yy = data_labels.to_numpy()
        for train_index, test_index in kf.split(xx):
            x_train, x_valid = xx[train_index], xx[test_index]
            y_train, y_valid = yy[train_index], yy[test_index]
            clf_opti = Classifier(**params_,
                                  loss='log_loss',
                                  class_weight='balanced', penalty='elasticnet'
                                  )
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

    space = {'alpha': Real(0.0001, 0.01),
             'l1_ratio': Real(0.1, 0.9),
             'max_iter': Integer(500, 1500),
             }
    opt = BayesSearchCV(Classifier(loss='log_loss',
                                   random_state=random_state,
                                   class_weight='balanced', penalty='elasticnet'
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

    space = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
             'l1_ratio': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6],
             'max_iter': [500, 750, 1000, 1250, 1500]
             }

    opt = GridSearchCV(Classifier(loss='log_loss',
                                  random_state=random_state,
                                  class_weight='balanced', penalty='elasticnet'
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

    parameters = {'alpha': 0.0001,
                  'max_iter': 1250,
                  'l1_ratio': 0.2,
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

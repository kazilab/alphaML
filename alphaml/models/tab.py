import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, make_scorer
import pickle
import optuna
from optuna import visualization
from optuna.samplers import TPESampler
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.io as pio
from ..utils.run_model import run_model
from ..utils.metrics import test_prediction, neglog2rmsl, custom_score, kappa_mcc_error
from ..utils.utils import plot_roc_auc_acc, batch_sizes
from ..utils.binary_tabnet import Classifier, TabPre
algorithm = 'TabNetClassifier'

# **** Build a model and calculate scores ****


def model_(data,
           test_data,
           data_labels,
           test_data_labels,
           path,
           parameters,
           sampling_method,
           param_search,
           unlabeled_data,
           test_size,
           random_state,
           max_epochs,
           patience,
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
        unlabeled_data: Unlabeled data
        test_size: The percentage of test size
        random_state: Random state number
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        control_fitting: If Yes, considers train scores.
        col_label: Label of class column.
    
    Returns:
        A tuple containing the unsupervised model, the supervised model, and a factor for performance.
    """
    logging.info(f'Building Model using {algorithm}')
    print(f'Building Model using {algorithm}')
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    x_train_un, x_test_un = train_test_split(unlabeled_data.fillna(0).to_numpy(),
                                             test_size=float(test_size),
                                             random_state=int(random_state)
                                             )
    x_train, x_valid, y_train, y_valid = train_test_split(data.to_numpy(),
                                                          data_labels.to_numpy(),
                                                          test_size=float(test_size),
                                                          random_state=int(random_state)
                                                          )

    unsupervised_model = TabPre(
                                **parameters
                                )
    logging.info(f'Fitting unsupervised {algorithm} model')
    print(f'Fitting unsupervised {algorithm} model')
    unsupervised_model.fit(
                           X_train=x_train_un,
                           eval_set=[x_test_un],
                           batch_size=batch_sizes(data)[0],
                           virtual_batch_size=batch_sizes(data)[1],
                           pretraining_ratio=0.8,
                           patience=patience,
                           max_epochs=max_epochs
                           )
    model = Classifier(**parameters
                       )
    logging.info(f'Fitting {algorithm} model')
    print(f'Fitting {algorithm} model')
    model.fit(X_train=x_train, y_train=y_train,
              eval_set=[(x_train, y_train), (x_valid, y_valid)],
              eval_name=['train', 'valid'],
              batch_size=batch_sizes(data)[0],
              virtual_batch_size=batch_sizes(data)[1],
              eval_metric=["accuracy"],
              patience=patience,
              max_epochs=max_epochs,
              weights=1,
              from_unsupervised=unsupervised_model
              )

    filename = f"{p_name}_model.pkl"
    pickle.dump(model, open(filename, 'wb'))
    # model.save_model(filename)
    logging.info(f"{algorithm} Model saved in alphaML_result folder")
    plt.plot(model.history['loss'], 'b', label='Train loss')
    plt.plot(model.history['train_accuracy'], 'r', label='Train accuracy')
    plt.plot(model.history['valid_accuracy'], 'g', label='Validation accuracy')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    filename1 = f"{p_name}_valid_accuracy_curve.pdf"
    fig.savefig(filename1)
    plt.close()
    logging.info(f"Validation accuracy curve for {algorithm} has been saved in alphaML_result folder")
    del fig

    # Export test result with probabilities
    combined_df = test_prediction(model, test_data.to_numpy(), test_data_labels)
    filename2 = f"{p_name}_test_prediction.xlsx"
    combined_df.to_excel(filename2)
    logging.info(f"Test predictions for {algorithm} has been saved in alphaML_result folder")
    del combined_df

    # Calculate scores for test and train data, and negative log2 RMSL
    all_score_df, factor = neglog2rmsl(model,
                                       data.to_numpy(),
                                       data_labels.to_numpy(),
                                       test_data.to_numpy(),
                                       test_data_labels.to_numpy()
                                       )
    all_score_df.to_csv(f"{p_name}_scores.csv")
    logging.info(f"Scores for {algorithm} have been saved in alphaML_result folder in a csv file")
    return unsupervised_model, model, factor

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
            max_epochs,
            patience,
            data,
            unsupervised_model,
            col_label,
            control_fitting
            ):
    """
    Draw AUC-ROC curve using 5-fold cv.
    
    Args:
        data: Data table
        labeled_data_s: Labeled data after sampling
        labels_s: Labels after sampling
        parameters: model parameter's
        path: Path to store model and log files.
        random_state: Random state for reproducibility.
        n_trials: Number of trials for hyperparameter search.
        cv: Number of cross-validation folds for hyperparameter search.
        sampling_method: Method for sampling data for cross-validation.
        param_search: Hyperparameter search method to use.
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        unsupervised_model: unsupervised model
        control_fitting: If Yes, considers train scores.
        col_label: Label of class column.

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
        x_train_tmp, x_test = xx[train], xx[test]
        y_train_tmp, y_test = yy[train], yy[test]
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_tmp,
                                                              y_train_tmp,
                                                              test_size=0.2,
                                                              random_state=int(random_state)
                                                              )
        clf_roc = Classifier(**parameters
                             )
        clf_roc.fit(X_train=x_train, y_train=y_train,
                    eval_set=[(x_valid, y_valid)],
                    eval_name=['valid'],
                    patience=patience,
                    max_epochs=max_epochs,
                    eval_metric=['accuracy'],
                    batch_size=batch_sizes(data)[0],
                    virtual_batch_size=batch_sizes(data)[1],
                    weights=1,
                    from_unsupervised=unsupervised_model
                    )
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
            max_epochs,
            patience,
            unlabeled_data,
            test_size,
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
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        unlabeled_data: Data matrix without labeling
        test_size: Test size percentage
        control_fitting: If Yes, considers train scores.
        col_label: Label of class column.

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Searching parameter for {algorithm} using Optuna')
    print(f'Searching parameter for {algorithm} using Optuna')
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting

    def objective(trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_d = trial.suggest_int("n_d", 8, 64, step=1)
        n_steps = trial.suggest_int("n_steps", 3, 5, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.5, step=0.1)
        lambda_sparse = trial.suggest_float("lambda_sparse", 0.00001, 0.0001, log=True)

        params_ = dict(n_d=n_d,
                       n_a=n_d,
                       n_steps=n_steps,
                       gamma=gamma,
                       n_independent=3,
                       n_shared=3,
                       mask_type=mask_type,
                       lambda_sparse=lambda_sparse,
                       seed=random_state
                       )
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        cv_score_array = []
        xx = data.to_numpy()
        yy = data_labels.to_numpy()
        for train_index, test_index in kf.split(xx):
            x_train, x_valid = xx[train_index], xx[test_index]
            y_train, y_valid = yy[train_index], yy[test_index]
            clf_opti = Classifier(**params_)
            clf_opti.fit(X_train=x_train, y_train=y_train,
                         eval_set=[(x_valid, y_valid)],
                         eval_name=['valid'],
                         patience=patience,
                         max_epochs=max_epochs,
                         eval_metric=['cohen_mcc'],
                         weights=1,
                         batch_size=batch_sizes(x_train)[0],
                         virtual_batch_size=batch_sizes(x_train)[1]
                         )
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
        'n_a': study.best_params['n_d'],
        'n_independent': 3,
        'n_shared': 3,
        'seed': random_state
    }
    parameters = study.best_params
    parameters.update(additional_parameters)

    unsupervised_model, model, performance = model_(data=data,
                                                    test_data=test_data,
                                                    data_labels=data_labels,
                                                    test_data_labels=test_data_labels,
                                                    path=path,
                                                    parameters=parameters,
                                                    sampling_method=sampling_method,
                                                    param_search=param_search,
                                                    unlabeled_data=unlabeled_data,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    max_epochs=max_epochs,
                                                    patience=patience,
                                                    col_label=col_label,
                                                    control_fitting=control_fitting
                                                    )
    opti_model = 'Optuna'
    return parameters, unsupervised_model, model, opti_model, performance


# ***** BayesSearchCV *****


def bayes_(data,
           test_data,
           data_labels,
           test_data_labels,
           path,
           random_state,
           sampling_method,
           param_search,
           max_epochs,
           patience,
           unlabeled_data,
           test_size,
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
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        unlabeled_data: Data matrix without labeling
        test_size: Test size percentage
        control_fitting: If Yes, considers train scores.
        col_label: Label of class column.

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Searching parameter for {algorithm} using BayesSearchCV')
    print(f'Searching parameter for {algorithm} using BayesSearchCV')

    x_train, x_valid, y_train, y_valid = train_test_split(data.to_numpy(),
                                                          data_labels.to_numpy(),
                                                          test_size=float(test_size),
                                                          random_state=int(random_state)
                                                          )
    custom_scorer = make_scorer(kappa_mcc_error, greater_is_better=False)

    space = {'mask_type': Categorical(["entmax", "sparsemax"]),
             'n_d': Integer(8, 64),
             'n_a': Integer(8, 64),
             'n_steps': Integer(3, 5),
             'gamma': Real(1.1, 1.5),
             'lambda_sparse': Real(0.00001, 0.0001, 'log-uniform'),
             }
    opt = BayesSearchCV(Classifier(n_independent=3,
                                   n_shared=3,
                                   seed=random_state
                                   ),
                        search_spaces=space,
                        scoring=custom_scorer,  # 'neg_log_loss',
                        n_iter=n_trials,
                        cv=cv,
                        verbose=0,
                        random_state=random_state
                        )
    opt.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_name=['valid'],
            batch_size=batch_sizes(x_train)[0],
            virtual_batch_size=batch_sizes(x_train)[1],
            patience=patience,
            max_epochs=max_epochs,
            eval_metric=['cohen_mcc'],
            weights=1
            )

    additional_parameters = {
        'n_independent': 3,
        'n_shared': 3,
        'seed': random_state
    }
    parameters = opt.best_params_
    parameters.update(additional_parameters)

    unsupervised_model, model, performance = model_(data=data,
                                                    test_data=test_data,
                                                    data_labels=data_labels,
                                                    test_data_labels=test_data_labels,
                                                    path=path,
                                                    parameters=parameters,
                                                    sampling_method=sampling_method,
                                                    param_search=param_search,
                                                    unlabeled_data=unlabeled_data,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    max_epochs=max_epochs,
                                                    patience=patience,
                                                    col_label=col_label,
                                                    control_fitting=control_fitting
                                                    )
    opti_model = 'Bayes'
    return parameters, unsupervised_model, model, opti_model, performance


# ****** GridSearchCV *******


def grid_(data,
          test_data,
          data_labels,
          test_data_labels,
          path,
          random_state,
          sampling_method,
          param_search,
          max_epochs,
          patience,
          unlabeled_data,
          test_size,
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
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        unlabeled_data: Data matrix without labeling
        test_size: Test size percentage.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    logging.info(f'Searching parameter for {algorithm} using GridSearchCV')
    print(f'Searching parameter for {algorithm} using GridSearchCV')

    x_train, x_valid, y_train, y_valid = train_test_split(data.to_numpy(),
                                                          data_labels.to_numpy(),
                                                          test_size=float(test_size),
                                                          random_state=int(random_state)
                                                          )
    custom_scorer = make_scorer(kappa_mcc_error, greater_is_better=False)
    space = {'mask_type': ["entmax", "sparsemax"],
             'n_d': [22],
             # 'n_d': np.linspace(8,64,28, endpoint=True, dtype=int).tolist(),
             'n_a': [22],
             # 'n_a': np.linspace(8,64,28, endpoint=True, dtype=int).tolist(),
             'n_steps': np.linspace(3, 5, 3, endpoint=True, dtype=int).tolist(),
             'gamma': [1.2],
             # 'gamma': np.linspace(1.1,1.2,2, endpoint=True, dtype=float).tolist(),
             'lambda_sparse': np.logspace(-5, -4, num=5, endpoint=True, base=10.0, dtype=float).tolist(),
             }

    opt = GridSearchCV(Classifier(n_independent=3,
                                  n_shared=3,
                                  seed=random_state,
                                  ),
                       param_grid=space,
                       scoring=custom_scorer,  # 'neg_log_loss',
                       cv=cv,
                       verbose=0
                       )
    opt.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_name=['valid'],
            batch_size=batch_sizes(x_train)[0],
            virtual_batch_size=batch_sizes(x_train)[1],
            patience=patience,
            max_epochs=max_epochs,
            eval_metric=['cohen_mcc'],
            weights=1
            )

    additional_parameters = {
        'n_independent': 3,
        'n_shared': 3,
        'seed': random_state
    }
    parameters = opt.best_params_
    parameters.update(additional_parameters)

    unsupervised_model, model, performance = model_(data=data,
                                                    test_data=test_data,
                                                    data_labels=data_labels,
                                                    test_data_labels=test_data_labels,
                                                    path=path,
                                                    parameters=parameters,
                                                    sampling_method=sampling_method,
                                                    param_search=param_search,
                                                    unlabeled_data=unlabeled_data,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    max_epochs=max_epochs,
                                                    patience=patience,
                                                    col_label=col_label,
                                                    control_fitting=control_fitting
                                                    )
    opti_model = 'Grid'
    return parameters, unsupervised_model, model, opti_model, performance


# ***** Predefined parameter *****


def predefined_(data,
                test_data,
                data_labels,
                test_data_labels,
                path,
                random_state,
                sampling_method,
                param_search,
                max_epochs,
                patience,
                unlabeled_data,
                test_size,
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
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        unlabeled_data: Data matrix without labeling
        test_size: Test size percentage.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores

    Returns:
        A tuple containing the best hyperparameter, the unsupervised model, the supervised model, and the performance.
    """
    # Hardcoded Predefined parameters
    # These parameters can be obtained from prior experiments, domain knowledge, or literature
    logging.info(f'{algorithm} Model will be built using Predefined parameters')
    print(f'{algorithm} Model will be built using Predefined parameters')

    parameters = {'mask_type': 'sparsemax',
                  'n_d': 22,
                  'n_a': 22,
                  'n_steps': 4,
                  'gamma': 1.2,
                  'n_independent': 3,
                  'n_shared': 3,
                  'lambda_sparse': 0.00005,
                  'seed': random_state
                  }
    unsupervised_model, model, performance = model_(data=data,
                                                    test_data=test_data,
                                                    data_labels=data_labels,
                                                    test_data_labels=test_data_labels,
                                                    path=path,
                                                    parameters=parameters,
                                                    sampling_method=sampling_method,
                                                    param_search=param_search,
                                                    unlabeled_data=unlabeled_data,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    max_epochs=max_epochs,
                                                    patience=patience,
                                                    col_label=col_label,
                                                    control_fitting=control_fitting
                                                    )
    opti_model = 'Predefined'
    return parameters, unsupervised_model, model, opti_model, performance


def optim_(data,
           test_data,
           data_labels,
           test_data_labels,
           path,
           random_state,
           sampling_method,
           param_search,
           max_epochs,
           patience,
           unlabeled_data,
           test_size,
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
        max_epochs: Maximum number of epochs
        patience: Used for early stop
        unlabeled_data: Data matrix without labeling
        test_size: Test size percentage.
        col_label: Label of class column.
        control_fitting: If Yes, considers train scores

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

    parameters, unsupervised_model, model, optim_model, pscore = search_method(data,
                                                                               test_data,
                                                                               data_labels,
                                                                               test_data_labels,
                                                                               path,
                                                                               random_state,
                                                                               sampling_method,
                                                                               param_search,
                                                                               max_epochs,
                                                                               patience,
                                                                               unlabeled_data,
                                                                               test_size,
                                                                               cv,
                                                                               n_trials,
                                                                               col_label,
                                                                               control_fitting
                                                                               )

    return parameters, unsupervised_model, model, optim_model, pscore


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

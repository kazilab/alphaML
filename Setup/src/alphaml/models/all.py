from ..utils.run_model import run_model

algorithm = 'all_ml'
optim_ = None
roc_auc = None

# run model


def building(labeled_data,
             labels,
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

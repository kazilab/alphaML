import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split

from . import __version__
from .config import ALGORITHMS, yes_no
from .models import MODELS, get_spec
from .models.automl import run_automl
from .models.engine import (feature_importance, fit_final, predict_single_threaded, roc_auc, search_parameters,
                            training_curve)
from .models.folds import build
from .utils.explain import lime_, shap_
from .utils.io import save_pickle, to_jsonable
from .utils.metrics import neglog2rmsl, prediction_table
from .utils.plots import plot_confusion_matrix, plot_parameters
from .utils.prep import preprocessing
from .utils.sampling import sampling

logger = logging.getLogger('alphaml')

BUNDLE_FORMAT = 'alphaml-bundle'


def _no_progress(stage, detail=''):
    pass


def train(labeled_data,
          data_labels,
          result_path,
          data_for_feature_selection=None,
          unlabeled_data=None,
          suggested_features=None,
          model_name='AutoML',
          automl_models=None,
          automl_metric='kappa_mcc',
          automl_top_k=3,
          automl_time_budget=10.0,
          automl_ensemble=True,
          sampling_method='no',
          param_search='Optuna',
          col_label='Trametinib',
          pos_class='resistant',
          neg_class='sensitive',
          test_size=0.2,
          random_state=12,
          n_trials=50,
          cv=5,
          features='HVF',
          features2='None',
          features3='None',
          features4='None',
          n_top_features=0,
          min_disp=1.5,
          log_type='log2',
          top_n_genes_pca=500,
          top_n_genes_rp=500,
          n_clusters=500,
          n_components=500,
          max_features=0,
          threshold='1.5*mean',
          min_sel_features_rfe=10,
          min_sel_features_sfs=10,
          normalization='min_max',
          run_feat_imp=False,
          run_shap=False,
          run_lime=False,
          run_roc=False,
          control_fitting=False,
          progress=None):
    """
    Build a binary classification model and save results, plots and the model bundle in result_path.

    model_name 'AutoML' selects the algorithm, its parameters and optionally an ensemble automatically (see
    models/automl.py); any other name trains that algorithm with the param_search method.
    Data arguments accept DataFrames or CSV/TSV file paths (sample names in the first column).
    Returns a summary dict with the NegLog2-RMSL score, parameters, test scores and the AutoML leaderboard.
    """
    progress = progress or _no_progress
    warnings.filterwarnings('ignore')
    os.makedirs(result_path, exist_ok=True)
    random_state = int(random_state)
    automl = model_name == 'AutoML'
    # The train/validation fitting control applies to Optuna searches
    control_fitting = bool(control_fitting) and (automl or param_search == 'Optuna')
    search_label = 'AutoML' if automl else param_search
    started = time.time()
    logger.info(f'* alphaML {__version__}: analysis started for {model_name}, sampling method {sampling_method} *')

    labels, labeled, unlabeled = preprocessing(
        labeled_data=labeled_data, data_labels=data_labels, result_path=result_path, column_name=col_label,
        pos_class=pos_class, neg_class=neg_class, features=features, features2=features2, features3=features3,
        features4=features4, normalization=normalization, data_for_feature_selection=data_for_feature_selection,
        unlabeled_data=unlabeled_data, suggested_features=suggested_features, random_state=random_state,
        progress=progress, n_top_features=int(n_top_features), min_disp=float(min_disp), log_type=log_type,
        top_n_genes_pca=int(top_n_genes_pca), top_n_genes_rp=int(top_n_genes_rp), n_clusters=int(n_clusters),
        n_components=int(n_components), max_features=int(max_features), threshold=threshold,
        min_sel_features_rfe=int(min_sel_features_rfe), min_sel_features_sfs=int(min_sel_features_sfs))
    feature_names = list(labeled.columns)

    # Separate test data before using sampling methods
    train_data, test_data, train_labels, test_labels = train_test_split(labeled, labels, test_size=float(test_size),
                                                                        random_state=random_state)
    x, y = sampling(train_data.to_numpy(), train_labels.to_numpy(), sampling_method, random_state)
    x_test, y_test = test_data.to_numpy(), test_labels.to_numpy()
    summary = {'model_name': model_name, 'n_samples': int(len(labels)), 'n_features': len(feature_names),
               'n_train': int(len(y)), 'n_test': int(len(y_test))}

    def file_prefix(algorithm):
        return os.path.join(result_path, f'{col_label}_{sampling_method}_{search_label}_{algorithm}_fit_'
                                         f'{yes_no(control_fitting)}')

    if automl:
        result = run_automl(x, y, models=list(automl_models or ALGORITHMS), metric=automl_metric,
                            top_k=int(automl_top_k), n_trials=int(n_trials), time_budget=float(automl_time_budget),
                            cv=int(cv), ensemble=bool(automl_ensemble), control_fitting=control_fitting,
                            random_state=random_state,
                            base=os.path.join(result_path, f'{col_label}_{sampling_method}_AutoML'),
                            progress=progress)
        recipe = result.recipe
        summary.update(selected_model=result.label, automl_metric=automl_metric,
                       leaderboard=to_jsonable(result.leaderboard.to_dict('records')))
    else:
        spec = get_spec(model_name)
        progress('Hyperparameter search', param_search)
        recipe = (spec.name, search_parameters(spec, x, y, param_search, int(n_trials), int(cv), random_state,
                                               control_fitting, file_prefix(spec.algorithm), progress))

    if isinstance(recipe, list):  # AutoML ensemble
        spec, algorithm = None, 'Ensemble'
        parameters = {name: params for name, params in recipe}
        summary['ensemble_members'] = [name for name, _ in recipe]
    else:
        spec = MODELS[recipe[0]]
        algorithm, parameters = spec.algorithm, recipe[1]
    prefix = file_prefix(algorithm)
    logger.info(f'Parameters: {parameters}')
    plot_parameters(parameters, f'{prefix}_parameters')
    with open(f'{prefix}_parameters.json', 'w') as fh:
        json.dump(to_jsonable(parameters), fh, indent=2)

    progress('Training final model', algorithm)
    logger.info(f'Building model using {algorithm}')
    if spec is not None:
        model = fit_final(spec, parameters, x, y, x_test, y_test)
    else:
        model = build(recipe, x, y, final=True)
    model = predict_single_threaded(model)
    save_pickle({'format': BUNDLE_FORMAT, 'alphaml_version': __version__, 'model': model,
                 'features': feature_names, 'normalization': normalization, 'col_label': col_label,
                 'pos_class': pos_class, 'neg_class': neg_class, 'model_name': model_name,
                 'algorithm': algorithm, 'parameters': parameters,
                 'created': datetime.now(timezone.utc).isoformat()},
                f'{prefix}_bundle.pkl')
    with open(os.path.join(result_path, 'selected_features.csv'), 'w') as fh:
        fh.write('Selected features\n' + '\n'.join(feature_names) + '\n')
    logger.info(f'{algorithm} model saved')

    if spec is not None:
        progress('Training curve')
        training_curve(spec, model, parameters, x, y, x_test, y_test, f'{prefix}_valid_accuracy_curve')

    # Test result with probabilities
    prediction_table(model, x_test, y_test, index=test_data.index, pos_class=pos_class,
                     neg_class=neg_class).to_excel(f'{prefix}_test_prediction.xlsx')
    # Scores for test and train data, and negative log2 RMSL
    scores, pscore = neglog2rmsl(model, x, y, x_test, y_test)
    scores.to_csv(f'{prefix}_scores.csv')
    logger.info(f'NegLog2RMSL: {pscore:.4f}')
    summary.update(algorithm=algorithm, parameters=to_jsonable(parameters), pscore=pscore,
                   scores=to_jsonable(scores.reset_index(names='Metric').to_dict('records')))

    if run_feat_imp:
        progress('Feature importance')
        logger.info(f'Calculating feature importance for {algorithm}')
        feature_importance(model, x_test, y_test, feature_names, random_state).to_csv(f'{prefix}_global_feat_imp.csv')

    plot_confusion_matrix(y_test, model.predict(x_test), (pos_class, neg_class), f'{prefix}_confusion_matrix')

    if run_shap:
        progress('SHAP explanations')
        logger.info(f'Feature importance using SHAP for {algorithm}')
        shap_(model, test_data, f'{prefix}_test_SHAP.pdf')
    if run_lime:
        progress('LIME explanations')
        logger.info(f'Feature importance using LIME for {algorithm}')
        lime_(model, test_data, f'{prefix}_test_LIME.pdf', class_names=(neg_class, pos_class),
              random_state=random_state)

    if run_roc:
        progress('ROC cross-validation', f'{int(cv)}-fold x {int(n_trials)} repeats')
        logger.info(f'Drawing ROC curve for {algorithm}')
        # Repeated cross-validation over all labeled data, sampled the same way as the training data
        x_all, y_all = sampling(labeled.to_numpy(), labels.to_numpy(), sampling_method, random_state)
        summary['roc_auc'] = float(roc_auc(recipe, x_all, y_all, int(n_trials), int(cv), random_state, prefix))

    if unlabeled is not None and len(unlabeled):
        progress('Predicting unlabeled samples')
        prediction_table(model, unlabeled.to_numpy(), index=unlabeled.index, pos_class=pos_class,
                         neg_class=neg_class).to_excel(f'{prefix}_unlabeled_prediction.xlsx')
        logger.info(f'Predictions for {len(unlabeled)} unlabeled samples saved')

    summary['runtime_seconds'] = round(time.time() - started, 1)
    with open(os.path.join(result_path, 'summary.json'), 'w') as fh:
        json.dump(to_jsonable(summary), fh, indent=2)
    logger.info(f"**** Analysis finished in {summary['runtime_seconds']} s ****")
    return summary

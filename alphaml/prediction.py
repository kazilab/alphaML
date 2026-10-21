import json
import logging
import os
import time
import warnings

from .run import BUNDLE_FORMAT
from .utils.explain import lime_, shap_
from .utils.io import as_frame, load_pickle, to_jsonable
from .utils.metrics import prediction_table
from .utils.prep import normalize_samples

logger = logging.getLogger('alphaml')


def load_bundle(model):
    """A model bundle saved by train(), given as the bundle dict or the path of its .pkl file."""
    if isinstance(model, (str, os.PathLike)):
        model = load_pickle(model)
    if not isinstance(model, dict) or model.get('format') != BUNDLE_FORMAT:
        raise ValueError('The model file is not an alphaML model bundle (the *_bundle.pkl file of a training run).')
    return model


def predict(test_data, result_path, model, normalization=None, col_label=None, pos_class=None, neg_class=None,
            run_shap=False, run_lime=False, progress=None):
    """
    Predict test samples with a trained model bundle and save predictions (and optional SHAP/LIME plots).

    Normalization and class names default to the values stored in the bundle.
    """
    progress = progress or (lambda stage, detail='': None)
    warnings.filterwarnings('ignore')
    os.makedirs(result_path, exist_ok=True)
    started = time.time()

    progress('Loading model')
    bundle = load_bundle(model)
    estimator = bundle['model']
    algorithm = bundle['algorithm']
    normalization = normalization or bundle['normalization']
    col_label = col_label or bundle['col_label']
    pos_class = pos_class or bundle['pos_class']
    neg_class = neg_class or bundle['neg_class']

    progress('Loading data')
    test = as_frame(test_data)
    selected = bundle['features']
    missing = [f for f in selected if f not in test.columns]
    if missing:
        raise ValueError(f'{len(missing)} of the {len(selected)} model features are missing from the test data, '
                         f"for example: {', '.join(missing[:10])}")
    test = normalize_samples(test[selected], normalization)
    test.index.name = 'Sample'

    prefix = os.path.join(result_path, f'{col_label}_{algorithm}')
    test.to_csv(f'{prefix}_normalized_test_data.csv')

    progress('Predicting')
    table = prediction_table(estimator, test.to_numpy(), index=test.index, pos_class=pos_class, neg_class=neg_class)
    table['predictions'] = table.pop('predicted_class')  # class names
    table.to_excel(f'{prefix}_test_prediction.xlsx')
    logger.info(f'Test predictions for {algorithm} saved')

    if run_shap:
        progress('SHAP explanations')
        shap_(estimator, test, f'{prefix}_test_SHAP.pdf')
    if run_lime:
        progress('LIME explanations')
        lime_(estimator, test, f'{prefix}_test_LIME.pdf', class_names=(neg_class, pos_class))

    counts = table['predictions'].value_counts()
    summary = {'algorithm': algorithm, 'n_samples': int(len(table)), 'n_features': len(selected),
               'normalization': normalization, 'col_label': col_label,
               'class_counts': {str(k): int(v) for k, v in counts.items()},
               'predictions': to_jsonable(table.reset_index().head(500).to_dict('records')),
               'runtime_seconds': round(time.time() - started, 1)}
    with open(os.path.join(result_path, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info('**** Prediction finished ****')
    return summary

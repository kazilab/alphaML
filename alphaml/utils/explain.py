import logging
import warnings

from matplotlib.backends.backend_pdf import PdfPages

from .plots import plt

logger = logging.getLogger('alphaml')

TREE_MODELS = ('XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier', 'GradientBoostingClassifier',
               'RandomForestClassifier', 'ExtraTreesClassifier')
LINEAR_MODELS = ('SGDClassifier',)


def _shap_values(model, x, feature_names):
    import shap

    name = type(model).__name__
    if name in TREE_MODELS:
        try:
            values = shap.TreeExplainer(model, x)(x)
        except Exception as exc:
            logger.info(f'TreeExplainer unavailable for {name} ({exc}), using the model-agnostic explainer')
            values = None
    elif name in LINEAR_MODELS:
        values = shap.LinearExplainer(model, x)(x)
    else:
        values = None
    if values is None:
        def f(data):
            return model.predict_proba(data)[:, 1]
        values = shap.Explainer(f, x)(x, max_evals=round(2 * x.shape[1] + 1))
    if values.values.ndim == 3:  # one output per class: explain the positive class
        values = values[..., 1]
    values.feature_names = list(feature_names)
    return values


def shap_(model, x_test, pdf_path):
    """SHAP summary plots and one waterfall plot per sample in a multi-page PDF."""
    import shap

    x = x_test.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        values = _shap_values(model, x, x_test.columns)
        summary_plots = [(dict(plot_type='layered_violin', color='#cccccc'), 'SHAP layered violin plot gray scale'),
                         (dict(plot_type='layered_violin'), 'SHAP layered violin plot'),
                         (dict(plot_type='violin'), 'SHAP violin plot'),
                         (dict(), 'SHAP dot plot')]
        with PdfPages(pdf_path) as pdf:
            for kwargs, title in summary_plots:
                try:
                    shap.summary_plot(values.values, x_test, show=False, **kwargs)
                    plt.title(title)
                    plt.grid(None)
                    pdf.savefig(bbox_inches='tight')
                except Exception as exc:
                    logger.info(f'{title} skipped: {exc}')
                plt.close('all')
            # Individual SHAP plots
            for i, sample in enumerate(x_test.index):
                try:
                    shap.plots.waterfall(values[i], show=False)
                    plt.title(f'SHAP Plot for Sample {sample}')
                    pdf.savefig(bbox_inches='tight')
                except Exception as exc:
                    logger.info(f'SHAP waterfall for {sample} skipped: {exc}')
                plt.close('all')
    logger.info(f'SHAP plots saved in {pdf_path}')


def lime_(model, x_test, pdf_path, class_names=('0', '1'), num_features=10, num_samples=500, random_state=None):
    """One LIME explanation per sample in a multi-page PDF."""
    from lime.lime_tabular import LimeTabularExplainer

    x = x_test.to_numpy()
    explainer = LimeTabularExplainer(x,
                                     feature_names=list(x_test.columns),
                                     class_names=list(class_names),
                                     mode='classification',
                                     verbose=False,
                                     random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with PdfPages(pdf_path) as pdf:
            for i, sample in enumerate(x_test.index):
                exp = explainer.explain_instance(x[i], model.predict_proba, num_features=num_features,
                                                 num_samples=num_samples)
                fig = exp.as_pyplot_figure(label=1)
                plt.title(f'LIME Explanation for Sample {sample}')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    logger.info(f'LIME explanations saved in {pdf_path}')

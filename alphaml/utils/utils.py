import warnings
import os
import math
import shap
import logging
import requests
import zipfile
import socket
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_selection import SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from itertools import combinations
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from scipy.special import comb
from .binary_tabnet.tab_model import TabNetClassifier
import matplotlib
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

# --------------------------------------------------------------------------#
# data: Gene names in column header and sample names in index (row header)
# sample BCL2 BCL2L1 PLK1 ... ... ... ... ... ... ... ... ... ... ... FLT3
# spl1    12    39    22  ... ... ... ... ... ... ... ... ... ... ... 30
# spl2    10    22    13  ... ... ... ... ... ... ... ... ... ... ... 25
# -------------------------------------------------------------------------#

# file ext


def ext(data_path):
    _, data_file_ext = os.path.splitext(data_path)
    if data_file_ext == '.csv':
        sep = ','
    else:
        sep = '\t'
    return sep

# Highly variable features adapted from scanpy.preprocessing._highly_variable_genes._highly_variable_genes_single_batch


def hvf(data, log_type='log2', n_top_features=0, min_disp=1, max_disp=np.inf, n_bins=20, min_mean=0.0125, max_mean=3):
    # for mean, we use np.log1p which is log10(x+1), so we set min_mean to log10(0.0292+1) and max_mean to log10(999+1)
    data = data.fillna(0)
    if log_type == 'log2':
        data = np.exp2(data)-1
    elif log_type == 'log10':
        data = np.expm1(data)
    elif log_type == 'ln':
        data = np.exp(data)
    else:
        data = data

    mean = np.mean(data, axis=0)
    mean[mean == 0] = 1e-6  # set entries equal to zero to small value
    variance = np.var(data, axis=0, ddof=1)
    dispersion = variance / mean
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)
    df = pd.DataFrame()
    df['means'] = mean
    df['dispersions'] = dispersion

    df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
    disp_grouped = df.groupby('mean_bin')['dispersions']
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)

    one_gene_per_bin = disp_std_bin.isnull()
    disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[one_gene_per_bin.values].values
    disp_mean_bin[one_gene_per_bin.values] = 0
    # actually do the normalization
    df['dispersions_norm'] = (df['dispersions'].values - disp_mean_bin[df['mean_bin'].values].values
                              ) / disp_std_bin[df['mean_bin'].values].values

    dispersion_norm = df['dispersions_norm'].values

    if n_top_features >= 2:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
        if n_top_features > data.shape[1]:
            logging.info('`n_top_genes` > `adata.n_var`, returning all genes.')
            selected_features = data.columns
        elif n_top_features > dispersion_norm.size:
            logging.info('`n_top_genes` > number of normalized dispersions, '
                         'returning all genes with normalized dispersions.')
            selected_features = df.index
        else:
            disp_cut_off = dispersion_norm[n_top_features - 1]
            feat_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
            df['highly_variable'] = feat_subset
            selected_features = df[df['highly_variable']].index  # similar to df[df['highly_variable'] == True].index
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0
        feat_subset = np.logical_and.reduce((mean > min_mean,
                                            mean < max_mean,
                                            dispersion_norm > min_disp,
                                            dispersion_norm < max_disp
                                             ))
        df['highly_variable'] = feat_subset
        selected_features = df[df['highly_variable']].index  # similar to df[df['highly_variable'] == True].index

    return selected_features.to_list()


# Min-Max normalization


def min_max(data, feature_min=0, feature_max=1):
    data = data
    data_min = np.nanmin(data, axis=0)
    data_max = np.nanmax(data, axis=0)
    x_std = (data - data_min) / (data_max - data_min)
    x_scaled = x_std*(feature_max - feature_min) + feature_min
    return x_scaled


# Standardization


def standardize_dataframe(df):
    # Standardize the data along axis=0 (columns)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Create a new DataFrame with the scaled values
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    return df_scaled

# ***************** Data sampling ***********************************
# Tasks                                                              #
# 1. Take data and labels as pd.DataFrame                            #
# 2. Use imblearn to undersample or over sample                       #
# 3. Run under sampling in two steps                                  #
# 4. Run over sample using SMOTE                                      #
# --------------------------------------------------------------------#


def sampling(x_train, y_train, sampling_method):
    # type= under=under sampling, over=over sampling
    # applicable only for train data
    # do not sample test data
    sampling_method = str(sampling_method)
    if sampling_method == 'under':
        sample1 = TomekLinks()
        x1, y1 = sample1.fit_resample(x_train, y_train)
        sample = RandomUnderSampler()
        x, y = sample.fit_resample(x1, y1)
    elif sampling_method == 'over':
        sample = SMOTE()
        x, y = sample.fit_resample(x_train, y_train)
    elif sampling_method == 'no':
        x, y = x_train, y_train
    else:
        error_message = "Sampling type error!"
        logging.error(error_message)
        raise ValueError(error_message)
    return x, y


# Batch size and virtual batch size
# Calculate batch_size from training data


def batch_sizes(x):
    size_factor = 2**int(math.floor(math.log2(x.shape[0]*0.1)))
    # batch_size = bs, virtual_batch_size = vbs
    if size_factor >= 1024:
        bs = 1024
        vbs = 128
    elif size_factor >= 128:
        bs = size_factor
        vbs = size_factor/8
    elif size_factor >= 16:
        bs = size_factor
        vbs = size_factor/4
    else:
        bs, vbs = 16, 4
    return bs, vbs

#  SHAPLY model independent


def shap_(model, x_test, path, sampling_method, param_search, algorithm, col_label, control_fitting):
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    if isinstance(model, (
                          GradientBoostingClassifier,
                          LGBMClassifier,
                          XGBClassifier)
                  ):
        explainer = shap.TreeExplainer(model, x_test)
        shap_values = explainer(x_test)
    elif isinstance(model, SGDClassifier
                    ):
        explainer = shap.LinearExplainer(model, x_test)
        shap_values = explainer(x_test)
    elif isinstance(model, (AdaBoostClassifier,
                            CatBoostClassifier,
                            ExtraTreesClassifier,
                            HistGradientBoostingClassifier,
                            MLPClassifier,
                            NuSVC,
                            RandomForestClassifier,
                            SVC,
                            TabNetClassifier)
                    ):
        explainer = shap.Explainer(model.predict, x_test.to_numpy())
        shap_values = explainer(x_test, max_evals=round(2*x_test.shape[1]+1))
    else:
        raise ValueError("Unsupported model type")

    with PdfPages(f"{p_name}_X_TEST_SHAP.pdf") as pdf:
        shap.summary_plot(shap_values, x_test, plot_type='layered_violin', color='#cccccc', show=False)
        plt.title('SHAP layered violin plot gray scale')
        plt.grid(None)
        pdf.savefig(transparent=False, facecolor='auto', edgecolor='auto')
        plt.close()
        shap.summary_plot(shap_values, x_test, plot_type='layered_violin', show=False)
        plt.title('SHAP layered violin plot')
        plt.grid(None)
        pdf.savefig(transparent=False, facecolor='w', edgecolor='auto')
        plt.close()
        shap.summary_plot(shap_values, x_test, plot_type='violin', show=False)
        plt.title('SHAP violin plot')
        plt.grid(None)
        pdf.savefig()
        plt.close()
        shap.summary_plot(shap_values, x_test, show=False)
        plt.title('SHAP dot plot')
        plt.grid(None)
        pdf.savefig()
        plt.close()
        # Generate individual SHAP plots
        for i in range(len(x_test)):
            fig, _ = plt.subplots()
            shap.plots.waterfall(shap_values[i], show=False)
            plt.title(f"SHAP Plot for Sample {i}")
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    logging.info(f"SHAP plots for {algorithm} have been saved in a pdf file in alphaML_results folder")


# Compute lime plots


def lime_(model, x_test, path, sampling_method, param_search, algorithm, col_label,
          control_fitting,  num_features=10, num_samples=500):
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    if isinstance(model, (AdaBoostClassifier,
                          CatBoostClassifier,
                          ExtraTreesClassifier,
                          GradientBoostingClassifier,
                          HistGradientBoostingClassifier,
                          LGBMClassifier,
                          MLPClassifier,
                          NuSVC,
                          RandomForestClassifier,
                          SGDClassifier,
                          SVC,
                          XGBClassifier)
                  ):
        explainer = LimeTabularExplainer(x_test.values,
                                         feature_names=x_test.columns,
                                         class_names=['0', '1'],
                                         mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                                         verbose=False)
    elif isinstance(model, TabNetClassifier):
        explainer = LimeTabularExplainer(x_test.to_numpy(),
                                         feature_names=x_test.columns,
                                         class_names=['0', '1'],
                                         mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                                         verbose=False)
    else:
        raise ValueError("Unsupported model type")

    with PdfPages(f"{p_name}_X_TEST_LIME.pdf") as pdf:
        for i in range(len(x_test)):
            if hasattr(model, 'predict_proba'):
                prob_func = model.predict_proba
            else:
                prob_func = model.predict

            exp = explainer.explain_instance(x_test.iloc[i].values,
                                             prob_func, num_features=num_features,
                                             num_samples=num_samples
                                             )
            fig = exp.as_pyplot_figure(label=1 if hasattr(model, 'predict_proba') else None)
            plt.title(f"LIME Explanation for Sample {i}")
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    logging.info(f"LIME explanations for {algorithm} have been saved in a pdf file in alphaML_results folder")

# compute sample weight


def sample_weight_(y):
    cw = compute_class_weight(class_weight='balanced',  classes=[0, 1], y=y)
    class_weight_ = {0: cw[0], 1: cw[1]}
    sample_weight = compute_sample_weight(class_weight=class_weight_, y=y)
    return sample_weight

# scale positive weight


def scale_pos_weight_(y):
    spw = (len(y)-sum(y))/sum(y)
    return spw

# Draw ROC AUC curve and


def plot_roc_auc_acc(n_trials, cv, results):
    c_fill = 'rgba(52, 152, 219, 0.2)'
    c_line = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid = 'rgba(189, 195, 199, 0.5)'
    fpr_mean = np.linspace(0, 1, n_trials*cv)
    interp_tprs = []
    for i in range(n_trials*cv):
        fpr = results['fpr'][i]
        tpr = results['tpr'][i]
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2*np.std(interp_tprs, axis=0)
    tpr_upper = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower = tpr_mean-tpr_std
    auc = np.mean(results['auc'])
    fig = go.Figure([
        go.Scatter(
            x=fpr_mean,
            y=tpr_upper,
            line=dict(color=c_line, width=1),
            hoverinfo="skip",
            showlegend=False,
            name='upper'),
        go.Scatter(
            x=fpr_mean,
            y=tpr_lower,
            fill='tonexty',
            fillcolor=c_fill,
            line=dict(color=c_line, width=1),
            hoverinfo="skip",
            showlegend=False,
            name='lower'),
        go.Scatter(
            x=fpr_mean,
            y=tpr_mean,
            line=dict(color=c_line_main, width=2),
            hoverinfo="skip",
            showlegend=True,
            name=f'AUC: {auc:.3f}')
    ])
    fig.add_shape(
        type='line',
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title="1 - Specificity",
        yaxis_title="Sensitivity",
        width=800,
        height=800,
        legend=dict(
            yanchor="bottom",
            xanchor="right",
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range=[0, 1],
        gridcolor=c_grid,
        scaleanchor="x",
        scaleratio=1,
        linecolor='black')
    fig.update_xaxes(
        range=[0, 1],
        gridcolor=c_grid,
        constrain='domain',
        linecolor='black')

    # *** Draw accuracy plot ***
    y_accuracy = results['accuracy']
    av_accuracy = np.mean(y_accuracy)
    sem_accuracy = pd.DataFrame(y_accuracy).sem()
    trace = go.Scatter(
                        x=np.arange(1, n_trials*cv+1),  # Generate auto number for X axis
                        y=y_accuracy,
                        mode='markers',
                        marker=dict(color='lightblue')
                        )
    # Create the layout for the plot
    layout = go.Layout(
            title='Accuracy plot from cross-validation',
            xaxis=dict(title='Repeated cross-validation number',
                       showgrid=False,
                       range=[0, n_trials*cv+0.5],
                       showline=True, linewidth=0.1, linecolor='black',  # Add x-axis frame
                       mirror=True,  # Reflect ticks and frame on the opposite side
                       ),
            yaxis=dict(title='Accuracy',
                       showgrid=False,
                       # range=[0, 1],
                       showline=True, linewidth=0.1, linecolor='black',  # Add y-axis frame
                       mirror=True,  # Reflect ticks and frame on the opposite side
                       ),
            plot_bgcolor='rgba(0,0,0,0)',  # Set transparent background
            )
    # Create the figure and plot it
    fig1 = go.Figure(data=[trace], layout=layout)
    fig1.add_annotation(
                        x=1, y=0,
                        xref="paper", yref="paper",  # use relative coordinates
                        text=f"Average accuracy: {av_accuracy:.3f}, SEM: {sem_accuracy[0]:.5f}",
                        showarrow=False,
                        font=dict(size=10),
                        xanchor="right", yanchor="bottom",
                        # bgcolor="rgba(255,255,255,08)" # to add a white background
                        )
    return fig, fig1


#  Check internet connection


def is_connected():
    try:
        socket.create_connection(("www.google.com", 80), timeout=2)
        return True
    except OSError:
        pass
    return False

#  Download data from google drive


def get_data(data_path):

    # 1. Check if alphaML_data folder exists, else create
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    csv_files = ["labeled_data.csv", "unlabeled_data.csv", "data_for_feature_selection.csv", "data_labels.csv"]
    missing_files = [file for file in csv_files if not os.path.isfile(os.path.join(data_path, file))]

    # 2. Check if CSV files are available, else download data.zip
    if missing_files:
        if not is_connected():
            raise ValueError("No internet connection. Please connect to the internet and try again.")

        url_ = "https://drive.google.com/uc?id=1RR26k1gcckinoqwxAPimPJRElpB3P84G&export=download&confirm=no_antivirus"
        response = requests.get(url_)

        zip_filename = os.path.join(data_path, "data.zip")
        with open(zip_filename, "wb") as zip_file:
            zip_file.write(response.content)

        # 3. Unzip and delete the zip file
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_filename)

    # 4. Confirm that all files are available, else raise ValueError
    for file in csv_files:
        if not os.path.isfile(os.path.join(data_path, file)):
            raise ValueError(f"{file} is missing. Please download the zip file manually from "
                             f"'https://drive.google.com/drive/folders/1RR26k1gcckinoqwxAPimPJRElpB3P84G?usp=sharing',"
                             f"unzip, and put the files in the alphaML_data folder.")
    logging.info("All files are available in the alphaML_data folder.")
    print("All files are available in the alphaML_data folder.")

# Data preprocessing
# 1. Select from model using Random Forest


def select_from_rf(x_train, y_train, threshold, max_features):
    if max_features > 1:
        max_features = max_features
    else:
        max_features = None
    selfrommodel = SelectFromModel(RandomForestClassifier(n_estimators=250,
                                                          max_depth=6,
                                                          random_state=18,
                                                          criterion='log_loss',
                                                          class_weight='balanced'),
                                   threshold=threshold,
                                   norm_order=1,
                                   max_features=max_features,
                                   importance_getter='auto'
                                   ).fit(x_train, y_train)

    selected_feat = x_train.columns[(selfrommodel.get_support())]
    return selected_feat.to_list()


# 2. Recursive feature elimination with cross-validation to select features


def recursive_feature_elimination(x_train, y_train, min_sel_features_rfe):

    rfe = RFECV(LinearSVC(C=0.7,
                          penalty='l1',
                          dual=False,
                          random_state=18,
                          class_weight='balanced'
                          ),
                step=1,
                min_features_to_select=min_sel_features_rfe,
                scoring='average_precision',
                verbose=0,
                ).fit(x_train, y_train)

    selected_feat = x_train.columns[np.where(rfe.support_)[0]]
    return selected_feat.to_list()


# 3. Sequential Feature Selection.


def seq_feat_sel(x_train, y_train, min_sel_features_sfs):

    sfs = SequentialFeatureSelector(XGBClassifier(learning_rate=0.025,
                                                  max_depth=6,
                                                  subsample=0.75,
                                                  colsample_bytree=0.75,
                                                  gamma=2,
                                                  reg_alpha=2.5,
                                                  reg_lambda=2,
                                                  min_child_weight=2.5,
                                                  n_estimators=75,
                                                  random_state=18,
                                                  objective='binary:logistic',
                                                  verbosity=0,
                                                  scale_pos_weight=scale_pos_weight_(y_train)
                                                  ),
                                    n_features_to_select=min_sel_features_sfs,
                                    scoring='average_precision',
                                    ).fit(x_train, y_train)

    features = sfs.get_support(indices=True)
    all_features = x_train.columns.tolist()
    selected_features = [all_features[i] for i in features]
    return selected_features

# 4. Select by median MAD


def median_absolute_deviation(data):
    if len(data) == 0:
        raise ValueError("The input data must be a non-empty list or array")
    median = np.median(data)
    absolute_deviations = np.abs(data - median)
    mad = np.median(absolute_deviations)
    return mad


def select_by_mad(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Calculate the MAD for each gene
    mads = df.apply(median_absolute_deviation, axis=0)

    # Find the median value of MADs
    median_mad = np.median(mads)

    # Select genes that pass the median value of MAD
    selected_genes = df.columns[mads > median_mad]

    return list(selected_genes)


# 5 Select by PCA


def select_by_pca(df, top_n_genes_pca=500, explained_variance_ratio=0.8):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Perform PCA
    pca = PCA()
    pca.fit(df)

    # Calculate the cumulative explained variance ratio
    cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components that explain a desired portion of the variance
    n_components = np.argmax(cum_explained_variance_ratio >= explained_variance_ratio) + 1

    # Calculate the aggregated loading scores from the selected components
    loading_scores = np.sum(np.abs(pca.components_[:n_components]), axis=0)
    loading_scores = pd.Series(loading_scores, index=df.columns)

    # Sort genes by their aggregated loading scores
    sorted_genes = loading_scores.sort_values(ascending=False)

    # Select the top N genes with the highest aggregated loading scores
    selected_genes = sorted_genes.head(top_n_genes_pca).index

    return list(selected_genes)


# 6 Random Projection


def select_by_random_projection(df, top_n_genes_rp=500, epsilon=0.5):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")
    # valu of epsilon may need to djust for small number of features
    # Johnson-Lindenstrauss lemma
    # https://doi.org/10.1002/rsa.10073
    # k ≥ (4 * log(n)) / (ε^2 / 2 - ε^3 / 3)
    # ε = 0.1 and 0.5
    # k = n_components (number of new features (dimensions))
    n_components = math.floor(4*np.log10(df.shape[1])/(epsilon ** 2 / 2 - epsilon ** 3 / 3))
    # print(n_components)
    # Perform Random Projection
    rp = GaussianRandomProjection(n_components=n_components)
    projected_data = rp.fit_transform(df)

    # Calculate the correlation between the original features and the projected features
    corr_matrix = np.corrcoef(df.T, projected_data.T)[:df.shape[1], df.shape[1]:]
    feature_importance = np.abs(corr_matrix).sum(axis=1)

    # Sort genes by their importance
    sorted_genes = pd.Series(feature_importance, index=df.columns).sort_values(ascending=False)

    # Select the top N genes with the highest importance
    selected_genes = sorted_genes.head(top_n_genes_rp).index

    return list(selected_genes)


# 7 Correlation-based method


def remove_highly_correlated_feat(df, correlation_threshold=0.9):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Calculate the pairwise correlation between all genes
    corr_matrix = df.corr()

    # Select genes with low pairwise correlation
    selected_genes = []
    for gene in corr_matrix.columns:
        # Check if the current gene has a high correlation with any of the already selected genes
        has_high_correlation = False
        for selected_gene in selected_genes:
            if np.abs(corr_matrix.loc[gene, selected_gene]) >= correlation_threshold:
                has_high_correlation = True
                break

        # If the current gene does not have a high correlation with any selected genes, add it to the list
        if not has_high_correlation:
            selected_genes.append(gene)

    return selected_genes


# 8 Clustering-based method


def select_by_clustering(df, n_clusters=500):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Transpose the DataFrame, so that rows represent genes and columns represent samples
    df_t = df.T

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df_t)

    # Assign each gene to a cluster
    gene_cluster_assignments = kmeans.labels_

    # Select a representative gene from each cluster
    selected_genes = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(gene_cluster_assignments == cluster_id)[0]
        cluster_center = kmeans.cluster_centers_[cluster_id]

        # Find the gene in the cluster that is closest to the cluster center
        min_distance = float("inf")
        representative_gene = None
        for idx in cluster_indices:
            gene = df_t.index[idx]
            distance = np.linalg.norm(df_t.loc[gene] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                representative_gene = gene

        selected_genes.append(representative_gene)

    return selected_genes


# 9 Non-negative Matrix Factorization (NMF)


def select_by_nmf(df, n_components=500, top_n_genes_per_component=1):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    # Ensure that the input data is non-negative
    if (df < 0).any().any():
        raise ValueError("Input data should be non-negative")

    # Perform NMF decomposition
    nmf = NMF(n_components=n_components, random_state=0)
    W = nmf.fit_transform(df)
    H = nmf.components_

    # Select the top N genes with the highest coefficients in each NMF component
    selected_genes = set()
    for component in H:
        top_gene_indices = component.argsort()[-top_n_genes_per_component:]
        for idx in top_gene_indices:
            selected_genes.add(df.columns[idx])

    return list(selected_genes)

# 10 Iterative Feature Selection


def iterative_feature_selector(x_train, y_train, min_features=1, max_features=3, cv=5):
    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("Input should be a pandas DataFrame")

    X = x_train.values
    feature_names = x_train.columns.tolist()
    initial_num = len(feature_names)
    total_iterations = comb(initial_num, 1) + comb(initial_num, 2) + comb(initial_num, 3)
    print('Total iterations: ', total_iterations)
    best_score = -np.inf
    best_features = None
    # Initialize the classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    for num_features in range(min_features, max_features + 1):
        for feature_combination in combinations(range(X.shape[1]), num_features):
            X_subset = X[:, feature_combination]
            score = np.mean(cross_val_score(clf, X_subset, y_train, cv=cv))
            if score > best_score:
                best_score = score
                best_features = feature_combination

    # Get the names of the selected features
    selected_feature_names = [feature_names[i] for i in best_features]

    return selected_feature_names


def get_pred_data(data_path):

    # 1. Check if alphaML_data folder exists, else create
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    csv_files = ["selected_features.csv", "test_data.csv", "model.pkl"]
    missing_files = [file for file in csv_files if not os.path.isfile(os.path.join(data_path, file))]

    # 2. Check if CSV files are available, else download data.zip
    if missing_files:
        if not is_connected():
            raise ValueError("No internet connection. Please connect to the internet and try again.")

        url_ = "https://drive.google.com/uc?id=1oIwrDJCOdVLq650O_bV2RrvnEHb50Uzm&export=download&confirm=no_antivirus"
        response = requests.get(url_)

        zip_filename = os.path.join(data_path, "predict.zip")
        with open(zip_filename, "wb") as zip_file:
            zip_file.write(response.content)

        # 3. Unzip and delete the zip file
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_filename)

    # 4. Confirm that all files are available, else raise ValueError
    for file in csv_files:
        if not os.path.isfile(os.path.join(data_path, file)):
            raise ValueError(f"{file} is missing. Please download the zip file manually from "
                             f"'https://drive.google.com/drive/folders/1oIwrDJCOdVLq650O_bV2RrvnEHb50Uzm?usp=sharing',"
                             f"unzip, and put the files in the alphaPred_data folder.")
    logging.info("All files are available in the alphaML_data folder.")
    print("All files are available in the alphaML_data folder.")

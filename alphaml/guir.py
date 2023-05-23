import tkinter as tk
from threading import Thread
import ctypes  # for forcefully terminating the thread
from .run import aml


def terminate_thread(thread):
    """Forcefully terminate a Python thread."""
    if not thread.is_alive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


root = tk.Tk()
root.title("alphaML GUI")
# Add a global flag to control the cancellation
cancel_run = False
aml_thread = None


def run_aml():
    global cancel_run, aml_thread
    cancel_run = False
    # Get the values from the dropdown list and input boxes
    model_name = model_name_var.get()  # 'XGBoost',
    sampling_method = sampling_var.get()  # "no",
    param_search = param_var.get()  # "Predefined",
    col_label = col_label_var.get()  # 'Sensitivity',
    pos_class = pos_var.get()  # 'resistant',
    neg_class = neg_var.get()  # 'sensitive',
    test_size = float(size_var.get())  # 0.2,
    random_state = int(random_var.get())  # 12,
    n_trials = int(trials_var.get())  # 10,
    cv = int(cv_var.get())  # 5,
    max_epochs = int(epochs_var.get())  # 100,
    patience = int(patience_var.get())  # 5,
    features = features_var.get()  # 'hvg',
    features2 = features2_var.get()  # 'None',
    features3 = features3_var.get()  # 'None',
    features4 = features4_var.get()  # 'None',
    n_top_features = int(topgenes_var.get())  # 20,
    min_disp = float(disp_var.get())  # 1.5,
    log_type = flavor_var.get()  # 'seurat',
    top_n_genes_pca = int(top_n_genes_pca_var.get())
    top_n_genes_rp = int(top_n_genes_rp_var.get())
    n_clusters = int(n_clusters_var.get())
    n_components = int(n_components_var.get())
    max_features = int(max_features_var.get())  # 0,
    threshold = threshold_var.get()  # '1.5*mean',
    min_sel_features_rfe = int(min_sel_features_rfe_var.get())
    min_sel_features_sfs = int(min_sel_features_sfs_var.get())
    latent_dim = int(latent_dim_var.get())  # 64,
    min_sel_features = int(min_sel_features_var.get())  # 10,
    fdr = float(fdr_var.get())  # 0.25,
    normalization = norm_var.get()  # 'min_max',
    run_feat_imp = featimp_var.get()  # 'Yes',
    run_shap = shap_var.get()  # 'Yes',
    run_lime = lime_var.get()  # 'Yes',
    run_roc = roc_var.get()  # 'Yes',
    control_fitting = control_var.get()  # 'Yes'

    # Call the runmodel function with the selected parameters
    def run_aml_thread():
        # Update the status label to display "Running..."
        status_label.config(text="Running...")
        run_cancelled = aml(
                            model_name=model_name,
                            sampling_method=sampling_method,
                            param_search=param_search,
                            col_label=col_label,
                            pos_class=pos_class,
                            neg_class=neg_class,
                            test_size=test_size,
                            random_state=random_state,
                            n_trials=n_trials,
                            cv=cv,
                            max_epochs=max_epochs,
                            patience=patience,
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
                            normalization=normalization,
                            run_feat_imp=run_feat_imp,
                            run_shap=run_shap,
                            run_lime=run_lime,
                            run_roc=run_roc,
                            control_fitting=control_fitting
                            )
        if run_cancelled:
            status_label.config(text="Run cancelled")
        else:
            status_label.config(text="Run completed")
    aml_thread = Thread(target=run_aml_thread)
    aml_thread.start()


def cancel_run_():
    global aml_thread, cancel_run

    # Stop the ongoing operation by forcefully terminating the aml_thread
    if aml_thread is not None and aml_thread.is_alive():
        terminate_thread(aml_thread)

    # Clear the status label
    status_label.config(text="")

    # Wait for a short period to ensure the termination takes effect
    # root.after(100, run_aml)  # Restart the aml function after a delay of 100 milliseconds


# Create a frame for the dropdown/entry fields
c01r11_frame = tk.Frame(root, borderwidth=0, relief="groove")
c01r11_frame.grid(row=1, column=0, columnspan=2, rowspan=1, padx=5, pady=5, sticky="w")
# Add a label to the frame for the title
c01r11_label = tk.Label(c01r11_frame, text="Build a CLETE Binary Classification Model", font=("Helvetica", 11),
                        fg="blue", anchor="sw")
c01r11_label.pack(pady=5, fill="both", expand=True)
c01r11_label.config(width=39)

# Create a frame for the dropdown/entry fields
c23r01_frame = tk.Frame(root, borderwidth=0, relief="groove")
c23r01_frame.grid(row=0, column=2, columnspan=2, rowspan=2, padx=5, pady=5)  # , sticky="w")
# Add a label to the frame for the title
c23r01_label = tk.Label(c23r01_frame, text="alphaML", font=("Helvetica", 36, "bold"), fg="blue")
c23r01_label.pack(pady=5)
c23r01_label.config(width=11)

# Create a frame for the dropdown/entry fields
c45r11_frame = tk.Frame(root, borderwidth=0, relief="groove")
c45r11_frame.grid(row=1, column=4, columnspan=2, rowspan=1, padx=5, pady=5, sticky="w")
# Add a label to the frame for the title
c45r11_label = tk.Label(c45r11_frame, text="Drug Sensitivity Prediction", font=("Helvetica", 11),
                        fg="blue", anchor="se")
c45r11_label.pack(pady=5, fill="both", expand=True)
c45r11_label.config(width=39)

# Create a frame for the dropdown/entry fields
c01r25_frame = tk.Frame(root, borderwidth=2, relief="groove")
c01r25_frame.grid(row=2, column=0, columnspan=2, rowspan=4, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c01r25_label = tk.Label(c01r25_frame, text="Binary class labels and column header", font=("Helvetica", 11, "bold"),
                        fg="blue")
c01r25_label.grid(row=0, column=0, columnspan=2, pady=5)
# main_label.config(width=50)

# Enter class information
col_label_label = tk.Label(c01r25_frame, text="Class header:", font=("Helvetica", 10, "bold"), anchor="e")
col_label_label.grid(row=1, column=0, padx=5, pady=10)
col_label_label.config(width=22)
col_label_var = tk.StringVar(root)
col_label_var.set('Trametinib')
col_label_entry = tk.Entry(c01r25_frame, textvariable=col_label_var)
col_label_entry.grid(row=1, column=1, padx=13, pady=10)
col_label_entry.config(width=22)

pos_label = tk.Label(c01r25_frame, text="Positive class:", font=("Helvetica", 10, "bold"), anchor="e")
pos_label.grid(row=2, column=0, padx=5, pady=9)
pos_label.config(width=22)
pos_var = tk.StringVar(root)
pos_var.set('resistant')
pos_entry = tk.Entry(c01r25_frame, textvariable=pos_var)
pos_entry.grid(row=2, column=1, padx=13, pady=9)
pos_entry.config(width=22)

neg_label = tk.Label(c01r25_frame, text="Negative class:", font=("Helvetica", 10, "bold"), anchor="e")
neg_label.grid(row=3, column=0, padx=5, pady=9)
neg_label.config(width=22)
neg_var = tk.StringVar(root)
neg_var.set('sensitive')
neg_entry = tk.Entry(c01r25_frame, textvariable=neg_var)
neg_entry.grid(row=3, column=1, padx=13, pady=9)
neg_entry.config(width=22)

# Create a frame for the dropdown/entry fields
c01r68_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c01r68_frame.grid(row=6, column=0, columnspan=2, rowspan=3, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c01r68_label = tk.Label(c01r68_frame, text="Other parameters", font=("Helvetica", 11, "bold"), fg="blue")
c01r68_label.grid(row=0, column=0, columnspan=2, pady=8)
# params_label.config(width=50)
# Test size for test-train distribution
size_label = tk.Label(c01r68_frame, text="Test size for splitting:", font=("Helvetica", 10, "bold"), anchor="e")
size_label.grid(row=1, column=0, padx=5, pady=8)
size_label.config(width=22)
size_var = tk.DoubleVar(root)
size_var.set(0.2)
size_entry = tk.Entry(c01r68_frame, textvariable=size_var)
size_entry.grid(row=1, column=1, padx=13, pady=8)
size_entry.config(width=22)

# Random state
random_label = tk.Label(c01r68_frame, text="Random seed:", font=("Helvetica", 10, "bold"), anchor="e")
random_label.grid(row=2, column=0, padx=5, pady=9)
random_label.config(width=22)
random_var = tk.IntVar(root)
random_var.set(12)
random_entry = tk.Entry(c01r68_frame, textvariable=random_var)
random_entry.grid(row=2, column=1, padx=13, pady=9)
random_entry.config(width=22)

# Create a frame for the dropdown/entry fields
c23r23_frames = tk.Frame(root, borderwidth=2, relief="groove")
c23r23_frames.grid(row=2, column=4, columnspan=2, rowspan=2, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c21r23_label = tk.Label(c23r23_frames, text="Sampling methods", font=("Helvetica", 11, "bold"), fg="blue")
c21r23_label.grid(row=0, column=0, columnspan=2, pady=5)
# main_label.config(width=50)

# Add a dropdown list for the sampling method
sampling_label = tk.Label(c23r23_frames, text="Select a sampling method:", font=("Helvetica", 10, "bold"), anchor="e")
sampling_label.config(width=22)
sampling_label.grid(row=1, column=0, padx=5, pady=5)
sampling_methods = ['no', 'over', 'under']
sampling_var = tk.StringVar(root)
sampling_var.set(sampling_methods[0])
sampling_dropdown = tk.OptionMenu(c23r23_frames, sampling_var, *sampling_methods)
sampling_dropdown.grid(row=1, column=1, padx=5, pady=5)
sampling_dropdown.config(width=18)  # Set the width of the widget

c23r48_frame = tk.Frame(root, borderwidth=2, relief="groove")
c23r48_frame.grid(row=4, column=2, columnspan=2, rowspan=5, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c23r48_label = tk.Label(c23r48_frame, text="Hyperparameter search", font=("Helvetica", 11, "bold"),
                        fg="blue")
c23r48_label.grid(row=0, column=0, columnspan=2, pady=7)
# main_label.config(width=50)

# Define a function to disable/enable the parameter widgets based on the selected feature method


def toggle_hyper_param_run(*args):
    if param_var.get() == "Predefined":
        control_dropdown.config(state="disabled")
        trials_entry.config(state="disabled")
        cv_entry.config(state="disabled")
    elif param_var.get() == "Optuna":
        control_dropdown.config(state="normal")
        trials_entry.config(state="normal")
        cv_entry.config(state="normal")
    else:
        control_dropdown.config(state="disabled")
        trials_entry.config(state="normal")
        cv_entry.config(state="normal")


# Add a dropdown list for the parameter search method
param_label = tk.Label(c23r48_frame, text="Hyperparameter search by:", font=("Helvetica", 10, "bold"), anchor="e")
param_label.grid(row=1, column=0, padx=5, pady=7)
param_label.config(width=22)
param_methods = ['Optuna', 'Bayes', 'Grid', 'Predefined']
param_var = tk.StringVar(root)
param_var.set(param_methods[0])
param_var.trace("w", toggle_hyper_param_run)
param_dropdown = tk.OptionMenu(c23r48_frame, param_var, *param_methods)
param_dropdown.grid(row=1, column=1, padx=5, pady=7)
param_dropdown.config(width=18)  # Set the width of the widge

# Add a dropdown list
control_label = tk.Label(c23r48_frame, text="Fitting Controls:", font=("Helvetica", 10, "bold"), anchor="e")
control_label.grid(row=2, column=0, padx=5, pady=7)
control_label.config(width=22)
control_methods = ['Yes', 'No']
control_var = tk.StringVar(root)
control_var.set(control_methods[1])
control_dropdown = tk.OptionMenu(c23r48_frame, control_var, *control_methods)
control_dropdown.grid(row=2, column=1, padx=5, pady=8)
control_dropdown.config(width=18)

# Number of trials
trials_label = tk.Label(c23r48_frame, text="Number of trials:", font=("Helvetica", 10, "bold"), anchor="e")
trials_label.grid(row=3, column=0, padx=5, pady=7)
trials_label.config(width=22)
trials_var = tk.IntVar(root)
trials_var.set(50)
trials_entry = tk.Entry(c23r48_frame, textvariable=trials_var)
trials_entry.grid(row=3, column=1, padx=13, pady=8)
trials_entry.config(width=22)

# Add an input box for the number of cross-validation folds
cv_label = tk.Label(c23r48_frame, text="Cross-Validation folds:", font=("Helvetica", 10, "bold"), anchor="e")
cv_label.grid(row=4, column=0, padx=5, pady=7)
cv_label.config(width=22)
cv_var = tk.IntVar(root)
cv_var.set(5)
cv_entry = tk.Entry(c23r48_frame, textvariable=cv_var)
cv_entry.grid(row=4, column=1, padx=13, pady=8)
cv_entry.config(width=22)

toggle_hyper_param_run()

# Create a frame for the dropdown/entry fields
c45r23_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c45r23_frame.grid(row=2, column=2, columnspan=2, rowspan=2, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r23_label = tk.Label(c45r23_frame, text="Data normalization", font=("Helvetica", 12, "bold"), fg="blue")
c45r23_label.grid(row=0, column=0, columnspan=2, pady=5)
# feature_label.config(width=50)

# Add a dropdown list for the normalization method
norm_label = tk.Label(c45r23_frame, text="Method:", font=("Helvetica", 10, "bold"), anchor="e")
norm_label.config(width=22)
norm_label.grid(row=1, column=0, padx=5, pady=5)
norm_methods = ['min_max', 'standardization', 'None']
norm_var = tk.StringVar(root)
norm_var.set(norm_methods[0])
norm_dropdown = tk.OptionMenu(c45r23_frame, norm_var, *norm_methods)
norm_dropdown.grid(row=1, column=1, padx=5, pady=5)
norm_dropdown.config(width=18)  # Set the width of the widget


# Create a frame for the dropdown/entry fields
c45r46_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c45r46_frame.grid(row=4, column=4, columnspan=2, rowspan=3, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r46_label = tk.Label(c45r46_frame, text="Parameters for iterations and early stopping",
                         font=("Helvetica", 11, "bold"), fg="blue")
c45r46_label.grid(row=0, column=0, columnspan=2, pady=5)
# params_label.config(width=50)
# Test size for test-train distribution

# Number of epochs
epochs_label = tk.Label(c45r46_frame, text="Maximum epochs:", font=("Helvetica", 10, "bold"), anchor="e")
epochs_label.grid(row=1, column=0, padx=5, pady=5)
epochs_label.config(width=22)
epochs_var = tk.IntVar(root)
epochs_var.set(500)
epochs_entry = tk.Entry(c45r46_frame, textvariable=epochs_var)
epochs_entry.grid(row=1, column=1, padx=13, pady=10)
epochs_entry.config(width=22)

# Patience for early stop
patience_label = tk.Label(c45r46_frame, text="Patience:", font=("Helvetica", 10, "bold"), anchor="e")
patience_label.grid(row=2, column=0, padx=5, pady=10)
patience_label.config(width=22)
patience_var = tk.IntVar(root)
patience_var.set(50)
patience_entry = tk.Entry(c45r46_frame, textvariable=patience_var)
patience_entry.grid(row=2, column=1, padx=13, pady=10)
patience_entry.config(width=22)

# Create a frame for the dropdown/entry fields
c45r78_frame = tk.Frame(root, borderwidth=2, relief="groove")
c45r78_frame.grid(row=7, column=4, columnspan=2, rowspan=2, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r78_label = tk.Label(c45r78_frame, text="Select an algorithm", font=("Helvetica", 11, "bold"), fg="blue")
c45r78_label.grid(row=0, column=0, columnspan=2, pady=5)

# Define a function to disable/enable the parameter widgets based on the selected feature method


def toggle_param_run(*args):
    if model_name_var.get() == "Test_Briefly" or model_name_var.get() == "None":
        featimp_dropdown.config(state="disabled")
        roc_dropdown.config(state="disabled")
        shap_dropdown.config(state="disabled")
        lime_dropdown.config(state="disabled")
    else:
        featimp_dropdown.config(state="normal")
        roc_dropdown.config(state="normal")
        shap_dropdown.config(state="normal")
        lime_dropdown.config(state="normal")


# Whether run a specific model
model_name_label = tk.Label(c45r78_frame, text="Algorithm:", font=("Helvetica", 10, "bold"), anchor="e")
model_name_label.grid(row=1, column=0, padx=5, pady=5)
model_name_label.config(width=22)
model_names = ['AdaBoost', 'CatBoost', 'ElasticNet', 'ExtraTrees', 'GradientBoosting', 'HistGradientBoosting', 'LASSO',
               'LightGBM', 'MLPC', 'RandomForest', 'Ridge', 'SVC', 'NuSVC', 'TabNet', 'XGBoost', 'Test_Briefly', 'None'
               ]
model_name_var = tk.StringVar(root)
model_name_var.set(model_names[14])
model_name_var.trace("w", toggle_param_run)  # Call the toggle_parameters function when the selected option changes
model_name_dropdown = tk.OptionMenu(c45r78_frame, model_name_var, *model_names)
model_name_dropdown.grid(row=1, column=1, padx=5, pady=5)
model_name_dropdown.config(width=18)

# Create a frame for the dropdown/entry fields
c45r913_frame = tk.Frame(root, borderwidth=2, relief="groove")
c45r913_frame.grid(row=9, column=4, columnspan=2, rowspan=5, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r913_label = tk.Label(c45r913_frame, text="Explain a model", font=("Helvetica", 11, "bold"), fg="blue")
c45r913_label.grid(row=0, column=0, columnspan=2, pady=5)
# Add a dropdown list for the Feature importance
featimp_label = tk.Label(c45r913_frame, text="Feature importance:", font=("Helvetica", 10, "bold"), anchor="e")
featimp_label.grid(row=1, column=0, padx=5, pady=5)
featimp_label.config(width=22)
featimp_methods = ['Yes', 'No']
featimp_var = tk.StringVar(root)
featimp_var.set(featimp_methods[1])
featimp_dropdown = tk.OptionMenu(c45r913_frame, featimp_var, *featimp_methods)
featimp_dropdown.grid(row=1, column=1, padx=5, pady=5)
featimp_dropdown.config(width=18)

# Add a dropdown list for the ROC-AUC
roc_label = tk.Label(c45r913_frame, text="ROC-AUC plot:", font=("Helvetica", 10, "bold"), anchor="e")
roc_label.grid(row=2, column=0, padx=5, pady=5)
roc_label.config(width=22)
roc_methods = ['Yes', 'No']
roc_var = tk.StringVar(root)
roc_var.set(roc_methods[1])
roc_dropdown = tk.OptionMenu(c45r913_frame, roc_var, *roc_methods)
roc_dropdown.grid(row=2, column=1, padx=5, pady=5)
roc_dropdown.config(width=18)

# Add a dropdown list for the SHAP
shap_label = tk.Label(c45r913_frame, text="SHAP plots:", font=("Helvetica", 10, "bold"), anchor="e")
shap_label.grid(row=3, column=0, padx=5, pady=5)
shap_label.config(width=22)
shap_methods = ['Yes', 'No']
shap_var = tk.StringVar(root)
shap_var.set(shap_methods[1])
shap_dropdown = tk.OptionMenu(c45r913_frame, shap_var, *shap_methods)
shap_dropdown.grid(row=3, column=1, padx=5, pady=5)
shap_dropdown.config(width=18)

# Add a dropdown list for the LIME plots
lime_label = tk.Label(c45r913_frame, text="LIME plots:", font=("Helvetica", 10, "bold"), anchor="e")
lime_label.grid(row=4, column=0, padx=5, pady=5)
lime_label.config(width=22)
lime_methods = ['Yes', 'No']
lime_var = tk.StringVar(root)
lime_var.set(lime_methods[1])
lime_dropdown = tk.OptionMenu(c45r913_frame, lime_var, *lime_methods)
lime_dropdown.grid(row=4, column=1, padx=5, pady=5)
lime_dropdown.config(width=18)

# Call the toggle_parameters function initially to set the state of the widgets
toggle_param_run()

# Create a frame for the dropdown/entry fields
c03r915_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c03r915_frame.grid(row=9, column=0, columnspan=4, rowspan=10, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c03r915_label = tk.Label(c03r915_frame, text="Data preprocessing", font=("Helvetica", 12, "bold"), fg="blue")
c03r915_label.grid(row=0, column=0, columnspan=4, pady=5)
# feature_label.config(width=50)


# Add a dropdown list for the Feature selection method
features_label = tk.Label(c03r915_frame, text="Feature selection method 1:", font=("Helvetica", 10, "bold"), anchor="e")
features_label.grid(row=1, column=0, padx=5, pady=3)
features_label.config(width=22)
features_methods = [ 'HVF', 'MedianMAD', 'SelByPCA', 'RandomProjection', 'SelByClustering', 'SelByNMF', 'SelectByRF',
                     'RecursiveFeatElim', 'SeqFeatSel', 'ModelX', 'Suggested']
features_var = tk.StringVar(root)
features_var.set(features_methods[0])
features_dropdown = tk.OptionMenu(c03r915_frame, features_var, *features_methods)
features_dropdown.grid(row=1, column=1, padx=5, pady=5)
features_dropdown.config(width=18)  # Set the width of the widget

features2_label = tk.Label(c03r915_frame, text="Method 2:", font=("Helvetica", 10, "bold"), anchor="e")
features2_label.grid(row=2, column=0, padx=5, pady=5)
features2_label.config(width=22)
features2_methods = ['HVF', 'MedianMAD', 'SelByPCA', 'RandomProjection', 'SelByClustering', 'SelByNMF', 'SelectByRF',
                     'RecursiveFeatElim', 'SeqFeatSel', 'ModelX', 'None']
features2_var = tk.StringVar(root)
features2_var.set(features2_methods[10])
features2_dropdown = tk.OptionMenu(c03r915_frame, features2_var, *features2_methods)
features2_dropdown.grid(row=2, column=1, padx=5, pady=5)
features2_dropdown.config(width=18)  # Set the width of the widget

features3_label = tk.Label(c03r915_frame, text="Method 3:", font=("Helvetica", 10, "bold"), anchor="e")
features3_label.grid(row=3, column=0, padx=5, pady=5)
features3_label.config(width=22)
features3_methods = ['HVF', 'MedianMAD', 'SelByPCA', 'RandomProjection', 'SelByClustering', 'SelByNMF', 'SelectByRF',
                     'RecursiveFeatElim', 'SeqFeatSel', 'ModelX', 'RemoveHighCorrFeat', 'IterativeFeatSel', 'None']
features3_var = tk.StringVar(root)
features3_var.set(features3_methods[12])
features3_dropdown = tk.OptionMenu(c03r915_frame, features3_var, *features3_methods)
features3_dropdown.grid(row=3, column=1, padx=5, pady=5)
features3_dropdown.config(width=18)  # Set the width of the widget

features4_label = tk.Label(c03r915_frame, text="Method 4:", font=("Helvetica", 10, "bold"), anchor="e")
features4_label.grid(row=4, column=0, padx=5, pady=5)
features4_label.config(width=22)
features4_methods = ['HVF', 'MedianMAD', 'SelByPCA', 'RandomProjection', 'SelByClustering', 'SelByNMF', 'SelectByRF',
                     'RecursiveFeatElim', 'SeqFeatSel', 'ModelX', 'RemoveHighCorrFeat', 'IterativeFeatSel', 'None']
features4_var = tk.StringVar(root)
features4_var.set(features3_methods[12])
features4_dropdown = tk.OptionMenu(c03r915_frame, features4_var, *features4_methods)
features4_dropdown.grid(row=4, column=1, padx=5, pady=5)
features4_dropdown.config(width=18)  # Set the width of the widget

top_n_genes_pca_label = tk.Label(c03r915_frame, text="PCA top features:", font=("Helvetica", 10, "bold"), anchor="e")
top_n_genes_pca_label.config(width=22)
top_n_genes_pca_label.grid(row=5, column=0, padx=5, pady=5)
top_n_genes_pca_var = tk.IntVar(root)
top_n_genes_pca_var.set(500)
top_n_genes_pca_entry = tk.Entry(c03r915_frame, textvariable=top_n_genes_pca_var)
top_n_genes_pca_entry.config(width=22)  # Set the width of the widget
top_n_genes_pca_entry.grid(row=5, column=1, padx=5, pady=5)

top_n_genes_rp_label = tk.Label(c03r915_frame, text="RandomProjection features:", font=("Helvetica", 10, "bold"), anchor="e")
top_n_genes_rp_label.config(width=22)
top_n_genes_rp_label.grid(row=6, column=0, padx=5, pady=5)
top_n_genes_rp_var = tk.IntVar(root)
top_n_genes_rp_var.set(500)
top_n_genes_rp_entry = tk.Entry(c03r915_frame, textvariable=top_n_genes_rp_var)
top_n_genes_rp_entry.config(width=22)  # Set the width of the widget
top_n_genes_rp_entry.grid(row=6, column=1, padx=5, pady=5)

n_clusters_label = tk.Label(c03r915_frame, text="Number of clusters:", font=("Helvetica", 10, "bold"), anchor="e")
n_clusters_label.config(width=22)
n_clusters_label.grid(row=7, column=0, padx=5, pady=5)
n_clusters_var = tk.IntVar(root)
n_clusters_var.set(500)
n_clusters_entry = tk.Entry(c03r915_frame, textvariable=n_clusters_var)
n_clusters_entry.config(width=22)  # Set the width of the widget
n_clusters_entry.grid(row=7, column=1, padx=5, pady=5)

n_components_label = tk.Label(c03r915_frame, text="NMF number of components:", font=("Helvetica", 9, "bold"), anchor="e")
n_components_label.config(width=25)
n_components_label.grid(row=8, column=0, padx=5, pady=5)
n_components_var = tk.IntVar(root)
n_components_var.set(500)
n_components_entry = tk.Entry(c03r915_frame, textvariable=n_components_var)
n_components_entry.config(width=22)  # Set the width of the widget
n_components_entry.grid(row=8, column=1, padx=5, pady=5)

min_sel_features_rfe_label = tk.Label(c03r915_frame, text="RFE Min features:", font=("Helvetica", 10, "bold"), anchor="e")
min_sel_features_rfe_label.config(width=22)
min_sel_features_rfe_label.grid(row=9, column=0, padx=5, pady=5)
min_sel_features_rfe_var = tk.IntVar(root)
min_sel_features_rfe_var.set(10)
min_sel_features_rfe_entry = tk.Entry(c03r915_frame, textvariable=min_sel_features_rfe_var)
min_sel_features_rfe_entry.config(width=22)  # Set the width of the widget
min_sel_features_rfe_entry.grid(row=9, column=1, padx=9, pady=5)


# Add input boxes for the parameters for HVG
flavor_label = tk.Label(c03r915_frame, text="Data has been transformed to:", font=("Helvetica", 9, "bold"), anchor="e")
flavor_label.config(width=25)
flavor_label.grid(row=1, column=2, padx=5, pady=5)
flavor_methods = ['log2', 'log10', 'ln', 'None']
flavor_var = tk.StringVar(root)
flavor_var.set(flavor_methods[0])
flavor_dropdown = tk.OptionMenu(c03r915_frame, flavor_var, *flavor_methods)
flavor_dropdown.config(width=18)
flavor_dropdown.grid(row=1, column=3, padx=5, pady=5)

topgenes_label = tk.Label(c03r915_frame, text="HVF Top features:", font=("Helvetica", 10, "bold"), anchor="e")
topgenes_label.config(width=22)
topgenes_label.grid(row=2, column=2, padx=5, pady=5)
topgenes_var = tk.IntVar(root)
topgenes_var.set(0)
topgenes_entry = tk.Entry(c03r915_frame, textvariable=topgenes_var)
topgenes_entry.config(width=22)  # Set the width of the widget
topgenes_entry.grid(row=2, column=3, padx=5, pady=5)

disp_label = tk.Label(c03r915_frame, text="HVG Min dispersion:", font=("Helvetica", 10, "bold"), anchor="e")
disp_label.config(width=22)
disp_label.grid(row=3, column=2, padx=5, pady=5)
disp_var = tk.DoubleVar(root)
disp_var.set(1.5)
disp_entry = tk.Entry(c03r915_frame, textvariable=disp_var)
disp_entry.config(width=22)  # Set the width of the widget
disp_entry.grid(row=3, column=3, padx=5, pady=5)


max_features_label = tk.Label(c03r915_frame, text="RF Maximum features:", font=("Helvetica", 10, "bold"), anchor="e")
max_features_label.config(width=22)
max_features_label.grid(row=4, column=2, padx=5, pady=5)
max_features_var = tk.IntVar(root)
max_features_var.set(0)
max_features_entry = tk.Entry(c03r915_frame, textvariable=max_features_var)
max_features_entry.config(width=22)  # Set the width of the widget
max_features_entry.grid(row=4, column=3, padx=9, pady=5)

threshold_label = tk.Label(c03r915_frame, text="RF Threshold(mean/median):", font=("Helvetica", 9, "bold"), anchor="e")
threshold_label.config(width=25)
threshold_label.grid(row=5, column=2, padx=5, pady=5)
threshold_var = tk.StringVar(root)
threshold_var.set("1.5*mean")
threshold_entry = tk.Entry(c03r915_frame, textvariable=threshold_var)
threshold_entry.config(width=22)  # Set the width of the widget
threshold_entry.grid(row=5, column=3, padx=9, pady=5)

# params_label.config(width=50)
min_sel_features_label = tk.Label(c03r915_frame, text="ModelX Min feature:", font=("Helvetica", 10, "bold"),
                                  anchor="e")
min_sel_features_label.config(width=22)
min_sel_features_label.grid(row=6, column=2, padx=12, pady=5)
min_sel_features_var = tk.IntVar(root)
min_sel_features_var.set(10)
min_sel_features_entry = tk.Entry(c03r915_frame, textvariable=min_sel_features_var)
min_sel_features_entry.config(width=22)  # Set the width of the widget
min_sel_features_entry.grid(row=6, column=3, padx=9, pady=5)

latent_dim_label = tk.Label(c03r915_frame, text="ModelX Latent dimensions:", font=("Helvetica", 10, "bold"), anchor="e")
latent_dim_label.config(width=22)
latent_dim_label.grid(row=7, column=2, padx=5, pady=5)
latent_dim_var = tk.IntVar(root)
latent_dim_var.set(32)
latent_dim_entry = tk.Entry(c03r915_frame, textvariable=latent_dim_var)
latent_dim_entry.config(width=22)  # Set the width of the widget
latent_dim_entry.grid(row=7, column=3, padx=9, pady=5)

fdr_label = tk.Label(c03r915_frame, text="ModelX FDR:", font=("Helvetica", 10, "bold"), anchor="e")
fdr_label.config(width=22)
fdr_label.grid(row=8, column=2, padx=5, pady=5)
fdr_var = tk.DoubleVar(root)
fdr_var.set(0.25)
fdr_entry = tk.Entry(c03r915_frame, textvariable=fdr_var)
fdr_entry.config(width=22)  # Set the width of the widget
fdr_entry.grid(row=8, column=3, padx=9, pady=5)

min_sel_features_sfs_label = tk.Label(c03r915_frame, text="SFS Min features:", font=("Helvetica", 10, "bold"), anchor="e")
min_sel_features_sfs_label.config(width=22)
min_sel_features_sfs_label.grid(row=9, column=2, padx=5, pady=3)
min_sel_features_sfs_var = tk.IntVar(root)
min_sel_features_sfs_var.set(10)
min_sel_features_sfs_entry = tk.Entry(c03r915_frame, textvariable=min_sel_features_sfs_var)
min_sel_features_sfs_entry.config(width=22)  # Set the width of the widget
min_sel_features_sfs_entry.grid(row=9, column=3, padx=9, pady=5)

# Add a button to run the model
run_button = tk.Button(root, text="Run Model", font=("Helvetica", 10, "bold"),  fg="green", command=run_aml)
run_button.grid(row=14, column=4, columnspan=2, rowspan=1, padx=5, pady=15)  # , sticky="w")
run_button.config(width=43)

# Create a button for cancelling the task
cancel_button = tk.Button(root, text="Cancel Run", font=("Helvetica", 10, "bold"),  fg="red", command=cancel_run_)
cancel_button.grid(row=15, column=4, columnspan=1, rowspan=1, padx=5, pady=5)  # , sticky="e")
cancel_button.config(width=20)


def close():
    root.destroy()  # Close the window


# Create a button for cancelling the task
close_button = tk.Button(root, text="Close", font=("Helvetica", 10, "bold"),  fg="brown", command=close)
close_button.grid(row=15, column=5, columnspan=1, rowspan=1, padx=5, pady=5)  # , sticky="e")
close_button.config(width=20)
# Create a Text widget
text_label = tk.Label(root, text="CLETE- Clear, Legible, Explainable, Transparent and Elucidative",
                      font=("Helvetica", 8, "bold"),  fg="blue")
text_label.grid(row=16, column=4, columnspan=2, rowspan=1, padx=10, pady=5, sticky="e")

status_label = tk.Label(root, text="")
status_label.grid(row=17, column=4, columnspan=1)

text_label2 = tk.Label(root, text="@KaziLab.se Lund University", font=("Helvetica", 8, "bold"),  fg="blue")
text_label2.grid(row=17, column=5, columnspan=1, rowspan=1, padx=10, pady=5, sticky="e")

# Start the GUI
root.mainloop()


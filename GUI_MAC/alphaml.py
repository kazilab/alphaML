import tkinter as tk
from threading import Thread
import ctypes  # Add this import for forcefully terminating the thread
# from tkinter import filedialog
from alphaml import aml


def close():
    root.destroy()  # Close the window


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
    n_top_genes = int(topgenes_var.get())  # 20,
    min_disp = float(disp_var.get())  # 1.5,
    flavor = flavor_var.get()  # 'seurat',
    threshold = threshold_var.get()  # '1.5*mean',
    max_features = int(max_features_var.get())  # 0,
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
                            n_top_genes=n_top_genes,
                            min_disp=min_disp,
                            flavor=flavor,
                            threshold=threshold,
                            max_features=max_features,
                            normalization=normalization,
                            latent_dim=latent_dim,
                            min_sel_features=min_sel_features,
                            fdr=fdr,
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
c01r11_label = tk.Label(c01r11_frame, text="Build a CLETE Binary Classification Model", font=("Helvetica", 14),
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
c45r11_label = tk.Label(c45r11_frame, text="Drug Sensitivity Prediction", font=("Helvetica", 14),
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
col_label_label.grid(row=1, column=0, padx=5, pady=5)
col_label_label.config(width=22)
col_label_var = tk.StringVar(root)
col_label_var.set('Trametinib')
col_label_entry = tk.Entry(c01r25_frame, textvariable=col_label_var)
col_label_entry.grid(row=1, column=1, padx=5, pady=5)
col_label_entry.config(width=22)

pos_label = tk.Label(c01r25_frame, text="Positive class:", font=("Helvetica", 10, "bold"), anchor="e")
pos_label.grid(row=2, column=0, padx=5, pady=5)
pos_label.config(width=22)
pos_var = tk.StringVar(root)
pos_var.set('resistant')
pos_entry = tk.Entry(c01r25_frame, textvariable=pos_var)
pos_entry.grid(row=2, column=1, padx=5, pady=5)
pos_entry.config(width=22)

neg_label = tk.Label(c01r25_frame, text="Negative class:", font=("Helvetica", 10, "bold"), anchor="e")
neg_label.grid(row=3, column=0, padx=5, pady=5)
neg_label.config(width=22)
neg_var = tk.StringVar(root)
neg_var.set('sensitive')
neg_entry = tk.Entry(c01r25_frame, textvariable=neg_var)
neg_entry.grid(row=3, column=1, padx=5, pady=5)
neg_entry.config(width=22)

# Create a frame for the dropdown/entry fields
c01r68_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c01r68_frame.grid(row=6, column=0, columnspan=2, rowspan=3, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c01r68_label = tk.Label(c01r68_frame, text="Other parameters", font=("Helvetica", 11, "bold"), fg="blue")
c01r68_label.grid(row=0, column=0, columnspan=2, pady=5)
# params_label.config(width=50)
# Test size for test-train distribution
size_label = tk.Label(c01r68_frame, text="Test size for splitting:", font=("Helvetica", 10, "bold"), anchor="e")
size_label.grid(row=1, column=0, padx=5, pady=5)
size_label.config(width=22)
size_var = tk.DoubleVar(root)
size_var.set(0.2)
size_entry = tk.Entry(c01r68_frame, textvariable=size_var)
size_entry.grid(row=1, column=1, padx=5, pady=5)
size_entry.config(width=22)

# Random state
random_label = tk.Label(c01r68_frame, text="Random seed:", font=("Helvetica", 10, "bold"), anchor="e")
random_label.grid(row=2, column=0, padx=5, pady=5)
random_label.config(width=22)
random_var = tk.IntVar(root)
random_var.set(12)
random_entry = tk.Entry(c01r68_frame, textvariable=random_var)
random_entry.grid(row=2, column=1, padx=5, pady=5)
random_entry.config(width=22)


c23r23_frame = tk.Frame(root, borderwidth=2, relief="groove")
c23r23_frame.grid(row=2, column=2, columnspan=2, rowspan=2, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c23r23_label = tk.Label(c23r23_frame, text="Sampling methods", font=("Helvetica", 11, "bold"),
                        fg="blue")
c23r23_label.grid(row=0, column=0, columnspan=2, pady=5)
# main_label.config(width=50)

# Add a dropdown list for the parameter search method
sampling_label = tk.Label(c23r23_frame, text="Select a sampling method:", font=("Helvetica", 10, "bold"), anchor="e")
sampling_label.grid(row=1, column=0, padx=5, pady=5)
sampling_label.config(width=22)
sampling_methods = ['no', 'over', 'under']
sampling_var = tk.StringVar(root)
sampling_var.set(sampling_methods[0])
sampling_dropdown = tk.OptionMenu(c23r23_frame, sampling_var, *sampling_methods)
sampling_dropdown.grid(row=1, column=1, padx=5, pady=5)
sampling_dropdown.config(width=18)  # Set the width of the widge


c23r48_frame = tk.Frame(root, borderwidth=2, relief="groove")
c23r48_frame.grid(row=4, column=2, columnspan=2, rowspan=5, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c23r48_label = tk.Label(c23r48_frame, text="Sampling & Hyperparameter search", font=("Helvetica", 11, "bold"),
                        fg="blue")
c23r48_label.grid(row=0, column=0, columnspan=2, pady=5)
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
param_label.grid(row=1, column=0, padx=5, pady=5)
param_label.config(width=22)
param_methods = ['Optuna', 'Bayes', 'Grid', 'Predefined']
param_var = tk.StringVar(root)
param_var.set(param_methods[0])
param_var.trace("w", toggle_hyper_param_run)
param_dropdown = tk.OptionMenu(c23r48_frame, param_var, *param_methods)
param_dropdown.grid(row=1, column=1, padx=5, pady=1)
param_dropdown.config(width=18)  # Set the width of the widge

# Add a dropdown list
control_label = tk.Label(c23r48_frame, text="Fitting Controls:", font=("Helvetica", 10, "bold"), anchor="e")
control_label.grid(row=2, column=0, padx=5, pady=5)
control_label.config(width=22)
control_methods = ['Yes', 'No']
control_var = tk.StringVar(root)
control_var.set(control_methods[1])
control_dropdown = tk.OptionMenu(c23r48_frame, control_var, *control_methods)
control_dropdown.grid(row=2, column=1, padx=5, pady=7)
control_dropdown.config(width=18)

# Number of trials
trials_label = tk.Label(c23r48_frame, text="Number of trials:", font=("Helvetica", 10, "bold"), anchor="e")
trials_label.grid(row=3, column=0, padx=5, pady=5)
trials_label.config(width=22)
trials_var = tk.IntVar(root)
trials_var.set(50)
trials_entry = tk.Entry(c23r48_frame, textvariable=trials_var)
trials_entry.grid(row=3, column=1, padx=5, pady=7)
trials_entry.config(width=22)

# Add an input box for the number of cross-validation folds
cv_label = tk.Label(c23r48_frame, text="Cross-Validation folds:", font=("Helvetica", 10, "bold"), anchor="e")
cv_label.grid(row=4, column=0, padx=5, pady=5)
cv_label.config(width=22)
cv_var = tk.IntVar(root)
cv_var.set(5)
cv_entry = tk.Entry(c23r48_frame, textvariable=cv_var)
cv_entry.grid(row=4, column=1, padx=5, pady=7)
cv_entry.config(width=22)

toggle_hyper_param_run()

# Create a frame for the dropdown/entry fields
c45r23_frame = tk.Frame(root, borderwidth=2, relief="groove")
c45r23_frame.grid(row=2, column=4, columnspan=2, rowspan=2, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r23_label = tk.Label(c45r23_frame, text="Select an algorithm", font=("Helvetica", 11, "bold"), fg="blue")
c45r23_label.grid(row=0, column=0, columnspan=2, pady=5)

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
model_name_label = tk.Label(c45r23_frame, text="Algorithm:", font=("Helvetica", 10, "bold"), anchor="e")
model_name_label.grid(row=1, column=0, padx=5, pady=5)
model_name_label.config(width=22)
model_names = ['AdaBoost', 'CatBoost', 'ElasticNet', 'ExtraTrees', 'GradientBoosting', 'HistGradientBoosting', 'LASSO',
               'LightGBM', 'MLPC', 'RandomForest', 'Ridge', 'SVC', 'NuSVC', 'TabNet', 'XGBoost', 'Test_Briefly', 'None'
               ]
model_name_var = tk.StringVar(root)
model_name_var.set(model_names[14])
model_name_var.trace("w", toggle_param_run)  # Call the toggle_parameters function when the selected option changes
model_name_dropdown = tk.OptionMenu(c45r23_frame, model_name_var, *model_names)
model_name_dropdown.grid(row=1, column=1, padx=5, pady=5)
model_name_dropdown.config(width=18)

# Create a frame for the dropdown/entry fields
c45r48_frame = tk.Frame(root, borderwidth=2, relief="groove")
c45r48_frame.grid(row=4, column=4, columnspan=2, rowspan=5, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r48_label = tk.Label(c45r48_frame, text="Explain a model", font=("Helvetica", 11, "bold"), fg="blue")
c45r48_label.grid(row=0, column=0, columnspan=2, pady=5)
# Add a dropdown list for the Feature importance
featimp_label = tk.Label(c45r48_frame, text="Feature importance:", font=("Helvetica", 10, "bold"), anchor="e")
featimp_label.grid(row=1, column=0, padx=5, pady=5)
featimp_label.config(width=22)
featimp_methods = ['Yes', 'No']
featimp_var = tk.StringVar(root)
featimp_var.set(featimp_methods[1])
featimp_dropdown = tk.OptionMenu(c45r48_frame, featimp_var, *featimp_methods)
featimp_dropdown.grid(row=1, column=1, padx=5, pady=5)
featimp_dropdown.config(width=18)

# Add a dropdown list for the ROC-AUC
roc_label = tk.Label(c45r48_frame, text="ROC-AUC plot:", font=("Helvetica", 10, "bold"), anchor="e")
roc_label.grid(row=2, column=0, padx=5, pady=5)
roc_label.config(width=22)
roc_methods = ['Yes', 'No']
roc_var = tk.StringVar(root)
roc_var.set(roc_methods[1])
roc_dropdown = tk.OptionMenu(c45r48_frame, roc_var, *roc_methods)
roc_dropdown.grid(row=2, column=1, padx=5, pady=7)
roc_dropdown.config(width=18)

# Add a dropdown list for the SHAP
shap_label = tk.Label(c45r48_frame, text="SHAP plots:", font=("Helvetica", 10, "bold"), anchor="e")
shap_label.grid(row=3, column=0, padx=5, pady=5)
shap_label.config(width=22)
shap_methods = ['Yes', 'No']
shap_var = tk.StringVar(root)
shap_var.set(shap_methods[1])
shap_dropdown = tk.OptionMenu(c45r48_frame, shap_var, *shap_methods)
shap_dropdown.grid(row=3, column=1, padx=5, pady=7)
shap_dropdown.config(width=18)

# Add a dropdown list for the LIME plots
lime_label = tk.Label(c45r48_frame, text="LIME plots:", font=("Helvetica", 10, "bold"), anchor="e")
lime_label.grid(row=4, column=0, padx=5, pady=5)
lime_label.config(width=22)
lime_methods = ['Yes', 'No']
lime_var = tk.StringVar(root)
lime_var.set(lime_methods[1])
lime_dropdown = tk.OptionMenu(c45r48_frame, lime_var, *lime_methods)
lime_dropdown.grid(row=4, column=1, padx=5, pady=7)
lime_dropdown.config(width=18)

# Call the toggle_parameters function initially to set the state of the widgets
toggle_param_run()

# Create a frame for the dropdown/entry fields
c03r915_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c03r915_frame.grid(row=9, column=0, columnspan=4, rowspan=7, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c03r915_label = tk.Label(c03r915_frame, text="Data preprocessing", font=("Helvetica", 12, "bold"), fg="blue")
c03r915_label.grid(row=0, column=0, columnspan=2, pady=5)
# feature_label.config(width=50)

# Add a dropdown list for the normalization method
norm_label = tk.Label(c03r915_frame, text="Normalization method:", font=("Helvetica", 10, "bold"), anchor="e")
norm_label.config(width=22)
norm_label.grid(row=1, column=0, padx=5, pady=5)
norm_methods = ['min_max', 'None']
norm_var = tk.StringVar(root)
norm_var.set(norm_methods[0])
norm_dropdown = tk.OptionMenu(c03r915_frame, norm_var, *norm_methods)
norm_dropdown.grid(row=1, column=1, padx=5, pady=5)
norm_dropdown.config(width=18)  # Set the width of the widget


# Define a function to disable/enable the parameter widgets based on the selected feature method

def toggle_param_feat(*args):
    if features_var.get() == "Suggested":
        flavor_dropdown.config(state="disabled")
        topgenes_entry.config(state="disabled")
        disp_entry.config(state="disabled")
        max_features_entry.config(state="disabled")
        threshold_entry.config(state="disabled")
        latent_dim_entry.config(state="disabled")
        min_sel_features_entry.config(state="disabled")
        fdr_entry.config(state="disabled")
    elif features_var.get() == 'HVG':
        flavor_dropdown.config(state="normal")
        topgenes_entry.config(state="normal")
        disp_entry.config(state="normal")
        max_features_entry.config(state="disabled")
        threshold_entry.config(state="disabled")
        latent_dim_entry.config(state="disabled")
        min_sel_features_entry.config(state="disabled")
        fdr_entry.config(state="disabled")
    elif features_var.get() == "SelectByRF":
        flavor_dropdown.config(state="disabled")
        topgenes_entry.config(state="disabled")
        disp_entry.config(state="disabled")
        max_features_entry.config(state="normal")
        threshold_entry.config(state="normal")
        latent_dim_entry.config(state="disabled")
        min_sel_features_entry.config(state="disabled")
        fdr_entry.config(state="disabled")
    elif features_var.get() == 'HVG_RF':
        flavor_dropdown.config(state="normal")
        topgenes_entry.config(state="normal")
        disp_entry.config(state="normal")
        max_features_entry.config(state="normal")
        threshold_entry.config(state="normal")
        latent_dim_entry.config(state="disabled")
        min_sel_features_entry.config(state="disabled")
        fdr_entry.config(state="disabled")
    elif features_var.get() == "RecursiveFeatElim" or features_var.get() == "SeqFeatSel":
        flavor_dropdown.config(state="disabled")
        topgenes_entry.config(state="disabled")
        disp_entry.config(state="disabled")
        max_features_entry.config(state="disabled")
        threshold_entry.config(state="disabled")
        latent_dim_entry.config(state="disabled")
        min_sel_features_entry.config(state="normal")
        fdr_entry.config(state="disabled")
    elif features_var.get() == "HVG_RFE" or features_var.get() == "HVG_SFS":
        flavor_dropdown.config(state="normal")
        topgenes_entry.config(state="normal")
        disp_entry.config(state="normal")
        max_features_entry.config(state="disabled")
        threshold_entry.config(state="disabled")
        latent_dim_entry.config(state="disabled")
        min_sel_features_entry.config(state="normal")
        fdr_entry.config(state="disabled")
    elif features_var.get() == "ModelX":
        flavor_dropdown.config(state="disabled")
        topgenes_entry.config(state="disabled")
        disp_entry.config(state="disabled")
        max_features_entry.config(state="disabled")
        threshold_entry.config(state="disabled")
        latent_dim_entry.config(state="normal")
        min_sel_features_entry.config(state="normal")
        fdr_entry.config(state="normal")
    elif features_var.get() == "HVG_ModelX":
        flavor_dropdown.config(state="normal")
        topgenes_entry.config(state="normal")
        disp_entry.config(state="normal")
        max_features_entry.config(state="disabled")
        threshold_entry.config(state="disabled")
        latent_dim_entry.config(state="normal")
        min_sel_features_entry.config(state="normal")
        fdr_entry.config(state="normal")
    elif features_var.get() == "RF_ModelX":
        flavor_dropdown.config(state="disabled")
        topgenes_entry.config(state="disabled")
        disp_entry.config(state="disabled")
        max_features_entry.config(state="normal")
        threshold_entry.config(state="normal")
        latent_dim_entry.config(state="normal")
        min_sel_features_entry.config(state="normal")
        fdr_entry.config(state="normal")
    else:
        flavor_dropdown.config(state="normal")
        topgenes_entry.config(state="normal")
        disp_entry.config(state="normal")
        max_features_entry.config(state="normal")
        threshold_entry.config(state="normal")
        latent_dim_entry.config(state="normal")
        min_sel_features_entry.config(state="normal")
        fdr_entry.config(state="normal")


# Add a dropdown list for the Feature selection method
features_label = tk.Label(c03r915_frame, text="Feature selection method:", font=("Helvetica", 10, "bold"), anchor="e")
features_label.grid(row=2, column=0, padx=5, pady=5)
features_label.config(width=22)
features_methods = ['HVG', 'SelectByRF', 'HVG_RF', 'RecursiveFeatElim', 'HVG_RFE', 'SeqFeatSel', 'HVG_SFS',
                    'ModelX', 'HVG_ModelX', 'RF_ModelX', 'HVG_RF_ModelX', 'Suggested']
features_var = tk.StringVar(root)
features_var.set(features_methods[0])
features_var.trace("w", toggle_param_feat)  # Call the toggle_parameters function when the selected option changes
features_dropdown = tk.OptionMenu(c03r915_frame, features_var, *features_methods)
features_dropdown.grid(row=2, column=1, padx=5, pady=5)
features_dropdown.config(width=18)  # Set the width of the widget

# Add a label to the frame for the title
c03r915_label = tk.Label(c03r915_frame, text="HVG parameters", font=("Helvetica", 11, "bold"), fg="blue")
c03r915_label.grid(row=3, column=0, columnspan=2, pady=5)
# params_label.config(width=50)

# Add input boxes for the parameters for HVG
flavor_label = tk.Label(c03r915_frame, text="HVG flavor:", font=("Helvetica", 10, "bold"), anchor="e")
flavor_label.config(width=22)
flavor_label.grid(row=4, column=0, padx=5, pady=5)
flavor_methods = ['seurat', 'cell_ranger', 'seurat_v3']
flavor_var = tk.StringVar(root)
flavor_var.set(flavor_methods[0])
flavor_dropdown = tk.OptionMenu(c03r915_frame, flavor_var, *flavor_methods)
flavor_dropdown.config(width=18)
flavor_dropdown.grid(row=4, column=1, padx=5, pady=5)

topgenes_label = tk.Label(c03r915_frame, text="Number of top genes:", font=("Helvetica", 10, "bold"), anchor="e")
topgenes_label.config(width=22)
topgenes_label.grid(row=5, column=0, padx=5, pady=5)
topgenes_var = tk.IntVar(root)
topgenes_var.set(0)
topgenes_entry = tk.Entry(c03r915_frame, textvariable=topgenes_var)
topgenes_entry.config(width=22)  # Set the width of the widget
topgenes_entry.grid(row=5, column=1, padx=5, pady=5)

disp_label = tk.Label(c03r915_frame, text="Min dispersion:", font=("Helvetica", 10, "bold"), anchor="e")
disp_label.config(width=22)
disp_label.grid(row=6, column=0, padx=5, pady=5)
disp_var = tk.DoubleVar(root)
disp_var.set(1.5)
disp_entry = tk.Entry(c03r915_frame, textvariable=disp_var)
disp_entry.config(width=22)  # Set the width of the widget
disp_entry.grid(row=6, column=1, padx=5, pady=5)

# Add a label to the frame for the title
params_label = tk.Label(c03r915_frame, text="SKLEARN model parameters",
                        font=("Helvetica", 11, "bold"), fg="blue")
params_label.grid(row=0, column=2, columnspan=2, pady=5)

max_features_label = tk.Label(c03r915_frame, text="Maximum features:", font=("Helvetica", 10, "bold"), anchor="e")
max_features_label.config(width=22)
max_features_label.grid(row=1, column=2, padx=5, pady=5)
max_features_var = tk.IntVar(root)
max_features_var.set(0)
max_features_entry = tk.Entry(c03r915_frame, textvariable=max_features_var)
max_features_entry.config(width=22)  # Set the width of the widget
max_features_entry.grid(row=1, column=3, padx=12, pady=5)

threshold_label = tk.Label(c03r915_frame, text="Threshold (mean/median):", font=("Helvetica", 10, "bold"), anchor="e")
threshold_label.config(width=22)
threshold_label.grid(row=2, column=2, padx=5, pady=5)
threshold_var = tk.StringVar(root)
threshold_var.set("1.0*mean")
threshold_entry = tk.Entry(c03r915_frame, textvariable=threshold_var)
threshold_entry.config(width=22)  # Set the width of the widget
threshold_entry.grid(row=2, column=3, padx=5, pady=5)

# Add a label to the frame for the title
params_label = tk.Label(c03r915_frame, text="ModelX knockoffs parameters",
                        font=("Helvetica", 11, "bold"), fg="blue")
params_label.grid(row=3, column=2, columnspan=2, pady=6)
# params_label.config(width=50)

latent_dim_label = tk.Label(c03r915_frame, text="Latent dimensions:", font=("Helvetica", 10, "bold"), anchor="e")
latent_dim_label.config(width=22)
latent_dim_label.grid(row=4, column=2, padx=5, pady=6)
latent_dim_var = tk.IntVar(root)
latent_dim_var.set(64)
latent_dim_entry = tk.Entry(c03r915_frame, textvariable=latent_dim_var)
latent_dim_entry.config(width=22)  # Set the width of the widget
latent_dim_entry.grid(row=4, column=3, padx=5, pady=6)

min_sel_features_label = tk.Label(c03r915_frame, text="Minimum feature number:", font=("Helvetica", 10, "bold"),
                                  anchor="e")
min_sel_features_label.config(width=22)
min_sel_features_label.grid(row=5, column=2, padx=5, pady=6)
min_sel_features_var = tk.IntVar(root)
min_sel_features_var.set(100)
min_sel_features_entry = tk.Entry(c03r915_frame, textvariable=min_sel_features_var)
min_sel_features_entry.config(width=22)  # Set the width of the widget
min_sel_features_entry.grid(row=5, column=3, padx=5, pady=6)

fdr_label = tk.Label(c03r915_frame, text="FDR:", font=("Helvetica", 10, "bold"), anchor="e")
fdr_label.config(width=22)
fdr_label.grid(row=6, column=2, padx=5, pady=5)
fdr_var = tk.DoubleVar(root)
fdr_var.set(0.25)
fdr_entry = tk.Entry(c03r915_frame, textvariable=fdr_var)
fdr_entry.config(width=22)  # Set the width of the widget
fdr_entry.grid(row=6, column=3, padx=5, pady=5)

# Call the toggle_parameters function initially to set the state of the widgets
toggle_param_feat()

# Create a frame for the dropdown/entry fields
c45r911_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c45r911_frame.grid(row=9, column=4, columnspan=2, rowspan=3, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r911_label = tk.Label(c45r911_frame, text="Parameters for iterations and early stopping",
                         font=("Helvetica", 11, "bold"), fg="blue")
c45r911_label.grid(row=0, column=0, columnspan=2, pady=5)
# params_label.config(width=50)
# Test size for test-train distribution


# Number of epochs
epochs_label = tk.Label(c45r911_frame, text="Maximum epochs:", font=("Helvetica", 10, "bold"), anchor="e")
epochs_label.grid(row=1, column=0, padx=5, pady=5)
epochs_label.config(width=22)
epochs_var = tk.IntVar(root)
epochs_var.set(500)
epochs_entry = tk.Entry(c45r911_frame, textvariable=epochs_var)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)
epochs_entry.config(width=22)

# Patience for early stop
patience_label = tk.Label(c45r911_frame, text="Patience:", font=("Helvetica", 10, "bold"), anchor="e")
patience_label.grid(row=2, column=0, padx=5, pady=5)
patience_label.config(width=22)
patience_var = tk.IntVar(root)
patience_var.set(50)
patience_entry = tk.Entry(c45r911_frame, textvariable=patience_var)
patience_entry.grid(row=2, column=1, padx=5, pady=5)
patience_entry.config(width=22)

# Add a button to run the model
run_button = tk.Button(root, text="Run Model", font=("Helvetica", 12, "bold"),  fg="green", command=run_aml)
run_button.grid(row=12, column=4, columnspan=2, rowspan=1, padx=5, pady=5)  # , sticky="w")
run_button.config(width=48)

# Create a button for cancelling the task
cancel_button = tk.Button(root, text="Cancel Run", font=("Helvetica", 12, "bold"),  fg="red", command=cancel_run_)
cancel_button.grid(row=13, column=4, columnspan=2, rowspan=1, padx=5, pady=5)  # , sticky="e")
cancel_button.config(width=48)

# Create a Text widget
text_label = tk.Label(root, text="CLETE- Clear, Legible, Explainable, Transparent and Elucidative",
                      font=("Helvetica", 12, "bold"),  fg="blue")
text_label.grid(row=14, column=4, columnspan=2, rowspan=1, padx=5, pady=5, sticky="e")

status_label = tk.Label(root, text="")
status_label.grid(row=15, column=4, columnspan=1)

text_label2 = tk.Label(root, text="@KaziLab.se", font=("Helvetica", 10, "bold"),  fg="blue")
text_label2.grid(row=15, column=5, columnspan=1, rowspan=1, padx=5, pady=5, sticky="e")

# Start the GUI
root.mainloop()
'''
browse_button = tk.Button(frame, text="Browse Data File", command=browse_data_file)
browse_button.grid(row=0, column=0, pady=5)

data_path_label = tk.Label(frame, text="", wraplength=300)
data_path_label.grid(row=1, column=0, pady=5)

execute_button = tk.Button(frame, text="Execute venDx", command=execute_venDx)
execute_button.grid(row=2, column=0, pady=5)

root.mainloop()
'''

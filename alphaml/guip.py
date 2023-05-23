import tkinter as tk
from threading import Thread
import ctypes  # for forcefully terminating the thread
from .pred import pred


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
root.title("alphaPred GUI")
# Add a global flag to control the cancellation
cancel_run = False
pred_thread = None


def run_pred():
    global cancel_run, pred_thread
    cancel_run = False
    # Get the values from the dropdown list and input boxes
    normalization = norm_var.get()  # 'min_max',
    col_label = col_label_var.get()  # 'Sensitivity',
    pos_class = pos_var.get()  # 'resistant',
    neg_class = neg_var.get()  # 'sensitive',
    run_shap = shap_var.get()  # 'Yes',
    run_lime = lime_var.get()  # 'Yes',

    # Call the runmodel function with the selected parameters

    def run_pred_thread():
        # Update the status label to display "Running..."
        status_label.config(text="Predicting...")
        run_cancelled = pred(
                             normalization=normalization,
                             col_label=col_label,
                             pos_class=pos_class,
                             neg_class=neg_class,
                             run_shap=run_shap,
                             run_lime=run_lime
                             )
        if run_cancelled:
            status_label.config(text="Prediction cancelled")
        else:
            status_label.config(text="Prediction completed")
    pred_thread = Thread(target=run_pred_thread)
    pred_thread.start()


def cancel_run_():
    global pred_thread, cancel_run

    # Stop the ongoing operation by forcefully terminating the pred_thread
    if pred_thread is not None and pred_thread.is_alive():
        terminate_thread(pred_thread)

    # Clear the status label
    status_label.config(text="")

    # Wait for a short period to ensure the termination takes effect
    # root.after(100, run_pred)  # Restart the pred function after a delay of 100 milliseconds


# Create a frame for the dropdown/entry fields
c01r11_frame = tk.Frame(root, borderwidth=0, relief="groove")
c01r11_frame.grid(row=1, column=0, columnspan=2, rowspan=1, padx=5, pady=5, sticky="w")
# Add a label to the frame for the title
c01r11_label = tk.Label(c01r11_frame, text="Predict test data using a CLETE Model", font=("Helvetica", 11),
                        fg="blue", anchor="sw")
c01r11_label.grid(row=1, column=0, columnspan=2, pady=5)
# c01r11_label.pack(pady=5, fill="both", expand=True)
# c01r11_label.config(width=39)

# Create a frame for the dropdown/entry fields
c23r01_frame = tk.Frame(root, borderwidth=0, relief="groove")
c23r01_frame.grid(row=0, column=2, columnspan=2, rowspan=2, padx=5, pady=5)  # , sticky="w")
# Add a label to the frame for the title
c23r01_label = tk.Label(c23r01_frame, text="alphaPred", font=("Helvetica", 36, "bold"), fg="blue")
c23r01_label.pack(pady=5)
c23r01_label.config(width=11)

# Create a frame for the dropdown/entry fields
c45r11_frame = tk.Frame(root, borderwidth=0, relief="groove")
c45r11_frame.grid(row=1, column=4, columnspan=2, rowspan=1, padx=5, pady=5, sticky="w")
# Add a label to the frame for the title
c45r11_label = tk.Label(c45r11_frame, text="CLETE- Clear, Legible, Explainable, Transparent and Elucidative",
                        font=("Helvetica", 9), fg="blue", anchor="se")
c45r11_label.grid(row=1, column=0, columnspan=2, pady=5)
# c45r11_label.pack(pady=5, fill="both", expand=True)
# c45r11_label.config(width=50)

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
col_label_entry.grid(row=1, column=1, padx=13, pady=12)
col_label_entry.config(width=22)

pos_label = tk.Label(c01r25_frame, text="Positive class:", font=("Helvetica", 10, "bold"), anchor="e")
pos_label.grid(row=2, column=0, padx=5, pady=9)
pos_label.config(width=22)
pos_var = tk.StringVar(root)
pos_var.set('resistant')
pos_entry = tk.Entry(c01r25_frame, textvariable=pos_var)
pos_entry.grid(row=2, column=1, padx=13, pady=12)
pos_entry.config(width=22)

neg_label = tk.Label(c01r25_frame, text="Negative class:", font=("Helvetica", 10, "bold"), anchor="e")
neg_label.grid(row=3, column=0, padx=5, pady=9)
neg_label.config(width=22)
neg_var = tk.StringVar(root)
neg_var.set('sensitive')
neg_entry = tk.Entry(c01r25_frame, textvariable=neg_var)
neg_entry.grid(row=3, column=1, padx=13, pady=12)
neg_entry.config(width=22)

# Create a frame for the dropdown/entry fields
c45r25_frame = tk.Frame(root, borderwidth=2, relief="groove")
c45r25_frame.grid(row=2, column=4, columnspan=2, rowspan=4, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c45r25_label = tk.Label(c45r25_frame, text="Explain prediction", font=("Helvetica", 11, "bold"), fg="blue")
c45r25_label.grid(row=0, column=0, columnspan=2, pady=5)

# Add a dropdown list for the SHAP
shap_label = tk.Label(c45r25_frame, text="SHAP plots:", font=("Helvetica", 10, "bold"), anchor="e")
shap_label.grid(row=1, column=0, padx=5, pady=5)
shap_label.config(width=22)
shap_methods = ['Yes', 'No']
shap_var = tk.StringVar(root)
shap_var.set(shap_methods[1])
shap_dropdown = tk.OptionMenu(c45r25_frame, shap_var, *shap_methods)
shap_dropdown.grid(row=1, column=1, padx=5, pady=10)
shap_dropdown.config(width=18)

# Add a dropdown list for the LIME plots
lime_label = tk.Label(c45r25_frame, text="LIME plots:", font=("Helvetica", 10, "bold"), anchor="e")
lime_label.grid(row=2, column=0, padx=5, pady=5)
lime_label.config(width=22)
lime_methods = ['Yes', 'No']
lime_var = tk.StringVar(root)
lime_var.set(lime_methods[1])
lime_dropdown = tk.OptionMenu(c45r25_frame, lime_var, *lime_methods)
lime_dropdown.grid(row=2, column=1, padx=5, pady=10)
lime_dropdown.config(width=18)

status_label = tk.Label(c45r25_frame, text="")
status_label.grid(row=3, column=0, columnspan=1)

text_label2 = tk.Label(c45r25_frame, text="@KaziLab.se", font=("Helvetica", 8, "bold"),  fg="blue")
text_label2.grid(row=3, column=1, columnspan=1, rowspan=1, padx=10, pady=5, sticky="e")

# Create a frame for the dropdown/entry fields
c23r23_frame = tk.Frame(root, borderwidth=2, relief="groove")
# params_frame = tk.Frame(root, bd=1, relief="sunken")
c23r23_frame.grid(row=2, column=2, columnspan=2, rowspan=2, padx=5, pady=5, sticky="w")

# Add a label to the frame for the title
c23r23_label = tk.Label(c23r23_frame, text="Data preprocessing", font=("Helvetica", 12, "bold"), fg="blue")
c23r23_label.grid(row=0, column=0, columnspan=2, pady=5)
# feature_label.config(width=50)

# Add a dropdown list for the normalization method
norm_label = tk.Label(c23r23_frame, text="Normalization method:", font=("Helvetica", 10, "bold"), anchor="e")
norm_label.config(width=22)
norm_label.grid(row=1, column=0, padx=5, pady=5)
norm_methods = ['min_max', 'standardization', 'None']
norm_var = tk.StringVar(root)
norm_var.set(norm_methods[0])
norm_dropdown = tk.OptionMenu(c23r23_frame, norm_var, *norm_methods)
norm_dropdown.grid(row=1, column=1, padx=15, pady=5)
norm_dropdown.config(width=18)  # Set the width of the widget

# Add a button to run the model
run_button = tk.Button(root, text="Predict", font=("Helvetica", 10, "bold"),  fg="green", command=run_pred)
run_button.grid(row=4, column=2, columnspan=2, rowspan=1, padx=5, pady=5)  # , sticky="w")
run_button.config(width=43)

# Create a button for cancelling the task
cancel_button = tk.Button(root, text="Cancel Prediction", font=("Helvetica", 10, "bold"),
                          fg="red", command=cancel_run_)
cancel_button.grid(row=5, column=2, columnspan=1, rowspan=1, padx=5, pady=5)  # , sticky="e")
cancel_button.config(width=20)


def close():
    root.destroy()  # Close the window


# Create a button for cancelling the task
close_button = tk.Button(root, text="Close", font=("Helvetica", 10, "bold"),  fg="brown", command=close)
close_button.grid(row=5, column=3, columnspan=1, rowspan=1, padx=5, pady=5)  # , sticky="e")
close_button.config(width=20)

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

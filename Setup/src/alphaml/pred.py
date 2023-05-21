import os
import logging
import pandas as pd
from .utils.prep import test_data_preprocessing
from .utils.utils import shap_, lime_


def pred(normalization='min_max',
         col_label='Trametinib',
         pos_class='resistant',
         neg_class='sensitive',
         run_shap='Yes',
         run_lime='Yes',
         ):

    # Check the availability of result and log folder otherwise create
    user_documents = os.path.expanduser("~/Documents")
    data_path = os.path.join(user_documents, "alphaPred_data/")
    result_path = os.path.join(user_documents, "alphaPred_results/")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Set the log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
    log_level = logging.INFO
    log_file_path = os.path.abspath(result_path + "alphaPred_run.log")
    # Configure logging settings
    logging.basicConfig(filename=log_file_path,
                        level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    model, test_data = test_data_preprocessing(data_path=data_path, normalization=normalization)
    test_data.index.name = 'IndexName'
    algorithm = model.__class__.__name__
    path = result_path
    sampling_method = ''
    param_search = ''
    control_fitting = ''
    p_name = path + col_label + '_' + sampling_method + '_' + param_search + '_' + algorithm + '_fit_' + control_fitting
    test_data_pred = model.predict(test_data.to_numpy())
    test_data_proba = model.predict_proba(test_data.to_numpy())[:, 1] if hasattr(model, "predict_proba") else None
    test_data_pred_df = pd.DataFrame(test_data_pred, columns=['predictions'])
    if test_data_proba is not None:
        test_data_proba_df = pd.DataFrame(test_data_proba, columns=['pred_proba'])
        combined_df = pd.concat([test_data_pred_df, test_data_proba_df], axis=1)
    else:
        combined_df = test_data_pred_df

    # Replace the index of combined_df with the index of test_data
    combined_df.index = test_data.index
    # Set the index name of combined_df to the index name of test_data
    combined_df.index.name = test_data.index.name
    # Replace 0 with 'Neg' and 1 with 'Pos' in the 'predictions' column
    combined_df['predictions'] = combined_df['predictions'].replace({0: neg_class, 1: pos_class})
    filename2 = f"{p_name}_test_prediction.xlsx"
    combined_df.to_excel(filename2)
    logging.info(f"Test predictions for {algorithm} has been saved in alphaPred_result folder")

    if run_shap == 'Yes':
        shap_(model=model,
              x_test=test_data,
              path=result_path,
              sampling_method=sampling_method,
              param_search=param_search,
              algorithm=algorithm,
              col_label=col_label,
              control_fitting=control_fitting
              )
    elif run_lime == 'Yes':
        lime_(model=model,
              x_test=test_data,
              path=result_path,
              sampling_method=sampling_method,
              param_search=param_search,
              algorithm=algorithm,
              col_label=col_label,
              control_fitting=control_fitting
              )
    else:
        pass
    del model, algorithm, combined_df, test_data
    print('\n **** Prediction finished **** @ ')

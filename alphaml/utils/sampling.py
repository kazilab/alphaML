import logging

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger('alphaml')

# ***************** Data sampling ***********************************
# Tasks                                                              #
# 1. Take data and labels as pd.DataFrame                            #
# 2. Use imblearn to undersample or over sample                      #
# 3. Run under sampling in two steps                                 #
# 4. Run over sample using SMOTE                                     #
# --------------------------------------------------------------------#


def sampling(x_train, y_train, sampling_method, random_state=None):
    # applicable only for train data, do not sample test data
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks

    sampling_method = str(sampling_method)
    if sampling_method == 'under':
        x1, y1 = TomekLinks().fit_resample(x_train, y_train)
        x, y = RandomUnderSampler(random_state=random_state).fit_resample(x1, y1)
    elif sampling_method == 'over':
        x, y = SMOTE(random_state=random_state).fit_resample(x_train, y_train)
    elif sampling_method == 'no':
        x, y = x_train, y_train
    else:
        error_message = 'Sampling type error!'
        logger.error(error_message)
        raise ValueError(error_message)
    return x, y


def sample_weight_(y):
    """Balanced sample weights for estimators without a class_weight option."""
    return compute_sample_weight(class_weight='balanced', y=np.asarray(y))


def scale_pos_weight_(y):
    y = np.asarray(y)
    return (len(y) - y.sum()) / y.sum()

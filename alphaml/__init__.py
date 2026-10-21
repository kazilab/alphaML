"""
***********************
***   alphaML 2.0   ***
***********************

AutoML for explainable (CLETE) binary classification models from tabular data
"""
__author__ = 'Julhash Kazi'
__email__ = 'aml@kazilab.se'
__description__ = 'alphaml'
__version__ = '0.2.0'
__url__ = 'https://www.kazilab.se'

__all__ = ['train', 'predict']


def __getattr__(name):
    # Imported on first use: worker processes import alphaml.models.folds and should not load everything
    if name == 'train':
        from .run import train
        return train
    if name == 'predict':
        from .prediction import predict
        return predict
    raise AttributeError(f"module 'alphaml' has no attribute '{name}'")

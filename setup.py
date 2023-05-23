"""
Setup alphaML
"""
import sys
try:
    from setuptools import setup, find_packages
except ImportError:
    raise ImportError("Please install `setuptools`")

if not (sys.version_info[0] == 3 and sys.version_info[1] >= 9):
    raise RuntimeError(
                'alphaML requires Python 3.9 or higher.\n'
                'You are using Python {0}.{1}'.format(
                    sys.version_info[0], sys.version_info[1])
                )

# main setup command
setup(
      name='alphaml',
      version='0.1.1', # Major.Minor.Patch
      author='Julhash Kazi',
      author_email='aml@kazilab.se',
      url='https://www.kazilab.se',
      description='Build a CLETE Binary Classification Model',
      license='Apache-2.0',
      install_requires=[
                        'catboost',
                        'imbalanced-learn',
                        'IPython',
                        'kaleido',
                        'lime',
                        'lightgbm',
                        'matplotlib',
                        'numpy',
                        'openpyxl',
                        'optuna',
                        'pandas',
                        'plotly',
                        'requests',
                        'scipy',
                        'scikit-learn',
                        'scikit-optimize',
                        'shap',
                        'tensorflow',
                        'torch',
                        'tqdm',
                        'xgboost'
                        ],
      platforms='any',
      packages=find_packages()
      )


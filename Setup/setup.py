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
    license='GPL',
    install_requires=[
        'catboost>=1.2',
        'imbalanced-learn>=0.10',
        'kaleido>=0.2',
        'lime>=0.2',
        'lightgbm>=3.3',
        'matplotlib>=3.7',
        'numpy>=1.23',
        'optuna>=3.1',
        'pandas>=1.5',
        'plotly>=5.9',
        'requests>=2.28',
        'scipy>=1.10',
        'scikit-learn>=1.2',
        'scikit-misc==0.2.0',
        'scikit-optimize>=0.9',
        'shap>=0.41',
        'tensorflow>=2.12',
        'torch>=1.12',
        'tqdm>=4.64',
        'xgboost>=1.7'
        ],
    platforms='any',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)


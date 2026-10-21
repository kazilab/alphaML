"""
Setup alphaML
"""
import sys
from pathlib import Path
try:
    from setuptools import setup, find_packages
except ImportError:
    raise ImportError("Please install `setuptools`")

if not (sys.version_info[0] == 3 and sys.version_info[1] >= 10):
    raise RuntimeError(
                'alphaML requires Python 3.10 or higher.\n'
                'You are using Python {0}.{1}'.format(
                    sys.version_info[0], sys.version_info[1])
                )

# main setup command
setup(
      name='alphaml',
      version='0.2.0', # Major.Minor.Patch
      author='Julhash Kazi',
      author_email='aml@kazilab.se',
      url='https://www.kazilab.se',
      description='alphaML 2.0: AutoML web application for explainable (CLETE) binary classification models',
      long_description=(Path(__file__).parent / 'README.md').read_text(encoding='utf-8'),
      long_description_content_type='text/markdown',
      license='Apache-2.0',
      python_requires='>=3.10',
      install_requires=[
                        'catboost',
                        'fastapi',
                        'imbalanced-learn',
                        'joblib',
                        'lightgbm',
                        'lime',
                        'matplotlib',
                        'numpy',
                        'openpyxl',
                        'optuna',
                        'pandas',
                        'psutil',
                        'python-multipart',
                        'requests',
                        'scikit-learn',
                        'scikit-optimize',
                        'scipy',
                        'shap',
                        'uvicorn',
                        'xgboost'
                        ],
      extras_require={'test': ['pytest', 'httpx']},
      entry_points={'console_scripts': ['alphaml-web=alphaml.web.__main__:main']},
      platforms='any',
      packages=find_packages(exclude=['tests']),
      package_data={'alphaml.web': ['static/*']},
      )

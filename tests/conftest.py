import numpy as np
import pandas as pd
import pytest


def make_dataset(folder, n_samples=90, n_features=120, n_unlabeled=12, seed=0):
    """Write a small synthetic expression data set in the alphaML input layout."""
    rng = np.random.default_rng(seed)
    genes = [f'GENE{i}' for i in range(n_features)]
    samples = [f'S{i}' for i in range(n_samples)]
    y = rng.integers(0, 2, n_samples)
    x = rng.gamma(2.0, 1.5, size=(n_samples, n_features))
    x[:, :6] *= np.where(y == 1, 3.0, 1.0)[:, None]  # informative genes: 3-fold higher in the positive class
    labeled = pd.DataFrame(np.log2(x + 1), index=samples, columns=genes)
    labels = pd.DataFrame({'Trametinib': np.where(y == 1, 'resistant', 'sensitive'),
                           'Other': rng.choice(['a', 'b'], n_samples)}, index=samples)
    labels.iloc[0, 0] = 'unknown'  # dropped during label encoding
    fs = pd.DataFrame(np.log2(rng.gamma(2.0, 1.5, size=(150, n_features)) + 1),
                      index=[f'F{i}' for i in range(150)], columns=genes)
    unlabeled = pd.DataFrame(np.log2(rng.gamma(2.0, 1.5, size=(n_unlabeled, n_features)) + 1),
                             index=[f'U{i}' for i in range(n_unlabeled)], columns=genes)
    folder.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(folder / 'labeled_data.csv')
    labels.to_csv(folder / 'data_labels.csv')
    fs.to_csv(folder / 'data_for_feature_selection.csv')
    unlabeled.to_csv(folder / 'unlabeled_data.csv')
    pd.DataFrame({'Selected features': genes[:10]}).to_csv(folder / 'suggested_features.csv', index=False)
    labeled.iloc[:15].to_csv(folder / 'test_data.csv')
    return folder


@pytest.fixture(scope='session')
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp('data'))

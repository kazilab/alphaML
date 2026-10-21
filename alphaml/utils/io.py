import os
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------#
# data: Gene names in column header and sample names in index (row header)
# sample BCL2 BCL2L1 PLK1 ... ... ... ... ... ... ... ... ... ... ... FLT3
# spl1    12    39    22  ... ... ... ... ... ... ... ... ... ... ... 30
# spl2    10    22    13  ... ... ... ... ... ... ... ... ... ... ... 25
# -------------------------------------------------------------------------#

MISSING_VALUE_INDICATORS = ['NAN', 'Nan', 'NA#', '#VALUE!', '#DIV/0!', '-', '_', '--', '---', '__', '___',
                            '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
                            '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null',
                            '#NUM!']

TRAIN_FILES = {
    'labeled_data': True,               # required
    'data_labels': True,                # required
    'data_for_feature_selection': False,
    'unlabeled_data': False,
    'suggested_features': False,
}
TABLE_SUFFIXES = ('.csv', '.tsv', '.txt')

# Example data set (pharmacogenomic cell line data) on figshare. The figshare.com download page answers
# scripts with a browser challenge, so the API and download hosts are used, in this order.
EXAMPLE_TRAIN_URLS = ('https://api.figshare.com/v2/file/download/42323292',
                      'https://ndownloader.figshare.com/files/42323292')


def _sep(path):
    return '\t' if Path(path).suffix.lower() in ('.tsv', '.txt') else ','


def read_table(path, numeric=True):
    """
    Read a CSV/TSV file with sample names in the first column.

    With numeric=True, missing-value indicators are parsed as NaN while reading (much faster than
    replacing them afterwards), any remaining text is coerced to NaN, and NaN is replaced by 0.
    """
    if numeric:
        df = pd.read_csv(path, sep=_sep(path), index_col=0, header=0,
                         na_values=MISSING_VALUE_INDICATORS, low_memory=False)
        non_numeric = df.columns[~df.dtypes.map(pd.api.types.is_numeric_dtype)]
        if len(non_numeric):
            df[non_numeric] = df[non_numeric].apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
    else:
        df = pd.read_csv(path, sep=_sep(path), index_col=0, header=0, low_memory=False)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def read_feature_list(path):
    """Read a one-column feature list (header in the first row)."""
    df = pd.read_csv(path, sep=_sep(path), index_col=None, header=0, low_memory=False)
    return [str(f) for f in df.iloc[:, 0].dropna().tolist()]


def as_frame(data, numeric=True):
    if isinstance(data, pd.DataFrame):
        return data
    return read_table(data, numeric=numeric)


def find_file(folder, stem):
    """Return the first <stem>.csv/.tsv/.txt in folder, or None."""
    for suffix in TABLE_SUFFIXES:
        candidate = Path(folder) / f'{stem}{suffix}'
        if candidate.is_file():
            return candidate
    return None


def save_pickle(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)


def load_pickle(path):
    # Unpickling can execute code: only load model files from trusted sources.
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def to_jsonable(value):
    """Convert numpy/pandas scalars and tuples so parameters can be stored as JSON."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ----------------------------- Example data ------------------------------ #


def safe_extract(zip_path, target):
    """Extract a zip archive, refusing members that would land outside target."""
    target = Path(target).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if target != dest and target not in dest.parents:
                raise ValueError(f'Unsafe path in archive: {member.filename}')
        zf.extractall(target)


def download_and_extract(urls, target, timeout=120):
    """Download a zip archive from the first of urls that serves one, and extract its files into target."""
    import requests

    os.makedirs(target, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / 'download.zip'
        errors = []
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    with open(zip_path, 'wb') as fh:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
            except requests.RequestException as exc:
                errors.append(f'{url}: {exc}')
                continue
            if zipfile.is_zipfile(zip_path):
                break
            errors.append(f'{url}: the response is not a zip archive')
        else:
            raise ValueError('Download failed. ' + '; '.join(errors))
        safe_extract(zip_path, tmp + '/x')
        # Archives may wrap the files in a folder: flatten everything into target.
        for root, _, files in os.walk(tmp + '/x'):
            for name in files:
                if not name.startswith('.') and '__MACOSX' not in root:
                    shutil.move(os.path.join(root, name), os.path.join(target, name))


"""Uploaded data sets, stored once and reused across training runs."""
import shutil
import uuid
from pathlib import Path

import pandas as pd

from ..utils.io import TRAIN_FILES, find_file, read_table
from .jobs import now, read_json, write_json

MAX_LABEL_VALUES = 50  # label columns with more distinct values are not offered as class columns


def _count_rows(path):
    with open(path, 'rb') as fh:
        lines = sum(chunk.count(b'\n') for chunk in iter(lambda: fh.read(1 << 20), b''))
    with open(path, 'rb') as fh:
        fh.seek(-1, 2)
        ends_with_newline = fh.read(1) == b'\n'
    return lines - 1 + (0 if ends_with_newline else 1)  # minus the header line


def _columns(path):
    return pd.read_csv(path, sep='\t' if path.suffix.lower() in ('.tsv', '.txt') else ',', nrows=0).columns


def summarize(folder, stats=None):
    """
    Shapes of the data files, and label columns with their values and counts.

    stats: {file stem: {'rows', 'columns', 'ids'}} counted while the files were uploaded; files without them
    are read again (slow for big files on a mapped Docker folder).
    """
    folder = Path(folder)
    stats = stats or {}
    labeled = find_file(folder, 'labeled_data')
    label_path = find_file(folder, 'data_labels')
    labels = read_table(label_path, numeric=False)
    if 'labeled_data' in stats:
        sample_ids = pd.Series(stats['labeled_data']['ids'], dtype=str)
    else:
        sample_ids = pd.read_csv(labeled, sep='\t' if labeled.suffix.lower() in ('.tsv', '.txt') else ',',
                                 usecols=[0]).iloc[:, 0].astype(str)

    files = {}
    for stem in TRAIN_FILES:
        path = find_file(folder, stem)
        if path is None:
            continue
        counted = stats.get(stem)
        info = {'file': path.name, 'size': path.stat().st_size}
        if stem != 'suggested_features':
            info['columns'] = counted['columns'] if counted else max(0, len(_columns(path)) - 1)
        info['rows'] = counted['rows'] if counted else _count_rows(path)
        files[stem] = info

    label_columns = []
    for column in labels.columns:
        counts = labels[column].dropna().astype(str).str.strip().value_counts()
        if 2 <= len(counts) <= MAX_LABEL_VALUES:
            label_columns.append({'name': str(column),
                                  'values': [{'value': v, 'count': int(c)} for v, c in counts.items()]})
    return {'files': files,
            'n_samples': files['labeled_data']['rows'],
            'n_features': files['labeled_data']['columns'],
            'n_matched_labels': int(sample_ids.isin(labels.index).sum()),
            'label_columns': label_columns}


class DatasetStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, dataset_id):
        if not dataset_id or not all(c in '0123456789abcdef' for c in dataset_id):
            raise KeyError(dataset_id)
        return self.root / dataset_id

    def new_folder(self):
        dataset_id = uuid.uuid4().hex[:12]
        folder = self.root / dataset_id
        folder.mkdir(parents=True)
        return dataset_id, folder

    def finalize(self, dataset_id, name, owner=None, public=False, stats=None):
        """
        Validate the files in a new data set folder and record its summary.

        owner: id of the user it belongs to (None without accounts). public: visible to every user (example data).
        stats: file statistics counted during a chunked upload (see summarize).
        """
        folder = self.folder(dataset_id)
        try:
            missing = [stem for stem, required in TRAIN_FILES.items() if required and not find_file(folder, stem)]
            if missing:
                raise ValueError(f"Missing required file(s): {', '.join(missing)}")
            info = summarize(folder, stats)
            if info['n_features'] < 1:
                raise ValueError('The labeled data has no feature columns.')
            if not info['label_columns']:
                raise ValueError('The labels file has no column with between 2 and '
                                 f'{MAX_LABEL_VALUES} distinct values.')
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        record = {'id': dataset_id, 'name': name, 'created': now(), 'owner': owner, 'public': public, **info}
        write_json(folder / 'dataset.json', record)
        return record

    def get(self, dataset_id):
        try:
            return read_json(self.folder(dataset_id) / 'dataset.json')
        except KeyError:
            return None

    def list(self):
        records = [read_json(d / 'dataset.json') for d in self.root.iterdir() if d.is_dir()]
        return sorted((r for r in records if r), key=lambda r: r['created'], reverse=True)

    def inputs(self, dataset_id):
        """File paths in the form train() expects (None for optional files that were not uploaded)."""
        folder = self.folder(dataset_id)
        return {stem: (str(p) if (p := find_file(folder, stem)) else None) for stem in TRAIN_FILES}

    def delete(self, dataset_id):
        shutil.rmtree(self.folder(dataset_id), ignore_errors=True)

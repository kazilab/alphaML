"""Regenerate the handbook's small, deterministic, entirely synthetic input files."""
import csv
import math
import random
from pathlib import Path


def build():
    root = Path(__file__).parent / 'static'
    rng = random.Random(20261019)
    features = [f'GENE{i:02d}' for i in range(1, 17)]

    def sample():
        latent = rng.gauss(0, 1)
        values = [max(0.1, 4 + (1 if j < 4 else -0.6 if j < 8 else 0) * latent
                      + rng.gauss(0, 0.6)) for j in range(16)]
        probability = 1 / (1 + math.exp(-2.0 * latent))
        label = 'resistant' if rng.random() < probability else 'sensitive'
        return [round(value, 5) for value in values], label

    with (root / 'example_labeled_data.csv').open('w', newline='', encoding='utf-8') as data, \
         (root / 'example_data_labels.csv').open('w', newline='', encoding='utf-8') as labels:
        dw, lw = csv.writer(data), csv.writer(labels)
        dw.writerow(['sample', *features])
        lw.writerow(['sample', 'response'])
        for i in range(120):
            values, label = sample()
            code = f'S{i + 1:03d}'
            dw.writerow([code, *values])
            lw.writerow([code, label])
    with (root / 'example_new_samples.csv').open('w', newline='', encoding='utf-8') as new:
        writer = csv.writer(new)
        writer.writerow(['sample', *reversed(features)])
        for i in range(16):
            values, _ = sample()
            writer.writerow([f'N{i + 1:03d}', *reversed(values)])
    with (root / 'example_suggested_features.csv').open('w', newline='', encoding='utf-8') as panel:
        writer = csv.writer(panel)
        writer.writerow(['feature'])
        writer.writerows([[name] for name in ('GENE01', 'GENE02', 'GENE05', 'GENE06')])


if __name__ == '__main__':
    build()

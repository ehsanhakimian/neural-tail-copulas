#!/usr/bin/env python3
"""Recompute the 8-versus-18-epoch statistics from archived weekly score vectors.

Example (run from the publisher folder):
    python scripts/analyze_18_epoch.py --package-root .

No model retraining, internet connection, or API key is required.
"""
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

def hac(x, lags=4):
    x = np.asarray(x, dtype=float)
    n = len(x)
    centered = x - x.mean()
    variance = np.dot(centered, centered) / n
    for h in range(1, lags + 1):
        variance += 2 * (1 - h / (lags + 1)) * np.dot(centered[h:], centered[:-h]) / n
    standard_error = np.sqrt(max(float(variance), 0) / n)
    mean = float(x.mean())
    return dict(mean_diff=mean, hac_se=standard_error,
                ci_lower=mean - 1.96 * standard_error,
                ci_upper=mean + 1.96 * standard_error,
                p_normal=float(2 * norm.sf(abs(mean / standard_error))) if standard_error else 1.,
                weeks_won=int(np.sum(x > 0)), n=n)

def run(root):
    directory = root / '18_EPOCH_ANALYSIS'
    archive = root / '00_User_Uploads' / 'ROLLING_VINE_18_EPOCH_RESULTS.zip'
    if not archive.is_file():
        raise FileNotFoundError(f'Original 18-epoch upload is missing: {archive}')
    if not directory.is_dir():
        raise FileNotFoundError(f'18-epoch audit inputs are missing: {directory}')
    output = directory / 'INDEPENDENT_RECHECK'
    output.mkdir(exist_ok=True)
    rows, paired = [], []
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Uploaded results ZIP has a damaged member')
        for design in ('full', 'price'):
            current = pd.read_csv(io.BytesIO(z.read(f'{design}/walkforward_test_scores.csv')))
            previous = pd.read_csv(directory / 'EIGHT_EPOCH_INPUTS' / f'{design}_walkforward_test_scores_8epoch.csv')
            if not current.Date.equals(previous.Date):
                raise ValueError(f'Forecast dates do not agree for {design}')
            if len(current) != 221 or current.isna().values.any():
                raise ValueError(f'Incomplete or invalid test observations for {design}')
            ranking = pd.read_csv(io.BytesIO(z.read(f'{design}/walkforward_ranking.csv')), index_col=0)
            for model in current.columns[1:]:
                if not np.isclose(current[model].mean(), ranking.loc[model, 'mean_log_score'], atol=1e-9):
                    raise ValueError(f'Ranking mismatch for {design}/{model}')
                rows.append(dict(spec=design, model=model, mean_8=float(previous[model].mean()),
                                 mean_18=float(current[model].mean()), **hac(current[model] - previous[model])))
            for model, reference in [('rolling_two_tree','rolling_neural'),
                                     ('expanding_two_tree','expanding_neural'),
                                     ('rolling_two_tree','gaussian_ewma'),
                                     ('rolling_two_tree','dynamic_student_t')]:
                paired.append(dict(spec=design,model=model,benchmark=reference,
                                   **hac(current[model] - current[reference])))
    pd.DataFrame(rows).to_csv(output / '8_vs_18.csv',index=False)
    pd.DataFrame(paired).to_csv(output / '18_epoch_pairs.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda n:f'{n:.5f}'))
    print(f'Rechecked {len(rows)} model/period comparisons. Files: {output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=Path('.'))
    args = parser.parse_args()
    run(args.package_root.resolve())

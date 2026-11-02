#!/usr/bin/env python3
"""Audit a synthetic experiment directory without modifying its source files."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def audit(path: Path):
    config = json.loads((path / 'config.json').read_text())
    scores = pd.read_csv(path / 'oos_log_scores.csv')
    advanced = pd.read_csv(path / 'oos_log_scores_enriched.csv') if (path / 'oos_log_scores_enriched.csv').exists() else None
    U = pd.read_csv(path / 'pseudo_observations.csv')
    returns = pd.read_csv(path / 'simulated_returns.csv')
    assert len(U) == config['T'] == len(returns)
    assert U.shape[1] == returns.shape[1] == config['d']
    assert 0 < config['train_end'] < config['val_end'] < config['T']
    assert np.isfinite(U.values).all() and ((U.values > 0) & (U.values < 1)).all()
    assert not scores['model'].duplicated().any()
    assert np.isfinite(scores['mean_log_score']).all()
    if advanced is not None:
        assert not advanced['model'].duplicated().any()
        bad_names = [n for n in advanced['model'] if 'Cholesky Gaussian GRU' in n]
        if bad_names:
            print('CAVEAT: archived output mislabels ridge Cholesky baseline as GRU:',bad_names)
    risk = pd.read_csv(path/'var_es_backtest.csv') if (path/'var_es_backtest.csv').exists() else None
    if risk is not None:
        for _,row in risk.iterrows():
            assert int(row['exceedances']) <= int(row['n'])
            print(f"VaR exceedances: {int(row['exceedances'])} / {int(row['n'])}; not evidence of calibration by itself")
    digest = hashlib.sha256((path/'config.json').read_bytes()).hexdigest()
    print('AUDIT PASS:', path, 'shape', U.shape, 'configuration_SHA256', digest)
    print(scores[['model','mean_log_score']].to_string(index=False))

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);args=ap.parse_args()
    audit(args.results)

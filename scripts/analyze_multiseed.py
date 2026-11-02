#!/usr/bin/env python3
"""Audit 50 seeded, simulated multivariate copula density scores.

Run from the publisher folder with `python scripts/analyze_multiseed.py`.
A paired comparison uses one replication as one statistical observation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
base = Path(__file__).resolve().parents[1] / 'empirical_results' / 'multiseed_50'
data = pd.read_csv(base/'all_seed_scores.csv')
p = data.pivot(index='seed', columns='model', values='mean_log_score').sort_index()
assert p.shape == (50,7) and p.notna().all().all(), 'Missing or duplicate model–seed pairs'
neural = 'Neural mixture C-vine'
competitors = ['State-linear mixture C-vine','Static mixture C-vine',
               'Static Student-t copula','Static Gaussian copula','EWMA Gaussian copula']
rows = []
for benchmark in competitors:
    d = p[neural]-p[benchmark]
    lo,hi = stats.t.interval(0.95, len(d)-1, loc=d.mean(), scale=stats.sem(d))
    rows.append(dict(benchmark=benchmark, mean_difference=d.mean(),
                     CI95_lower=lo, CI95_upper=hi, wins=int((d>0).sum()),
                     losses=int((d<0).sum()), paired_t_p=stats.ttest_1samp(d, 0).pvalue))
r = pd.DataFrame(rows)
ix = np.argsort(r.paired_t_p.to_numpy())
q = r.paired_t_p.to_numpy()
adj = np.zeros(len(q))
adj[ix] = np.minimum(1, np.maximum.accumulate(q[ix]*(len(q)-np.arange(len(q)))))
r['holm_p_adjusted'] = adj
r.to_csv(base/'paired_comparisons_recomputed.csv', index=False)
print(r.to_string(index=False))

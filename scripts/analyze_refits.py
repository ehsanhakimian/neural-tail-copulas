#!/usr/bin/env python3
"""Independent analysis of fixed test scores; inference is exploratory.

No hyperparameter is selected by this script. It only audits saved predictions.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def hholm(pvals):
    p=np.asarray(pvals);order=np.argsort(p);out=np.ones(len(p));last=0
    for pos,k in enumerate(order):
        val=min(1.,(len(p)-pos)*p[k]);last=max(last,val);out[k]=last
    return out

def block_ci(x,seed=20261102,B=2400,block=8):
    rng=np.random.default_rng(seed);x=np.asarray(x,dtype=float);n=len(x)
    starts=np.arange(n-block+1);m=int(np.ceil(n/block));means=np.empty(B)
    for b in range(B):
        pos=rng.choice(starts,size=m,replace=True)
        idx=np.concatenate([np.arange(k,k+block) for k in pos])[:n]
        means[b]=x[idx].mean()
    return np.quantile(means,[.025,.975]).tolist()

def main(full,price,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(Path(full)/'walkforward_test_scores.csv',parse_dates=['Date'])
    b=pd.read_csv(Path(price)/'walkforward_test_scores.csv',parse_dates=['Date'])
    assert a.Date.equals(b.Date)
    assert len(a)==221 and a.notna().all().all() and b.notna().all().all()
    comparisons=[('rolling_two_tree','rolling_neural'),('rolling_two_tree','gaussian_ewma'),
                 ('rolling_two_tree','dynamic_student_t'),('rolling_neural','gaussian_ewma'),
                 ('expanding_neural','gaussian_ewma'),('dynamic_student_t','gaussian_ewma')]
    rows=[]
    for x,y in comparisons:
        d=(a[x]-a[y]).to_numpy(float);n=len(d);z=d-d.mean();lag=4
        v=float(z.dot(z)/n)
        for j in range(1,lag+1):v+=2*(1-j/(lag+1))*float(z[j:].dot(z[:-j])/n)
        se=np.sqrt(max(v,0)/n);p=2*stats.norm.sf(abs(float(d.mean())/se)) if se else 1.
        ci=block_ci(d,seed=20261102+len(rows))
        rows.append(dict(model=x,benchmark=y,mean_difference=float(d.mean()),hac4_se=se,
            hac4_p=p,lower_hac95=float(d.mean()-1.96*se),upper_hac95=float(d.mean()+1.96*se),
            block_bootstrap_lo=ci[0],block_bootstrap_hi=ci[1],wins=int((d>0).sum()),n=n))
    c=pd.DataFrame(rows);c['holm_p']=hholm(c.hac4_p.to_numpy());c.to_csv(out/'walkforward_independent_comparisons.csv',index=False)
    sensitivity=[]
    for model in ('rolling_neural','rolling_two_tree','expanding_neural','expanding_two_tree'):
        d=b[model]-a[model];sensitivity.append(dict(model=model,
          with_macro=float(a[model].mean()),price_only=float(b[model].mean()),
          price_minus_macro=float(d.mean()),weekly_max_abs_change=float(np.max(np.abs(d)))))
    pd.DataFrame(sensitivity).to_csv(out/'no_fred_sensitivity.csv',index=False)
    risk=Path(full).parent/'risk'/'historically_lagged_risk_summary.csv'
    if not risk.exists():
        # The analysis can still run when risk calculations have not been executed.
        raise FileNotFoundError(f'Missing risk summary: {risk}. Run risk_price_only.py first.')
    summary=pd.read_csv(risk)
    summary['expected_exceptions']=summary.n*.05
    summary.to_csv(out/'risk_only_price_available.csv',index=False)
    # These are exploratory paired tests on an already examined chronological holdout.
    models=['dynamic_student_t','gaussian_ewma','rolling_two_tree','rolling_neural','expanding_two_tree','expanding_neural']
    means=a[models].mean().sort_values(ascending=True)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.barh(range(len(means)),means.values)
    ax.set_yticks(range(len(means)),means.index)
    ax.set_xlabel('Mean weekly out-of-sample copula log score (higher is better)')
    ax.set_title('Walk-forward computational pilot, 221 weeks')
    fig.tight_layout();fig.savefig(out/'walkforward_ranking.pdf');fig.savefig(out/'walkforward_ranking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ['rolling_two_tree','rolling_neural','dynamic_student_t']:
        (a[name]-a['gaussian_ewma']).cumsum().plot(ax=ax,label=name)
    ax.axhline(0,linestyle=':');ax.set_ylabel('Cumulative score difference versus EWMA Gaussian')
    ax.set_title('Fixed 2022–2026 holdout: exploratory comparison')
    ax.legend();fig.tight_layout();fig.savefig(out/'walkforward_cumulative_differences.pdf');plt.close(fig)
    audit=dict(n=len(a),test_start=str(a.Date.min().date()),test_end=str(a.Date.max().date()),
        first_test_previously_inspected=True,claims_only_exploratory=True,
        four_tiers=['original 50-seed synthetic','original observed one-shot','eight-epoch walk-forward pilot','as-of FRED not yet downloaded'],
        observed_macro_data='latest revised; not verified vintages',
        statistical_methods='HAC(4) mean differences, moving block bootstrap size 8; Holm within six exploratory contrasts')
    (out/'SCIENTIFIC_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(c.round(4).to_string(index=False))
    print(pd.DataFrame(sensitivity).round(4).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',required=True);p.add_argument('--price',required=True);p.add_argument('--out',required=True)
    z=p.parse_args();main(z.full,z.price,z.out)

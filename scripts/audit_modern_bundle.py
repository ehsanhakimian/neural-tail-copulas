#!/usr/bin/env python3
"""Recalculate weekly score and VaR summaries from the bundled modern-market results.

Usage (from a publisher edition):
    python scripts/audit_modern_bundle.py --base modern_etf_fred --out audit_check
This is an output audit, not a neural-network refit.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

def hac(series, lag=3):
    x=np.asarray(series,dtype=float)
    z=x-x.mean(); n=len(x)
    lr=float(z@z/n)
    for k in range(1,min(lag,n-1)+1):
        lr+=2*(1-k/(lag+1))*float(z[k:]@z[:-k])/n
    se=float(np.sqrt(max(lr,0)/n))
    return se,float(2*norm.sf(abs(x.mean()/se))) if se else float('nan')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='modern_etf_fred')
    p.add_argument('--out',default='modern_etf_fred/audit_check')
    a=p.parse_args()
    path=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    obs=pd.read_csv(path/'data/observations.csv',parse_dates=['Date'])
    test=pd.read_csv(path/'results/test_scores.csv',parse_dates=['Date'])
    risk=pd.read_csv(path/'results/risk_forecasts_95.csv',parse_dates=['Date'])
    if len(test)!=221 or len(obs)!=1126 or len(risk)!=221*6:
        raise RuntimeError('Unexpected observed sample size; inspect the dataset before using results.')
    if not (test.Date.values==obs.Date.iloc[-len(test):].values).all():
        raise ValueError('Held-out test dates are not the final chronological block.')
    model_cols=test.columns.drop('Date')
    mean=test[model_cols].mean().sort_values(ascending=False)
    mean.rename('mean_copula_log_score').to_csv(out/'model_scores.csv',header=True)
    pair=[]
    for m in model_cols.drop('neural_mixture'):
        d=test.neural_mixture-test[m]
        se,pval=hac(d)
        pair.append({'benchmark':m,'mean_diff':float(d.mean()),'HAC3_SE':se,'p_approx':pval,'lower95':float(d.mean()-1.96*se),'upper95':float(d.mean()+1.96*se),'neural_wins':int((d>0).sum())})
    pd.DataFrame(pair).to_csv(out/'paired_tests.csv',index=False)
    rs=[]
    for m,g in risk.groupby('model'):
        n=len(g);b=int(g.breach.sum());alpha=.05;phat=np.clip(b/n,1e-12,1-1e-12)
        uc=-2*((n-b)*np.log((1-alpha)/(1-phat))+b*np.log(alpha/phat))
        rs.append({'model':m,'breaches':b,'expected':n*alpha,'breach_rate':b/n,'quantile_loss_mean':g.pinball95.mean(),'Kupiec_p':chi2.sf(uc,1)})
    pd.DataFrame(rs).to_csv(out/'var95_audit.csv',index=False)
    print('SUCCESS: audited raw weekly outcomes and saved model_scores.csv, paired_tests.csv and var95_audit.csv')
    print(mean.round(4).to_string())

if __name__=='__main__':main()

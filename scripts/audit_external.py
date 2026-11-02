#!/usr/bin/env python3
"""Locked external-universe validation audit (never selects a model on test scores).

Input score files must have exactly the same forecast dates. Results are reported
as observed, including unfavorable differences. Missing results cause failure.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

UNIVERSE=['EFA','EEM','GLD','IEF','TLT','VNQ']
MODELS=['rolling_two_tree','rolling_neural','dynamic_student_t','gaussian_ewma']

def hac(x, lags=4):
    x=np.asarray(x,float); n=len(x)
    if n<20 or not np.isfinite(x).all(): raise ValueError('Invalid paired sample')
    a=x-x.mean(); v=float(a@a/n)
    for k in range(1,min(lags+1,n)):
        v+=2*(1-k/(lags+1))*float(a[k:]@a[:-k]/n)
    se=float(np.sqrt(max(v,0)/n)); m=float(x.mean())
    return dict(mean=m,hac_se=se,lo=m-1.96*se,hi=m+1.96*se,
                p=float(2*stats.norm.sf(abs(m/se))) if se>1e-14 else (0. if m else 1.),
                wins=int((x>0).sum()),n=n)

def kupiec(exceed, n, alpha=.05):
    if n==0:return np.nan
    p=exceed/n
    x=exceed; y=n-exceed
    l0=x*np.log(alpha)+y*np.log(1-alpha)
    l1=(x*np.log(p) if x else 0.)+(y*np.log(1-p) if y else 0.)
    return float(stats.chi2.sf(2*(l1-l0),1))

def run(panel,scorefile,riskfile,out):
    panel=Path(panel);scorefile=Path(scorefile);riskfile=Path(riskfile);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    x=pd.read_csv(panel,parse_dates=['Date']);s=pd.read_csv(scorefile,parse_dates=['Date'])
    required=['r_MKT','r_EEM','r_GLD','r_IEF','r_TLT','r_VNQ',
      'x_lag_mkt_return','x_lag_market_abs_4w','x_lag_market_return_12w']
    if list(x.columns)!=['Date']+required:raise ValueError(f'Unexpected panel columns: {list(x.columns)}')
    if not x.Date.is_monotonic_increasing or x.Date.duplicated().any():raise ValueError('Invalid input dates')
    if x[required].isna().any().any() or not np.isfinite(x[required].to_numpy()).all():raise ValueError('Nonfinite observations')
    if any(z not in s.columns for z in ['Date']+MODELS):raise ValueError('Missing forecast-score columns')
    if s.Date.duplicated().any() or not s.Date.is_monotonic_increasing:raise ValueError('Duplicate/unsorted score dates')
    if not s.Date.isin(x.Date).all():raise ValueError('Forecast dates do not match external panel')
    if not np.isfinite(s[MODELS].to_numpy()).all():raise ValueError('Nonfinite scores')
    ranks=s[MODELS].agg(['mean','std']).T.sort_values('mean',ascending=False)
    ranks.to_csv(out/'external_logscore_ranking.csv')
    comparisons=[]
    for other in ['dynamic_student_t','gaussian_ewma','rolling_neural']:
        comparisons.append(dict(model='rolling_two_tree',comparator=other,
            **hac(s['rolling_two_tree']-s[other])))
    comps=pd.DataFrame(comparisons)
    # Holm correction across the three prespecified external comparisons.
    order=np.argsort(comps.p.to_numpy());adj=np.empty(len(order));prev=0
    for rank, idx in enumerate(order):
        prev=max(prev, min(1., (len(order)-rank)*float(comps.loc[idx,'p'])))
        adj[idx]=prev
    comps['holm_p']=adj
    comps.to_csv(out/'external_paired_hac.csv',index=False)
    risk=pd.read_csv(riskfile,parse_dates=['Date'])
    need={'Date','model','loss_realized','VaR95','ES95','breach','pinball95'}
    if not need.issubset(risk.columns):raise ValueError('Risk file columns missing')
    wanted=set(['neural_two_tree','neural_first_tree','dynamic_student_t','ewma_gaussian'])
    if set(risk.model)!=wanted:raise ValueError('Unexpected risk models')
    if risk.duplicated(['Date','model']).any():raise ValueError('Duplicated risk predictions')
    if any(set(risk.loc[risk.model==k,'Date'])!=set(s.Date) for k in wanted):
        raise ValueError('Risk and score forecast dates differ')
    outrows=[]
    for k,g in risk.groupby('model'):
        n=len(g);ex=int(g.breach.sum())
        outrows.append(dict(model=k,n=n,breaches=ex,expected_breaches=.05*n,
               observed_breach_rate=ex/n,kupiec_p=kupiec(ex,n),
               mean_pinball=float(g.pinball95.mean()),mean_VaR=float(g.VaR95.mean()),
               mean_ES=float(g.ES95.mean())))
    pd.DataFrame(outrows).to_csv(out/'external_risk_audit.csv',index=False)
    meta=dict(source_panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
              n_input=len(x),n_test=len(s),test_start=str(s.Date.iloc[0].date()),
              test_end=str(s.Date.iloc[-1].date()),
              asset_universe=UNIVERSE,status='externally observed test panel, retrospective adjusted prices',
              holdout_caveat='new cross-sectional universe; overlapping historical calendar; not prospective future data',
              no_test_tuning=True,selection_policy='locked before opening test outcomes',
              risk_model_name_map={'rolling_two_tree':'neural_two_tree','gaussian_ewma':'ewma_gaussian'})
    (out/'EXTERNAL_AUDIT_METADATA.json').write_text(json.dumps(meta,indent=2))
    print('PASS: external data, score and risk integrity.\n',ranks.to_string(), '\nPaired tests\n',comps.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--scores',required=True)
    p.add_argument('--risk',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run(a.panel,a.scores,a.risk,a.out)

#!/usr/bin/env python3
"""Independent audit of uploaded weekly ETF/FRED forecasting outputs.

All estimates derive from supplied CSV files; the fitted neural network is not rerun.
"""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm, chi2, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent/'modern_uploaded'
OUT=Path(__file__).resolve().parent/'audit'
OUT.mkdir(exist_ok=True)
obs=pd.read_csv(ROOT/'modern_data/observations.csv',parse_dates=['Date']);test=pd.read_csv(ROOT/'modern_results/test_scores.csv',parse_dates=['Date']); risk=pd.read_csv(ROOT/'modern_results/risk_forecasts_95.csv',parse_dates=['Date'])
metadata=json.loads((ROOT/'modern_results/experiment.json').read_text());prov=json.loads((ROOT/'modern_data/DATA_PROVENANCE.json').read_text())
assert len(obs)==1126 and obs.Date.is_unique and obs.Date.is_monotonic_increasing
assert len(test)==metadata['n_test']==221 and test.Date.is_unique
assert len(risk)==len(test)*6 and risk.Date.nunique()==221
assert list(test.Date)==list(obs.Date.iloc[-221:]); assert obs.isna().sum().sum()==0
assert test.notna().all().all() and np.isfinite(test.drop(columns='Date')).all().all()
for name in ('modern_data/observations.csv','modern_data/DATA_PROVENANCE.json','modern_data/daily_adjusted_closes.csv','modern_data/fred_latest_revised.csv'):
    with zipfile.ZipFile('/mnt/data/MODERN_DATA_BACKUP.zip') as z: a=z.read(name)
    assert hashlib.sha256(a).digest()==hashlib.sha256((ROOT/name).read_bytes()).digest(),name

models=[x for x in test.columns if x!='Date'];n=len(test)

def hac(x, lag):
    x=np.asarray(x,float); cen=x-x.mean(); N=len(x)
    v=float(cen@cen)/N
    for k in range(1,min(lag,N-1)+1): v+=2*(1-k/(lag+1))*float(cen[k:]@cen[:-k])/N
    se=np.sqrt(max(v,0)/N); p=2*norm.sf(abs(x.mean()/se)) if se>0 else 0
    return se,p,float(x.mean()-1.96*se),float(x.mean()+1.96*se)

means=test[models].mean(); stds=test[models].std(ddof=1)
assert np.allclose(means.sort_values(ascending=False).values,pd.read_csv(ROOT/'modern_results/test_model_comparison.csv').mean_test_log_score.values,atol=1e-6)
mt=pd.DataFrame({'model':models,'mean_score':[float(means[m]) for m in models], 'sd_score':[float(stds[m]) for m in models],'n_test':n}).sort_values('mean_score',ascending=False)
mt.to_csv(OUT/'verified_scores.csv',index=False)
rows=[]
for m in models:
    if m=='neural_mixture':continue
    diff=np.asarray(test.neural_mixture-test[m]); se,p,lo,hi=hac(diff,3)
    row={'benchmark':m,'mean_neural_minus_benchmark':diff.mean(),'hac3_se':se,'ci95_lower':lo,'ci95_upper':hi,'p_hac3':p,'neural_weeks_won':int((diff>0).sum()),'share_weeks_won':float((diff>0).mean()),'median_diff':float(np.median(diff))}
    for lag in [0,8,12,26]:
        _,pv,lower,upper=hac(diff,lag);row[f'p_hac{lag}']=pv;row[f'ci95_hac{lag}_lo']=lower;row[f'ci95_hac{lag}_hi']=upper
    rows.append(row)
comp=pd.DataFrame(rows).sort_values('mean_neural_minus_benchmark',ascending=False)
# Holm: adjusted p in ascending p order, monotone correction.
ordering=np.argsort(comp.p_hac3.to_numpy()); raw=comp.p_hac3.to_numpy();adj=np.empty_like(raw);maxval=0.;K=len(raw)
for j,i in enumerate(ordering):
    maxval=max(maxval,(K-j)*raw[i]);adj[i]=min(1.,maxval)
comp['holm_adjusted_p_hac3']=adj
comp.to_csv(OUT/'verified_pairwise_with_sensitivity.csv',index=False)

# Paired circular moving block bootstrap, deterministic conditional on uploaded scores.
rng=np.random.default_rng(20261102); B=4000;block=8
rows=[]
for m in models:
    if m=='neural_mixture':continue
    v=(test.neural_mixture-test[m]).to_numpy()
    num_blocks=int(np.ceil(n/block))
    starts=rng.integers(0,n,size=(B,num_blocks))
    inds=(starts[:,:,None]+np.arange(block)[None,None,:])%n
    draws=v[inds.reshape(B,-1)[:,:n]].mean(axis=1)
    rows.append({'benchmark':m,'block_length':block,'B':B,'mean_difference':v.mean(),'bootstrap_ci_lo':np.quantile(draws,.025),'bootstrap_ci_hi':np.quantile(draws,.975),'bootstrap_prob_gain_positive':np.mean(draws>0)})
pd.DataFrame(rows).to_csv(OUT/'block_bootstrap.csv',index=False)

# Subperiod comparisons, descriptive only, cut before seeing the outcomes would be ideal.
bins=[('2022-08-12','2023-12-29','2022-2023'),('2024-01-01','2025-12-31','2024-2025'),('2026-01-01','2026-12-31','2026')]
ss=[]
for st,en,label in bins:
    x=test.loc[(test.Date>=st)&(test.Date<=en)]
    if len(x):
        out={'period':label,'n_weeks':len(x),'neural_mixture':x.neural_mixture.mean()}
        for m in ('gaussian_ewma','student_t_static','state_linear_mixture','static_mixture'):
            out[m]=x[m].mean();out['diff_neural_minus_'+m]=float((x.neural_mixture-x[m]).mean())
        ss.append(out)
pd.DataFrame(ss).to_csv(OUT/'subperiod_scores.csv',index=False)

risk_summary=[]
for name,g in risk.groupby('model'):
    n_r=len(g);b=int(g.breach.sum()); expected=n_r*.05;exceed=b/n_r
    p_twosided=binomtest(b,n_r,.05).pvalue
    # Kupiec LRuc for observed proportions
    a=.05;phat=min(max(exceed,1e-12),1-1e-12)
    LR=-2*((n_r-b)*np.log((1-a)/(1-phat))+b*np.log(a/phat))
    risk_summary.append({'model':name,'n_test':n_r,'breaches':b,'breach_rate':exceed,'expected':expected,'mean_VaR95':g.VaR95.mean(),'mean_ES95':g.ES95.mean(),'mean_pinball95':g.pinball95.mean(),'kupiec_LRuc':LR,'kupiec_p':chi2.sf(LR,1),'binomial_exact_two_sided_p':p_twosided})
rt=pd.DataFrame(risk_summary).sort_values('mean_pinball95')
rt.to_csv(OUT/'verified_var_backtest.csv',index=False)
# Paired weekly loss scoring difference for quantile score of neural - EWMA.
pivot=risk.pivot(index='Date',columns='model',values='pinball95').sort_index()
for m in ['gaussian_ewma','student_t_static']:
    diff=(pivot.neural_mixture-pivot[m]).values
    se,p,lo,hi=hac(diff,3)
    (OUT/f'risk_pinball_comparison_{m}.json').write_text(json.dumps({'neural_minus':m,'mean_diff':float(diff.mean()),'HAC_se':se,'p_approx':p,'CI95':[lo,hi]},indent=2))
# plot cumulative differential: neural minus comparators
fig,ax=plt.subplots(figsize=(9,4.8))
for col in ('gaussian_ewma','student_t_static','state_linear_mixture','static_mixture'):
    ax.plot(test.Date,(test.neural_mixture-test[col]).cumsum(),label='Neural − '+col.replace('_',' '))
ax.axhline(0,color='black',lw=.7);ax.set(title='Modern ETF/FRED test: cumulative paired copula log-score difference',ylabel='Cumulative score difference',xlabel='Week');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
fig.savefig(OUT/'modern_cumulative_score_differences.png',dpi=170);fig.savefig(OUT/'modern_cumulative_score_differences.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,4.4))
view=mt.sort_values('mean_score')
ax.barh(view.model.str.replace('_',' '),view.mean_score);ax.set(xlabel='Mean out-of-sample copula log score',title='Modern ETF/FRED: model ranking, 221 weeks');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT/'modern_model_ranking.png',dpi=170);fig.savefig(OUT/'modern_model_ranking.pdf');plt.close(fig)
report={'n_observations_raw':len(obs),'n_fit_after_24_lags':len(obs)-24,'n_train':metadata['n_train'],'n_val':metadata['n_validation'],'n_test':n,'start':str(obs.Date.min().date()),'end':str(obs.Date.max().date()),'train_end':metadata['train_end'],'val_end':metadata['validation_end'],'test_start':metadata['test_start'],'test_end':metadata['test_end'],'data_backup_sha_identical':True,'winning_model':mt.iloc[0]['model'],'source':'Uploaded REAL_MARKET_MODERN_RESULTS_v2.zip; no model refit performed'}
(OUT/'independent_audit.json').write_text(json.dumps(report,indent=2))
print(mt.to_string(index=False));print('\nPAIRED\n',comp[['benchmark','mean_neural_minus_benchmark','ci95_lower','ci95_upper','p_hac3','holm_adjusted_p_hac3','neural_weeks_won']].to_string(index=False));print('\nRISK\n',rt[['model','breaches','breach_rate','mean_pinball95','kupiec_p']].to_string(index=False));print('\nSUBPERIOD\n',pd.DataFrame(ss).to_string(index=False))

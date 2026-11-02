#!/usr/bin/env python3
"""One-week-ahead VaR/ES for adaptive Gaussian and Student-t copulas.

Inputs are historical returns, train-only fitted marginal Student-t laws and
covariance states updated with strictly preceding observations. No FRED
indicator enters either forecast. This cannot verify historical corporate-
action revisions in the supplied adjusted ETF price histories.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'code'))
import core_engine as core
from run_real_market import prep_observations
from run_walkforward import locked_t_df

def calc(inp,out,draws=3000,seed=314159):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data,R,X,U,loc,sigma,pars,tr,val,te,desc=prep_observations(inp,24,.6,.8)
    nu,_=locked_t_df(U,tr,val)
    zr=stats.norm.ppf(core.clip_u(U));zt=stats.t.ppf(core.clip_u(U),df=nu)
    cov={}
    cov['ewma_gaussian']=np.cov(zr[tr].T)
    cov['dynamic_student_t']=np.cov(zt[tr].T)
    rng=np.random.default_rng(seed);recs=[]
    for t in range(len(tr),len(U)):
        if t>len(tr):
            for key,z in [('ewma_gaussian',zr),('dynamic_student_t',zt)]:
                cov[key]=.96*cov[key]+.04*np.outer(z[t-1],z[t-1])
        if t<te[0]:continue
        for key in ['ewma_gaussian','dynamic_student_t']:
            S=cov[key];sd=np.sqrt(np.maximum(np.diag(S),1e-10));corr=S/np.outer(sd,sd)
            corr=.985*corr+.015*np.eye(U.shape[1])
            if key=='ewma_gaussian':UU=core.sample_gaussian(draws,corr,rng)
            else:UU=core.sample_tcopula(draws,corr,nu,rng)
            scen=np.column_stack([loc[t,j]+sigma[t,j]*(par['location']+
                 par['scale']*stats.t.ppf(UU[:,j],df=par['df']))
                 for j,par in enumerate(pars)])
            losses=-scen.mean(axis=1)
            var=float(np.quantile(losses,.95));es=float(losses[losses>=var].mean())
            loss=-float(R[t].mean())
            recs.append(dict(Date=data.index[t],model=key,loss_realized=loss,
                 VaR95=var,ES95=es,breach=int(loss>var),
                 pinball95=float((.95-(loss<=var))*(loss-var))))
    frame=pd.DataFrame(recs);frame.to_csv(out/'historically_lagged_risk_forecasts.csv',index=False)
    summary=frame.groupby('model').agg(n=('breach','size'),exceedances=('breach','sum'),
        mean_VaR=('VaR95','mean'),mean_ES=('ES95','mean'),mean_pinball=('pinball95','mean')).reset_index()
    summary.to_csv(out/'historically_lagged_risk_summary.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'source':'uploaded ETF returns',
        'covariance_updates':'strictly earlier copula scores only',
        'indicators':'no FRED indicators enter forecasts',
        'margins':'original train-only Student-t laws, lagged 24-week mean and volatility',
        'selected_t_df_from_original_validation':nu,'draws_per_forecast':draws,'seed':seed,
        'caution':'adjusted prices may be retrospectively adjusted; not complete as-of-price certification'},indent=2))
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--out',required=True);a.add_argument('--draws',type=int,default=3000)
    z=a.parse_args();calc(z.input,z.out,z.draws)

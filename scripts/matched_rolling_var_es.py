#!/usr/bin/env python3
"""Matched one-week-ahead 95% VaR/ES, using a rolling neural two-tree vine.

A single locked 221-week ETF test panel and estimated Student-t margins are used
for all models. Neural fit, second-tree edge estimation, and elliptical covariance
updates never include the forecast week's realized returns. The experiment is
EXPLORATORY: the 2022--2026 period was examined in previous analyses.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'code'))
from run_real_market import prep_observations
from run_walkforward import fit_origin, conditional_pits, fitted_rho, locked_t_df
import core_engine as core

def mixture_h(root,v,l,up,w):
    # Vectorized first-tree conditional CDF; monotone in v.
    r=np.clip(root,1e-8,1-1e-8); v=np.clip(v,1e-8,1-1e-8)
    hl=np.power(r,-l-1)*np.power(np.maximum(np.power(r,-l)+np.power(v,-l)-1,1e-30),-1/l-1)
    q=1-r; s=1-v
    hu=1-np.power(q,-up-1)*np.power(np.maximum(np.power(q,-up)+np.power(s,-up)-1,1e-30),-1/up-1)
    return np.clip(w*hl+(1-w)*hu,0,1)

def invert_mixture_h(root,h,l,up,w,iterations=37):
    lo=np.full(h.shape,1e-8);hi=np.full(h.shape,1-1e-8)
    for _ in range(iterations):
        mid=(lo+hi)/2
        hmid=mixture_h(root,mid,l,up,w)
        lo=np.where(hmid<h,mid,lo)
        hi=np.where(hmid>=h,mid,hi)
    return np.clip((lo+hi)/2,1e-7,1-1e-7)

def vine_sim(n,param,rho,rng,second_tree=True):
    l,up,w=param
    J=len(l);out=np.empty((n,J+1));root=rng.uniform(1e-6,1-1e-6,size=n);out[:,0]=root
    if second_tree:
        z=np.empty((n,J));z[:,0]=rng.normal(size=n)
        for j in range(1,J):
            rr=float(rho[j-1]);z[:,j]=rr*z[:,j-1]+np.sqrt(max(1-rr*rr,1e-8))*rng.normal(size=n)
        h=stats.norm.cdf(z)
    else:h=rng.uniform(size=(n,J))
    for j in range(J):out[:,j+1]=invert_mixture_h(root,h[:,j],l[j],up[j],w[j])
    return out

def loss_scenarios(U,loc,sigma,pars):
    x=np.empty_like(U)
    for j,p in enumerate(pars):
        x[:,j]=loc[j]+sigma[j]*(p['location']+p['scale']*stats.t.ppf(np.clip(U[:,j],1e-7,1-1e-7),df=p['df']))
    return -x.mean(axis=1)

def score_risk(losses,observed,alpha):
    var=float(np.quantile(losses,alpha));es=float(losses[losses>=var].mean())
    pin=float((alpha-int(observed<=var))*(observed-var))
    return var,es,int(observed>var),pin

def run(inp,out,epochs=18,cadence=26,window=260,draws=3000,features='price',seed=20261102):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data,R,X,U,loc,sigma,pars,tr,val,te,desc=prep_observations(inp,24,.60,.80)
    cols=desc['feature_columns'];raw=pd.read_csv(inp)
    if features=='price':
        mask=[i for i,c in enumerate(cols) if 'macro_' not in c]
        assert len(mask)==3,(cols,mask)
        state=raw[[cols[j] for j in mask]].to_numpy(float)[24:]
        mu=state[tr].mean(0);sd=np.maximum(state[tr].std(0),1e-9)
        X=np.clip((state-mu)/sd,-7,7)
    elif features!='all':raise ValueError(features)
    x=torch.as_tensor(X,dtype=core.DTYPE);u=torch.as_tensor(U,dtype=core.DTYPE)
    nu,_=locked_t_df(U,tr,val)
    ntest=len(te);d=U.shape[1];n0=int(te[0]);N=len(U)
    zG=stats.norm.ppf(core.clip_u(U));zT=stats.t.ppf(core.clip_u(U),df=nu)
    covG=np.cov(zG[tr].T);covT=np.cov(zT[tr].T)
    # Store benchmark covariance paths without ever using the realized current return.
    historical=[]
    for t in range(len(tr),N):
        if t>len(tr):
            covG=.96*covG+.04*np.outer(zG[t-1],zG[t-1]);covT=.96*covT+.04*np.outer(zT[t-1],zT[t-1])
        if t>=n0:historical.append((covG.copy(),covT.copy()))
    assert len(historical)==ntest
    allrows=[]; refits=[];start=time.monotonic()
    for origin in range(n0,N,cadence):
        until=min(origin+cadence,N)
        filename=out/f'block_{origin}_{until}_ep{epochs}_N{draws}_{features}.csv'
        if filename.exists():
            block=pd.read_csv(filename)
            required={'Date','model','VaR95','ES95','breach','pinball95'}
            if required.issubset(block.columns) and len(block)==(until-origin)*4:
                allrows.append(block);print('Reused',filename.name,flush=True);continue
        model,info=fit_origin(x,u,origin,window,epochs,seed=113)
        model.eval()
        with torch.no_grad():
            fullpars=tuple(p.detach().cpu().numpy() for p in model(x[:until]))
        H=conditional_pits(U[:origin],tuple(p[:origin] for p in fullpars))
        rho=fitted_rho(H[max(0,origin-window):origin]);refits.append(dict(**info,until=until))
        rows=[]
        for t in range(origin,until):
            rr=np.random.default_rng(seed+t)
            param=tuple(p[t] for p in fullpars)
            models={
                'neural_first_tree':vine_sim(draws,param,rho,np.random.default_rng(seed+t),False),
                'neural_two_tree':vine_sim(draws,param,rho,np.random.default_rng(seed+t),True),
            }
            Sg,St=historical[t-n0]
            for key,S in [('ewma_gaussian',Sg),('dynamic_student_t',St)]:
                scales=np.sqrt(np.maximum(np.diag(S),1e-8));corr=S/np.outer(scales,scales)
                corr=.985*corr+.015*np.eye(d)
                if key=='ewma_gaussian':models[key]=core.sample_gaussian(draws,corr,np.random.default_rng(seed+t))
                else:models[key]=core.sample_tcopula(draws,corr,nu,np.random.default_rng(seed+t))
            realized=-R[t].mean()
            for name,scenU in models.items():
                simulated_loss=loss_scenarios(scenU,loc[t],sigma[t],pars)
                var,es,breach,pin=score_risk(simulated_loss,realized,.95)
                rows.append(dict(Date=str(data.index[t].date()),model=name,loss_realized=realized,VaR95=var,ES95=es,breach=breach,pinball95=pin))
        frame=pd.DataFrame(rows);frame.to_csv(filename,index=False);allrows.append(frame)
        print(f'Completed {origin}:{until} in {time.monotonic()-start:.1f}s',flush=True)
    result=pd.concat(allrows,ignore_index=True).sort_values(['Date','model'])
    result.to_csv(out/'matched_risk_forecasts.csv',index=False)
    summary=result.groupby('model').agg(n=('breach','size'),exceedances=('breach','sum'),mean_VaR=('VaR95','mean'),mean_ES=('ES95','mean'),mean_pinball=('pinball95','mean')).reset_index()
    summary.to_csv(out/'matched_risk_summary.csv',index=False)
    if refits:
        prev=pd.read_csv(out/'refit_audit.csv') if (out/'refit_audit.csv').exists() else pd.DataFrame()
        pd.concat([prev,pd.DataFrame(refits)],ignore_index=True).drop_duplicates(['origin'],keep='last').to_csv(out/'refit_audit.csv',index=False)
    (out/'methodology.json').write_text(json.dumps(dict(n_test=ntest,test_start=desc['test_start'],test_end=desc['test_end'],epochs_max=epochs,cadence=cadence,rolling_window=window,features=features,draws=draws,df_dynamic_t=nu,margin_fitting='original training sample only; past 24-week mean and standard deviation',no_as_of_fred_claim=True,original_holdout_previously_inspected=True,measured_model_accuracy='Monte Carlo VaR and ES, equal-weight all six assets, same margins/test dates for four models',notes='Exploratory and not independent confirmation; adjusted prices are retrospectively adjusted'),indent=2))
    print(summary.to_string(index=False,float_format=lambda z:f'{z:.5f}'),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--cadence',type=int,default=26)
    p.add_argument('--window',type=int,default=260);p.add_argument('--draws',type=int,default=3000)
    p.add_argument('--features',choices=['all','price'],default='price')
    x=p.parse_args();run(x.input,x.out,x.epochs,x.cadence,x.window,x.draws,x.features)

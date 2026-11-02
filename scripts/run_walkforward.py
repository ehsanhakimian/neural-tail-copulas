#!/usr/bin/env python3
"""Fixed-protocol weekly walk-forward audit of the observed ETF panel.

The original 2022-2026 test window is never used for hyperparameter selection.
Each model fit uses only observations strictly preceding its forecast origin;
subsequent within-block model predictions use a causal (forward) GRU.
This is an experimental follow-up to an already inspected test set, not a new
untouched confirmatory study. See PROTOCOL.md for limitations.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'code'))
sys.path.insert(0,str(BASE/'scripts'))
import core_engine as core
from run_real_market import prep_observations


def clayton_h(root, leaf, theta):
    r=np.clip(root,1e-8,1-1e-8)
    v=np.clip(leaf,1e-8,1-1e-8)
    th=np.maximum(theta,1e-5)
    q=np.maximum(np.power(r,-th)+np.power(v,-th)-1,1e-30)
    return np.power(r,-th-1)*np.power(q,-1/th-1)


def conditional_pits(u,pars):
    l,up,omega=pars
    root=u[:,0,None];leaf=u[:,1:]
    h1=clayton_h(root,leaf,l)
    h2=1-clayton_h(1-root,1-leaf,up)
    return np.clip(omega*h1+(1-omega)*h2,1e-7,1-1e-7)


def pair_gaussian_scores(h, rho):
    z=stats.norm.ppf(np.clip(h,1e-7,1-1e-7))
    a=z[:,:-1];b=z[:,1:]
    r=rho.reshape(1,-1); d=1-r*r
    return (-.5*np.log(d) - .5*((r*r)*(a*a+b*b)-2*r*a*b)/d).sum(axis=1)


def fitted_rho(h):
    z=stats.norm.ppf(np.clip(h,1e-7,1-1e-7))
    out=[]
    for j in range(z.shape[1]-1):
        x=z[:,j];y=z[:,j+1];r=np.corrcoef(x,y)[0,1]
        out.append(float(np.clip(.98*r,-.90,.90)) if np.isfinite(r) else 0.0)
    return np.array(out)


def ewma_t_scores(U, initial, nu, lam=.96):
    d=U.shape[1];z=stats.t.ppf(np.clip(U,1e-6,1-1e-6),df=nu)
    S=np.cov(z[:initial].T)
    out=np.full(len(U),np.nan)
    for t in range(initial,len(U)):
        if t>initial:
            q=z[t-1];S=lam*S+(1-lam)*np.outer(q,q)
        sd=np.sqrt(np.maximum(np.diag(S),1e-8));R=S/np.outer(sd,sd)
        R=.985*R+.015*np.eye(d)
        out[t]=core.t_copula_ll(U[t:t+1],R,nu)[0]
    return out


def locked_t_df(U,tr,val):
    options=[4.,5.,6.,8.,12.,20.]
    scores=[]
    for nu in options:
        ss=ewma_t_scores(U,len(tr),nu)
        scores.append((float(np.mean(ss[val])),nu))
    return max(scores)[1], scores


def fit_origin(x,u,origin,history,epochs,seed=113):
    start=0 if history is None else max(0,origin-history)
    n=origin-start
    if n<90:raise ValueError('Insufficient past observations')
    # Final 20% of each historical window is used ONLY to choose epoch via early stopping.
    split=start+int(.80*n)
    train=np.arange(start,split);valid=np.arange(split,origin)
    cfg=core.Config(d=u.shape[1],hidden=7,emb=2,max_epochs=epochs,
                    patience=9,lr=.008,theta_min=.05,theta_max=6.0)
    core.set_seed(seed)
    mdl=core.NeuralMixtureVine(x.shape[1],u.shape[1]-1,cfg)
    hist,v=core.fit_model(mdl,x[:origin],u[:origin],train,valid,cfg,'neural',max_epochs=epochs)
    return mdl,dict(origin=int(origin),start=int(start),train_count=int(len(train)),
                    validation_count=int(len(valid)),best_val_nll=float(v),n_epochs=int(len(hist)))


def score_fold(model,x,u,origin,until,second_tree=True,history=None):
    model.eval()
    with torch.no_grad():
        pars=tuple(z.detach().cpu().numpy() for z in model(x[:until]))
    U=u.detach().cpu().numpy()
    root=core.star_loglik_np(U[origin:until],*(p[origin:until] for p in pars))
    if not second_tree:return root,None
    H=conditional_pits(U[:until],pars)
    start=0 if history is None else max(0,origin-history)
    rho=fitted_rho(H[start:origin])
    return root,root+pair_gaussian_scores(H[origin:until],rho)


def hac_ci(x,lags=4):
    a=np.asarray(x,dtype=float);n=len(a);mu=float(a.mean());c=a-mu
    v=float(np.mean(c*c))
    for j in range(1,min(n,lags+1)):
        v+=2*(1-j/(lags+1))*float(np.dot(c[j:],c[:-j])/n)
    se=np.sqrt(max(v,0)/n)
    return dict(mean=mu,hac_se=float(se),lower_95=mu-1.96*se,upper_95=mu+1.96*se,
                p_normal=float(2*stats.norm.sf(abs(mu/se))) if se>0 else np.nan,
                weeks_won=int((a>0).sum()),n=n)


def evaluate(input_csv,out,cadence=26,epochs=18,rolling_window=260,features='all'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    df,R,X,U,loc,sigma,pars,tr,val,te,desc=prep_observations(input_csv,24,.60,.80)
    cols=desc['feature_columns']
    if features=='price':
        mask=[i for i,c in enumerate(cols) if 'macro_' not in c]
        if len(mask)!=3:raise ValueError('Expected 3 historical price-only state features')
        # Standardize only on the original training sample.
        p=X[:,mask].copy()
        original=pd.read_csv(input_csv)
        raw=original[[cols[j] for j in mask]].to_numpy(float)[24:]
        mu=raw[tr].mean(axis=0); sd=np.maximum(raw[tr].std(axis=0),1e-9)
        X=np.clip((raw-mu)/sd,-7,7)
    x=torch.tensor(X,dtype=core.DTYPE);u=torch.tensor(U,dtype=core.DTYPE)
    test_start=int(te[0]);nt=len(U);dates=df.index
    store={key:np.full(nt,np.nan) for key in
           ['expanding_neural','rolling_neural','expanding_two_tree','rolling_two_tree']}
    fit_details=[]
    cache=out/'refit_cache';cache.mkdir(exist_ok=True)
    for origin in range(test_start,nt,cadence):
        until=min(nt,origin+cadence)
        for mode,history in [('expanding',None),('rolling',rolling_window)]:
            t0=time.time()
            checkpoint=cache/f'{features}_{mode}_{origin}_{until}_ep{epochs}_cad{cadence}_win{rolling_window}.npz'
            if checkpoint.exists():
                z=np.load(checkpoint)
                star=z['star'];tree=z['tree']
                detail=json.loads(str(z['meta'].item()))
                print(f'Reused saved checkpoint {checkpoint.name}',flush=True)
            else:
                mdl,detail=fit_origin(x,u,origin,history,epochs)
                star,tree=score_fold(mdl,x,u,origin,until,True,history)
                np.savez_compressed(checkpoint,star=star,tree=tree,meta=json.dumps(detail))
            store[mode+'_neural'][origin:until]=star
            store[mode+'_two_tree'][origin:until]=tree
            detail.update(strategy=mode,forecast_start=str(dates[origin].date()),
                          forecast_end=str(dates[until-1].date()),
                          elapsed_seconds=float(time.time()-t0))
            fit_details.append(detail)
            print(f'{mode} origin {origin} weeks {origin}:{until} '
                  f'logscore={np.mean(star):.3f} +2tree={np.mean(tree):.3f}',flush=True)
    # The degrees of freedom are selected entirely in the pre-test validation sample.
    nu,nu_scores=locked_t_df(U,tr,val)
    store['dynamic_student_t']=ewma_t_scores(U,len(tr),nu)
    store['gaussian_ewma']=core.ewma_gaussian_ll(U,len(tr),lam=.96)
    # Read-only reference: historical one-shot model from the original experiment.
    row=pd.DataFrame({'Date':dates[te]})
    for k,v in store.items():row[k]=v[te]
    if not row.drop(columns=['Date']).notna().all().all():raise ValueError('Missing score values')
    row.to_csv(out/'walkforward_test_scores.csv',index=False)
    ranks=(row.drop(columns=['Date']).agg(['mean','std']).T
           .rename(columns={'mean':'mean_log_score','std':'sd_log_score'})
           .sort_values('mean_log_score',ascending=False))
    ranks.to_csv(out/'walkforward_ranking.csv')
    tests=[]
    for mode in ('expanding','rolling'):
        for comparator in ('gaussian_ewma','dynamic_student_t'):
            tests.append(dict(model=mode+'_neural',benchmark=comparator,
                **hac_ci(row[mode+'_neural']-row[comparator])))
        tests.append(dict(model=mode+'_two_tree',benchmark=mode+'_neural',
            **hac_ci(row[mode+'_two_tree']-row[mode+'_neural'])))
    pd.DataFrame(tests).to_csv(out/'locked_protocol_paired_tests.csv',index=False)
    pd.DataFrame(fit_details).to_csv(out/'refit_audit.csv',index=False)
    metadata=dict(input_file=str(input_csv),feature_ablation=features,
        original_test_start=desc['test_start'],original_test_end=desc['test_end'],
        n_test=len(te),cadence_weeks=cadence,epochs=epochs,
        rolling_window_weeks=rolling_window,expanding_start=desc['train_start'],
        dynamic_student_t_df=nu,df_validation_scores=nu_scores,
        validation_selection_only=True,test_tuning_permitted=False,
        second_tree='fixed adjacent sector edges, Gaussian conditional pair copulas; not full R-vine',
        comparison_status='post-hoc follow-up on a previously examined 2022-2026 test period',
        historic_vintage_verification='NOT established by this script',
        marginal_law='original train-only fitted univariate t with past 24-week moments')
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('\nFINAL SCORES:\n'+ranks.to_string(float_format=lambda z:f'{z:.4f}'),flush=True)
    return ranks

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--cadence',type=int,default=26)
    p.add_argument('--epochs',type=int,default=18)
    p.add_argument('--rolling-window',type=int,default=260)
    p.add_argument('--features',choices=['all','price'],default='all')
    a=p.parse_args()
    evaluate(a.input,a.out,a.cadence,a.epochs,a.rolling_window,a.features)

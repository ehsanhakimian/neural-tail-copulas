#!/usr/bin/env python3
"""Observed-market density-forecast study: temporally honest split and train-only margins.

For each asset, the PIT uses a Student-t distribution fitted to training-window
standardized residuals. The conditional location and volatility inputs are computed
from PRIOR monthly or weekly returns. Tail-dependence models are trained on the
training split and selected on an untouched chronological validation split.
The final test split is never used for fitting or model selection.
"""
import argparse, sys, json, math, time
from pathlib import Path
from dataclasses import asdict
import numpy as np, pandas as pd
from scipy import stats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
# The notebook and local repository ship the same core_engine module in ../code/.
sys.path.insert(0,str(HERE.parent/'code'))
import core_engine as core


def prep_observations(file, lag_window, tr, va):
    d=pd.read_csv(file,parse_dates=['Date']).set_index('Date').sort_index()
    assert d.index.is_unique and d.index.is_monotonic_increasing
    rc=[c for c in d if c.startswith('r_')]; xc=[c for c in d if c.startswith('x_')]
    assert len(rc)>=3 and len(xc)>=2
    # MKT must be the root. No interpolation of missing returns.
    assert rc[0]=='r_MKT', f'Root return must be first, found {rc[0]}'
    assert d[rc+xc].notna().all().all()
    R=d[rc].to_numpy(float); X=d[xc].to_numpy(float); N=len(d)
    # All risk parameters at t derive from returns strictly before t.
    loc=np.zeros_like(R); sigma=np.zeros_like(R)
    for i in range(N):
        past=R[max(0,i-lag_window):i]
        if len(past)<lag_window:
            loc[i]=np.nan;sigma[i]=np.nan
        else:
            loc[i]=past.mean(axis=0)
            sigma[i]=np.maximum(past.std(axis=0,ddof=1), 1e-4)
    keep=np.isfinite(loc).all(axis=1)
    d=d.iloc[np.where(keep)[0]]; R=R[keep]; X=X[keep];loc=loc[keep];sigma=sigma[keep];N=len(d)
    trn=int(N*tr); val=int(N*va)
    assert 80<=trn<val<N and (val-trn)>=15 and (N-val)>=20, (N,trn,val)
    train=np.arange(trn);valid=np.arange(trn,val);test=np.arange(val,N)
    muX=X[train].mean(axis=0);sdX=X[train].std(axis=0);sdX=np.where(sdX<1e-7,1.,sdX)
    X=np.clip((X-muX)/sdX,-7,7)
    Z=(R-loc)/sigma
    pars=[];U=np.zeros_like(Z)
    for j in range(Z.shape[1]):
        # The truncated t fit must never use validation or test observations.
        df, tloc, ts=stats.t.fit(Z[train,j],floc=0.0)
        df=float(np.clip(df,3.0,60.0));ts=float(max(ts,0.1))
        U[:,j]=np.clip(stats.t.cdf((Z[:,j]-tloc)/ts,df=df),1e-5,1-1e-5)
        pars.append({'asset':rc[j][2:],'df':df,'location':tloc,'scale':ts})
    return d,R,X,U,loc,sigma,pars,train,valid,test,{'feature_columns':xc,'return_columns':rc,'train_start':str(d.index[0].date()),'train_end':str(d.index[trn-1].date()),'validation_start':str(d.index[trn].date()),'validation_end':str(d.index[val-1].date()),'test_start':str(d.index[val].date()),'test_end':str(d.index[-1].date()),'n_train':len(train),'n_validation':len(valid),'n_test':len(test),'marginal_lag_periods':lag_window,'standardization_fit':'train only','features_mean_train':muX.tolist(),'features_std_train':sdX.tolist()}


def evaluate(file,out,max_epochs=150,init_seeds=(31,71,113),risk_draws=4000):
    t0=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    core.set_seed(20261102)
    data,R,X,U,loc,sigma,pars,tr,val,te,desc=prep_observations(file,24,.6,.8)
    cfg=core.Config(d=U.shape[1],hidden=7,emb=2,max_epochs=max_epochs,patience=22,lr=.008,theta_min=.05,theta_max=6.0)
    x=torch.tensor(X,dtype=core.DTYPE);u=torch.tensor(U,dtype=core.DTYPE);J=U.shape[1]-1
    models={};val_results=[]
    for name,model in [('static_mixture',core.StaticMixture(J,cfg)),('state_linear_mixture',core.StateLinearMixture(X.shape[1],J,cfg))]:
        hist,v=core.fit_model(model,x,u,tr,val,cfg,name,max_epochs=max_epochs)
        models[name]=model;hist.to_csv(out/f'training_{name}.csv',index=False)
        val_results.append({'model':name,'val_nll':v})
    # Select neural initialization exclusively by validation likelihood, not the test score.
    selected=None
    for seed in init_seeds:
        core.set_seed(seed)
        model=core.NeuralMixtureVine(X.shape[1],J,cfg)
        hist,v=core.fit_model(model,x,u,tr,val,cfg,'neural',max_epochs=max_epochs)
        hist.to_csv(out/f'training_neural_init{seed}.csv',index=False)
        val_results.append({'model':f'neural_seed_{seed}','val_nll':v})
        if selected is None or v<selected[0]:selected=(v,seed,model)
    models['neural_mixture']=selected[2]
    scores={}
    with torch.no_grad():
        for k,model in models.items():
            model.eval()
            # run causal GRU full sequence for valid state updates, scoring only test.
            thL,thU,om=model(x)
            ss=core.mixture_log_density_torch(u[:,0],u[:,1:],thL,thU,om).sum(1).cpu().numpy()
            scores[k]=ss
            if k=='neural_mixture':
                pd.DataFrame({'Date':data.index,'lower_tail_root_edge1':(om[:,0]*2**(-1/thL[:,0])).numpy(),'upper_tail_root_edge1':((1-om[:,0])*2**(-1/thU[:,0])).numpy()}).to_csv(out/'estimated_tail_path.csv',index=False)
    Rg=core.gaussian_corr(U[tr]);scores['gaussian_static']=core.gaussian_copula_ll(U,Rg)
    best=(-np.inf,None,None)
    for nu in [4.,5.,6.,8.,12.,20.,40.]:
        xtr=stats.t.ppf(U[tr],df=nu)
        Rc=np.corrcoef(xtr,rowvar=False); Rc=.98*Rc+.02*np.eye(len(Rc)); Rc=(Rc+Rc.T)/2
        try:
            lv=core.t_copula_ll(U[val],Rc,nu).mean()
            if lv>best[0]: best=(lv,nu,Rc)
        except (ValueError, np.linalg.LinAlgError):continue
    assert best[1] is not None
    scores['student_t_static']=core.t_copula_ll(U,best[2],best[1])
    # Dynamic Gaussian: EWMA updated through time with historical scores only.
    scores['gaussian_ewma']=core.ewma_gaussian_ll(U,len(tr),lam=.96)
    for k,v in scores.items():
        if not np.isfinite(v[te]).all():raise ValueError(f'non-finite test scores for {k}')
    frame=pd.DataFrame({'Date':data.index})
    for k,v in scores.items():frame[k]=v
    frame.to_csv(out/'all_period_scores.csv',index=False)
    test_frame=frame.iloc[te].copy();test_frame.to_csv(out/'test_scores.csv',index=False)
    avgs=pd.DataFrame([{'model':k,'mean_test_log_score':float(np.mean(v[te])),'test_log_score_sd':float(np.std(v[te],ddof=1)),'n_test':len(te)} for k,v in scores.items()]).sort_values('mean_test_log_score',ascending=False)
    avgs.to_csv(out/'test_model_comparison.csv',index=False)
    # Newey-West standard errors on test-date paired differences, sensitivity to one lag choice.
    rows=[]
    for k,v in scores.items():
        if k=='neural_mixture':continue
        diff=np.asarray(scores['neural_mixture'][te]-v[te]);n=len(diff);center=diff-diff.mean();lag=min(3,n-1)
        s=float(center@center/n)
        for a in range(1,lag+1):s+=2*(1-a/(lag+1))*float(center[a:]@center[:-a]/n)
        se=np.sqrt(max(s,0.0)/n)
        z=diff.mean()/se if se>0 else np.nan
        p=2*stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append({'benchmark':k,'mean_difference':float(diff.mean()),'hac_se_lag3':float(se),'ci95_lower':float(diff.mean()-1.96*se),'ci95_upper':float(diff.mean()+1.96*se),'p_approx':float(p),'periods_won':int((diff>0).sum()),'n_test':n})
    comp=pd.DataFrame(rows);comp.to_csv(out/'paired_hac_comparisons.csv',index=False)
    # Feasible one-period-ahead equal-weight portfolio risk forecasts from estimated
    # marginal laws. These small samples are exploratory; no backtest approval claimed.
    if risk_draws:
        risk_rows=[]; rng=np.random.default_rng(891011)
        ps={}
        with torch.no_grad():
            for k,model in models.items():
                pp=model(x);ps[k]=tuple(z.detach().numpy() for z in pp)
        for t in te:
            actual_loss=-float(R[t].mean())
            for k in ['neural_mixture','state_linear_mixture','static_mixture','gaussian_static','student_t_static','gaussian_ewma']:
                if k in ps:
                    pp=ps[k]
                    UU=core.sample_star_snapshot(risk_draws,pp[0][t],pp[1][t],pp[2][t],rng)
                elif k=='gaussian_static': UU=core.sample_gaussian(risk_draws,Rg,rng)
                elif k=='student_t_static': UU=core.sample_tcopula(risk_draws,best[2],best[1],rng)
                else:
                    # EWMA covariance updated from strictly earlier PITs (no return_t).
                    zz=stats.norm.ppf(core.clip_u(U[:t])); cov=np.cov(zz[:len(tr)].T)
                    # Only PITs observed before the forecast date may update covariance.
                    for v1 in zz[len(tr):t]: cov=.96*cov+.04*np.outer(v1,v1)
                    ss0=np.sqrt(np.diag(cov)); corr=cov/np.outer(ss0,ss0)
                    corr=.99*corr+.01*np.eye(len(corr))
                    UU=core.sample_gaussian(risk_draws,corr,rng)
                S=np.empty_like(UU)
                for j,par in enumerate(pars):
                    S[:,j]=loc[t,j]+sigma[t,j]*(par['location']+par['scale']*stats.t.ppf(UU[:,j],df=par['df']))
                simulated_loss=-S.mean(axis=1)
                var=float(np.quantile(simulated_loss,.95))
                es=float(simulated_loss[simulated_loss>=var].mean())
                # Quantile loss for loss outcome: (alpha-I[y<=q])*(y-q)
                score=(.95-(actual_loss<=var))*(actual_loss-var)
                risk_rows.append({'Date':data.index[t],'model':k,'loss_realized':actual_loss,'VaR95':var,'ES95':es,'breach':int(actual_loss>var),'pinball95':float(score)})
        risk=pd.DataFrame(risk_rows);risk.to_csv(out/'risk_forecasts_95.csv',index=False)
        risk.groupby('model').agg(n=('breach','size'),exceedances=('breach','sum'),mean_VaR=('VaR95','mean'),mean_ES=('ES95','mean'),mean_pinball=('pinball95','mean')).reset_index().to_csv(out/'risk_backtest_summary.csv',index=False)
    pd.DataFrame(pars).to_csv(out/'train_marginal_fits.csv',index=False)
    pd.DataFrame(val_results).to_csv(out/'validation_model_selection.csv',index=False)
    desc.update({'dataset':str(file),'neural_selected_seed':int(selected[1]),'val_selected_nll':float(selected[0]),'copula_model':'first tree Clayton/survival-Clayton mixture','t_copula_selected_df':best[1],'note':'pilot evaluation with a single chronological test period; results must not be called statistically robust','elapsed_seconds':round(time.time()-t0,2)})
    (out/'experiment.json').write_text(json.dumps(desc,indent=2))
    fig,ax=plt.subplots(figsize=(8.0,4.0))
    test_frame.set_index('Date')[['neural_mixture','state_linear_mixture','gaussian_static','student_t_static']].cumsum().plot(ax=ax)
    ax.set_title('Historical test period: cumulative copula log scores');ax.set_ylabel('Cumulative log score');ax.set_xlabel('Date');ax.grid(alpha=.22)
    fig.tight_layout();fig.savefig(out/'cumulative_oos_scores.pdf');fig.savefig(out/'cumulative_oos_scores.png',dpi=160);plt.close(fig)
    print('REAL-MARKET EXPERIMENT:')
    print(json.dumps({'test_start':desc['test_start'],'test_end':desc['test_end'],'n_test':len(te),'selected_neural_seed':int(selected[1])},indent=2))
    print(avgs.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(comp[['benchmark','mean_difference','ci95_lower','ci95_upper','p_approx']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return avgs,comp,desc

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--out',required=True);a.add_argument('--epochs',type=int,default=150);a.add_argument('--neural-seeds',default='31,71,113');a.add_argument('--risk-draws',type=int,default=4000);args=a.parse_args()
    evaluate(args.input,args.out,args.epochs,[int(z) for z in args.neural_seeds.split(',')],risk_draws=args.risk_draws)

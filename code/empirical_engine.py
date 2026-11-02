#!/usr/bin/env python3
"""
Enriched empirical engine for a macro-conditioned neural tail-asymmetric copula model.

The core experiment is inherited from the accompanying `core_engine.py`: a regime-switching
10-dimensional truncated C-vine whose market-root edges have dynamic Clayton/survival-Clayton
mixtures learned by a GRU.  This enrichment layer adds three diagnostics motivated by the three
recent reference papers supplied with the project, without copying their models:

1. A GARCH-MIDAS/EVT-style marginal diagnostic and Christoffersen conditional-coverage test
   (inspired by Yao, Chen & Chen, 2025).
2. A Clayton-Gaussian-rotated-Clayton (CGRC) static benchmark plus random-forest permutation
   importance for tail-dependence drivers (inspired by Conlon, Cotter & Ropotos, 2026).
3. A macro-conditioned Cholesky ridge Gaussian copula benchmark and likelihood-sensitivity feature analysis
   (inspired by Huh, Ha & Jeong, 2025).

Results depend on code version, package versions, and seed. New runs are saved independently from the archived manuscript outputs.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import torch
from torch import nn
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core_engine as core


def bivar_gaussian_density(u, v, rho):
    u = core.clip_u(np.asarray(u)); v = core.clip_u(np.asarray(v))
    rho = np.clip(rho, -0.97, 0.97)
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    den = math.sqrt(1-rho*rho)
    expo = -0.5/(1-rho*rho)*(x*x - 2*rho*x*y + y*y) + 0.5*(x*x+y*y)
    return np.exp(expo)/den


def cgrc_negll(par, u, v):
    # unconstrained parameters -> simplex weights, positive Clayton parameters, bounded rho
    logits = par[:3]
    ex = np.exp(logits - np.max(logits)); w = ex/ex.sum()
    th_l = 0.02 + np.log1p(np.exp(par[3]))
    rho = 0.97*np.tanh(par[4])
    th_u = 0.02 + np.log1p(np.exp(par[5]))
    c_l = core.clayton_density_np(u,v,th_l)
    c_g = bivar_gaussian_density(u,v,rho)
    c_u = core.clayton_density_np(1-u,1-v,th_u)
    dens = np.maximum(w[0]*c_l+w[1]*c_g+w[2]*c_u,1e-300)
    return -np.log(dens).mean()


def fit_cgrc_root_vine(U, train_idx, val_idx, test_idx):
    root=U[:,0]; J=U.shape[1]-1
    rows=[]; test_ll=np.zeros(len(test_idx))
    for j in range(J):
        uv=np.r_[train_idx,val_idx]
        x0=np.array([0.,0.,0.,-0.3,0.,-0.8])
        res=optimize.minimize(cgrc_negll,x0,args=(root[uv],U[uv,j+1]),method='L-BFGS-B',options={'maxiter':120})
        p=res.x; ex=np.exp(p[:3]-np.max(p[:3])); w=ex/ex.sum()
        thl=0.02+np.log1p(np.exp(p[3])); rho=.97*np.tanh(p[4]); thu=0.02+np.log1p(np.exp(p[5]))
        u=root[test_idx]; v=U[test_idx,j+1]
        dens=np.maximum(w[0]*core.clayton_density_np(u,v,thl)+w[1]*bivar_gaussian_density(u,v,rho)+w[2]*core.clayton_density_np(1-u,1-v,thu),1e-300)
        test_ll += np.log(dens)
        rows.append([j+1,*w,thl,rho,thu,w[0]*2**(-1/thl),w[2]*2**(-1/thu),res.success])
    cols=['asset','w_lower','w_gaussian','w_upper','theta_lower','rho_gaussian','theta_upper','lambda_lower','lambda_upper','success']
    return pd.DataFrame(rows,columns=cols), test_ll


class CholeskyGaussianGRU(nn.Module):
    """Recurrent copula benchmark that guarantees a positive-definite correlation matrix by construction."""
    def __init__(self,p,d,hidden=18):
        super().__init__(); self.d=d
        self.gru=nn.GRU(p,hidden,batch_first=True)
        self.head=nn.Linear(hidden,d*(d+1)//2)
        self.register_buffer('tri',torch.tril_indices(d,d))
    def forward(self,x):
        h,_=self.gru(x[None,:,:]); raw=self.head(h[0])
        T=x.shape[0]; L=torch.zeros((T,self.d,self.d),dtype=x.dtype,device=x.device)
        L[:,self.tri[0],self.tri[1]]=raw
        di=torch.arange(self.d,device=x.device)
        L[:,di,di]=torch.nn.functional.softplus(L[:,di,di])+0.08
        S=L@L.transpose(1,2)
        sd=torch.sqrt(torch.diagonal(S,dim1=1,dim2=2).clamp_min(1e-8))
        R=S/(sd[:,:,None]*sd[:,None,:])
        return R


def gaussian_copula_loglik_torch(Z,R):
    d=Z.shape[1]; eye=torch.eye(d,device=Z.device,dtype=Z.dtype)[None,:,:]
    sign,logdet=torch.linalg.slogdet(R)
    inv=torch.linalg.inv(R)
    quad=torch.einsum('ti,tij,tj->t',Z,inv-eye,Z)
    return -0.5*logdet-0.5*quad


def fit_dynamic_cholesky(X,U,train,val,test,seed):
    torch.manual_seed(seed+99)
    xt=torch.tensor(X,dtype=torch.float32); zt=torch.tensor(stats.norm.ppf(core.clip_u(U)),dtype=torch.float32)
    model=CholeskyGaussianGRU(X.shape[1],U.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=4e-3,weight_decay=1e-5)
    best=None; bestv=1e30; wait=0; hist=[]
    for ep in range(12):
        model.train(); opt.zero_grad(); R=model(xt); ll=gaussian_copula_loglik_torch(zt,R); loss=-ll[train].mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        model.eval()
        with torch.no_grad():
            R2=model(xt); ll2=gaussian_copula_loglik_torch(zt,R2); tr=float(-ll2[train].mean()); va=float(-ll2[val].mean())
        hist.append((ep,tr,va))
        if va<bestv-1e-5: bestv=va; best={k:v.detach().clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=5: break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        R=model(xt); ll=gaussian_copula_loglik_torch(zt,R).cpu().numpy()
    return model,pd.DataFrame(hist,columns=['epoch','train_nll','val_nll']),ll,R.detach().cpu().numpy()


def christoffersen_cc(exceed,alpha):
    x=np.asarray(exceed,dtype=int); n=len(x); z=int(x.sum())
    eps=1e-12
    # unconditional coverage likelihood ratio
    phat=np.clip(z/max(n,1),eps,1-eps)
    ll0=(n-z)*math.log(1-alpha+eps)+z*math.log(alpha+eps)
    ll1=(n-z)*math.log(1-phat)+z*math.log(phat)
    lr_uc=-2*(ll0-ll1)
    if n<2:
        return lr_uc,np.nan,np.nan
    n00=n01=n10=n11=0
    for a,b in zip(x[:-1],x[1:]):
        if a==0 and b==0:n00+=1
        elif a==0 and b==1:n01+=1
        elif a==1 and b==0:n10+=1
        else:n11+=1
    p01=n01/max(n00+n01,1); p11=n11/max(n10+n11,1); p=(n01+n11)/max(n-1,1)
    def L(a,b,pv): return a*math.log(max(1-pv,eps))+b*math.log(max(pv,eps))
    lr_ind=-2*(L(n00+n10,n01+n11,p)-L(n00,n01,p01)-L(n10,n11,p11))
    lr_cc=lr_uc+lr_ind
    return lr_uc,lr_ind,1-stats.chi2.cdf(lr_cc,2)


def midas_weights(K,w1=1.4,w2=4.0):
    k=np.arange(1,K+1); x=k/K
    raw=np.power(x,w1-1)*np.power(1-x+1e-9,w2-1); return raw/raw.sum()


def garch_midas_diagnostic(returns,z,regimes,train_end):
    # A diagnostic, not a re-estimation of the DGP: link realized monthly variance to a slowly moving uncertainty index.
    d=returns.shape[1]; days=21; nmon=math.ceil(len(returns)/days)
    # Fit the descriptive MIDAS regression on complete training months only.
    nmon=min(nmon, train_end // days)
    epu=np.array([z[m*days:(m+1)*days,0].mean() for m in range(nmon)])
    rv=np.zeros((nmon,d))
    for m in range(nmon): rv[m]=np.mean(returns[m*days:(m+1)*days]**2,axis=0)
    K=12; phi=midas_weights(K)
    X=[]; Y=[]
    for m in range(K,nmon):
        X.append(np.dot(phi,epu[m-K:m][::-1])); Y.append(np.log(rv[m]+1e-12))
    X=np.asarray(X); Y=np.asarray(Y)
    if len(X) < 3:
        # Twelve monthly lags require a substantially longer training history.
        # Return unavailable diagnostics rather than fitting an unidentified regression.
        return pd.DataFrame({
            'series': np.arange(d), 'midas_intercept': np.nan,
            'uncertainty_slope': np.nan, 'r2': np.nan,
        }), epu, phi
    rows=[]
    for i in range(d):
        b=np.polyfit(X,Y[:,i],1); fitted=b[1]+b[0]*X; r2=1-np.sum((Y[:,i]-fitted)**2)/np.sum((Y[:,i]-Y[:,i].mean())**2)
        rows.append([i,b[1],b[0],r2])
    return pd.DataFrame(rows,columns=['series','midas_intercept','uncertainty_slope','r2']),epu,phi


def fit_evt_diagnostics(returns,sigma,train_end):
    Z=returns[:train_end]/np.maximum(sigma[:train_end],1e-12)
    rows=[]
    for i in range(Z.shape[1]):
        zi=Z[:,i]; ql,qu=np.quantile(zi,[.10,.90])
        yl=ql-zi[zi<ql]; yu=zi[zi>qu]-qu
        cl,locl,sl=stats.genpareto.fit(yl,floc=0); cu,locu,su=stats.genpareto.fit(yu,floc=0)
        rows.append([i,ql,cl,sl,qu,cu,su])
    return pd.DataFrame(rows,columns=['series','u_lower','xi_lower','scale_lower','u_upper','xi_upper','scale_upper'])



def fit_macro_cholesky_regression(X,U,train,val,test,window=80,ridge=1e-2):
    """Fast Huh-inspired benchmark: macro features predict rolling Cholesky factors.

    Rolling Gaussian copula correlations are converted to Cholesky coordinates.  Each coordinate
    is regressed on the macro state with ridge regularization; predictions are reconstructed as
    positive-definite correlations, so no ex-post eigenvalue repair is needed.
    """
    Z=stats.norm.ppf(core.clip_u(U)); d=U.shape[1]
    idx=np.r_[train,val]
    tri=np.tril_indices(d)
    Y=[]; XX=[]
    for t in idx[idx>=window]:
        R=np.corrcoef(Z[t-window:t],rowvar=False); R=.99*R+.01*np.eye(d)
        L=np.linalg.cholesky(R)
        Y.append(L[tri]); XX.append(np.r_[1.0,X[t]])
    Y=np.asarray(Y); XX=np.asarray(XX)
    B=np.linalg.solve(XX.T@XX+ridge*np.eye(XX.shape[1]),XX.T@Y)
    ll=np.full(len(U),np.nan); Rpath=np.zeros((len(U),d,d))
    for t in test:
        pred=np.r_[1.0,X[t]]@B; L=np.zeros((d,d)); L[tri]=pred
        di=np.diag_indices(d); L[di]=np.maximum(np.abs(L[di]),.08)
        S=L@L.T; sd=np.sqrt(np.diag(S)); R=S/np.outer(sd,sd); R=.995*R+.005*np.eye(d)
        Rpath[t]=R; ll[t]=core.gaussian_copula_ll(U[t:t+1],R)[0]
    return ll,Rpath,B

def main():
    ap=argparse.ArgumentParser(description='Diagnostics for synthetic neural-copula study')
    ap.add_argument('--out',required=True)
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    ap.add_argument('--reuse-core',action='store_true')
    ap.add_argument('--seed',type=int,default=None)
    ap.add_argument('--risk-draws',type=int,default=20000,
                    help='Simulation draws per VaR/ES origin (minimum 10000)')
    args=ap.parse_args()
    if args.risk_draws < 10000:
        raise ValueError('Use at least 10000 risk draws for 99 percent ES estimation')
    default_seed=json.loads(args.config.read_text()).get('seed',20260926)
    if args.seed is None: args.seed=default_seed
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # Run the audited core experiment unless precomputed deterministic core outputs are supplied.
    if not args.reuse_core:
        subprocess.run([sys.executable,str(HERE/'core_engine.py'),'--out',str(out),'--seed',str(args.seed),'--config',str(args.config)],check=True)
    R=out/'results'; F=out/'figures'
    cfg=json.load(open(R/'config.json')); summary=json.load(open(R/'summary.json'))
    U=pd.read_csv(R/'pseudo_observations.csv').values
    ret=pd.read_csv(R/'simulated_returns.csv').values
    state=pd.read_csv(R/'latent_state_path.csv'); z=state[['volatility_state','credit_state','liquidity_state']].values; regimes=state['regime'].values.astype(int)
    train=np.arange(cfg['train_end']); val=np.arange(cfg['train_end'],cfg['val_end']); test=np.arange(cfg['val_end'],cfg['T'])
    Xraw=core.make_features(z,U); fs=pd.read_csv(R/'feature_standardization.csv'); X=(Xraw-fs['mean'].values)/fs['sd'].values

    # Conlon-style convex-combination benchmark.
    cgrc,ll_cgrc=fit_cgrc_root_vine(U,train,val,test); cgrc.to_csv(R/'cgrc_static_parameters.csv',index=False)
    cgrc_score=float(np.mean(ll_cgrc))

    # Macro-conditioned Cholesky ridge regression. This is NOT the defined GRU.
    ll_chol,Rpath,Bchol=fit_macro_cholesky_regression(X,U,train,val,test)
    chol_score=float(np.mean(ll_chol[test]))
    np.save(R/'dynamic_cholesky_correlation_path.npy',Rpath)
    pd.DataFrame(Bchol).to_csv(R/'dynamic_cholesky_coefficients.csv',index=False)

    # Load the fitted flagship network and calculate its tail functional for interpretability.
    J=cfg['d']-1; cc=core.Config(**cfg); neural=core.NeuralMixtureVine(X.shape[1],J,cc); neural.load_state_dict(torch.load(R/'neural_model_state.pt',map_location='cpu'))
    xt=torch.tensor(X,dtype=core.DTYPE); p=core.params_from_model(neural,xt); lamL=core.lambda_lower(p[0],p[2]); lamU=core.lambda_upper(p[1],p[2]); yL=lamL.mean(axis=1)

    # Random forest driver diagnostic on held-out lower-tail dependence.
    feature_names=['uncertainty','credit_stress','liquidity','lag_market_pit','lag_tail_frequency','lag_abs_market_score']
    rf=RandomForestRegressor(n_estimators=120,min_samples_leaf=8,max_features='sqrt',random_state=args.seed,n_jobs=1)
    rf.fit(X[train],yL[train]); perm=permutation_importance(rf,X[test],yL[test],n_repeats=4,random_state=args.seed,n_jobs=1)
    imp=pd.DataFrame({'feature':feature_names,'permutation_importance':perm.importances_mean,'importance_sd':perm.importances_std}).sort_values('permutation_importance',ascending=False)
    imp.to_csv(R/'rf_tail_driver_importance.csv',index=False)

    # Huh-style likelihood-sensitive ablation at test time. No retraining: isolates local dependence on each input.
    base_ll=core.star_loglik_np(U,*p)
    ab=[]
    for k,name in enumerate(feature_names):
        Xa=X.copy(); Xa[:,k]=0.0; pa=core.params_from_model(neural,torch.tensor(Xa,dtype=core.DTYPE)); lla=core.star_loglik_np(U,*pa)
        ab.append([name,float(base_ll[test].mean()),float(lla[test].mean()),float(base_ll[test].mean()-lla[test].mean())])
    pd.DataFrame(ab,columns=['feature','full_log_score','zeroed_feature_log_score','score_drop']).sort_values('score_drop',ascending=False).to_csv(R/'likelihood_feature_ablation.csv',index=False)

    # Yao-style slow-moving macro/EVT diagnostics.
    # Reconstruct the conditional sigma used in the core simulator.
    dfs=pd.read_csv(R/'marginal_parameters.csv')['df'].values; base_vol=pd.read_csv(R/'marginal_parameters.csv')['base_vol'].values
    reg_mult=np.array([0.88,1.08,1.48])[regimes]; state_mult=np.exp(.055*z[:,0]+.030*z[:,1]-.018*z[:,2]); sigma=reg_mult[:,None]*state_mult[:,None]*base_vol[None,:]
    midas,epu,phi=garch_midas_diagnostic(ret,z,regimes,cfg['train_end']); midas.to_csv(R/'midas_diagnostics.csv',index=False)
    evt=fit_evt_diagnostics(ret,sigma,cfg['train_end']); evt.to_csv(R/'evt_tail_diagnostics.csv',index=False)
    pd.DataFrame({'lag':np.arange(1,len(phi)+1),'beta_weight':phi}).to_csv(R/'midas_beta_weights.csv',index=False)

    # Oracle-margin demonstration only: sigma uses the simulator's hidden regime.
    # The higher Monte Carlo count reduces numerical tail noise but does not make
    # these forecasts available to an investor without the true volatility law.
    rng=np.random.default_rng(args.seed+700); realized_loss=-ret[test,1:].mean(axis=1)
    var=[]; es=[]
    for t in test:
        Us=core.sample_star_snapshot(args.risk_draws,p[0][t],p[1][t],p[2][t],rng)
        Xs=np.empty_like(Us)
        for i in range(cfg['d']):
            nu=dfs[i]; scale=math.sqrt((nu-2)/nu); Xs[:,i]=sigma[t,i]*scale*stats.t.ppf(Us[:,i],df=nu)
        loss=-Xs[:,1:].mean(axis=1); q=np.quantile(loss,.99); var.append(q); es.append(loss[loss>=q].mean())
    var=np.asarray(var); es=np.asarray(es); exc=realized_loss>var
    lruc,lrind,pcc=christoffersen_cc(exc,.01)
    back=pd.DataFrame([['Neural mixture C-vine (oracle margins)',.01,int(exc.sum()),len(exc),float(exc.mean()),lruc,lrind,pcc,float(np.mean(var)),float(np.mean(es))]],columns=['model','alpha','exceedances','n','exceedance_rate','LR_uc','LR_ind','CC_pvalue','mean_VaR','mean_ES'])
    back['draws_per_forecast']=args.risk_draws
    back['margins']='true_simulator_volatility_not_feasible_forecast'
    back.to_csv(R/'var_es_backtest.csv',index=False)
    pd.DataFrame({'time':test,'realized_loss':realized_loss,'VaR_99':var,'ES_99':es,'exceedance':exc.astype(int)}).to_csv(R/'var_es_path.csv',index=False)

    # Add benchmark scores to a unified table.
    old=pd.read_csv(R/'oos_log_scores.csv')
    extra=pd.DataFrame([
        ['Static CGRC root-vine',cgrc_score,np.nan,np.nan],
        ['Macro-conditioned Cholesky ridge Gaussian copula',chol_score,np.nan,np.nan],
    ],columns=old.columns)
    unified=pd.concat([old,extra],ignore_index=True).sort_values('mean_log_score',ascending=False); unified.to_csv(R/'oos_log_scores_enriched.csv',index=False)

    # Figures for the new diagnostics.
    plt.figure(figsize=(6.8,3.6)); ii=imp.sort_values('permutation_importance'); plt.barh(ii.feature,ii.permutation_importance); plt.xlabel('permutation importance'); plt.tight_layout(); plt.savefig(F/'driver_importance.pdf'); plt.savefig(F/'driver_importance.png',dpi=180); plt.close()
    abl=pd.read_csv(R/'likelihood_feature_ablation.csv').sort_values('score_drop'); plt.figure(figsize=(6.8,3.6)); plt.barh(abl.feature,abl.score_drop); plt.xlabel('drop in held-out copula log score'); plt.tight_layout(); plt.savefig(F/'feature_ablation.pdf'); plt.savefig(F/'feature_ablation.png',dpi=180); plt.close()
    plt.figure(figsize=(6.8,3.2)); plt.plot(test,var,label='99% VaR'); plt.plot(test,realized_loss,label='realized loss',alpha=.65); plt.xlabel('test index'); plt.ylabel('loss'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(F/'var_backtest.pdf'); plt.savefig(F/'var_backtest.png',dpi=180); plt.close()

    summary.update({'cgrc_static_test_log_score':cgrc_score,'dynamic_cholesky_test_log_score':chol_score,'christoffersen_cc_pvalue':float(pcc),'var_exceedance_rate':float(exc.mean()),'rf_top_feature':str(imp.iloc[0].feature),'rf_top_importance':float(imp.iloc[0].permutation_importance)})
    with open(R/'summary_enriched.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
    print(unified.to_string(index=False))

if __name__=='__main__': main()

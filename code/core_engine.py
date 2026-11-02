#!/usr/bin/env python3
"""
Advanced empirical engine for a regime-switching neural mixture-C-vine copula.

The model is a truncated C-vine with a market root U_0. Each root-asset link is a
state-dependent convex mixture of a lower-tail Clayton copula and its survival
(rotation) counterpart. A GRU with asset embeddings maps observable macro state
and lagged market diagnostics to compactly constrained copula parameters.

Outputs are deterministic under the configured seed and are written to an output
folder for direct ingestion by the LaTeX manuscript builder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn
import matplotlib.pyplot as plt

DTYPE = torch.float32
EPS = 1e-9

@dataclass
class Config:
    seed: int = 20260926
    T: int = 1800
    d: int = 10
    train_end: int = 1200
    val_end: int = 1450
    theta_min: float = 0.05
    theta_max: float = 8.0
    omega_eps: float = 0.02
    hidden: int = 14
    emb: int = 4
    lr: float = 0.015
    max_epochs: int = 220
    patience: int = 30
    risk_mc: int = 10000
    matrix_mc: int = 12000
    q_tail: float = 0.05


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def softplus_np(x):
    return np.logaddexp(0.0, x)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_u(u):
    return np.clip(u, 1e-8, 1 - 1e-8)


def clayton_cdf(u, v, theta):
    u, v = clip_u(u), clip_u(v)
    return np.maximum(u**(-theta) + v**(-theta) - 1.0, 1e-300)**(-1.0 / theta)


def clayton_log_density_np(u, v, theta):
    """Evaluate the Clayton log density without exponentiating large tail powers."""
    u, v = clip_u(u), clip_u(v)
    theta = np.asarray(theta)
    a = np.logaddexp(-theta * np.log(u), -theta * np.log(v))
    log_base = a + np.log1p(-np.exp(-a))
    return (np.log1p(theta) - (theta + 1.0) * (np.log(u) + np.log(v))
            - (2.0 + 1.0 / theta) * log_base)


def clayton_density_np(u, v, theta):
    """Return a density for legacy callers; likelihoods use log densities directly."""
    return np.exp(clayton_log_density_np(u, v, theta))


def clayton_h_inv(p, u, theta):
    """Inverse conditional CDF v -> P(V<=v | U=u) for Clayton."""
    p = clip_u(p); u = clip_u(u)
    a = (p * u**(theta + 1.0)) ** (-theta / (theta + 1.0))
    inner = np.maximum(a - u**(-theta) + 1.0, 1e-300)
    return clip_u(inner ** (-1.0 / theta))


def survival_clayton_h_inv(p, u, theta):
    return clip_u(1.0 - clayton_h_inv(1.0 - p, 1.0 - u, theta))


def lambda_lower(theta_l, omega):
    return omega * 2.0 ** (-1.0 / np.maximum(theta_l, 1e-12))


def lambda_upper(theta_u, omega):
    return (1.0 - omega) * 2.0 ** (-1.0 / np.maximum(theta_u, 1e-12))


def stationary_dist(P):
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi /= pi.sum()
    return pi


def simulate_regimes(T, P, seed):
    rng = np.random.default_rng(seed)
    pi = stationary_dist(P)
    s = np.zeros(T, dtype=int)
    s[0] = rng.choice(len(pi), p=pi)
    for t in range(1, T):
        s[t] = rng.choice(P.shape[0], p=P[s[t-1]])
    return s, pi


def simulate_states(T, regimes, seed):
    rng = np.random.default_rng(seed)
    A = np.array([[0.82, 0.06, 0.00],
                  [0.03, 0.86, 0.04],
                  [0.00, 0.06, 0.78]])
    mus = np.array([
        [-0.55, -0.35,  0.20],
        [ 0.05,  0.10,  0.05],
        [ 0.95,  0.75, -0.45],
    ])
    covs = np.array([
        [[0.08,0.02,0.00],[0.02,0.07,-0.01],[0.00,-0.01,0.06]],
        [[0.14,0.04,0.00],[0.04,0.12,-0.02],[0.00,-0.02,0.10]],
        [[0.28,0.10,-0.03],[0.10,0.24,-0.05],[-0.03,-0.05,0.16]],
    ])
    z = np.zeros((T,3))
    z[0] = mus[regimes[0]] + rng.multivariate_normal(np.zeros(3), covs[regimes[0]])
    for t in range(1,T):
        r = regimes[t]
        target = mus[r]
        z[t] = target + A @ (z[t-1] - target) + rng.multivariate_normal(np.zeros(3), covs[r])
    return z, A, mus, covs


def true_params(z, regimes, J, theta_min=0.05, theta_max=8.0, omega_eps=0.02):
    T = len(z)
    load = np.linspace(-0.65, 0.85, J)
    sector = np.repeat(np.arange(3), math.ceil(J/3))[:J]
    sec_eff = np.array([-0.20, 0.08, 0.26])[sector]
    regL = np.array([-0.25, 0.35, 1.25])[regimes][:,None]
    regU = np.array([ 0.35, 0.08,-0.18])[regimes][:,None]
    regW = np.array([-0.75, 0.10, 1.35])[regimes][:,None]
    zz0,zz1,zz2 = z[:,0,None],z[:,1,None],z[:,2,None]
    aa = load[None,:]
    rawL = (-0.35 + 0.55*aa + sec_eff[None,:] + 0.62*zz0 + 0.33*zz1 - 0.20*zz2
            + regL + 0.22*np.sin(1.15*zz0 + 0.7*aa))
    rawU = (-1.05 + 0.18*aa - 0.10*sec_eff[None,:] - 0.12*zz0 + 0.16*zz2
            + regU + 0.14*np.cos(0.8*zz1 - aa))
    rawW = (0.18 + 0.35*aa + 0.20*sec_eff[None,:] + 0.70*zz0 + 0.36*zz1 -0.10*zz2
            + regW + 0.12*np.tanh(zz0*zz1))
    thL = theta_min + (theta_max-theta_min)*sigmoid_np(rawL)/3.2  # safely interior
    thU = theta_min + (theta_max-theta_min)*sigmoid_np(rawU)/4.2
    om = omega_eps + (1-2*omega_eps)*sigmoid_np(rawW)
    return thL, thU, om, load, sector


def sample_star_vine(params: Tuple[np.ndarray,np.ndarray,np.ndarray], rng):
    thL, thU, om = params
    # params shapes T x J
    T,J = thL.shape
    U = np.empty((T,J+1))
    root = rng.uniform(size=T)
    U[:,0]=root
    p = rng.uniform(size=(T,J))
    chooseL = rng.uniform(size=(T,J)) < om
    for j in range(J):
        vL = clayton_h_inv(p[:,j], root, thL[:,j])
        vU = survival_clayton_h_inv(p[:,j], root, thU[:,j])
        U[:,j+1] = np.where(chooseL[:,j], vL, vU)
    return clip_u(U)


def simulate_returns(U, z, regimes, seed):
    rng = np.random.default_rng(seed)
    T,d = U.shape
    dfs = np.linspace(5.2, 10.5, d)
    base_vol = np.linspace(0.010, 0.019, d)
    base_vol[0] = 0.012
    reg_mult = np.array([0.88,1.08,1.48])[regimes]
    state_mult = np.exp(0.055*z[:,0] + 0.030*z[:,1] - 0.018*z[:,2])
    sigma = reg_mult[:,None]*state_mult[:,None]*base_vol[None,:]
    X = np.empty_like(U)
    for i in range(d):
        nu=dfs[i]
        scale = math.sqrt((nu-2.0)/nu)
        X[:,i] = sigma[:,i] * scale * stats.t.ppf(U[:,i], df=nu)
    # tiny predictable-free drift for numerical realism, not used in copula fit
    X += rng.normal(0,1e-5,size=X.shape)
    return X, dfs, base_vol, sigma


def make_features(z, U):
    T = len(z)
    # standardize macro state using training prefix later externally; raw here
    lag_root = np.r_[0.5, U[:-1,0]]
    lag_tail = np.r_[0.0, (U[:-1,1:] < 0.08).mean(axis=1)]
    lag_abs = np.r_[0.0, np.abs(stats.norm.ppf(clip_u(U[:-1,0])))]
    return np.column_stack([z, lag_root-0.5, lag_tail, lag_abs])


def clayton_log_density_torch(u, v, theta):
    u = torch.clamp(u, 1e-7, 1-1e-7)
    v = torch.clamp(v, 1e-7, 1-1e-7)
    logu, logv = torch.log(u), torch.log(v)
    # log a = log(exp(-theta log u)+exp(-theta log v)-1), direct is stable enough in bounded theta
    a = torch.exp(-theta*logu) + torch.exp(-theta*logv) - 1.0
    a = torch.clamp(a, min=1e-12)
    return torch.log1p(theta) + (-1.0-theta)*(logu+logv) + (-2.0-1.0/theta)*torch.log(a)


def mixture_log_density_torch(root, assets, thL, thU, om):
    # root T, assets T x J
    r = root[:,None].expand_as(assets)
    logL = clayton_log_density_torch(r, assets, thL)
    logU = clayton_log_density_torch(1-r, 1-assets, thU)
    logw = torch.log(torch.clamp(om,1e-8,1-1e-8))
    log1w = torch.log(torch.clamp(1-om,1e-8,1-1e-8))
    return torch.logsumexp(torch.stack([logw+logL, log1w+logU],dim=-1), dim=-1)


class ParamTransform(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__(); self.cfg=cfg
    def forward(self, raw):
        thL = self.cfg.theta_min + (self.cfg.theta_max-self.cfg.theta_min)*torch.sigmoid(raw[...,0])
        thU = self.cfg.theta_min + (self.cfg.theta_max-self.cfg.theta_min)*torch.sigmoid(raw[...,1])
        om = self.cfg.omega_eps + (1-2*self.cfg.omega_eps)*torch.sigmoid(raw[...,2])
        return thL,thU,om

class StaticMixture(nn.Module):
    def __init__(self, J,cfg):
        super().__init__(); self.raw=nn.Parameter(torch.zeros(J,3,dtype=DTYPE)); self.tr=ParamTransform(cfg)
        with torch.no_grad(): self.raw[:,0]= -1.0; self.raw[:,1]=-1.5; self.raw[:,2]=0.0
    def forward(self, x):
        T=x.shape[0]; raw=self.raw[None,:,:].expand(T,-1,-1); return self.tr(raw)

class StateLinearMixture(nn.Module):
    def __init__(self, p,J,cfg):
        super().__init__(); self.intercept=nn.Parameter(torch.zeros(J,3,dtype=DTYPE)); self.beta=nn.Parameter(torch.zeros(J,3,p,dtype=DTYPE)); self.tr=ParamTransform(cfg)
        with torch.no_grad(): self.intercept[:,0]=-1.0; self.intercept[:,1]=-1.5
    def forward(self,x):
        raw = self.intercept[None,:,:] + torch.einsum('tp,jkp->tjk',x,self.beta)
        return self.tr(raw)

class NeuralMixtureVine(nn.Module):
    def __init__(self,p,J,cfg):
        super().__init__(); self.cfg=cfg; self.J=J
        self.gru=nn.GRU(p,cfg.hidden,batch_first=True)
        self.emb=nn.Embedding(J,cfg.emb)
        self.mlp=nn.Sequential(nn.Linear(cfg.hidden+cfg.emb,24),nn.Tanh(),nn.Linear(24,16),nn.Tanh(),nn.Linear(16,3))
        self.tr=ParamTransform(cfg)
        
    def forward(self,x):
        h,_=self.gru(x[None,:,:]); h=h[0]
        e=self.emb(torch.arange(self.J,device=x.device))
        hh=h[:,None,:].expand(-1,self.J,-1); ee=e[None,:,:].expand(h.shape[0],-1,-1)
        raw=self.mlp(torch.cat([hh,ee],dim=-1))
        return self.tr(raw)


def fit_model(model, x, U, train_idx, val_idx, cfg, name, max_epochs=None, lr=None):
    max_epochs=max_epochs or cfg.max_epochs; lr=lr or cfg.lr
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=18,min_lr=8e-5)
    root=U[:,0]; assets=U[:,1:]
    best=None; best_val=float('inf'); patience=0; history=[]
    for epoch in range(max_epochs):
        model.train(); opt.zero_grad()
        thL,thU,om=model(x)
        ll=mixture_log_density_torch(root,assets,thL,thU,om).sum(dim=1)
        loss=-ll[train_idx].mean()
        # mild temporal smoothness only for non-static sequence models
        if len(train_idx)>1 and name=='neural':
            # Penalize variation only inside the training window. Using the entire
            # sequence would let held-out covariates affect the training objective.
            a,b,c=thL[train_idx],thU[train_idx],om[train_idx]
            smooth=((a[1:]-a[:-1])**2).mean()+((b[1:]-b[:-1])**2).mean()+0.5*((c[1:]-c[:-1])**2).mean()
            loss=loss+2e-4*smooth
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad():
            thL2,thU2,om2=model(x); ll2=mixture_log_density_torch(root,assets,thL2,thU2,om2).sum(dim=1)
            val=float((-ll2[val_idx].mean()).cpu())
            tr=float((-ll2[train_idx].mean()).cpu())
        scheduler.step(val); history.append((epoch,tr,val,opt.param_groups[0]['lr']))
        if val < best_val - 1e-5:
            best_val=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
        if patience>=cfg.patience: break
    if best is not None: model.load_state_dict(best)
    return pd.DataFrame(history,columns=['epoch','train_nll','val_nll','lr']),best_val


def star_loglik_np(U,thL,thU,om):
    root=U[:,0,None]; assets=U[:,1:]
    ll_lower=clayton_log_density_np(root,assets,thL)
    ll_upper=clayton_log_density_np(1-root,1-assets,thU)
    log_mix=np.logaddexp(np.log(om)+ll_lower, np.log1p(-om)+ll_upper)
    return log_mix.sum(axis=1)


def gaussian_corr(U):
    Z=stats.norm.ppf(clip_u(U)); R=np.corrcoef(Z,rowvar=False)
    R=0.985*R+0.015*np.eye(R.shape[0]);
    # ensure PD
    vals,vecs=np.linalg.eigh(R); vals=np.maximum(vals,1e-5); R=vecs@np.diag(vals)@vecs.T
    s=np.sqrt(np.diag(R)); R=R/np.outer(s,s)
    return R


def gaussian_copula_ll(U,R):
    Z=stats.norm.ppf(clip_u(U)); d=U.shape[1]
    sign,logdet=np.linalg.slogdet(R); inv=np.linalg.inv(R)
    quad=np.einsum('ti,ij,tj->t',Z,inv-np.eye(d),Z)
    return -0.5*logdet-0.5*quad


def t_copula_ll(U,R,nu):
    X=stats.t.ppf(clip_u(U),df=nu)
    mv=stats.multivariate_t(loc=np.zeros(U.shape[1]),shape=R,df=nu)
    return mv.logpdf(X)-stats.t.logpdf(X,df=nu).sum(axis=1)


def select_t_copula(U, train, val):
    best=None
    for nu in [4.0,5.0,6.0,8.0,12.0,20.0,40.0]:
        X=stats.t.ppf(clip_u(U[train]),df=nu); R=np.corrcoef(X,rowvar=False); R=.985*R+.015*np.eye(R.shape[0])
        ll=t_copula_ll(U[val],R,nu).mean()
        if best is None or ll>best[0]: best=(ll,nu,R)
    nu=best[1]; idx=np.r_[train,val]
    X=stats.t.ppf(clip_u(U[idx]),df=nu); R=np.corrcoef(X,rowvar=False); R=.985*R+.015*np.eye(R.shape[0])
    return nu,R


def ewma_gaussian_ll(U, train_end, lam=0.97):
    Z=stats.norm.ppf(clip_u(U)); d=U.shape[1]
    S=np.cov(Z[:train_end].T); ll=np.full(len(U),np.nan)
    for t in range(train_end,len(U)):
        if t > train_end:
            zprev=Z[t-1][:,None]
            S=lam*S+(1.0-lam)*(zprev@zprev.T)
        sd=np.sqrt(np.diag(S)); R=S/np.outer(sd,sd)
        R=.99*R+.01*np.eye(d)
        ll[t]=gaussian_copula_ll(U[t:t+1],R)[0]
    return ll


def params_from_model(model,x):
    model.eval();
    with torch.no_grad():
        a,b,c=model(x)
    return a.cpu().numpy(),b.cpu().numpy(),c.cpu().numpy()


def metric_rows(name,true, pred, mask):
    rows=[]
    for kind, tarr, parr in [('lambda_L',true[0],pred[0]),('lambda_U',true[1],pred[1])]:
        a=tarr[mask].ravel(); b=parr[mask].ravel()
        rmse=np.sqrt(np.mean((a-b)**2)); mae=np.mean(np.abs(a-b)); corr=np.corrcoef(a,b)[0,1] if np.std(b)>1e-12 else np.nan
        rows.append([name,kind,rmse,mae,corr,a.mean(),b.mean()])
    return rows


def auc_rank(y,score):
    y=np.asarray(y).astype(int); score=np.asarray(score)
    pos=score[y==1]; neg=score[y==0]
    if len(pos)==0 or len(neg)==0:return np.nan
    ranks=stats.rankdata(np.r_[pos,neg]); n1=len(pos); n0=len(neg)
    return (ranks[:n1].sum()-n1*(n1+1)/2)/(n1*n0)


def sample_star_snapshot(n,thL,thU,om,rng):
    J=len(thL); root=rng.uniform(size=n); U=np.empty((n,J+1)); U[:,0]=root
    for j in range(J):
        p=rng.uniform(size=n); choose=rng.uniform(size=n)<om[j]
        a=clayton_h_inv(p,root,np.full(n,thL[j])); b=survival_clayton_h_inv(p,root,np.full(n,thU[j]))
        U[:,j+1]=np.where(choose,a,b)
    return clip_u(U)


def sample_gaussian(n,R,rng):
    z=rng.multivariate_normal(np.zeros(R.shape[0]),R,size=n); return clip_u(stats.norm.cdf(z))


def sample_tcopula(n,R,nu,rng):
    # multiv t = normal / sqrt(chi2/nu)
    z=rng.multivariate_normal(np.zeros(R.shape[0]),R,size=n); g=rng.chisquare(nu,size=n)
    x=z/np.sqrt(g[:,None]/nu); return clip_u(stats.t.cdf(x,df=nu))


def risk_from_U(U, sigma_row, dfs, q=0.99):
    n,d=U.shape; X=np.empty_like(U)
    for i in range(d):
        nu=dfs[i]; scale=math.sqrt((nu-2)/nu); X[:,i]=sigma_row[i]*scale*stats.t.ppf(U[:,i],df=nu)
    port=X[:,1:].mean(axis=1); loss=-port
    var=np.quantile(loss,q); es=loss[loss>=var].mean()
    crash=(U[:,1:]<0.05).sum(axis=1)>=math.ceil((d-1)/3)
    return float(var),float(es),float(crash.mean()),float(port.std())


def cocrasch_matrix(U,q=0.05):
    I=(U<q).astype(float); return (I.T@I)/len(U)/q


def write_matrix_csv(path,M,prefix='A'):
    labels=[f'{prefix}{i}' for i in range(M.shape[0])]
    pd.DataFrame(M,index=labels,columns=labels).to_csv(path)


def plot_outputs(out, regimes, test_mask, lam_true, lam_neural, hist, regime_summary, coc_true, coc_neural, cfg):
    figs=out/'figures'; figs.mkdir(exist_ok=True)
    # average lower-tail path
    idx=np.where(test_mask)[0]
    plt.figure(figsize=(8.2,3.8)); plt.plot(idx,lam_true[test_mask].mean(1),label='True'); plt.plot(idx,lam_neural[test_mask].mean(1),label='Neural mixture C-vine',alpha=.9)
    crisis=(regimes[idx]==2); y0,y1=plt.ylim()
    # mark contiguous crisis regions
    start=None
    for k,flag in enumerate(crisis):
        if flag and start is None: start=idx[k]
        if start is not None and (not flag or k==len(crisis)-1):
            end=idx[k] if not flag else idx[k]+1; plt.axvspan(start,end,alpha=.08); start=None
    plt.xlabel('time index'); plt.ylabel('mean lower-tail dependence'); plt.legend(frameon=False); plt.tight_layout();
    plt.savefig(figs/'tail_path.pdf'); plt.savefig(figs/'tail_path.png',dpi=180); plt.close()
    # training
    plt.figure(figsize=(6.6,3.8)); plt.plot(hist.epoch,hist.train_nll,label='train'); plt.plot(hist.epoch,hist.val_nll,label='validation'); plt.xlabel('epoch'); plt.ylabel('negative log copula likelihood'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(figs/'training_curve.pdf'); plt.savefig(figs/'training_curve.png',dpi=180); plt.close()
    # regime tail bars
    piv=regime_summary.pivot(index='regime',columns='source',values='lambda_L_mean')
    piv.plot(kind='bar',figsize=(6.6,3.8)); plt.ylabel('mean lower-tail dependence'); plt.xlabel('regime'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(figs/'regime_tail_dependence.pdf'); plt.savefig(figs/'regime_tail_dependence.png',dpi=180); plt.close()
    # heatmaps separate
    for name,M in [('true_cocrash',coc_true),('neural_cocrash',coc_neural),('cocrash_error',coc_neural-coc_true)]:
        plt.figure(figsize=(5.2,4.6)); im=plt.imshow(M,aspect='auto'); plt.colorbar(im); plt.xlabel('asset index'); plt.ylabel('asset index'); plt.tight_layout(); plt.savefig(figs/f'{name}.pdf'); plt.savefig(figs/f'{name}.png',dpi=180); plt.close()


def main():
    ap=argparse.ArgumentParser(description='Controlled synthetic copula experiment')
    ap.add_argument('--out',default='new_run',help='Fresh output directory; do not overwrite archived results')
    ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    ap.add_argument('--seed',type=int,default=None,help='Override the JSON experiment seed')
    args=ap.parse_args()
    config_values=json.loads(args.config.read_text()) if args.config.exists() else {}
    cfg=Config(**config_values)
    if args.seed is not None: cfg.seed=args.seed
    if not (0 < cfg.train_end < cfg.val_end < cfg.T):
        raise ValueError('Require 0 < train_end < val_end < T')
    set_seed(cfg.seed)
    out=Path(args.out); (out/'results').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    t0=time.time(); rng=np.random.default_rng(cfg.seed)
    P=np.array([[0.965,0.030,0.005],[0.045,0.915,0.040],[0.012,0.078,0.910]])
    regimes,pi=simulate_regimes(cfg.T,P,cfg.seed+1)
    z,A,mus,covs=simulate_states(cfg.T,regimes,cfg.seed+2)
    J=cfg.d-1
    thL0,thU0,om0,load,sector=true_params(z,regimes,J,cfg.theta_min,cfg.theta_max,cfg.omega_eps)
    U=sample_star_vine((thL0,thU0,om0),np.random.default_rng(cfg.seed+3))
    returns,dfs,base_vol,sigma=simulate_returns(U,z,regimes,cfg.seed+4)
    Xraw=make_features(z,U)
    # standardize features using train only
    muX=Xraw[:cfg.train_end].mean(0); sdX=Xraw[:cfg.train_end].std(0); sdX[sdX<1e-8]=1
    X=(Xraw-muX)/sdX
    train=np.arange(0,cfg.train_end); val=np.arange(cfg.train_end,cfg.val_end); test=np.arange(cfg.val_end,cfg.T)
    train_mask=np.zeros(cfg.T,bool); train_mask[train]=1; val_mask=np.zeros(cfg.T,bool); val_mask[val]=1; test_mask=np.zeros(cfg.T,bool); test_mask[test]=1
    xt=torch.tensor(X,dtype=DTYPE); ut=torch.tensor(U,dtype=DTYPE)

    # Fit nested benchmark set
    static=StaticMixture(J,cfg); h_static,_=fit_model(static,xt,ut,train,val,cfg,'static',max_epochs=140,lr=.03)
    linear=StateLinearMixture(X.shape[1],J,cfg); h_linear,_=fit_model(linear,xt,ut,train,val,cfg,'linear',max_epochs=170,lr=.022)
    neural=NeuralMixtureVine(X.shape[1],J,cfg); h_neural,_=fit_model(neural,xt,ut,train,val,cfg,'neural')

    p_static=params_from_model(static,xt); p_linear=params_from_model(linear,xt); p_neural=params_from_model(neural,xt)
    lam0=(lambda_lower(thL0,om0),lambda_upper(thU0,om0))
    lam_static=(lambda_lower(p_static[0],p_static[2]),lambda_upper(p_static[1],p_static[2]))
    lam_linear=(lambda_lower(p_linear[0],p_linear[2]),lambda_upper(p_linear[1],p_linear[2]))
    lam_neural=(lambda_lower(p_neural[0],p_neural[2]),lambda_upper(p_neural[1],p_neural[2]))

    # log scores
    ll_true=star_loglik_np(U,thL0,thU0,om0); ll_static=star_loglik_np(U,*p_static); ll_linear=star_loglik_np(U,*p_linear); ll_neural=star_loglik_np(U,*p_neural)
    Rg=gaussian_corr(U[np.r_[train,val]]); ll_gauss=gaussian_copula_ll(U,Rg)
    nu_t,Rt=select_t_copula(U,train,val); ll_t=t_copula_ll(U,Rt,nu_t)
    ll_ewma=ewma_gaussian_ll(U,cfg.val_end,lam=.97)

    scores=[]
    for name,ll in [('Oracle true copula',ll_true),('Neural mixture C-vine',ll_neural),('State-linear mixture C-vine',ll_linear),('Static mixture C-vine',ll_static),('Static Student-t copula',ll_t),('Static Gaussian copula',ll_gauss),('EWMA Gaussian copula',ll_ewma)]:
        arr=ll[test]; scores.append([name,np.nanmean(arr),np.nanstd(arr)/math.sqrt(np.sum(np.isfinite(arr))),np.nanmean(arr<0)])
    scores_df=pd.DataFrame(scores,columns=['model','mean_log_score','se_log_score','fraction_negative_log_score']).sort_values('mean_log_score',ascending=False)
    scores_df.to_csv(out/'results'/'oos_log_scores.csv',index=False)

    # tail metrics
    rows=[]
    for name,p in [('Neural mixture C-vine',lam_neural),('State-linear mixture C-vine',lam_linear),('Static mixture C-vine',lam_static)]: rows+=metric_rows(name,lam0,p,test_mask)
    tailmetrics=pd.DataFrame(rows,columns=['model','tail','rmse','mae','correlation','true_mean','pred_mean']); tailmetrics.to_csv(out/'results'/'tail_metrics.csv',index=False)

    # regime summary on test
    rrows=[]
    for r,rname in enumerate(['calm','transition','crisis']):
        mask=test_mask & (regimes==r)
        for src,laL,laU in [('true',lam0[0],lam0[1]),('neural',lam_neural[0],lam_neural[1]),('state-linear',lam_linear[0],lam_linear[1]),('static',lam_static[0],lam_static[1])]:
            rrows.append([rname,src,int(mask.sum()),laL[mask].mean(),laU[mask].mean(),(laL[mask]-laU[mask]).mean()])
    regime_summary=pd.DataFrame(rrows,columns=['regime','source','n','lambda_L_mean','lambda_U_mean','asymmetry_mean']); regime_summary.to_csv(out/'results'/'regime_tail_summary.csv',index=False)

    # crisis detection with mean lower tail
    y=(regimes[test]==2).astype(int); auc=auc_rank(y,lam_neural[0][test].mean(1)); threshold=np.quantile(lam_neural[0][train].mean(1),.80); pred=(lam_neural[0][test].mean(1)>threshold).astype(int)
    precision=(pred[y==1].sum()/max(pred.sum(),1)); recall=(pred[y==1].sum()/max(y.sum(),1))

    # per-asset recovery
    arows=[]
    for j in range(J):
        a=lam0[0][test,j]; b=lam_neural[0][test,j]; au=lam0[1][test,j]; bu=lam_neural[1][test,j]
        arows.append([j+1,int(sector[j]),load[j],np.sqrt(np.mean((a-b)**2)),np.corrcoef(a,b)[0,1],np.sqrt(np.mean((au-bu)**2)),np.corrcoef(au,bu)[0,1],a.mean(),b.mean()])
    pd.DataFrame(arows,columns=['asset','sector','loading','rmse_lambda_L','corr_lambda_L','rmse_lambda_U','corr_lambda_U','mean_true_lambda_L','mean_pred_lambda_L']).to_csv(out/'results'/'asset_recovery.csv',index=False)

    # representative regime stress tests
    risk_rows=[]; rep_times={}
    for r,rname in enumerate(['calm','transition','crisis']):
        candidates=test[regimes[test]==r]
        if len(candidates)==0: continue
        stress=z[candidates,0]+z[candidates,1]-0.5*z[candidates,2]; tt=int(candidates[np.argmin(np.abs(stress-np.median(stress)))]); rep_times[rname]=tt
        models={
            'Oracle true copula':(thL0[tt],thU0[tt],om0[tt]),
            'Neural mixture C-vine':(p_neural[0][tt],p_neural[1][tt],p_neural[2][tt]),
            'State-linear mixture C-vine':(p_linear[0][tt],p_linear[1][tt],p_linear[2][tt]),
            'Static mixture C-vine':(p_static[0][tt],p_static[1][tt],p_static[2][tt]),
        }
        rr=np.random.default_rng(cfg.seed+100+r)
        for m,pp in models.items():
            Us=sample_star_snapshot(cfg.risk_mc,*pp,rr); va,es,cp,sd=risk_from_U(Us,sigma[tt],dfs); risk_rows.append([rname,tt,m,va,es,cp,sd])
        for m,Us in [('Static Gaussian copula',sample_gaussian(cfg.risk_mc,Rg,rr)),('Static Student-t copula',sample_tcopula(cfg.risk_mc,Rt,nu_t,rr))]:
            va,es,cp,sd=risk_from_U(Us,sigma[tt],dfs); risk_rows.append([rname,tt,m,va,es,cp,sd])
    risk_df=pd.DataFrame(risk_rows,columns=['regime','time','model','VaR_99','ES_99','joint_crash_prob','portfolio_sd']); risk_df.to_csv(out/'results'/'stress_risk_metrics.csv',index=False)

    # crisis co-crash matrices
    tt=rep_times.get('crisis',test[-1]); rr=np.random.default_rng(cfg.seed+222)
    Utrue=sample_star_snapshot(cfg.matrix_mc,thL0[tt],thU0[tt],om0[tt],rr); Uneu=sample_star_snapshot(cfg.matrix_mc,p_neural[0][tt],p_neural[1][tt],p_neural[2][tt],rr)
    Mtrue=cocrasch_matrix(Utrue,cfg.q_tail); Mneu=cocrasch_matrix(Uneu,cfg.q_tail); Mdiff=Mneu-Mtrue
    write_matrix_csv(out/'results'/'cocrash_true_crisis.csv',Mtrue); write_matrix_csv(out/'results'/'cocrash_neural_crisis.csv',Mneu); write_matrix_csv(out/'results'/'cocrash_error_crisis.csv',Mdiff)
    coc_rmse=float(np.sqrt(np.mean((Mtrue-Mneu)**2)))

    # conditional parameter matrix at representative crisis time
    par_rows=[]
    for j in range(J):
        par_rows.append([j+1,thL0[tt,j],p_neural[0][tt,j],thU0[tt,j],p_neural[1][tt,j],om0[tt,j],p_neural[2][tt,j],lam0[0][tt,j],lam_neural[0][tt,j],lam0[1][tt,j],lam_neural[1][tt,j]])
    pd.DataFrame(par_rows,columns=['asset','thetaL_true','thetaL_neural','thetaU_true','thetaU_neural','omega_true','omega_neural','lambdaL_true','lambdaL_neural','lambdaU_true','lambdaU_neural']).to_csv(out/'results'/'crisis_parameter_matrix.csv',index=False)

    # source matrices and data samples
    pd.DataFrame(P,index=['calm','transition','crisis'],columns=['calm','transition','crisis']).to_csv(out/'results'/'regime_transition_matrix.csv')
    pd.DataFrame(A).to_csv(out/'results'/'state_VAR_matrix.csv',index=False,header=False)
    pd.DataFrame(z,columns=['volatility_state','credit_state','liquidity_state']).assign(regime=regimes).to_csv(out/'results'/'latent_state_path.csv',index=False)
    pd.DataFrame(returns,columns=['market']+[f'asset_{j+1}' for j in range(J)]).to_csv(out/'results'/'simulated_returns.csv',index=False)
    pd.DataFrame(U,columns=['market']+[f'asset_{j+1}' for j in range(J)]).to_csv(out/'results'/'pseudo_observations.csv',index=False)
    pd.DataFrame(Rg).to_csv(out/'results'/'gaussian_correlation.csv',index=False,header=False)
    pd.DataFrame(Rt).to_csv(out/'results'/'student_t_correlation.csv',index=False,header=False)
    h_static.to_csv(out/'results'/'training_static.csv',index=False); h_linear.to_csv(out/'results'/'training_state_linear.csv',index=False); h_neural.to_csv(out/'results'/'training_neural.csv',index=False)
    torch.save(neural.state_dict(),out/'results'/'neural_model_state.pt')

    summary={
        'seed':cfg.seed,'T':cfg.T,'d':cfg.d,'train_n':len(train),'validation_n':len(val),'test_n':len(test),
        'stationary_regime_probabilities':pi.tolist(),'student_t_df_baseline':float(nu_t),
        'neural_epochs':int(h_neural.epoch.iloc[-1]+1),'neural_best_val_nll':float(h_neural.val_nll.min()),
        'crisis_detection_auc':float(auc),'crisis_detection_precision_at_train80':float(precision),'crisis_detection_recall_at_train80':float(recall),
        'crisis_cocrash_matrix_rmse':coc_rmse,'representative_times':rep_times,
        'runtime_seconds':float(time.time()-t0),
    }
    # score deltas
    sd=scores_df.set_index('model')['mean_log_score']
    summary['neural_logscore_gain_vs_gaussian']=float(sd['Neural mixture C-vine']-sd['Static Gaussian copula'])
    summary['neural_logscore_gain_vs_t']=float(sd['Neural mixture C-vine']-sd['Static Student-t copula'])
    summary['neural_logscore_gain_vs_state_linear']=float(sd['Neural mixture C-vine']-sd['State-linear mixture C-vine'])
    summary['neural_logscore_efficiency_vs_oracle']=float(sd['Neural mixture C-vine']/sd['Oracle true copula'])
    with open(out/'results'/'summary.json','w') as f: json.dump(summary,f,indent=2)
    with open(out/'results'/'config.json','w') as f: json.dump(asdict(cfg),f,indent=2)
    # save feature stats and marginal params
    pd.DataFrame({'feature':['z_vol','z_credit','z_liquidity','lag_root','lag_tail','lag_abs_root'],'mean':muX,'sd':sdX}).to_csv(out/'results'/'feature_standardization.csv',index=False)
    pd.DataFrame({'series':['market']+[f'asset_{j+1}' for j in range(J)],'df':dfs,'base_vol':base_vol}).to_csv(out/'results'/'marginal_parameters.csv',index=False)

    plot_outputs(out,regimes,test_mask,lam0[0],lam_neural[0],h_neural,regime_summary,Mtrue,Mneu,cfg)
    print(json.dumps(summary,indent=2))
    print(scores_df.to_string(index=False))
    print(tailmetrics.to_string(index=False))

if __name__=='__main__': main()

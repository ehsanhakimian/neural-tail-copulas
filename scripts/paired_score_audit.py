import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D/'code'))
import core_engine as core
R=D/'empirical_results'
import json
cfg=json.loads((R/'config.json').read_text())
U=pd.read_csv(R/'pseudo_observations.csv').to_numpy()
state=pd.read_csv(R/'latent_state_path.csv')
z=state[['volatility_state','credit_state','liquidity_state']].to_numpy()
Xraw=core.make_features(z,U)
fs=pd.read_csv(R/'feature_standardization.csv')
X=(Xraw-fs['mean'].to_numpy())/fs['sd'].to_numpy()
mdl=core.NeuralMixtureVine(X.shape[1],cfg['d']-1,core.Config(**cfg))
mdl.load_state_dict(torch.load(R/'neural_model_state.pt',map_location='cpu',weights_only=True))
p=core.params_from_model(mdl,torch.tensor(X,dtype=core.DTYPE))
idx=np.arange(cfg['val_end'],cfg['T'])
ll_neu=core.star_loglik_np(U,*p)
Rg=pd.read_csv(R/'gaussian_correlation.csv',header=None).to_numpy()
ll_g=core.gaussian_copula_ll(U,Rg)
nu=float(json.loads((R/'summary.json').read_text())['student_t_df_baseline'])
Rt=pd.read_csv(R/'student_t_correlation.csv',header=None).to_numpy()
ll_t=core.t_copula_ll(U,Rt,nu)
params=pd.read_csv(R/'cgrc_static_parameters.csv')
import empirical_engine as e
cc=np.zeros(len(idx))
for _,r in params.iterrows():
    j=int(r.asset)
    dens=r.w_lower*core.clayton_density_np(U[idx,0],U[idx,j],r.theta_lower)+r.w_gaussian*e.bivar_gaussian_density(U[idx,0],U[idx,j],r.rho_gaussian)+r.w_upper*core.clayton_density_np(1-U[idx,0],1-U[idx,j],r.theta_upper)
    cc+=np.log(np.maximum(dens,1e-300))

def hacmean(x,lag=10):
    x=np.asarray(x); n=len(x); m=np.mean(x); y=x-m
    v=np.dot(y,y)/n
    for h in range(1,lag+1):v+=2*(1-h/(lag+1))*np.dot(y[h:],y[:-h])/n
    se=np.sqrt(max(v,0)/n)
    return m,se,m/se if se>0 else np.nan
print('total records',len(U),'test',len(idx)); print('recalc neural',np.mean(ll_neu[idx]),'cgrc',np.mean(cc),'gauss',np.mean(ll_g[idx]),'t',np.mean(ll_t[idx]))
for name,base in [('CGRC',cc),('Gaussian',ll_g[idx]),('Student t',ll_t[idx])]:
    d=ll_neu[idx]-base; m,se,t=hacmean(d)
    print(name,'delta',round(m,6),'HAC10SE',round(se,6),'t',round(t,3),'CI95',(round(m-1.96*se,5),round(m+1.96*se,5)))
print('CHECK variance of marg PIT unconditional',U.mean(axis=0).round(3))

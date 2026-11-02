#!/usr/bin/env python3
"""Numerical, information-set and archival checks for the extension."""
from pathlib import Path
import tempfile,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_walkforward import clayton_h,conditional_pits,pair_gaussian_scores,fitted_rho
from merge_alfred_pit import run as merge_vintage

def main():
    rng=np.random.default_rng(4)
    u=rng.uniform(.05,.95,(120,6))
    theta=np.ones((120,5))*.85
    w=np.ones((120,5))*.5
    H=conditional_pits(u,(theta,theta,w))
    assert H.shape==(120,5) and ((H>0)&(H<1)).all()
    assert np.allclose(pair_gaussian_scores(H,np.zeros(4)),0,atol=1e-12)
    rho=fitted_rho(H);assert (np.abs(rho)<=.90).all()
    # Check h_C as a numerical partial derivative of the Clayton CDF.
    x=.42;y=.67;th=.91;eps=1e-6
    c=lambda v,z:(v**(-th)+z**(-th)-1)**(-1/th)
    deriv=(c(x+eps,y)-c(x-eps,y))/(2*eps)
    assert abs(deriv-clayton_h(np.array([x]),np.array([y]),np.array([th]))[0])<1e-8
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);dates=pd.date_range('2022-01-07',periods=2,freq='W-FRI')
        pd.DataFrame({'Date':dates,'r_MKT':[0.,.1],
           **{'x_macro_'+k+'_prior_week':[.1,.2] for k in ['VIXCLS','DGS10','BAA10Y']}}).to_csv(p/'observations.csv',index=False)
        (p/'cache').mkdir()
        pd.DataFrame({'Date':[dates[0]],'as_of_date':['2021-12-31'],
           **{'x_macro_'+k+'_prior_week':[1.] for k in ['VIXCLS','DGS10','BAA10Y']}}).to_csv(p/'cache'/'alfred_batch_0000_0001.csv',index=False)
        try:merge_vintage(p/'observations.csv',p/'cache',p/'merged.csv')
        except RuntimeError as ex:
            assert 'incomplete' in str(ex)
        else:raise AssertionError('Missing ALFRED vintage should fail closed')
    print('PASSED: conditional PIT, Gaussian independence, analytic derivative, ALFRED fail-closed')

if __name__=='__main__':main()

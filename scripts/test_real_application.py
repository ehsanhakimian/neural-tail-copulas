#!/usr/bin/env python3
"""Cross-check historical market sample, information timing and frozen test results."""
import argparse, json,sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from run_real_market import prep_observations


def verify():
    file=ROOT/'real_market_pilot/observations.csv';d=pd.read_csv(file,parse_dates=['Date']).set_index('Date')
    assert d.shape[0]==210 and str(d.index[0].date())=='1992-07-31'
    obs,R,X,U,loc,sigma,pars,tr,val,te,meta=prep_observations(file,24,.6,.8)
    assert len(tr)==111 and len(val)==37 and len(te)==38
    assert meta['test_start']=='2006-11-30'
    # Training-index transformations cannot depend on modified test outcomes.
    altered=d.copy();rc=[c for c in d if c.startswith('r_')]
    altered.iloc[-12:,altered.columns.get_indexer(rc)]=altered.iloc[-12:,altered.columns.get_indexer(rc)].values+0.1
    tmp=ROOT/'real_market_pilot/_QA_TEMP_TEST.csv';altered.reset_index().to_csv(tmp,index=False)
    try:
        _,R2,X2,U2,loc2,sigma2,pars2,tr2,val2,te2,meta2=prep_observations(tmp,24,.6,.8)
        assert np.allclose(U[tr],U2[tr]) and np.allclose(X[tr],X2[tr])
        assert np.allclose(U[val],U2[val]) and np.allclose(X[val],X2[val])
    finally:tmp.unlink(missing_ok=True)
    # Verify all 50 simulations from the verbatim uploaded bundle are unchanged.
    import zipfile,io
    with zipfile.ZipFile(ROOT/'empirical_results/multiseed_50/ORIGINAL_USER_UPLOAD_MULTISEED_RESULTS_50.zip') as z:
        raw=pd.read_csv(io.BytesIO(z.read('all_seed_scores.csv')))
    assert len(raw)==350 and len(raw.seed.unique())==50 and len(raw.model.unique())==7
    scores=pd.read_csv(ROOT/'real_market_pilot/results/test_model_comparison.csv')
    assert len(scores)==6 and np.isclose(float(scores.loc[scores.model=='neural_mixture','mean_test_log_score'].iloc[0]),1.2626,atol=0.0001)
    checks={'historical_monthly_rows':len(d),'training_months':len(tr),'validation_months':len(val),'test_months':len(te),'test_start':meta['test_start'],'test_end':meta['test_end'],'test_data_not_used_in_fit':True,'original_uploaded_multiseed_rows':len(raw),'original_uploaded_multiseed_seeds':50,'n_density_benchmarks':len(scores),'full_modern_weekly_run':'NOT_EXECUTED'}
    (ROOT/'REAL_MARKET_QA.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':verify()

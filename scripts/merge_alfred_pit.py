#!/usr/bin/env python3
"""Fail-closed merge of vintage batches into the ETF input panel.

Only returns a full panel when every requested model date and macro value has
an as-of ALFRED record. Never silently substitute latest-revised FRED values.
"""
import argparse,json
from pathlib import Path
import pandas as pd

COLS=['VIXCLS','DGS10','BAA10Y']

def run(observations,cache,out):
    raw=pd.read_csv(observations,parse_dates=['Date'])
    files=sorted(Path(cache).glob('alfred_batch_*.csv'))
    if not files:raise FileNotFoundError('No vintage batches found; first run fetch_alfred_pit.py')
    a=pd.concat([pd.read_csv(p,parse_dates=['Date']) for p in files],ignore_index=True)
    repeated=a[a.duplicated(['Date'],keep=False)]
    if len(repeated):
        for date,g in repeated.groupby('Date'):
            if not g.fillna('NA').drop(columns=['Date']).eq(g.fillna('NA').drop(columns=['Date']).iloc[0]).all().all():
                raise ValueError('Conflicting vintage data for '+str(date))
    a=a.drop_duplicates(['Date']).sort_values('Date')
    key=['x_macro_'+sid+'_prior_week' for sid in COLS]
    if set(raw.Date)!=set(a.Date):
        missing=sorted(set(raw.Date)-set(a.Date));raise RuntimeError(f'Vintage panel incomplete: {len(missing)} missing weeks, e.g. {missing[:3]}')
    if a[key].isna().any().any():raise RuntimeError('Missing vintage values; do not substitute revised data without disclosure.')
    raw=raw.drop(columns=['x_macro_'+s+'_prior_week' for s in COLS]).merge(a[['Date']+key],on='Date',validate='one_to_one')
    raw.to_csv(out,index=False)
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'ALFRED date-specific vintages','observations':len(raw),'source_batches':len(files),'latest_revised_values_substituted':False},indent=2))
    print('SAVED',out,'with',len(raw),'historically available records')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    x=p.parse_args();run(x.input,x.cache,x.out)

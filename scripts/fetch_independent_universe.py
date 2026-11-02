#!/usr/bin/env python3
"""Frozen external-market panel, prepared without any FRED or original US-sector ETFs.

Universe is EFA (root), EEM, GLD, IEF, TLT, VNQ. The final 20% is a new
cross-sectional holdout, *not* a prospective post-study time period. No model
choice or parameter change should be made after inspecting its performance.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
TICKERS=['EFA','EEM','GLD','IEF','TLT','VNQ']

def build_from_prices(price,out,start='2008-01-01',end='2026-11-01'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    price=price.copy();price.index=pd.to_datetime(price.index).tz_localize(None)
    price=price.sort_index()
    if any(t not in price for t in TICKERS):raise ValueError('Expected tickers '+repr(TICKERS))
    price=price[TICKERS]
    if price.index.has_duplicates:raise ValueError('Duplicate daily dates')
    # Never fill a missing price with a future observation.
    price=price.dropna()
    if not (price>0).all().all():raise ValueError('Nonpositive prices')
    price.index.name='Date';price.to_csv(out/'daily_adjusted_closes.csv')
    weekly=price.resample('W-FRI').last().dropna()
    weekly=weekly[weekly.index<=price.index.max()]
    rets=np.log(weekly).diff().dropna()
    outdata=rets.rename(columns={'EFA':'MKT'}).add_prefix('r_')
    p=rets['EFA'].shift(1)
    outdata['x_lag_mkt_return']=p
    outdata['x_lag_market_abs_4w']=p.abs().rolling(4,min_periods=4).mean()
    outdata['x_lag_market_return_12w']=p.rolling(12,min_periods=12).sum()
    outdata=outdata.dropna().replace([np.inf,-np.inf],np.nan).dropna()
    if len(outdata)<350:raise ValueError('Too few weekly observations: '+str(len(outdata)))
    # Exactly match the first-position-root and lagged-feature contract.
    assert list(outdata.filter(like='r_').columns) == ['r_MKT','r_EEM','r_GLD','r_IEF','r_TLT','r_VNQ']
    raw=outdata.to_csv(index_label='Date',float_format='%.12g')
    (out/'observations.csv').write_text(raw)
    meta={'tickers':TICKERS,'root':'EFA','training_fraction':.60,'validation_fraction':.20,
          'test_fraction':.20,'rolling_window':260,'refit_cadence':26,'max_epochs':18,
          'features':'price only','no_original_sector_ETFs':True,
          'evaluation_rule':'locked before first inspection: evaluate all predefined models once',
          'data_source':'Yahoo Finance adjusted prices (retrospectively adjusted), downloaded in Colab',
          'date_range_observed':[str(outdata.index.min().date()),str(outdata.index.max().date())],
          'n_weeks':len(outdata),'prices_as_of_warning':'Adjusted historic prices are not reconstructible as real-time price vintages',
          'feature_timing':'each x feature depends only on EFA returns observed before forecast week',
          'input_sha256':hashlib.sha256(raw.encode()).hexdigest(),'start_requested':start,'end_exclusive':end,
          'holdout_note':'This is a new asset-universe holdout, NOT an untouched calendar period; do not retune after inspecting it.'}
    (out/'INDEPENDENT_UNIVERSE_PROVENANCE.json').write_text(json.dumps(meta,indent=2))
    print('Frozen external-market panel:',len(outdata),'weeks',meta['date_range_observed'],flush=True)
    return meta

def run(out,start,end):
    try:import yfinance as yf
    except ImportError:raise RuntimeError('Install with: pip install yfinance')
    raw=yf.download(TICKERS,start=start,end=end,auto_adjust=True,group_by='ticker',threads=False,progress=False,interval='1d')
    if raw.empty:raise RuntimeError('Yahoo returned no data; retry after checking access')
    pr=pd.DataFrame(index=pd.to_datetime(raw.index))
    for t in TICKERS:
        if not isinstance(raw.columns,pd.MultiIndex) or (t,'Close') not in raw.columns:
            raise ValueError('Missing adjusted close for '+t+'; downloaded columns='+str(raw.columns[:8]))
        pr[t]=raw[(t,'Close')].to_numpy(dtype=float)
    return build_from_prices(pr,out,start,end)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--start',default='2008-01-01');a.add_argument('--end',default='2026-11-01')
    x=a.parse_args();run(x.out,x.start,x.end)

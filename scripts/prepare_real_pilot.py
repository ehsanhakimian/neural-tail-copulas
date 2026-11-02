#!/usr/bin/env python3
"""Build a historical-data pilot from Matplotlib stock prices and Statsmodels macrodata.

The installed datasets are observations from external vendors, not synthetic samples.
Prices are monthly. Macroeconomic observations are conservatively assumed available
90 days after the end of each reference quarter, then lagged one more month for forecasting.
These ex-post revised macro series are NOT real-time vintages; do not describe the output
as a point-in-time production backtest.
"""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pandas as pd

ASSETS={'MKT':'^GSPC','AAPL':'AAPL','IBM':'IBM','MSFT':'MSFT','ADBE':'ADBE'}

def build(stock_file, macro_file, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(stock_file,comment='#'); s['Date']=pd.to_datetime(s['Date'])
    s=s.dropna(subset=list(ASSETS.values())).copy()
    s=s.loc[(s.Date >= '1990-01-01')&(s.Date <= '2009-12-31')].copy()
    s['Date']=s.Date.dt.to_period('M').dt.to_timestamp('M')
    s=s.drop_duplicates('Date').sort_values('Date').set_index('Date')
    prices=s[list(ASSETS.values())].rename(columns={v:k for k,v in ASSETS.items()})
    r=np.log(prices).diff().dropna()
    # Preserve original adjusted-price data; the source snapshot is revised historically.
    m=pd.read_csv(macro_file)
    q=pd.PeriodIndex.from_fields(year=m.year.astype(int), quarter=m.quarter.astype(int),freq='Q')
    m['reference_quarter_end']=q.to_timestamp(how='end').normalize()
    m['conservative_available_date']=m.reference_quarter_end+pd.Timedelta(days=90)
    m['gdp_yoy']=100*(np.log(m.realgdp)-np.log(m.realgdp.shift(4)))
    m=m[['conservative_available_date','tbilrate','unemp','infl','gdp_yoy']].sort_values('conservative_available_date')
    # Forecast date t uses information as of the previous month-end, no contemporaneous market return.
    state=pd.DataFrame(index=r.index)
    state['lag_mkt_return']=r.MKT.shift(1)
    state['lag_market_abs_return_3m']=r.MKT.abs().shift(1).rolling(3,min_periods=3).mean()
    state['lag_market_return_6m']=r.MKT.shift(1).rolling(6,min_periods=6).sum()
    macro=pd.merge_asof(pd.DataFrame({'Date':r.index,'asof':r.index-pd.offsets.MonthEnd(1)}).sort_values('asof'),m,left_on='asof',right_on='conservative_available_date',direction='backward')
    macro=macro.set_index('Date').reindex(state.index)
    for k in ['tbilrate','unemp','infl','gdp_yoy']:
        state['macro_'+k]=macro[k]
    # Restrict to data after the first 24 monthly returns, because the risk engine fits
    # marginal distributions to 24-month lagged rolling standardized returns.
    combined=pd.concat([r.add_prefix('r_'),state.add_prefix('x_')],axis=1)
    combined=combined.replace([np.inf,-np.inf],np.nan).dropna()
    combined=combined.loc[combined.index >= '1992-07-31']
    combined.index.name='Date'
    combined.to_csv(out/'observations.csv',float_format='%.12g')
    prices.to_csv(out/'source_stock_prices_1990_2009.csv')
    m.to_csv(out/'macro_release_proxy.csv',index=False)
    manifest={'dataset':'historical-market pilot','asset_names':list(ASSETS),'assets':ASSETS,
        'return_definition':'natural logarithm of consecutive monthly sample prices',
        'monthly_dates':'source observations labelled first calendar day interpreted as month-end closes',
        'n':len(combined),'start':str(combined.index.min().date()),'end':str(combined.index.max().date()),
        'source_stock':'Matplotlib mpl-data/sample_data/Stocks.csv (declares Yahoo Finance as source); sample snapshot through 2022-06',
        'source_macro':'Statsmodels datasets/macrodata/macrodata.csv (U.S. quarterly macro observations ending 2009Q3)',
        'macro_availability':'conservative reference-quarter-end plus 90 days, as-of previous month-end; NOT authentic data vintages',
        'qualification':'Revised macro data; no point-in-time vintages; small sample; no contemporary markets.',
        'stock_original_sha256':hashlib.sha256(Path(stock_file).read_bytes()).hexdigest(),
        'macro_original_sha256':hashlib.sha256(Path(macro_file).read_bytes()).hexdigest()}
    (out/'DATA_PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
    print('Pilot prepared',len(combined),'months',manifest['start'],manifest['end'])
    return combined

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stocks',required=True);p.add_argument('--macro',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    build(args.stocks,args.macro,args.out)

#!/usr/bin/env python3
"""Download historically observed US ETF returns and publicly available FRED states.

Requires internet access (Google Colab is recommended). Adjusted ETF closes are
retrieved from Yahoo Finance through yfinance. The FRED CSV endpoint supplies
VIX, 10-year Treasury yields and a corporate bond yield spread. We aggregate
all inputs to Fridays, and shift the macro-financial covariates one whole week
so that each forecast uses an earlier observation. FRED's latest revised records
are NOT point-in-time vintages; this is a research backtest, not a live strategy.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd

TICKERS=['SPY','XLF','XLK','XLE','XLI','XLU']

def fetch(out,start='2005-01-01',end='2026-11-01'):
    try:import yfinance as yf
    except ImportError as e:raise SystemExit('Install yfinance: pip install yfinance') from e
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=yf.download(TICKERS,start=start,end=end,auto_adjust=True,threads=False,progress=False,group_by='ticker',interval='1d')
    if raw.empty:raise RuntimeError('Yahoo Finance returned no prices; check internet, symbol limits, or retry later.')
    price=pd.DataFrame(index=pd.DatetimeIndex(raw.index).tz_localize(None))
    for ticker in TICKERS:
        try:p=raw[(ticker,'Close')] if isinstance(raw.columns,pd.MultiIndex) else raw['Close']
        except KeyError:raise RuntimeError(f'Adjusted close unavailable: {ticker}')
        price[ticker]=p.to_numpy(dtype=float)
    if price.isna().mean().max()>.05:raise RuntimeError('Missing-price fraction exceeds 5%; inspect downloaded symbols.')
    price=price.dropna()
    price.to_csv(out/'daily_adjusted_closes.csv')
    # Completed Friday weeks only, to avoid publishing future-dated week-ends.
    weekly=price.resample('W-FRI').last().dropna()
    weekly=weekly[weekly.index <= price.index.max()]
    rets=np.log(weekly).diff().dropna()
    if len(rets)<300:raise RuntimeError(f'Insufficient weeks: {len(rets)}')
    # Download each FRED series separately and align their dates.
    try:
        frames = []

        for sid in ('VIXCLS', 'DGS10', 'BAA10Y'):

            url = (
                'https://fred.stlouisfed.org/graph/'
                f'fredgraph.csv?id={sid}'
            )

            part = pd.read_csv(url, na_values=['.'])

            date_col = (
                'observation_date'
                if 'observation_date' in part.columns
                else 'DATE'
            )

            if date_col not in part.columns or sid not in part.columns:
                raise ValueError(
                    f'Unexpected FRED columns for {sid}: '
                    f'{list(part.columns)}'
                )

            part[date_col] = pd.to_datetime(
                part[date_col],
                errors='raise'
            )

            frame = (
                part.set_index(date_col)[[sid]]
                .apply(pd.to_numeric, errors='coerce')
            )

            frames.append(frame)

            print(
                f'FRED {sid}: {len(frame)} records downloaded',
                flush=True
            )

        fred = (
            pd.concat(frames, axis=1, join='outer')
            .sort_index()
            .reset_index()
        )

    except Exception as e:
        raise RuntimeError(
            f'FRED per-series download failed: {e}'
        ) from e
    dc='DATE' if 'DATE' in fred else ('observation_date' if 'observation_date' in fred else fred.columns[0])
    fred[dc]=pd.to_datetime(fred[dc]);fred=fred.set_index(dc).sort_index()
    for name in ['VIXCLS','DGS10','BAA10Y']:
        if name not in fred:raise RuntimeError('Missing FRED series: '+name)
        fred[name]=pd.to_numeric(fred[name],errors='coerce')
    fred[['VIXCLS','DGS10','BAA10Y']].to_csv(out/'fred_latest_revised.csv')
    m=fred[['VIXCLS','DGS10','BAA10Y']].resample('W-FRI').last().ffill(limit=2).shift(1)
    # Lag every non-macro predictor as well: returns on Friday t are NOT inputs to forecast t.
    state=pd.DataFrame(index=rets.index)
    past=rets['SPY'].shift(1)
    state['x_lag_mkt_return']=past
    state['x_lag_market_abs_4w']=past.abs().rolling(4,min_periods=4).mean()
    state['x_lag_market_return_12w']=past.rolling(12,min_periods=12).sum()
    state['x_macro_VIX_prior_week']=m.VIXCLS.reindex(state.index)
    state['x_macro_DGS10_prior_week']=m.DGS10.reindex(state.index)
    state['x_macro_BAA10Y_prior_week']=m.BAA10Y.reindex(state.index)
    outdata=pd.concat([rets.rename(columns={'SPY':'MKT'}).add_prefix('r_'),state],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(outdata)<300:raise RuntimeError(f'Insufficient aligned observations: {len(outdata)}')
    outdata.index.name='Date';outdata.to_csv(out/'observations.csv',float_format='%.12g')
    (out/'DATA_PROVENANCE.json').write_text(json.dumps({'dataset':'US ETF/FRED weekly observed-data study','start':start,'end_exclusive':end,
        'tickers':TICKERS,'return_frequency':'weekly Friday log returns from Yahoo adjusted closes','n':len(outdata),
        'first_week':str(outdata.index.min().date()),'last_week':str(outdata.index.max().date()),
        'macro_source':'FRED, VIXCLS DGS10 BAA10Y (latest revised readings)',
        'macro_availability':'one fully completed week lag; no future values included; FRED series are ex-post revisions and NOT real-time vintages',
        'stock_source':'Yahoo Finance adjusted close via yfinance, not separately licensed for redistribution',
        'warning':'Data downloaded in user runtime, not verified by authors of supplied ZIP until results and provenance returned.'},indent=2))
    print(f'Prepared {len(outdata)} weekly observed return/state rows: {outdata.index.min().date()} to {outdata.index.max().date()}')
    return outdata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--start',default='2005-01-01');p.add_argument('--end',default='2026-11-01');a=p.parse_args()
    fetch(a.out,a.start,a.end)

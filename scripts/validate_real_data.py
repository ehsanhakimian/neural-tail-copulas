#!/usr/bin/env python3
"""Validate externally supplied market observations; this is NOT a fitted forecasting model.

Expected returns CSV: date, ROOT, ASSET_1, ...; columns hold arithmetic daily
returns, not prices. Expected macro CSV: observation_date, release_timestamp,
feature_1, ... (UTC ISO timestamps). This script never downloads data.
"""
import argparse
import pandas as pd
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--returns',required=True)
    ap.add_argument('--macro',required=True);args=ap.parse_args()
    ret=pd.read_csv(args.returns);mac=pd.read_csv(args.macro)
    assert 'date' in ret and len(ret.columns)>2
    ret['date']=pd.to_datetime(ret['date'],utc=True,errors='raise')
    assert ret.date.is_unique and ret.date.is_monotonic_increasing
    values=ret.drop(columns='date').apply(pd.to_numeric,errors='raise')
    assert np.isfinite(values.values).all(), 'Missing or nonfinite returns require a documented treatment'
    assert (values.abs()<1.0).all().all(), 'Returns above 100%: check price/return units'
    assert {'observation_date','release_timestamp'}.issubset(mac.columns)
    mac['observation_date']=pd.to_datetime(mac['observation_date'],utc=True,errors='raise')
    mac['release_timestamp']=pd.to_datetime(mac['release_timestamp'],utc=True,errors='raise')
    assert mac.release_timestamp.notna().all()
    assert (mac.release_timestamp>=mac.observation_date).all(), 'Information released before observation date: check semantics'
    print('INPUT VALIDATION PASS; this file does not fit a neural copula or establish any risk results')
    print(f'{len(ret)} trading dates, {values.shape[1]} assets, {len(mac)} macro releases')

if __name__=='__main__': main()

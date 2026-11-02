#!/usr/bin/env python3
"""Retrieve date-specific ALFRED/FRED daily observations without future revisions.

Requires a personal FRED_API_KEY (never write it into the code or a ZIP).
Each query requests information *known on the preceding completed Friday*.
The collected values are not assumed present if a series is missing in that vintage.
Batch execution with a disk cache supports restart after API rate limiting.
"""
import argparse,json,os,time,urllib.parse,urllib.request,urllib.error
from pathlib import Path
import pandas as pd

IDS=['VIXCLS','DGS10','BAA10Y']

def fetch_one(series,cutoff,api_key,cache_dir):
    path=cache_dir/f'{series}_{cutoff}.json'
    if path.exists():
        resp=json.loads(path.read_text())
    else:
        params=dict(series_id=series,api_key=api_key,file_type='json',
            realtime_start=str(cutoff),realtime_end=str(cutoff),
            observation_start=str((pd.Timestamp(cutoff)-pd.Timedelta(days=25)).date()),
            observation_end=str(cutoff),sort_order='desc',limit='50')
        url='https://api.stlouisfed.org/fred/series/observations?'+urllib.parse.urlencode(params)
        for k in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-provenance/1.0'}),timeout=35) as r:
                    resp=json.loads(r.read().decode('utf-8'))
                break
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as ex:
                if k==3:raise RuntimeError(f'ALFRED request failed for {series} at {cutoff}: {type(ex).__name__} (no key included)') from ex
                time.sleep(1.5*(k+1))
        if 'observations' not in resp:raise RuntimeError('ALFRED returned no observations field: '+str(resp.get('error_message','unknown')))
        path.write_text(json.dumps(resp))
        time.sleep(.15)
    candidate=[]
    for x in resp.get('observations',[]):
        if x.get('value') in (None,'.'):continue
        try:v=float(x['value'])
        except (ValueError,TypeError):continue
        date=pd.Timestamp(x['date'])
        if date>pd.Timestamp(cutoff):raise ValueError('ALFRED value uses future date')
        candidate.append((date,v))
    if not candidate:return None,None
    d,v=max(candidate,key=lambda z:z[0])
    if (pd.Timestamp(cutoff)-d).days>14:return None,None
    return v,str(d.date())

def run(observations,out,from_index=0,count=25):
    api_key=os.environ.get('FRED_API_KEY','').strip()
    if len(api_key)!=32 or not api_key.isalnum():
        raise RuntimeError('FRED_API_KEY must contain your own 32-character API key. Do not publish it.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cache=out/'alfred_cache';cache.mkdir(exist_ok=True)
    frame=pd.read_csv(observations,parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    if count<=0:raise ValueError('count must be positive')
    rows=[]
    for _,row in frame.iloc[from_index:from_index+count].iterrows():
        week=pd.Timestamp(row.Date)
        prior=(week-pd.Timedelta(days=7)).date()
        line={'Date':str(week.date()),'as_of_date':str(prior)}
        for sid in IDS:
            v,obs_date=fetch_one(sid,prior,api_key,cache)
            line['x_macro_'+sid+'_prior_week']=v
            line['observation_date_'+sid]=obs_date
        rows.append(line)
        print('Verified historical info for',week.date(),'as of',prior,flush=True)
    path=out/f'alfred_batch_{from_index:04d}_{from_index+len(rows):04d}.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    (out/'ALFRED_METADATA.json').write_text(json.dumps({
        'method':'FRED series/observations with realtime_start=realtime_end=prior week Friday',
        'series':IDS,'api_key_stored':False,
        'api_docs':'https://fred.stlouisfed.org/docs/api/fred/series_observations.html',
        'scope':'date-specific vintage read. Batches must cover entire 1126-week panel before full-vintage training can be claimed.'},indent=2))
    print('SAVED',path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--from-index',type=int,default=0);p.add_argument('--count',type=int,default=25)
    a=p.parse_args();run(a.input,a.out,a.from_index,a.count)

#!/usr/bin/env python3
"""Run independent complete synthetic experiments and summarize held-out scores.

By default uses the same protocol as the paper, changing only the seed.
This code produces evidence only after it has actually completed.
"""
import argparse, subprocess, sys, json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent.parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=HERE/'code'/'config.json')
    ap.add_argument('--out',type=Path,default=HERE/'multiseed_runs')
    ap.add_argument('--seeds',type=int,default=50)
    ap.add_argument('--start-seed',type=int,default=20261001)
    args=ap.parse_args()
    if args.seeds<=0:raise ValueError('--seeds must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    completed=[]
    for n in range(args.seeds):
        seed=args.start_seed+n
        folder=args.out/f'seed_{seed}'
        if not (folder/'results'/'oos_log_scores.csv').exists():
            subprocess.run([sys.executable,str(HERE/'code'/'core_engine.py'),
                            '--config',str(args.config),'--seed',str(seed),
                            '--out',str(folder)],check=True)
        tab=pd.read_csv(folder/'results'/'oos_log_scores.csv')
        for _,row in tab.iterrows():
            completed.append({'seed':seed,'model':row['model'],'mean_log_score':row['mean_log_score']})
        pd.DataFrame(completed).to_csv(args.out/'all_seed_scores.csv',index=False)
    df=pd.DataFrame(completed)
    summary=df.groupby('model')['mean_log_score'].agg(['count','mean','std']).reset_index()
    summary.to_csv(args.out/'summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':main()

import os; os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import numpy as np, pandas as pd, json, generate_dataset as G
from sklearn.metrics import roc_auc_score
nc=pd.Series(G.NIBSS_CH_COUNT); nc=nc/nc.sum()
rows=[]
for s in range(5):
    d=G.generate(8000,0.001,seed=100+s); f=d[d.is_fraud==1]
    cs=f.channel.value_counts(normalize=True).reindex(G.CHANNELS); dev=float((cs-nc.reindex(G.CHANNELS)).abs().max())
    rows.append(dict(seed=100+s,rows=len(d),fraud=len(f),prev=len(f)/len(d),max_chan_dev=dev,
       auc_velocity=roc_auc_score(d.is_fraud,d.velocity_score),auc_sum24=roc_auc_score(d.is_fraud,d.amount_sum_24h),auc_amount=roc_auc_score(d.is_fraud,d.amount),
       cv=float(d.sort_values(['customer_id','timestamp']).groupby('customer_id').timestamp.diff().dt.total_seconds().pipe(lambda x:x[x>0].std()/x[x>0].mean())),
       mean_ibloss=float(f[f.channel=='IB'].amount.mean())))
r=pd.DataFrame(rows); print(r.round(4).to_string()); r.to_json('results/seeds.json',orient='records')
print(r.drop(columns='seed').agg(['mean','std']).round(4).to_string())

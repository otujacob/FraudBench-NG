import os; os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import pandas as pd, numpy as np, json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve
import generate_dataset as G
plt.rcParams.update({'font.size':10,'figure.dpi':200,'axes.spines.top':False,'axes.spines.right':False})
C={'syn':'#2A9D8F','nibss':'#E76F51','legit':'#264653','fraud':'#E9C46A','x':'#8D5FD3'}
d=pd.read_parquet('data/nibss_synthetic_fraud.parquet'); f=d[d.is_fraud==1]; l=d[d.is_fraud==0]
R={}
def save(n): plt.tight_layout(); plt.savefig(f'figures/{n}.png',bbox_inches='tight'); plt.close()
# ---- NIBSS comparisons
chs=G.CHANNELS; nc=pd.Series(G.NIBSS_CH_COUNT); nc=nc/nc.sum(); sc=f[f.is_followon==0].channel.value_counts(normalize=True).reindex(chs)
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); x=np.arange(6)
ax[0].bar(x-.2,nc.reindex(chs)*100,.4,color=C['nibss'],label='NIBSS 2023'); ax[0].bar(x+.2,sc*100,.4,color=C['syn'],label='Synthetic')
ax[0].set_xticks(x); ax[0].set_xticklabels(chs); ax[0].set_ylabel('Share of fraud cases (%)'); ax[0].legend(frameon=False); ax[0].set_title('(a) Channel mix')
lm={c:G.NIBSS_CH_LOSS_M[c]*1e6/G.NIBSS_CH_COUNT[c] for c in chs}; inc_loss=f.groupby('incident_id').agg(ch=('channel','first'),loss=('amount','sum')); sm=inc_loss.groupby('ch').loss.mean().reindex(chs)
ax[1].bar(x-.2,[lm[c]/1e3 for c in chs],.4,color=C['nibss'],label='NIBSS 2023'); ax[1].bar(x+.2,sm/1e3,.4,color=C['syn'],label='Synthetic')
ax[1].set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels(chs); ax[1].set_ylabel('Mean loss per incident (NGN thousand)'); ax[1].set_title('(b) Mean loss per incident'); save('fig_nibss_channel')
R['chi2_channel']=float(stats.chisquare(f.groupby('incident_id').channel.first().value_counts().reindex(chs).values, nc.reindex(chs).values*f.incident_id.nunique())[1])
R['incident_mean_loss']={c:float(sm[c]) for c in chs}; R['nibss_mean_loss']={c:float(lm[c]) for c in chs}; R['row_mean_amount']={c:float(v) for c,v in f.groupby('channel').amount.mean().items()}
R['channel_share_incidents']={c:float(v) for c,v in f.groupby('incident_id').channel.first().value_counts(normalize=True).reindex(chs).items()}
nm=np.array(G.NIBSS_MONTH_COUNT)/sum(G.NIBSS_MONTH_COUNT); smm=f[f.is_followon==0].timestamp.dt.month.value_counts(normalize=True).sort_index()
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); m=np.arange(1,13)
ax[0].plot(m,nm*100,'o-',color=C['nibss'],label='NIBSS 2023'); ax[0].plot(m,smm.values*100,'s-',color=C['syn'],label='Synthetic'); ax[0].set_xticks(m); ax[0].set_xlabel('Month'); ax[0].set_ylabel('Share of fraud cases (%)'); ax[0].legend(frameon=False); ax[0].set_title('(a) Monthly profile')
tn=pd.Series(G.NIBSS_TECH_COUNT); tn=tn/tn.sum(); ts_=f[f.is_followon==0].fraud_technique.value_counts(normalize=True).reindex(G.TECHNIQUES); xx=np.arange(6)
ax[1].bar(xx-.2,tn.reindex(G.TECHNIQUES)*100,.4,color=C['nibss'],label='NIBSS 2023'); ax[1].bar(xx+.2,ts_*100,.4,color=C['syn'],label='Synthetic'); ax[1].set_xticks(xx); ax[1].set_xticklabels([t.replace('_','\n') for t in G.TECHNIQUES],fontsize=7); ax[1].set_ylabel('Share (%)'); ax[1].set_title('(b) Technique mix'); save('fig_nibss_month_technique')
fi=f[f.is_followon==0]; R['chi2_month']=float(stats.chisquare(fi.timestamp.dt.month.value_counts().sort_index().values, nm*len(fi))[1])
R['chi2_technique']=float(stats.chisquare(f[f.is_followon==0].fraud_technique.value_counts().reindex(G.TECHNIQUES).values, tn.reindex(G.TECHNIQUES).values*int((f.is_followon==0).sum()))[1])
# ---- prevalence vs other datasets
fig,ax=plt.subplots(figsize=(5.2,3.2)); names=['NIBSS 2023\n(real, volume)','ULB card\n(real)','PaySim\n(synthetic)','This dataset','IEEE-CIS\n(real)']; v=[0.0008,0.172,0.129,0.1,3.499]
ax.barh(names,v,color=[C['nibss'],C['legit'],C['legit'],C['syn'],C['legit']]); ax.set_xscale('log'); ax.set_xlabel('Fraud prevalence (%)'); ax.invert_yaxis(); save('fig_prevalence')
# ---- amounts
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); bins=np.logspace(2,7.5,70)
ax[0].hist(l.amount,bins,density=True,alpha=.6,color=C['legit'],label='Legitimate'); ax[0].hist(f.amount,bins,density=True,alpha=.7,color=C['fraud'],label='Fraud'); ax[0].set_xscale('log'); ax[0].set_xlabel('Amount (NGN)'); ax[0].set_ylabel('Density'); ax[0].legend(frameon=False); ax[0].set_title('(a) Amount by class')
cm=d.groupby('customer_id').amount.apply(lambda s:np.log(s).mean()); ax[1].hist(cm,60,color=C['x'],alpha=.8); ax[1].set_xlabel('Customer mean log-amount'); ax[1].set_ylabel('Customers'); ax[1].set_title('(b) Spend heterogeneity'); save('fig_amounts')
R['n_incidents']=int(f.incident_id.nunique()); R['rows_per_incident']=float(len(f)/f.incident_id.nunique()); R['followon_share']=float(f.is_followon.mean()); R['atm_withdrawal_share_by_channel']={c:float((d[d.channel==c].merchant_category=='ATM Withdrawal').mean()) for c in chs}
R['ks_amount_fraud_vs_legit']=float(stats.ks_2samp(l.amount.sample(200000,random_state=0),f.amount)[0])
# ---- temporal
fig,ax=plt.subplots(1,3,figsize=(11,3.2)); hh=d.groupby(d.timestamp.dt.hour).size()/len(d)*100
ax[0].bar(hh.index,hh.values,color=C['syn']); ax[0].set_xlabel('Hour of day'); ax[0].set_ylabel('% of transactions'); ax[0].set_title('(a) Diurnal profile')
dd=d.groupby(d.timestamp.dt.dayofweek).size()/len(d)*100; ax[1].bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],dd.values,color=C['x']); ax[1].set_title('(b) Weekly profile'); ax[1].set_ylabel('%')
dl=d.sort_values(['customer_id','timestamp']); dt=dl.groupby('customer_id').timestamp.diff().dt.total_seconds().dropna(); dt=dt[dt>0]
ax[2].hist(np.log10(dt),80,color=C['nibss']); ax[2].set_xlabel('log10 inter-arrival (s)'); ax[2].set_title('(c) Inter-arrival times'); save('fig_temporal')
R['interarrival_cv']=float(dt.std()/dt.mean()); R['hour_night_share']=float(hh.loc[0:5].sum()/100)
# ---- features by class
fig,ax=plt.subplots(1,4,figsize=(12,3)); 
for a,(c,lab) in zip(ax,[('amount_vs_mean_ratio','Amount / running mean'),('velocity_score','Velocity score'),('tx_count_24h','Tx in last 24h'),('channel_diversity','Distinct channels used')]):
    hi=np.quantile(d[c],.995); b=np.linspace(0,hi,40) if c!='channel_diversity' and c!='tx_count_24h' else np.arange(0,min(hi,12)+2)-.5
    a.hist(l[c].clip(upper=hi),b,density=True,alpha=.6,color=C['legit'],label='Legit'); a.hist(f[c].clip(upper=hi),b,density=True,alpha=.7,color=C['fraud'],label='Fraud'); a.set_xlabel(lab); a.set_yscale('log')
ax[0].legend(frameon=False); save('fig_features_by_class')
# ---- univariate AUC
num=[c for c in d.columns if c not in ['transaction_id','customer_id','timestamp','is_fraud','fraud_technique','incident_id','is_followon','channel','bank','merchant_category','location','age_group']]
s=d.sample(600000,random_state=1) if False else d
uni={c:float(roc_auc_score(d.is_fraud,d[c])) for c in num}; R['univariate_auc']=uni
u=pd.Series(uni); u=(u-.5).abs().sort_values()+.5
fig,ax=plt.subplots(figsize=(5.5,4.6)); ax.barh([G.label(c) for c in u.index],u.values,color=C['syn']); ax.axvline(.5,color='k',lw=.8); ax.set_xlim(.45,1); ax.set_xlabel('Single-feature ROC-AUC (folded about 0.5)'); save('fig_univariate_auc')
# ---- correlation
cr=d[num].corr(); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cr,cmap='RdBu_r',vmin=-1,vmax=1); ax.set_xticks(range(len(num))); ax.set_xticklabels([G.label(c) for c in num],rotation=90,fontsize=6); ax.set_yticks(range(len(num))); ax.set_yticklabels([G.label(c) for c in num],fontsize=6); plt.colorbar(im,fraction=.04); save('fig_corr')
# ---- models
M=json.load(open('results/models.json')); te=np.load('results/te_idx.npy'); yt=d.is_fraud.values[te]
fig,ax=plt.subplots(1,2,figsize=(9,3.5)); col={'Logistic':'#E76F51','Random':'#2A9D8F','XGBoost':'#8D5FD3'}
for k,cc in col.items():
    p=np.load(f'results/p_{k}.npy'); pr,rc,_=precision_recall_curve(yt,p); ax[0].plot(rc,pr,color=cc,label=k)
ax[0].axhline(yt.mean(),color='gray',ls='--',label='No skill'); ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].set_yscale('log'); ax[0].legend(frameon=False); ax[0].set_title('(a) Precision-recall (test)')
names=['Logistic regression','Random forest','XGBoost']; x=np.arange(3)
ax[1].bar(x-.2,[M[n]['rec_fpr01'] for n in names],.4,color='#2A9D8F',label='FPR 0.1%'); ax[1].bar(x+.2,[M[n]['rec_fpr1'] for n in names],.4,color='#E9C46A',label='FPR 1%'); ax[1].set_xticks(x); ax[1].set_xticklabels(['LR','RF','XGB']); ax[1].set_ylabel('Recall'); ax[1].legend(frameon=False); ax[1].set_title('(b) Recall at fixed FPR'); save('fig_models')
# ---- tests
# causality brute force (prior-only features) on random customers
full=pd.read_parquet('data/nibss_synthetic_fraud.parquet')
full_b=G.generate(4000,0.001,seed=7,drop_burnin=False)   # includes burn-in so full history is available
ids=np.random.default_rng(0).choice(full_b.customer_id.unique(),40,replace=False); mx=0
for cid in ids:
    g=full_b[full_b.customer_id==cid].sort_values('timestamp'); a=g.amount.values; t=g.timestamp.values.astype('datetime64[ns]').astype('int64')/1e9
    for i in range(1,len(g)):
        pm=a[:i].mean(); w=(t[:i]>=t[i]-86400); w7=(t[:i]>=t[i]-7*86400)
        mx=max(mx,abs(g.amount_vs_mean_ratio.values[i]-a[i]/pm)/(a[i]/pm),abs(g.amount_sum_24h.values[i]-a[:i][w].sum())/max(a[:i][w].sum(),1),
               abs(g.amount_mean_7d.values[i]-(a[:i][w7].mean() if w7.any() else pm))/max(pm,1),abs(g.tx_count_24h.values[i]-w.sum())/max(w.sum(),1))
R['causality_max_rel_err']=float(mx)
tr=full.timestamp<'2023-08-06'; R['median_history_train']=float(full.tx_count_total[tr].median()); R['median_history_test']=float(full.tx_count_total[full.timestamp>='2023-10-18'].median()); R['share_n24_zero']=float((full.tx_count_24h==0).mean()); R['corr_velocity_ratio']=float(full[['velocity_score','amount_vs_mean_ratio']].corr(method='spearman').iloc[0,1]); R['first_tx_share']=float((full.tx_count_total==0).mean())
R['n_rows']=len(d); R['n_fraud']=int(len(f)); R['prevalence']=float(d.is_fraud.mean()); R['n_customers']=int(d.customer_id.nunique())
R['missing_values_numeric']=int(d[num].isna().sum().sum()); R['dup_ids']=int(d.transaction_id.duplicated().sum())
R['fraud_customers_share']=float(f.customer_id.nunique()/d.customer_id.nunique())
# customer heterogeneity eta2 of log amount
la=np.log(l.amount); gm=la.mean(); gs=l.assign(la=la).groupby('customer_id').la.agg(['mean','count']); R['eta2_customer_logamount']=float(((gs['mean']-gm)**2*gs['count']).sum()/((la-gm)**2).sum())
# label permutation
yy=np.random.default_rng(0).permutation(d.is_fraud.values); R['perm_auc']=float(roc_auc_score(yy,d.amount_vs_mean_ratio))
json.dump(R,open('results/validation.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in R.items() if k!='univariate_auc'},indent=1)); print({k:round(v,3) for k,v in uni.items()})

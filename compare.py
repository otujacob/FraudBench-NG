import os; os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import pandas as pd, numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve, roc_auc_score
import xgboost as xgb
d=pd.read_parquet(sys.argv[1] if len(sys.argv)>1 else 'data/nibss_synthetic_fraud.parquet')
CAT=['channel','merchant_category','bank','location','age_group']
DROP=['transaction_id','customer_id','timestamp','is_fraud','fraud_technique']
X=pd.get_dummies(d.drop(columns=DROP),columns=CAT).astype(float); y=d.is_fraud.values
t=d.timestamp.values; q1,q2=np.quantile(t.astype('int64'),[.6,.8]); ti=t.astype('int64')
tr,va,te=ti<q1,(ti>=q1)&(ti<q2),ti>=q2
print('fraud tr/va/te',y[tr].sum(),y[va].sum(),y[te].sum())
rng=np.random.default_rng(0)
li=np.flatnonzero(tr&(y==0)); keep=np.r_[np.flatnonzero(tr&(y==1)),rng.choice(li,300000,replace=False)]
Xtr,ytr=X.values[keep],y[keep]
sc=StandardScaler().fit(Xtr)
def rec_at_fpr(yt,p,f):
    fpr,tpr,_=roc_curve(yt,p); return float(np.interp(f,fpr,tpr))
def best_thr(yt,p):
    pr,rc,th=precision_recall_curve(yt,p); f=2*pr*rc/(pr+rc+1e-12); return th[np.argmax(f[:-1])]
models={'Logistic regression':LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000),
 'Random forest':RandomForestClassifier(300,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=2,random_state=0),
 'XGBoost':xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=.8,colsample_bytree=.8,scale_pos_weight=50,n_jobs=2,random_state=0,eval_metric='aucpr')}
out={}; preds={}
for n,m in models.items():
    if n=='Logistic regression': m.fit(sc.transform(Xtr),ytr); f=lambda A:m.predict_proba(sc.transform(A))[:,1]
    else: m.fit(Xtr,ytr); f=lambda A,m=m:m.predict_proba(A)[:,1]
    pv,pt=f(X.values[va]),f(X.values[te]); preds[n]=pt
    th=best_thr(y[va],pv); yp=pt>=th
    tp=int((yp&(y[te]==1)).sum()); fp=int((yp&(y[te]==0)).sum()); fn=int((~yp&(y[te]==1)).sum())
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    # bootstrap
    yt=y[te]
    bs=[]; yt=y[te]; idx=np.arange(len(yt)); r=np.random.default_rng(1)
    pos=np.flatnonzero(yt==1); neg=np.flatnonzero(yt==0)
    for _ in range(200):
        s=np.r_[r.choice(pos,len(pos)),r.choice(neg,len(neg))]
        bs.append((average_precision_score(yt[s],pt[s]),rec_at_fpr(yt[s],pt[s],.001),rec_at_fpr(yt[s],pt[s],.01)))
    bs=np.array(bs)
    out[n]=dict(pr_auc=average_precision_score(yt,pt),pr_ci=list(np.percentile(bs[:,0],[2.5,97.5])),
      roc_auc=roc_auc_score(yt,pt),rec_fpr01=rec_at_fpr(yt,pt,.001),rec_fpr1=rec_at_fpr(yt,pt,.01),
      prec_top1=float(yt[np.argsort(-pt)[:int(.01*len(pt))]].mean()),rec1_ci=list(np.percentile(bs[:,2],[2.5,97.5])),f1=2*prec*rec/max(prec+rec,1e-12),prec=prec,rec=rec,thr=float(th),tp=tp,fp=fp,fn=fn)
    print(n,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[n].items()})
out['prevalence_test']=float(y[te].mean()); out['n_test']=int(te.sum()); out['n_fraud_test']=int(y[te].sum())
json.dump(out,open('results/models.json','w'),indent=1)
np.save('results/te_idx.npy',np.flatnonzero(te)); 
for n,p in preds.items(): np.save(f'results/p_{n.split()[0]}.npy',p)

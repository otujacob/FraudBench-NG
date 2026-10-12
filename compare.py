"""Benchmark classifiers on a chronological split, with customer-clustered bootstrap intervals and separate
results for first fraud events and follow-on transactions.
usage: python compare.py [data/nibss_synthetic_fraud.parquet] [--with-merchant]
Writes results/models.json, results/te_idx.npy, results/p_*.npy"""
import os; os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import pandas as pd, numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve, roc_auc_score
import xgboost as xgb
import generate_dataset as G
args = [a for a in sys.argv[1:] if not a.startswith('--')]
WITH_M = '--with-merchant' in sys.argv
d = pd.read_parquet(args[0] if args else 'data/nibss_synthetic_fraud.parquet')
X = G.model_matrix(d, include_merchant=WITH_M); y = d.is_fraud.values
fo = d.is_followon.values.astype(bool)
ti = d.timestamp.values.astype('int64'); q1, q2 = np.quantile(ti, [.6, .8])
tr, va, te = ti < q1, (ti >= q1) & (ti < q2), ti >= q2
print('features', X.shape[1], '| fraud tr/va/te', y[tr].sum(), y[va].sum(), y[te].sum())
rng = np.random.default_rng(0)
li = np.flatnonzero(tr & (y == 0)); keep = np.r_[np.flatnonzero(tr & (y == 1)), rng.choice(li, 300000, replace=False)]
Xtr, ytr = X.values[keep], y[keep]
sc = StandardScaler().fit(Xtr)

def rec_at_fpr(yt, p, f, w=None):
    fpr, tpr, _ = roc_curve(yt, p, sample_weight=w); return float(np.interp(f, fpr, tpr))
def best_thr(yt, p):
    pr, rc, th = precision_recall_curve(yt, p); f = 2 * pr * rc / (pr + rc + 1e-12); return th[np.argmax(f[:-1])]

models = {'Logistic regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000),
          'Random forest': RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=2, random_state=0),
          'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=.8, colsample_bytree=.8,
                                       scale_pos_weight=50, n_jobs=2, random_state=0, eval_metric='aucpr')}
yt = y[te]; fot = fo[te]
cust = pd.factorize(d.customer_id.values[te])[0]; ncust = cust.max() + 1
inc = pd.factorize(d.incident_id.values[te][yt == 1])[0]; ninc = inc.max() + 1
B = 200; rb = np.random.default_rng(1)
out = {}; preds = {}
for n, m in models.items():
    if n == 'Logistic regression': m.fit(sc.transform(Xtr), ytr); f = lambda A: m.predict_proba(sc.transform(A))[:, 1]
    else: m.fit(Xtr, ytr); f = lambda A, m=m: m.predict_proba(A)[:, 1]
    pv, pt = f(X.values[va]), f(X.values[te]); preds[n] = pt
    th = best_thr(y[va], pv); yp = pt >= th
    tp = int((yp & (yt == 1)).sum()); fp = int((yp & (yt == 0)).sum()); fn = int((~yp & (yt == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    # customer-clustered bootstrap: resample customers (all their rows move together) via multinomial weights
    bs = []; bi = []
    posidx = np.flatnonzero(yt == 1)
    for _ in range(B):
        w = np.bincount(rb.integers(0, ncust, ncust), minlength=ncust)[cust].astype(float)
        bs.append((average_precision_score(yt, pt, sample_weight=w), rec_at_fpr(yt, pt, .001, w), rec_at_fpr(yt, pt, .01, w)))
        # incident-clustered bootstrap for recall: resample incidents among the positives, negatives fixed
        wi = np.ones(len(yt)); wi[posidx] = np.bincount(rb.integers(0, ninc, ninc), minlength=ninc)[inc]
        bi.append(rec_at_fpr(yt, pt, .01, wi))
    bs = np.array(bs)
    # first events vs follow-ons: positives restricted to one kind, all legitimate rows kept
    sub = {}
    for lab, mask in (('first', ~fot), ('followon', fot | (yt == 0))):
        keepm = (~fot) if lab == 'first' else (fot | (yt == 0))
        sub[lab] = dict(n_pos=int(yt[keepm].sum()), pr_auc=float(average_precision_score(yt[keepm], pt[keepm])),
                        roc_auc=float(roc_auc_score(yt[keepm], pt[keepm])), rec_fpr1=rec_at_fpr(yt[keepm], pt[keepm], .01),
                        base_rate=float(yt[keepm].mean()))
    out[n] = dict(pr_auc=average_precision_score(yt, pt), pr_ci=list(np.percentile(bs[:, 0], [2.5, 97.5])),
                  roc_auc=roc_auc_score(yt, pt), rec_fpr01=rec_at_fpr(yt, pt, .001), rec_fpr1=rec_at_fpr(yt, pt, .01),
                  prec_top1=float(yt[np.argsort(-pt)[:int(.01 * len(pt))]].mean()),
                  rec1_ci=list(np.percentile(bs[:, 2], [2.5, 97.5])), rec1_ci_incident=list(np.percentile(bi, [2.5, 97.5])),
                  rec01_ci=list(np.percentile(bs[:, 1], [2.5, 97.5])),
                  f1=2 * prec * rec / max(prec + rec, 1e-12), prec=prec, rec=rec, thr=float(th), tp=tp, fp=fp, fn=fn, by_event=sub)
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[n].items() if k != 'by_event'}, sub, flush=True)
out['prevalence_test'] = float(yt.mean()); out['n_test'] = int(te.sum()); out['n_fraud_test'] = int(yt.sum())
out['n_followon_test'] = int(fot.sum() if False else (fot & (yt == 1)).sum()); out['n_incidents_test'] = int(ninc); out['n_customers_test'] = int(ncust)
out['features'] = list(X.columns); out['with_merchant'] = WITH_M
suf = '_with_merchant' if WITH_M else ''
json.dump(out, open(f'results/models{suf}.json', 'w'), indent=1)
if not WITH_M:
    np.save('results/te_idx.npy', np.flatnonzero(te))
    for n, p in preds.items(): np.save(f'results/p_{n.split()[0]}.npy', p)

"""Takeover-burst experiments (10 seeds x 8,000 customers each), run in parallel.

1. Cross-assumption transfer: a random forest trained at one continuation probability phi is tested on data
   generated at another (phi = 0, 0.4, 0.7; all 3 x 3 combinations).
2. Why is performance higher at phi = 0 than at phi = 0.4? Ablations at phi = 0.4:
     no_split   every row of an incident carries the full incident loss (switch G.FRAUD_SPLIT = False)
     same_inc   prevalence raised so that the number of incidents equals that at phi = 0
   plus the single-feature ROC-AUC of the relative amount and the amount for first fraud events.
Writes results/phi_experiments.json. usage: python phi_experiments.py  (about 1 hour on 2 cores)"""
import os; os.makedirs('results', exist_ok=True)
import json, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
import generate_dataset as G

SEEDS = list(range(11, 21)); N_CUST = 8000
# name -> (phi, prevalence, split losses)
SETS = {'phi0': (0.0, 0.001, True), 'phi04': (0.4, 0.001, True), 'phi07': (0.7, 0.001, True),
        'phi04_nosplit': (0.4, 0.001, False), 'phi04_sameinc': (0.4, 0.001 / 0.6, True)}
MAIN = ['phi0', 'phi04', 'phi07']

def rec(y, p, f=.01):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(f, fpr, tpr))

def build(name, seed):
    phi, prev, split = SETS[name]
    G.FRAUD_FOLLOW_P = phi; G.FRAUD_SPLIT = split
    d = G.generate(N_CUST, prevalence=prev, seed=seed)
    G.FRAUD_FOLLOW_P = 0.40; G.FRAUD_SPLIT = True
    y = d.is_fraud.values; fo = d.is_followon.values.astype(bool)
    ti = d.timestamp.astype('int64').values; q1, q2 = np.quantile(ti, [.6, .8]); tr, te = ti < q1, ti >= q2
    X = G.model_matrix(d).values.astype(np.float32)
    first = (y == 1) & ~fo; leg = y == 0
    feat = dict(auc_ratio_first=float(roc_auc_score(np.r_[np.ones(first.sum()), np.zeros(leg.sum())],
                                                      np.r_[d.amount_vs_mean_ratio.values[first], d.amount_vs_mean_ratio.values[leg]])),
                auc_amount_first=float(roc_auc_score(np.r_[np.ones(first.sum()), np.zeros(leg.sum())],
                                                       np.r_[d.amount.values[first], d.amount.values[leg]])),
                mean_amount_first=float(d.amount.values[first].mean()), incidents=int(first.sum()), fraud_rows=int(y.sum()),
                followon_share=float(fo[y == 1].mean()))
    rng = np.random.default_rng(0)
    neg = rng.choice(np.flatnonzero(tr & (y == 0)), 150_000, replace=False); idx = np.r_[np.flatnonzero(tr & (y == 1)), neg]
    return dict(Xtr=X[idx], ytr=y[idx], Xte=X[te], yte=y[te].astype(np.int8), fote=fo[te], feat=feat)

def run(seed):
    D = {n: build(n, seed) for n in SETS}
    M = {}
    for n, v in D.items():
        m = RandomForestClassifier(150, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=1, random_state=0)
        M[n] = m.fit(v['Xtr'], v['ytr'])
    cells = {}; preds = {}
    pairs = [(a, b) for a in MAIN for b in MAIN] + [(n, n) for n in SETS if n not in MAIN]
    for a, b in pairs:
        p = M[a].predict_proba(D[b]['Xte'])[:, 1]; yt = D[b]['yte']; fot = D[b]['fote']
        k = ~fot   # first events: positives restricted to first events, all legitimate rows kept
        cells[f'{a}>{b}'] = dict(pr_auc=float(average_precision_score(yt, p)), pr_first=float(average_precision_score(yt[k], p[k])),
                                 rec1=rec(yt, p), base=float(yt.mean()), n_fraud=int(yt.sum()), n_first=int(yt[k].sum()))
        preds[f'{a}>{b}'] = (yt, fot, p.astype(np.float32))
    out = dict(seed=seed, cells=cells, feat={n: D[n]['feat'] for n in SETS})
    print(seed, {k: round(v['pr_auc'], 3) for k, v in cells.items()}, flush=True)
    return out, preds

if __name__ == '__main__':
    with Pool(2) as pool: res = pool.map(run, SEEDS, chunksize=1)
    keys = list(res[0][0]['cells'])
    summ = {}
    for k in keys:
        rows = pd.DataFrame([r[0]['cells'][k] for r in res])
        yy = np.concatenate([r[1][k][0] for r in res]); ff = np.concatenate([r[1][k][1] for r in res]); pp = np.concatenate([r[1][k][2] for r in res])
        kk = ~ff
        summ[k] = dict(pr_auc_mean=float(rows.pr_auc.mean()), pr_auc_sd=float(rows.pr_auc.std()), pr_first_mean=float(rows.pr_first.mean()),
                       pr_first_sd=float(rows.pr_first.std()), rec1_mean=float(rows.rec1.mean()), rec1_sd=float(rows.rec1.std()),
                       pooled_pr_auc=float(average_precision_score(yy, pp)), pooled_pr_first=float(average_precision_score(yy[kk], pp[kk])),
                       pooled_rec1=rec(yy, pp), base=float(rows.base.mean()), n_fraud_total=int(rows.n_fraud.sum()), n_first_total=int(rows.n_first.sum()))
    feat = {}
    for n in SETS:
        f = pd.DataFrame([r[0]['feat'][n] for r in res]); feat[n] = {c: [float(f[c].mean()), float(f[c].std())] for c in f.columns}
    json.dump(dict(summary=summ, features=feat, per_seed=[r[0] for r in res], seeds=SEEDS, n_customers=N_CUST), open('results/phi_experiments.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1)); print(json.dumps(feat, indent=1))

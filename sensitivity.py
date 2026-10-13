"""Sensitivity of the benchmark to the generator's assumptions (10 seeds x 7 settings), run in parallel.
For each setting a random forest is trained on a chronological split with the default features (no merchant
columns) and with the merchant columns. Results are also reported separately for first fraud events and
follow-on transactions, and as pooled (all-seed) PR-AUC. Writes results/sensitivity.json."""
import os; os.makedirs('results', exist_ok=True)
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
import generate_dataset as G

F = G.MERCHANT_FRAUD_FACTOR
order = sorted(F, key=F.get)
VARIANTS = [('assumed', None, {}),
            ('flat', {m: 1.0 for m in F}, {}),                                           # no merchant effect
            ('reversed', {m: F[order[::-1][i]] for i, m in enumerate(order)}, {}),       # opposite ordering
            ('no_followon', None, {'FRAUD_FOLLOW_P': 0.0}), ('followon_0.7', None, {'FRAUD_FOLLOW_P': 0.7}),
            ('fraud_sigma_0.5', None, {'FRAUD_SIGMA': 0.5}), ('fraud_sigma_1.3', None, {'FRAUD_SIGMA': 1.3})]
SEEDS = list(range(11, 21))

def rec(y, p, f=.01):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(f, fpr, tpr))

def run(job):
    name, fac, over, seed = job
    saved = {k: getattr(G, k) for k in over}
    for k, v in over.items(): setattr(G, k, v)
    d = G.generate(8000, seed=seed, merchant_factors=fac)
    for k, v in saved.items(): setattr(G, k, v)
    d['merchant_risk_score'] = d['merchant_risk_score'].fillna(0.5)
    y = d.is_fraud.values; fo = d.is_followon.values.astype(bool)
    ti = d.timestamp.astype('int64').values; q1, q2 = np.quantile(ti, [.6, .8]); tr, te = ti < q1, ti >= q2
    top3 = d[d.is_fraud == 1].merchant_category.isin(['Transfer', 'Electronics', 'Bill Payment']).mean()
    row = dict(variant=name, seed=seed, fraud=int(y.sum()), fraud_test=int(y[te].sum()), followon_share=float(fo[y == 1].mean()),
               merchant_auc=float(roc_auc_score(y, d.merchant_risk_score)), top3_fraud_share=float(top3),
               velocity_auc=float(roc_auc_score(y, d.velocity_score)), amount_auc=float(roc_auc_score(y, d.amount)))
    rng = np.random.default_rng(0)
    neg = rng.choice(np.flatnonzero(tr & (y == 0)), 150_000, replace=False); idx = np.r_[np.flatnonzero(tr & (y == 1)), neg]
    for tag, inc in (('', False), ('_m', True)):
        X = G.model_matrix(d, include_merchant=inc).values
        m = RandomForestClassifier(150, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=1, random_state=0)
        m.fit(X[idx], y[idx]); p = m.predict_proba(X[te])[:, 1]; yt = y[te]; fot = fo[te]
        row['rf_pr_auc' + tag] = float(average_precision_score(yt, p)); row['rf_recall_1pct' + tag] = rec(yt, p)
        if tag == '':
            k1 = ~fot; k2 = fot | (yt == 0)
            row['pr_first'] = float(average_precision_score(yt[k1], p[k1])); row['pr_followon'] = float(average_precision_score(yt[k2], p[k2]))
            row['_y'] = yt.astype(np.int8).tolist() if False else None
            np.save(f'results/_sens_{name}_{seed}_y.npy', yt.astype(np.int8)); np.save(f'results/_sens_{name}_{seed}_p.npy', p.astype(np.float32))
    print(row, flush=True)
    return row

if __name__ == '__main__':
    jobs = [(n, f, o, s) for n, f, o in VARIANTS for s in SEEDS]
    with Pool(2) as pool: rows = pool.map(run, jobs, chunksize=1)
    r = pd.DataFrame(rows).drop(columns='_y')
    # pooled PR-AUC over all seeds of a setting (about 10x the test fraud of a single run)
    pooled = {}
    for n, _, _ in VARIANTS:
        yy = np.concatenate([np.load(f'results/_sens_{n}_{s}_y.npy') for s in SEEDS]); pp = np.concatenate([np.load(f'results/_sens_{n}_{s}_p.npy') for s in SEEDS])
        pooled[n] = dict(pr_auc=float(average_precision_score(yy, pp)), recall_1pct=rec(yy, pp), n_fraud=int(yy.sum()))
    for f in os.listdir('results'):
        if f.startswith('_sens_'): os.remove(os.path.join('results', f))
    r.to_json('results/sensitivity.json', orient='records'); json.dump(pooled, open('results/sensitivity_pooled.json', 'w'), indent=1)
    print(r.drop(columns='seed').groupby('variant').agg(['mean', 'std']).round(4).T.to_string()); print(pooled)

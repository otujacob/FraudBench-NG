"""Alternative customer-population scenarios: the assumed legitimate channel mix is replaced by a mobile-heavy mix
and by the NIBSS fraud channel mix (which removes any difference in channel mix between fraud and legitimate rows).
Age group, location and bank are inert in the generator, so only the channel mix is varied.
10 seeds x 3 scenarios, random forest as in sensitivity.py. Writes results/mix_scenarios.json."""
import os; os.makedirs('results', exist_ok=True)
import json, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
import generate_dataset as G

SEEDS = list(range(11, 21))
fr = pd.Series(G.NIBSS_CH_COUNT)[G.CHANNELS]; fr = (fr / fr.sum()).values
SCEN = {'assumed': G.CH_P.copy(), 'mobile_heavy': np.array([0.03, 0.07, 0.07, 0.60, 0.12, 0.11]), 'nibss_fraud_mix': fr}
def rec(y, p, f=.01):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(f, fpr, tpr))
def run(job):
    name, seed = job
    G.CH_P = SCEN[name]; d = G.generate(8000, seed=seed); G.CH_P = SCEN['assumed']
    y = d.is_fraud.values; fo = d.is_followon.values.astype(bool)
    ti = d.timestamp.astype('int64').values; q1, q2 = np.quantile(ti, [.6, .8]); tr, te = ti < q1, ti >= q2
    Xdf = G.model_matrix(d); X = Xdf.values.astype(np.float32)
    rng = np.random.default_rng(0); idx = np.r_[np.flatnonzero(tr & (y == 1)), rng.choice(np.flatnonzero(tr & (y == 0)), 150_000, replace=False)]
    m = RandomForestClassifier(150, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=1, random_state=0).fit(X[idx], y[idx])
    p = m.predict_proba(X[te])[:, 1]; yt = y[te]; k = ~fo[te]
    imp = pd.Series(m.feature_importances_, index=Xdf.columns); ch_imp = float(imp[[c for c in imp.index if c.startswith('channel_')]].sum())
    row = dict(scenario=name, seed=seed, pr_auc=float(average_precision_score(yt, p)), pr_first=float(average_precision_score(yt[k], p[k])),
               rec1=rec(yt, p), channel_importance=ch_imp, fraud_test=int(yt.sum()))
    print(row, flush=True); return row, (yt, p.astype(np.float32))
if __name__ == '__main__':
    jobs = [(n, s) for n in SCEN for s in SEEDS]
    with Pool(2) as pool: res = pool.map(run, jobs, chunksize=1)
    r = pd.DataFrame([x[0] for x in res]); pooled = {}
    for n in SCEN:
        ii = [i for i, j in enumerate(jobs) if j[0] == n]
        yy = np.concatenate([res[i][1][0] for i in ii]); pp = np.concatenate([res[i][1][1] for i in ii])
        pooled[n] = dict(pr_auc=float(average_precision_score(yy, pp)), rec1=rec(yy, pp), n_fraud=int(yy.sum()))
    g = r.drop(columns='seed').groupby('scenario').agg(['mean', 'std'])
    json.dump(dict(per_seed=r.to_dict('records'), pooled=pooled, mixes={k: list(map(float, v)) for k, v in SCEN.items()}), open('results/mix_scenarios.json', 'w'), indent=1)
    print(g.round(4).T.to_string()); print(pooled)

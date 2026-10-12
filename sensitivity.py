import os; os.makedirs('results', exist_ok=True)
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
import generate_dataset as G

F = G.MERCHANT_FRAUD_FACTOR
order = sorted(F, key=F.get)
MERCH = {
    'assumed': None,                                              # factors used in the released dataset
    'flat': {m: 1.0 for m in F},                                  # no merchant effect at all
    'reversed': {m: F[order[::-1][i]] for i, m in enumerate(order)},  # same strength, opposite ordering
}
# (name, merchant factors, module-constant overrides)
VARIANTS = [(k, v, {}) for k, v in MERCH.items()] + [
    ('no_followon', None, {'FRAUD_FOLLOW_P': 0.0}), ('followon_0.7', None, {'FRAUD_FOLLOW_P': 0.7}),
    ('fraud_sigma_0.5', None, {'FRAUD_SIGMA': 0.5}), ('fraud_sigma_1.3', None, {'FRAUD_SIGMA': 1.3}),
    ('no_bursts', None, {'BURST_PROB': 0.0}), ('bursts_0.5', None, {'BURST_PROB': 0.5})]
CAT = ['channel', 'merchant_category', 'bank', 'location', 'age_group']
DROP = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud', 'fraud_technique']
rows = []
for name, fac, over in VARIANTS:
    for seed in (11, 12, 13):
        saved = {k: getattr(G, k) for k in over}
        for k, v in over.items(): setattr(G, k, v)
        d = G.generate(8000, seed=seed, merchant_factors=fac)
        for k, v in saved.items(): setattr(G, k, v)
        d['merchant_risk_score'] = d['merchant_risk_score'].fillna(0.5)     # flat case: score undefined
        y = d.is_fraud.values
        auc_m = roc_auc_score(y, d.merchant_risk_score)
        top3 = d[d.is_fraud == 1].merchant_category.isin(['Transfer', 'Electronics', 'Bill Payment']).mean()
        X = pd.get_dummies(d.drop(columns=DROP), columns=CAT).astype(float).values
        ti = d.timestamp.astype('int64').values; q1, q2 = np.quantile(ti, [.6, .8])
        tr, te = ti < q1, ti >= q2
        rng = np.random.default_rng(0)
        neg = rng.choice(np.flatnonzero(tr & (y == 0)), 150_000, replace=False)
        idx = np.r_[np.flatnonzero(tr & (y == 1)), neg]
        m = RandomForestClassifier(150, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=0)
        m.fit(X[idx], y[idx]); p = m.predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y[te], p)
        rows.append(dict(variant=name, seed=seed, fraud=int(y.sum()), merchant_auc=auc_m, top3_fraud_share=top3,
                         velocity_auc=roc_auc_score(y, d.velocity_score), amount_auc=roc_auc_score(y, d.amount),
                         rf_pr_auc=average_precision_score(y[te], p), rf_recall_1pct=float(np.interp(.01, fpr, tpr))))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
s = r.groupby('variant').agg(['mean', 'std']).round(4)
print(s.to_string())
r.to_json('results/sensitivity.json', orient='records')

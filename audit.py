"""Audit of an earlier synthetic NIBSS benchmark (the authors' first release) against FraudBench-NG.
usage: python audit.py earlier.csv fraudbench_ng.parquet [intermediate.csv]
Writes results/audit.json and figures/fig_audit_hours.png"""
import os, sys, json
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

old = pd.read_csv(sys.argv[1], low_memory=False); new = pd.read_parquet(sys.argv[2])
old['timestamp'] = pd.to_datetime(old['timestamp'], format='%d/%m/%Y %H:%M')
mid = None
if len(sys.argv) > 3:
    mid = pd.read_csv(sys.argv[3], low_memory=False); mid['timestamp'] = pd.to_datetime(mid['timestamp'])
R = {}
NIBSS_RATIO = 95620 / 11274704942                      # fraud cases / transactions, 2023

def interarrival_cv(d):
    d = d.sort_values(['customer_id', 'timestamp'])
    g = d.groupby('customer_id').timestamp.diff().dt.total_seconds().dropna()
    g = g[g > 0] if (g <= 0).mean() > 0.01 else g
    return float(g.std() / g.mean())

def eta2(d):
    la = np.log1p(d.amount); tot = la.var() * (len(la) - 1)
    within = ((la - la.groupby(d.customer_id).transform('mean')) ** 2).sum()
    return float(1 - within / tot)

DS = [('old', old)] + ([('mid', mid)] if mid is not None else []) + [('new', new)]
for name, d in DS:
    r = {}
    hr = d.timestamp.dt.hour
    r['rows'] = int(len(d)); r['fraud'] = int(d.is_fraud.sum()); r['prevalence'] = float(d.is_fraud.mean())
    r['prevalence_vs_nibss'] = r['prevalence'] / NIBSS_RATIO
    r['share_00_06'] = float((hr < 6).mean())
    r['hour_share_max_min'] = float(hr.value_counts(normalize=True).max() / hr.value_counts(normalize=True).min())
    r['eta2_customer'] = eta2(d)
    r['interarrival_cv'] = interarrival_cv(d.sample(min(len(d), 400000), random_state=0).sort_values('customer_id')
                                           if False else d)
    r['share_tx24_eq1'] = float((d.tx_count_24h == 1).mean()); r['share_tx24_eq0'] = float((d.tx_count_24h == 0).mean())
    r['min_tx24'] = int(d.tx_count_24h.min())
    r['spearman_velocity_ratio'] = float(stats.spearmanr(d.velocity_score.values[:300000], d.amount_vs_mean_ratio.values[:300000])[0])
    r['share_velocity_eq_ratio'] = float(np.isclose(d.velocity_score, d.amount_vs_mean_ratio, rtol=1e-3, atol=1e-6).mean())
    r['share_ratio_eq_1'] = float(np.isclose(d.amount_vs_mean_ratio, 1.0).mean())
    # whole-history leakage: is tx_count_total constant within a customer and equal to the customer's total row count?
    g = d.groupby('customer_id')
    r['share_cust_txcount_constant'] = float((g.tx_count_total.nunique() == 1).mean())
    r['share_cust_txcount_eq_rows'] = float((g.tx_count_total.max() == g.size()).mean())
    r['share_cust_mean_total_constant'] = float((g.amount_mean_total.nunique() == 1).mean())
    first = d.sort_values('timestamp').groupby('customer_id').head(1)
    r['median_prior_tx_at_first_row'] = float(first.tx_count_total.median())
    # merchant score vs realised fraud rate
    mr = d.groupby('merchant_category').agg(score=('merchant_risk_score', 'mean'), rate=('is_fraud', 'mean'))
    r['corr_merchant_score_rate'] = float(mr.score.corr(mr.rate))
    # month effect
    mo = d.groupby(d.timestamp.dt.month).is_fraud.mean()
    r['month_rate_max_min'] = float(mo.max() / mo.min())
    # bank / location association with fraud (chi2)
    for c in ('bank', 'location'):
        ct = pd.crosstab(d[c], d.is_fraud); r[f'chi2_p_{c}'] = float(stats.chi2_contingency(ct)[1])
        cu = d.groupby('customer_id').agg(a=(c, 'first'), f=('is_fraud', 'max'))        # customer-level (rows are clustered)
        r[f'chi2_p_{c}_customer'] = float(stats.chi2_contingency(pd.crosstab(cu.a, cu.f))[1])
    # single feature auc
    feats = [f for f in ('composite_risk', 'velocity_score', 'amount_vs_mean_ratio', 'amount', 'tx_count_total',
                         'amount_std_total', 'amount_mean_total', 'amount_std_7d', 'tx_count_24h', 'merchant_risk_score') if f in d.columns]
    r['auc'] = {f: float(max(roc_auc_score(d.is_fraud, d[f].fillna(0)), 1 - roc_auc_score(d.is_fraud, d[f].fillna(0)))) for f in feats}
    if 'composite_risk' in d.columns:
        X = d[['merchant_risk_score', 'amount_vs_mean_ratio', 'velocity_score']].values
        r['composite_linear_r2'] = float(LinearRegression().fit(X, d.composite_risk).score(X, d.composite_risk))
    R[name] = r

# chronological benchmark with each dataset's own feature set (random forest, class-weighted)
def bench(d, drop):
    y = d.is_fraud.values
    Xd = d.drop(columns=[c for c in drop if c in d.columns])
    X = pd.get_dummies(Xd, columns=[c for c in ('channel', 'merchant_category', 'bank', 'location', 'age_group') if c in Xd.columns]).astype(float)
    ti = d.timestamp.astype('int64').values; q1, q2 = np.quantile(ti, [.6, .8]); tr, te = ti < q1, ti >= q2
    rng = np.random.default_rng(0)
    neg = rng.choice(np.flatnonzero(tr & (y == 0)), 200000, replace=False); idx = np.r_[np.flatnonzero(tr & (y == 1)), neg]
    m = RandomForestClassifier(150, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=-1, random_state=0).fit(X.values[idx], y[idx])
    p = m.predict_proba(X.values[te])[:, 1]
    return dict(roc_auc=float(roc_auc_score(y[te], p)), pr_auc=float(average_precision_score(y[te], p)), base_rate=float(y[te].mean()),
                top_features=list(pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False).head(5).round(3).to_dict().items()))
# merchant columns are left out for every dataset: in a synthetic benchmark they only return the generator's own assumption
BASE = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud', 'fraud_technique', 'incident_id', 'is_followon',
        'merchant_risk_score', 'merchant_category']
R['old']['rf'] = bench(old, BASE)
if mid is not None: R['mid']['rf'] = bench(mid, BASE)
R['new']['rf'] = bench(new, BASE)
# ablation on the earlier benchmark: transaction-level features only (no customer-history aggregates)
HIST = ['tx_count_24h', 'amount_sum_24h', 'amount_mean_7d', 'amount_std_7d', 'tx_count_total', 'amount_mean_total', 'amount_std_total',
        'channel_diversity', 'location_diversity', 'amount_vs_mean_ratio', 'online_channel_ratio', 'velocity_score', 'composite_risk']
R['old']['rf_no_history'] = bench(old, BASE + HIST)
R['new']['rf_no_history'] = bench(new, BASE + [h for h in HIST if h in new.columns])
json.dump(R, open('results/audit.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))

fig, ax = plt.subplots(figsize=(5.5, 3.2)); h = np.arange(24)
for lab, d, c in [('Earlier benchmark', old, '#E76F51')] + ([('Intermediate version', mid, '#8D5FD3')] if mid is not None else []) + [('FraudBench-NG', new, '#2A9D8F')]:
    s = d.timestamp.dt.hour.value_counts(normalize=True).sort_index() * 100; ax.plot(h, s.values, 'o-', ms=3, color=c, label=lab)
ax.axhline(100 / 24, color='grey', ls=':', lw=.8); ax.set(xlabel='hour of day', ylabel='% of transactions'); ax.legend()
plt.tight_layout(); plt.savefig('figures/fig_audit_hours.png', dpi=200); plt.close()

"""TreeSHAP attribution of the Table-4 XGBoost model (same training sample and settings as compare.py).
usage: python shap_compare.py fraudbench.parquet   -> results/shap.json (plotted by make_figures.py)"""
import os, sys, json
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import numpy as np, pandas as pd, shap, xgboost as xgb
import generate_dataset as G
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

new = pd.read_parquet(sys.argv[1])
CAT = ['channel', 'merchant_category', 'bank', 'location', 'age_group']
DROP = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud', 'fraud_technique', 'incident_id', 'is_followon',
        'merchant_risk_score', 'merchant_category']      # merchant columns only return the generator's own assumption
AMOUNT = {'amount', 'amount_log', 'amount_rounded'}
HIST = {'amount_vs_mean_ratio', 'amount_sum_24h', 'amount_mean_7d', 'amount_std_7d', 'tx_count_24h', 'amount_mean_total', 'amount_std_total',
        'tx_count_total', 'channel_diversity', 'location_diversity', 'online_channel_ratio', 'velocity_score', 'composite_risk'}
TIME = {'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'}
def group(col):
    if col in AMOUNT: return 'Amount'
    if col in HIST: return 'Customer history'
    if col in TIME: return 'Time'
    return 'Channel, bank, location, age'

def run(d):
    y = d.is_fraud.values
    X = G.model_matrix(d)                                  # the same feature matrix as compare.py
    ti = d.timestamp.astype('int64').values; q1, q2 = np.quantile(ti, [.6, .8]); tr, te = ti < q1, ti >= q2
    rng = np.random.default_rng(0)
    idx = np.r_[np.flatnonzero(tr & (y == 1)), rng.choice(np.flatnonzero(tr & (y == 0)), 300000, replace=False)]
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8, scale_pos_weight=50,
                          eval_metric='aucpr', n_jobs=2, random_state=0).fit(X.values[idx], y[idx])
    pt = m.predict_proba(X.values[te])[:, 1]
    tei = np.flatnonzero(te)
    samp = np.r_[tei[y[tei] == 1], rng.choice(tei[y[tei] == 0], 4000, replace=False)]      # all test fraud + 4,000 legitimate
    sv = shap.TreeExplainer(m).shap_values(X.values[samp])
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns)
    g = imp.groupby(group).sum(); g = g / g.sum()
    return dict(pr_auc=float(average_precision_score(y[te], pt)), roc_auc=float(roc_auc_score(y[te], pt)), base_rate=float(y[te].mean()),
                group_share=g.round(4).to_dict(), top=imp.sort_values(ascending=False).head(8).round(4).to_dict(),
                n_fraud=int((y[samp] == 1).sum()), n_rows=int(len(samp)))
R = {'fraudbench': run(new)}
json.dump(R, open('results/shap.json', 'w'), indent=1)
print(json.dumps(R, indent=1))

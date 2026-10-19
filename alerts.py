"""What the benchmark results mean for a bank: alert burden at a fixed false-positive rate and how early
each fraud incident is flagged. Uses the test-period scores saved by compare.py.
usage: python alerts.py [data/nibss_synthetic_fraud.parquet]   -> results/alerts.json"""
import json, sys, numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'data/nibss_synthetic_fraud.parquet')
te = np.load('results/te_idx.npy'); t = d.iloc[te].reset_index(drop=True)
y = t.is_fraud.values.astype(bool); n_leg = int((~y).sum())
out = {'n_test': int(len(t)), 'n_legit': n_leg, 'n_fraud': int(y.sum()), 'n_incidents': int(t[y].incident_id.nunique())}
for model in ('Logistic', 'Random', 'XGBoost'):
    p = np.load(f'results/p_{model}.npy'); res = {}
    for fpr in (0.01, 0.001):
        thr = np.quantile(p[~y], 1 - fpr); a = p > thr
        tp = int((a & y).sum()); fp = int((a & ~y).sum())
        r = dict(false_alerts_per_100k_legit=fp / n_leg * 1e5, false_alerts=fp, fraud_rows_flagged=tp, fraud_rows_total=int(y.sum()),
                 false_alerts_per_fraud_row_caught=fp / max(tp, 1), precision=tp / max(tp + fp, 1))
        f = t[y].assign(alert=a[y]).sort_values(['incident_id', 'timestamp'])
        g = f.groupby('incident_id')
        first_hit = g.apply(lambda x: bool(x.alert.iloc[0]), include_groups=False)
        any_hit = g.alert.any()
        later_only = any_hit & ~first_hit
        # fraud amount that occurs after the first alert of an incident (what a real-time block could still stop)
        def stoppable(x):
            if not x.alert.any(): return 0.0
            k = int(np.argmax(x.alert.values)); return float(x.amount.values[k + 1:].sum())
        stop = g.apply(stoppable, include_groups=False).sum(); tot = float(f.amount.sum())
        def rows_before(x):
            return int(np.argmax(x.alert.values)) if x.alert.any() else np.nan
        rb = g.apply(rows_before, include_groups=False)
        r.update(incidents=int(len(any_hit)), incidents_flagged_first_tx=float(first_hit.mean()), incidents_flagged_only_later=float(later_only.mean()),
                 incidents_never_flagged=float((~any_hit).mean()), incidents_flagged_any=float(any_hit.mean()),
                 mean_fraud_rows_before_first_alert=float(np.nanmean(rb)), share_fraud_amount_after_first_alert=stop / tot,
                 multi_row_incidents=int((g.size() > 1).sum()), flagged_first_among_multi=float(first_hit[g.size() > 1].mean()),
                 flagged_any_among_multi=float(any_hit[g.size() > 1].mean()))
        res[f'fpr_{fpr}'] = r
    out[model] = res
json.dump(out, open('results/alerts.json', 'w'), indent=1); print(json.dumps(out, indent=1))

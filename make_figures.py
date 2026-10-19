"""Schematic and summary figures for the paper: generator architecture, causal-feature diagram, SHAP attribution. Run after shap_compare.py."""
import generate_dataset as G
import os, json
os.makedirs('figures', exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
TEAL, CORAL, PURP, GOLD, BLUE, GREY = '#2A9D8F', '#E76F51', '#8D5FD3', '#E9C46A', '#4C78A8', '#8A8F98'

def box(ax, x, y, w, h, text, fc, fs=8, tc='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec='#333', lw=0.8, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)
def arrow(ax, p, q, c='#333', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1, color=c, connectionstyle=f'arc3,rad={rad}'))

# ---- 1. generator architecture ------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.set_xlim(0, 21); ax.set_ylim(0, 8.8); ax.axis('off')
box(ax, 0.3, 6.6, 6.2, 1.9, 'NIBSS 2023 tables\nchannel, month, technique shares\nmean loss per case', '#BFD4EA')
box(ax, 8.0, 6.6, 7.2, 1.9, 'Declared assumptions\ncustomer spend and activity, diurnal and weekly\nprofile, bursts, merchant factors, prevalence', '#F6D5C9')
xs = [0.3, 3.8, 7.3, 10.8, 14.3, 17.8]; w = 3.0
labels = ['1. Customers\nage, location, bank,\npreferred channel,\nspend level, activity', '2. Legitimate\ntransactions\ntimestamps, bursts,\nchannel, merchant\n(channel-compatible),\namount',
          '3. Fraud incidents\nincident quotas, month x\nmerchant weights, bursts,\none loss per incident,\nsplit over transactions', '4. Causal features\nprior transactions only,\nexact 24 h / 7 d\nwindows, current\nexcluded',
          '5. Burn-in\n90 days simulated\nthen discarded', '6. FraudBench-NG\n2.1 M rows\n0.1% fraud']
cols = [TEAL] * 5 + [PURP]
for x, t, c in zip(xs, labels, cols): box(ax, x, 2.6, w, 2.7, t, c, fs=7.5, tc='white' if c == PURP else 'black')
for a, b in zip(xs[:-1], xs[1:]): arrow(ax, (a + w, 3.95), (b, 3.95))
arrow(ax, (3.4, 6.6), (8.8, 5.35), BLUE); arrow(ax, (3.4, 6.6), (11.8, 5.35), BLUE, 0.0)
arrow(ax, (9.0, 6.6), (5.3, 5.35), CORAL); arrow(ax, (10.5, 6.6), (2.1, 5.35), CORAL, -0.1); arrow(ax, (12.0, 6.6), (8.6, 5.35), CORAL)
box(ax, 2.5, 0.2, 16.5, 1.5, 'Validation: NIBSS agreement, causality brute force, seeds and sensitivity, benchmark classifiers, SHAP', '#E8E8E8', fs=8)
arrow(ax, (19.3, 2.6), (17.5, 1.7))
plt.tight_layout(); plt.savefig('figures/fig_architecture.png', dpi=200); plt.close()

# ---- 2. causal feature diagram -------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True)
tx = np.array([1, 2, 3, 4.5, 5.2, 6, 7.2, 8, 9]); cur = 5
for k, (a, title) in enumerate(zip(ax, ['Earlier benchmark: whole-year aggregates', 'FraudBench-NG: prior transactions only'])):
    a.set_xlim(0.3, 9.8); a.set_ylim(-1.6, 1.4); a.axis('off'); a.set_title(title, fontsize=9, loc='left')
    a.annotate('', (9.7, 0), (0.4, 0), arrowprops=dict(arrowstyle='-|>', color='#333'))
    for i, x in enumerate(tx):
        c = CORAL if i == cur else (TEAL if i < cur else GREY)
        a.scatter(x, 0, s=130 if i == cur else 70, color=c, zorder=3, marker='*' if i == cur else 'o')
    a.text(tx[cur], 0.45, 'current\ntransaction', ha='center', fontsize=7.5, color=CORAL)
    a.text(1.2, 0.45, 'past', fontsize=7.5, color=TEAL); a.text(8.2, 0.45, 'future', fontsize=7.5, color=GREY)
ax[0].plot([0.6, 9.4], [-0.6, -0.6], color=CORAL, lw=3); ax[0].text(5, -1.05, 'mean, count, diversity over the whole year: includes the current and all future transactions', ha='center', fontsize=7.5, color=CORAL)
ax[1].plot([0.6, tx[cur] - 0.15], [-0.6, -0.6], color=TEAL, lw=3); ax[1].text(2.8, -1.0, 'baseline: all prior transactions', ha='center', fontsize=7.5, color=TEAL)
ax[1].plot([3.7, tx[cur] - 0.15], [-0.95, -0.95], color=BLUE, lw=3); ax[1].text(3.6, -1.35, 'prior 7 days (604,800 s)', ha='right', fontsize=7.5, color=BLUE)
ax[1].plot([4.9, tx[cur] - 0.15], [-1.3, -1.3], color=PURP, lw=3); ax[1].text(4.8, -1.6, 'prior 24 h (86,400 s)', ha='right', fontsize=7.5, color=PURP)
plt.tight_layout(); plt.savefig('figures/fig_causal.png', dpi=200); plt.close()

# ---- 3. SHAP attribution ---------------------------------------------------------
S = json.load(open('results/shap.json'))['fraudbench']
order = ['Customer history', 'Channel, bank, location, age', 'Time', 'Amount']; cols = [CORAL, GREY, GOLD, TEAL]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={'width_ratios': [1, 1.15]})
v = [S['group_share'].get(o, 0) * 100 for o in order]
ax[0].barh(order[::-1], v[::-1], color=cols[::-1])
for i, x in enumerate(v[::-1]): ax[0].text(x + 1, i, f'{x:.0f}%', va='center', fontsize=8)
ax[0].set(xlabel='share of mean |SHAP| (%)', xlim=(0, 70), title='(a) importance by feature group'); ax[0].tick_params(axis='y', labelsize=8)
top = S['top']; names = [G.label(n) for n in list(top)[::-1]]; vals = np.array([top[n] for n in list(top)[::-1]]); vals = vals / vals.max()
ax[1].barh(names, vals, color=TEAL); ax[1].tick_params(axis='y', labelsize=8); ax[1].set(xlabel='mean |SHAP|, relative to top feature', title='(b) eight largest features', xlim=(0, 1.05))
plt.tight_layout(); plt.savefig('figures/fig_shap.png', dpi=200, bbox_inches='tight'); plt.close()
print('figures written')

# ---- 4. phi ablation: why is PR-AUC higher at phi = 0 than at phi = 0.4? -----------------------------
import os
if os.path.exists('results/phi_experiments.json'):
    P = json.load(open('results/phi_experiments.json')); ps = P['per_seed']
    V = [('phi0>phi0', 'phi0', '$\\phi=0$'), ('phi04>phi04', 'phi04', '$\\phi=0.4$'), ('phi04_sameinc>phi04_sameinc', 'phi04_sameinc', '$\\phi=0.4$\nsame no. of\nincidents'),
         ('phi04_nosplit>phi04_nosplit', 'phi04_nosplit', '$\\phi=0.4$\nfull loss\nper row'), ('phi07>phi07', 'phi07', '$\\phi=0.7$')]
    COL = [TEAL, CORAL, GOLD, PURP, BLUE]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    def lift(s, cell, key):
        c = s['cells'][cell]; n_test = c['n_fraud'] / c['base']
        return c[key] / (c['base'] if key == 'pr_auc' else c['n_first'] / n_test)
    for a, key, ttl in ((ax[0], 'pr_auc', '(a) PR-AUC lift, all fraud rows'), (ax[1], 'pr_first', '(b) PR-AUC lift, first fraud events only')):
        for i, (cell, _, lab) in enumerate(V):
            v = np.array([lift(s, cell, key) for s in ps]); a.bar(i, v.mean(), color=COL[i], alpha=.85)
            a.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color='k', capsize=3, lw=1)
            a.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.18, .18, len(v)), v, s=7, color='k', alpha=.45, zorder=3)
        a.set_xticks(range(len(V))); a.set_xticklabels([l for *_, l in V], fontsize=7); a.set_title(ttl, fontsize=9)
        a.set_ylabel('PR-AUC / base rate (mean of 10 seeds; bars: s.e.)', fontsize=7)
    F = P['features']
    for i, (_, f, _) in enumerate(V):
        m, s_ = F[f]['auc_ratio_first']; ax[2].bar(i, m, color=COL[i], alpha=.85); ax[2].errorbar(i, m, s_ / np.sqrt(len(ps)), color='k', capsize=3, lw=1)
    ax[2].axhline(.5, color='grey', ls=':', lw=.8); ax[2].set_ylim(.4, .8); ax[2].set_xticks(range(len(V))); ax[2].set_xticklabels([l for *_, l in V], fontsize=7)
    ax[2].set_title('(c) ROC-AUC of relative amount, first events', fontsize=9); ax[2].set_ylabel('ROC-AUC', fontsize=7.5)
    plt.tight_layout(); plt.savefig('figures/fig_phi_ablation.png', dpi=200, bbox_inches='tight'); plt.close(); print('phi figure written')

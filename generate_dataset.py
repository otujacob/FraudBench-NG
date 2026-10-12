"""
NIBSS-calibrated synthetic inter-bank fraud dataset generator.

Fully vectorised; every behavioural feature is computed from the final
transaction amounts using only information available up to (and including)
the current transaction of the same customer (strictly causal).

Parameters that are taken from the NIBSS 2023 Annual Fraud Landscape are marked
[NIBSS]; parameters that are modelling assumptions without a published source
are marked [ASSUMED] and are documented as such in the accompanying paper.

Usage:  python generate_dataset.py --customers 25000 --prevalence 0.001 --out data
"""
import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CHANNELS = ['ATM', 'ECOM', 'IB', 'Mobile', 'POS', 'Web']
BANKS = [f'Bank_{i}' for i in range(1, 11)]
LOCATIONS = ['Lagos', 'Abuja', 'Ogun', 'Oyo', 'Rivers', 'Other']
LOC_P = [0.28, 0.20, 0.12, 0.14, 0.11, 0.15]                       # [ASSUMED]
AGE_GROUPS = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
AGE_P = [0.18, 0.30, 0.24, 0.15, 0.08, 0.05]                       # [ASSUMED]
MERCHANTS = ['Medical', 'ATM Withdrawal', 'Retail', 'Grocery', 'Education',
             'Transfer', 'Electronics', 'Transport', 'Airtime', 'Fashion',
             'Restaurant', 'Bill Payment', 'Entertainment']
MERCHANT_FRAUD_FACTOR = {                                           # [ASSUMED]
    'Medical': 0.3, 'ATM Withdrawal': 0.4, 'Retail': 0.7, 'Grocery': 0.5,
    'Education': 0.4, 'Transfer': 1.4, 'Electronics': 1.3, 'Transport': 0.9,
    'Airtime': 1.1, 'Fashion': 1.0, 'Restaurant': 0.8, 'Bill Payment': 1.2,
    'Entertainment': 1.1}
# [ASSUMED] which merchant categories can occur on which channel (a transaction's merchant is drawn
# uniformly from the categories allowed for its channel). Cash withdrawal only at ATMs and POS agents;
# transfers only on digital channels; internet banking restricted to institutional payees.
_ALL = [m for m in MERCHANTS]
CH_MERCHANTS = {
    'ATM': ['ATM Withdrawal'],
    'POS': [m for m in _ALL if m != 'Transfer'],
    'ECOM': [m for m in _ALL if m not in ('ATM Withdrawal', 'Transfer')],
    'Mobile': [m for m in _ALL if m != 'ATM Withdrawal'],
    'Web': [m for m in _ALL if m != 'ATM Withdrawal'],
    'IB': ['Transfer', 'Bill Payment', 'Airtime', 'Education', 'Medical', 'Entertainment'],
}
TECHNIQUES = ['SOCIAL_ENGINEERING', 'ROBBERY', 'CARD_THEFT', 'PIN_COMPROMISE',
              'PHISHING', 'OTHER']

# [NIBSS] 2023 fraud counts and losses (NGN million) by channel.
# NOTE: NIBSS's own channel table sums to 95,525 cases and NGN 16,636.1m, whereas the stated totals
# (and the monthly and technique tables) are 95,620 cases and NGN 17,666.8m. The generator uses each
# table only through shares or per-case means normalised within that table, so the tables never
# need to agree with one another (see the paper, Section on reconciling the NIBSS tables).
NIBSS_CH_COUNT = {'Mobile': 47526, 'Web': 21887, 'POS': 17560,
                  'IB': 5381, 'ECOM': 2449, 'ATM': 722}
NIBSS_CH_LOSS_M = {'Mobile': 5691.7, 'Web': 2223.6, 'POS': 4415.8,
                   'IB': 4097.3, 'ECOM': 144.1, 'ATM': 63.6}
# [NIBSS] 2023 fraud counts by calendar month
NIBSS_MONTH_COUNT = [9234, 9492, 8829, 8234, 11716, 6240, 8649, 8207, 7503,
                     6860, 6367, 4289]
# [NIBSS] 2023 fraud counts by technique ("OTHER" = all remaining categories)
NIBSS_TECH_COUNT = {'SOCIAL_ENGINEERING': 62901, 'ROBBERY': 10179,
                    'CARD_THEFT': 6825, 'PIN_COMPROMISE': 5273,
                    'PHISHING': 4457, 'OTHER': 5985}

CH_P = np.array([0.05, 0.11, 0.11, 0.42, 0.16, 0.15])             # [ASSUMED] legit channel mix
CH_LEGIT_SHIFT = {'ATM': -0.30, 'ECOM': -0.10, 'IB': 0.50,         # [ASSUMED] log-amount shift
                  'Mobile': 0.0, 'POS': -0.10, 'Web': 0.10}
HOUR_W = np.array([.4, .3, .3, .3, .4, .5, .8, 1.2, 1.6, 2.0, 2.0, 2.0,
                   2.2, 2.2, 2.2, 2.0, 1.9, 2.1, 2.2, 2.0, 1.6, 1.2, .8, .5])  # [ASSUMED]
DOW_W = np.array([1.0, 1.0, 1.0, 1.05, 1.25, 1.1, 0.8])           # [ASSUMED] Mon..Sun
FRAUD_SIGMA = 0.9                                                  # [ASSUMED]
BURST_PROB = 0.25      # [ASSUMED] prob. a legit tx follows the previous one within minutes
BURST_MEAN_S = 900.0   # [ASSUMED]
FRAUD_FOLLOW_P = 0.40  # [ASSUMED] prob. that a fraud event is followed by another (takeover bursts)
FRAUD_SPLIT_ALPHA = 2.0  # [ASSUMED] an incident's loss is split across its transactions in Dirichlet(alpha) proportions
WARM_DAYS = 90         # burn-in simulated before 1 Jan 2023 and discarded, so that history features
                       # are on comparable footing in every part of the released year
TOTAL_S = (365 + WARM_DAYS) * 86400
WARM_S = WARM_DAYS * 86400
KEY = 1e9              # customer key spacing (> seconds in a year)


# Columns that are bookkeeping, not features, and the merchant columns that carry the generator's own
# merchant assumption. model_matrix() leaves all of them out by default.
NON_FEATURES = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud', 'fraud_technique',
                'incident_id', 'is_followon']
MERCHANT_COLS = ['merchant_category', 'merchant_risk_score']
CATEGORICAL = ['channel', 'merchant_category', 'bank', 'location', 'age_group']


def model_matrix(d, include_merchant=False):
    """Default feature matrix for modelling: drops identifiers, labels, incident bookkeeping and (unless
    include_merchant) the merchant columns, which only reflect the generator's merchant assumption."""
    drop = [c for c in NON_FEATURES + ([] if include_merchant else MERCHANT_COLS) if c in d.columns]
    cats = [c for c in CATEGORICAL if c in d.columns and c not in drop]
    return pd.get_dummies(d.drop(columns=drop), columns=cats).astype(float)


def draw_merchants(ch, rng):
    out = np.empty(len(ch), dtype=object)
    for c in CHANNELS:
        m = ch == c
        out[m] = rng.choice(CH_MERCHANTS[c], int(m.sum()))
    return out


ORIGIN = pd.Timestamp('2023-01-01') - pd.Timedelta(days=WARM_DAYS)   # Monday 3 Oct 2022


def month_of(sec):
    return (pd.to_datetime(sec, unit='s', origin=ORIGIN).month).values


def generate(n_customers=25000, prevalence=0.001, seed=42, drop_burnin=True,
             merchant_factors=None):
    """merchant_factors: optional dict {category: weight} replacing MERCHANT_FRAUD_FACTOR
    (used for the sensitivity analysis; the default reproduces the released dataset)."""
    mfac = MERCHANT_FRAUD_FACTOR if merchant_factors is None else merchant_factors
    rng = np.random.default_rng(seed)

    # ---- customers ---------------------------------------------------------
    cust = pd.DataFrame({
        'customer_id': [f'CUST{i:07d}' for i in range(n_customers)],
        'age_group': rng.choice(AGE_GROUPS, n_customers, p=AGE_P),
        'location': rng.choice(LOCATIONS, n_customers, p=LOC_P),
        'bank': rng.choice(BANKS, n_customers),
        'pref': rng.choice(CHANNELS, n_customers, p=CH_P),
        'mu': rng.normal(10.8, 0.6, n_customers),         # customer spend level (log-NGN)
        'sig': rng.uniform(0.4, 0.9, n_customers),        # customer within-spend dispersion
        'n': np.clip(rng.lognormal(4.2, 0.7, n_customers).astype(int), 5, 1000)
             * (365 + WARM_DAYS) // 365,
    })
    cid = np.repeat(np.arange(n_customers), cust['n'].values)
    N = len(cid)

    # ---- legitimate timestamps: diurnal + weekly profile, bursty -----------
    nd = 365 + WARM_DAYS
    day_w = DOW_W[np.arange(nd) % 7]                   # day 0 = ORIGIN, a Monday
    day = rng.choice(nd, N, p=day_w / day_w.sum())
    hour = rng.choice(24, N, p=HOUR_W / HOUR_W.sum())
    t = day * 86400.0 + hour * 3600.0 + rng.uniform(0, 3600, N)
    order = np.lexsort((t, cid)); cid, t = cid[order], t[order]
    first = np.r_[True, cid[1:] != cid[:-1]]
    leader = first | (rng.random(N) > BURST_PROB)
    cl = np.cumsum(leader) - 1
    off = np.where(leader, 0.0, rng.exponential(BURST_MEAN_S, N))
    cs = np.cumsum(off)
    lead_idx = np.flatnonzero(leader)
    off = cs - cs[lead_idx][cl] + off * 0              # cumulative offset within cluster
    t = np.minimum(t[lead_idx][cl] + off, TOTAL_S - 1)

    # ---- legit channel / merchant / amount ---------------------------------
    pref = cust['pref'].values[cid]
    ch = np.where(rng.random(N) < 0.70, pref, rng.choice(CHANNELS, N, p=CH_P))
    mer = draw_merchants(ch, rng)
    shift = pd.Series(ch).map(CH_LEGIT_SHIFT).values
    amt = np.exp(rng.normal(cust['mu'].values[cid] + shift, cust['sig'].values[cid]))
    fraud = np.zeros(N, bool)
    tech = np.full(N, '', dtype=object)

    # ---- fraud initiation (weights from NIBSS channel & month mix) ---------
    ch_share = pd.Series(NIBSS_CH_COUNT) / sum(NIBSS_CH_COUNT.values())
    mdays = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    m_share = np.array(NIBSS_MONTH_COUNT) / sum(NIBSS_MONTH_COUNT)
    m_factor = m_share / (mdays / 365)
    month = month_of(t)
    # NIBSS counts and losses are per CASE (one incident). The generator therefore works in incidents:
    # the channel quota is a number of incidents (NIBSS channel share of all incidents); each incident
    # has one initiating transaction plus a geometric number of follow-on transactions, and ONE incident
    # loss is drawn per incident and then split across its transactions. The expected number of rows per
    # incident is 1/(1-phi), so the number of incidents is target*(1-phi) and the expected prevalence stays
    # at `prevalence` whatever phi is. Initiators are sampled without replacement on month x merchant weights.
    kept = t >= WARM_S                                  # fraud is injected only into the released year
    target = int(round(prevalence * kept.sum()))
    n_incidents = int(round(target * (1 - FRAUD_FOLLOW_P)))
    init = np.zeros(N, bool); n_extra = np.zeros(N, int)
    for c in CHANNELS:
        quota = int(round(ch_share[c] * n_incidents))
        pool = np.flatnonzero((ch == c) & kept)
        wc = (m_factor[month[pool] - 1] * pd.Series(mer[pool]).map(mfac).values)
        cand = rng.choice(pool, size=min(len(pool), quota), replace=False, p=wc / wc.sum())
        init[cand] = True; n_extra[cand] = rng.geometric(1 - FRAUD_FOLLOW_P, len(cand)) - 1

    # ---- incident losses, technique, takeover bursts -----------------------
    loss_mean = {c: NIBSS_CH_LOSS_M[c] * 1e6 / NIBSS_CH_COUNT[c] for c in CHANNELS}   # NGN per case
    tp = np.array([NIBSS_TECH_COUNT[k] for k in TECHNIQUES], float); tp /= tp.sum()

    init_idx = np.flatnonzero(init); n_inc = len(init_idx)
    pos = np.full(N, -1); pos[init_idx] = np.arange(n_inc)          # initiating row -> incident number
    ridx = np.repeat(np.arange(N), n_extra)                         # follow-on rows -> initiating row
    inc_ex = pos[ridx]
    gap = rng.exponential(300.0, len(ridx))
    cg = np.cumsum(gap)                                  # cumulative gaps within each incident (ridx is sorted)
    if len(ridx):
        fst = np.r_[True, ridx[1:] != ridx[:-1]]
        fi = np.flatnonzero(fst)
        gi = np.cumsum(fst) - 1
        cg = cg - (cg[fi] - gap[fi])[gi]
    t_ex = np.minimum(t[ridx] + cg, TOTAL_S - 1)

    # one loss per incident (mean = NIBSS loss per case), split across the incident's transactions
    tm = pd.Series(ch[init_idx]).map(loss_mean).values
    loss = np.exp(rng.normal(np.log(tm) - FRAUD_SIGMA ** 2 / 2, FRAUD_SIGMA))
    g0 = rng.gamma(FRAUD_SPLIT_ALPHA, 1.0, n_inc); g1 = rng.gamma(FRAUD_SPLIT_ALPHA, 1.0, len(ridx))
    tot = g0 + np.bincount(inc_ex, weights=g1, minlength=n_inc)
    fraud[init] = True
    tech[init] = rng.choice(TECHNIQUES, n_inc, p=tp)
    amt[init_idx] = np.maximum(loss * g0 / tot, 1.0)
    e_amt = np.maximum(loss[inc_ex] * g1 / tot[inc_ex], 1.0)
    e_ch, e_tech = ch[ridx], tech[ridx]
    inc = np.full(N, -1); inc[init_idx] = np.arange(n_inc)
    cid = np.r_[cid, cid[ridx]]; t = np.r_[t, t_ex]
    ch = np.r_[ch, e_ch]; mer = np.r_[mer, draw_merchants(e_ch, rng)]
    amt = np.r_[amt, e_amt]; fraud = np.r_[fraud, np.ones(len(ridx), bool)]
    tech = np.r_[tech, e_tech]
    inc = np.r_[inc, inc_ex]; isfo = np.r_[np.zeros(N, int), np.ones(len(ridx), int)]
    N = len(cid)

    # ---- sort chronologically within customer; causal features -------------
    order = np.lexsort((t, cid))
    cid, t, ch, mer, amt, fraud, tech, inc, isfo = (a[order] for a in (cid, t, ch, mer, amt, fraud, tech, inc, isfo))
    amt = np.round(amt, 2)
    first = np.r_[True, cid[1:] != cid[:-1]]
    gstart = np.flatnonzero(first)
    gid = np.cumsum(first) - 1

    def gcum(x):                                       # within-customer cumulative sum
        c = np.cumsum(x)
        return c - (c[gstart] - x[gstart])[gid]

    # All behavioural features use PRIOR transactions of the same customer only
    # (the current transaction is excluded from every baseline and window).
    cnt = gcum(np.ones(N)) - 1                          # number of prior transactions
    s1, s2 = gcum(amt) - amt, gcum(amt ** 2) - amt ** 2
    has = cnt > 0
    mean_tot = np.where(has, s1 / np.maximum(cnt, 1), amt)        # no history -> current amount
    std_tot = np.where(has, np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - (s1 / np.maximum(cnt, 1)) ** 2, 0)), 0.0)

    key = cid * KEY + t
    P1, P2 = np.r_[0, np.cumsum(amt)], np.r_[0, np.cumsum(amt ** 2)]
    idx = np.arange(N)
    lo24 = np.searchsorted(key, key - 86400, 'left')    # prior rows j with t - t_j <= 24 h, j < i
    lo7 = np.searchsorted(key, key - 7 * 86400, 'left')  # prior rows with t - t_j <= 7 days (exact seconds)
    n24 = idx - lo24; sum24 = P1[idx] - P1[lo24]
    n7 = idx - lo7
    m7 = np.where(n7 > 0, (P1[idx] - P1[lo7]) / np.maximum(n7, 1), mean_tot)
    sd7 = np.where(n7 > 0, np.sqrt(np.maximum((P2[idx] - P2[lo7]) / np.maximum(n7, 1) - ((P1[idx] - P1[lo7]) / np.maximum(n7, 1)) ** 2, 0)), 0.0)

    ratio = np.where(has, amt / np.where(mean_tot > 0, mean_tot, 1), 1.0)
    velocity = np.where(has, n24 * sum24 / np.where(mean_tot > 0, mean_tot, 1), 0.0)
    chdiv = np.zeros(N)
    for c in CHANNELS:
        chdiv += (gcum((ch == c).astype(float)) - (ch == c) > 0)
    online = np.where(has, (gcum(np.isin(ch, ['Mobile', 'Web', 'ECOM']).astype(float))
                            - np.isin(ch, ['Mobile', 'Web', 'ECOM'])) / np.maximum(cnt, 1), 0.0)

    ts = pd.to_datetime(t, unit='s', origin=ORIGIN)
    hr, dw, mo = ts.hour.values, ts.dayofweek.values, ts.month.values
    mf = pd.Series(mfac)
    mrs = ((mf - mf.min()) / (mf.max() - mf.min()) * 0.8 + 0.1).round(6)

    df = pd.DataFrame({
        'customer_id': cust['customer_id'].values[cid],
        'timestamp': ts,
        'amount': amt, 'channel': ch,
        'bank': cust['bank'].values[cid],
        'merchant_category': mer,
        'location': cust['location'].values[cid],
        'age_group': cust['age_group'].values[cid],
        'is_fraud': fraud.astype(int),
        'fraud_technique': np.where(fraud, tech, None),
        'incident_id': np.where(inc >= 0, 'INC' + pd.Series(inc).astype(str).str.zfill(7).values, None),
        'is_followon': isfo,
        'amount_log': np.log1p(amt),
        'amount_vs_mean_ratio': ratio.round(6),
        'amount_sum_24h': sum24.round(2), 'amount_mean_7d': m7.round(2),
        'amount_std_7d': sd7.round(2), 'tx_count_24h': n24,
        'amount_mean_total': mean_tot.round(2), 'amount_std_total': std_tot.round(2),
        'tx_count_total': cnt.astype(int), 'channel_diversity': chdiv.astype(int),
        'online_channel_ratio': online.round(6), 'velocity_score': velocity.round(6),
        'merchant_risk_score': pd.Series(mer).map(mrs).values,
        'hour_sin': np.sin(2 * np.pi * hr / 24), 'hour_cos': np.cos(2 * np.pi * hr / 24),
        'day_sin': np.sin(2 * np.pi * dw / 7), 'day_cos': np.cos(2 * np.pi * dw / 7),
        'month_sin': np.sin(2 * np.pi * mo / 12), 'month_cos': np.cos(2 * np.pi * mo / 12),
        'is_peak_hour': ((hr >= 9) & (hr <= 18)).astype(int),
    })
    if drop_burnin:
        df = df[df['timestamp'] >= '2023-01-01']
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    df.insert(0, 'transaction_id', [f'TXN{i:09d}' for i in range(len(df))])
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--customers', type=int, default=25000)
    ap.add_argument('--prevalence', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    d = generate(a.customers, a.prevalence, a.seed)
    print(f'{len(d):,} rows, {d.is_fraud.sum():,} fraud ({d.is_fraud.mean():.4%})')
    d.to_parquet(f'{a.out}/nibss_synthetic_fraud.parquet', index=False)

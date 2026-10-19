# FraudBench-NG: a NIBSS-calibrated synthetic dataset and generator for Nigerian inter-bank fraud detection

Generator, validation code and benchmark for a synthetic, transaction-level dataset of Nigerian inter-bank payments with fraud. The fraud channel mix, monthly profile, technique mix and mean loss per case follow the **2023 NIBSS Annual Fraud Landscape**. Fraud is generated as *incidents* (NIBSS counts and loss are per case): one loss per incident, split over its transactions. Every other parameter is declared as an assumption in the paper (Table "provenance") and in the code (`[ASSUMED]` comments in `generate_dataset.py`).

> **Read this first.** This is synthetic data. We did not have access to real NIBSS transaction records, so nothing here shows that a model trained on this data will work on real transactions. Channel, monthly and technique profiles match NIBSS *by construction*. Fraud prevalence is set to 0.1%, about 115x the real national rate by volume, so that the data are usable for benchmarking. FraudBench-NG is meant to be *transparent, auditable and assumption-aware*, not realistic in every detail.

Paper: *A NIBSS-Calibrated Synthetic Dataset for Nigerian Inter-Bank Fraud Detection* (Otu Jacob, Colin Kuka, Adam Knowles; Wrexham University). Citation details will be added on publication.

## Why this exists

Synthetic fraud benchmarks can look rigorous and still hand the model the answer, through features computed over a customer's whole year, windows that contain the transaction being judged, redundant scores, or the generator's own assumptions used as inputs. FraudBench-NG is a transparent parametric generator designed to avoid these: fraud status and amounts are fixed first, every behavioural feature uses only prior transactions, and every parameter is marked as NIBSS or assumed. This repository holds the generator and the validation code.

## What you get

| Property | Value |
|---|---|
| Rows | 2,102,151 transactions, 1 Jan to 31 Dec 2023 |
| Customers | 25,000 (fixed age group, location, bank, preferred channel, spending level) |
| Fraud | 2,059 rows (0.098%) in 1,261 incidents; 39% of fraud rows are follow-ons |
| Channels | ATM, E-commerce, Internet banking, Mobile, POS, Web |
| Format | Parquet (about 160 MB) |

Customer behaviour has heterogeneity (log-normal spend and activity), diurnal and weekly rhythm, bursts, and account-takeover fraud bursts. Behavioural features are **strictly causal**: each uses only a customer's earlier transactions, never the current one. Time windows are exact (24 h = 86,400 s; 7 d = 604,800 s). A 90-day burn-in is simulated before 1 January and discarded.

## Quick start

```bash
pip install -r requirements.txt
python generate_dataset.py --customers 25000 --prevalence 0.001 --seed 42 --out data
```

This writes `data/nibss_synthetic_fraud.parquet` (about 30 seconds). Same seed gives the same file.

```python
import generate_dataset as G
df = G.generate(n_customers=25000, prevalence=0.001, seed=42)
```

Or open `demo.ipynb` (also runs in Colab). It is a thin wrapper around `generate_dataset.py` that generates the data, checks it against NIBSS, brute-force verifies causality, plots distributions and runs a small benchmark.

## Reproducing the paper's results

Run from the repository root, in this order:

```bash
python generate_dataset.py --out data     # the dataset
python compare.py                         # LR / RF / XGBoost benchmark, customer- and incident-clustered bootstrap CIs, first-event vs follow-on -> results/models.json
python compare.py --with-merchant         # same, with merchant columns added -> results/models_with_merchant.json
python validate.py                        # tests and figures -> results/validation.json, figures/
python seeds.py                           # stability over 10 seeds -> results/seeds.json
python sensitivity.py                     # merchant, takeover (phi), fraud-dispersion and burst sensitivity (7 settings x 10 seeds, about 1-2 hours) -> results/sensitivity.json
python checks.py data/nibss_synthetic_fraud.parquet   # leakage, redundancy and history-ablation checks -> results/checks.json
python shap_compare.py data/nibss_synthetic_fraud.parquet   # TreeSHAP attribution -> results/shap.json
python phi_experiments.py                 # train at one takeover probability, test at another; phi=0 vs 0.4 ablation (about 1 hour) -> results/phi_experiments.json
python mix_scenarios.py                   # alternative legitimate channel mixes -> results/mix_scenarios.json
python alerts.py                          # alert burden and detection timing from the saved test scores -> results/alerts.json
python make_figures.py                    # architecture, causal, SHAP and phi-ablation figures
```

`validate.py` needs the output of `compare.py`, so run them in that order.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | The generator (all logic lives here) |
| `compare.py` | Chronological split; class-weighted logistic regression, random forest, XGBoost; PR-AUC, recall at fixed false-positive rate, precision in top 1%, customer- and incident-clustered bootstrap intervals; results by first event and follow-on |
| `validate.py` | Comparison with NIBSS, real-data properties, statistical tests, causality brute force, figures |
| `seeds.py` | Repeats generation across seeds to check stability |
| `checks.py` | Leakage, redundancy and customer-history ablation checks on FraudBench-NG |
| `sensitivity.py` | Regenerates under alternative merchant factors, takeover continuation, fraud dispersion and burst settings |
| `phi_experiments.py` | Cross-assumption transfer (train at phi=0.4, test at 0 and 0.7) and the phi=0 vs 0.4 ablation |
| `mix_scenarios.py` | Sensitivity to the assumed legitimate channel mix |
| `alerts.py` | False alerts per fraud caught and how early incidents are flagged |
| `shap_compare.py` | TreeSHAP attribution of XGBoost trained on FraudBench-NG |
| `make_figures.py` | Architecture, causal-feature and SHAP figures from the results JSONs |
| `demo.ipynb` | Walk-through notebook |
| `figures/`, `results/` | Outputs reported in the paper |

## Columns

| Group | Columns |
|---|---|
| Identifiers | `transaction_id`, `customer_id`, `timestamp` |
| Transaction | `amount`, `channel`, `bank`, `merchant_category`, `location`, `age_group` |
| Labels and bookkeeping | `is_fraud`; `fraud_technique` (fraud rows only); `incident_id` (fraud rows; **none of these three are features**); `is_followon` (1 for follow-on rows of an incident) |
| Amount | `amount_log`, `amount_vs_mean_ratio` |
| Customer history (prior transactions only) | `amount_sum_24h`, `amount_mean_7d`, `amount_std_7d`, `tx_count_24h`, `amount_mean_total`, `amount_std_total`, `tx_count_total`, `channel_diversity`, `online_channel_ratio`, `velocity_score` |
| Context | `merchant_risk_score` (**assumption, not a default feature**), `is_peak_hour`, `hour_sin/cos`, `day_sin/cos`, `month_sin/cos` |

First-ever transaction of a customer: ratio = 1, velocity = 0, mean = amount.

## Using it for evaluation: please note

- **Split by time, not at random.** The paper uses train to 7 Aug, validation to 18 Oct, test to 31 Dec.
- **Do not report accuracy or a fixed 0.5 threshold.** At 0.1% prevalence they are uninformative. Use PR-AUC, recall at a fixed false-positive rate, or precision in the top 1% of scores, and class weighting.
- **Use `generate_dataset.model_matrix(df)`** for the default feature set: it drops identifiers, `is_fraud`, `fraud_technique`, `incident_id`, `is_followon`, `merchant_category` and `merchant_risk_score`. The merchant score is the generator's own assumption, so a model that uses it learns the assumption (`include_merchant=True` / `compare.py --with-merchant` to test).
- **Report first events and follow-ons separately** (`is_followon`); follow-ons are shaped by the assumed takeover-burst probability (0.4) and drive much of the benchmark difficulty.
- **Bootstrap by customer** (or by incident for fraud-specific metrics), not by row.
- **Do not use `fraud_technique`** as a feature (it exists only for fraud rows). Consider also excluding **`tx_count_total`**: history length grows through the year (median prior transactions about 56 in training vs 125 in test), so it carries no fraud signal (ROC-AUC 0.49) but differs between splits. `compare.py` keeps it to match the paper's table; `demo.ipynb` drops it.

## Calibration data

Fraud channel, month and technique profiles and mean loss per case come from the NIBSS *Annual Fraud Landscape 2023*, the latest NIBSS publication we found with those breakdowns. NIBSS has since reported only totals (2024: 70,111 cases, NGN 52.26bn, including a single NGN 31.1bn incident; 2025: 67,518 cases, NGN 25.85bn). The 2023 report's own tables disagree slightly (channel rows sum to 95,525 cases, whereas the stated total, monthly and technique tables give 95,620); the generator normalises each table separately. See the paper, section "Calibration data and its limits".

## What is not modelled

Merchant preferences of customer groups (merchants are drawn uniformly within a simple channel-compatibility table); relationships among location, bank and channel; amount-merchant relationships; regular (recurring) payments; individual versus business customers; demographic and geographic fraud patterns (reported Lagos concentration, victims aged 40+); time-of-day of fraud; seasonality; customer churn; devices, locations and beneficiaries; fraud rings; detection and label delay. Age group, location and bank are inert attributes. `merchant_risk_score` is a rescaling of the generator's own assumed merchant factors, not independent evidence of merchant risk. Full list and consequences: paper, Discussion (Table "Structure not modelled"); every parameter and its justification: paper, Table "Register of design choices".

## Known limitations

- No validation against real transactions; many parameters (customer mix, diurnal and weekly shape, burst parameters, merchant factors) are assumptions.
- Prevalence is about 115x enriched. The takeover-burst probability (0.4), the loss split (Dirichlet 2) and the channel-merchant table are assumptions. Within-day timing of fraud is not modelled.
- Residual history drift described above.
- Single-feature separability is modest (maximum ROC-AUC about 0.64) and benchmark scores are low (PR-AUC 0.016 to 0.036 against 0.00066 for random guessing); customer-clustered intervals are wide and overlap, so models cannot be ranked.

See the paper's Discussion section for the full list.

## Dataset deposit

The Parquet file is too large for GitHub. The archived copy and DOI will be added here once deposited. Until then, regenerate it with the command above.

## Licence

To be confirmed with the authors before release (suggested: MIT for code, CC BY 4.0 for data).

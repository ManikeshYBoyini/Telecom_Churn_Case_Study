# Methodology

## Problem framing

Customers rarely churn instantly — the original assignment models three
phases: a "good" phase, an "action" phase (service friction starts), and
the "churn" phase. This dataset covers months 6-8 (the "good"/"action"
window) and predicts churn in month 9. See `original_work/ManikeshBoyini.ipynb`
for the original framing this project builds on.

## Data cleaning

- Columns with >60% missing values are dropped — at that level, imputation
  would mostly be inventing data.
- Constant columns (`nunique() == 1`) and date-string columns are dropped;
  the dates are redundant with the recharge-amount columns already in the
  data.
- The drop list is computed **once, from train**, and applied identically
  to the unseen set — a column's fate never depends on the unseen set's
  own missingness pattern.
- Remaining nulls (all under 7% of any column) are median-imputed, not
  mean-imputed — the distributions are heavily skewed by high-usage
  outliers, so the mean would be pulled off-center.

## Feature engineering

`avg_customer_Spend = (age_on_network_in_months) * (mean ARPU across the
three months)` combines loyalty and revenue into one number, replacing 4
raw columns (`aon`, `arpu_6/7/8`) with 1 engineered one. Customers are then
bucketed into `customer_value` (HVC/MVC/LVC) by that number's 65th/20th
percentile on **train** — the same two threshold values are reused
unchanged for the unseen set, so a customer's segment never depends on
where they happen to fall in the unseen set's own distribution.

## Why PCA only for Logistic Regression

Logistic regression assumes linearly independent predictors and the
remaining correlated usage columns (122 after cleaning) violate that
badly — PCA (70 components, ~95% of variance) fixes it. Random Forest and XGBoost don't share that assumption
and handle correlated features natively; running them on the original
features instead of PCA components keeps feature importances tied to real,
named columns instead of an opaque linear combination of columns — which
matters directly for the business recommendations below.

Statistical significance for the logistic regression's PCA-component
coefficients is checked with `statsmodels.Logit` (notebook section 8) —
the original notebook's README claimed this was done, but the original
code never actually imported or called statsmodels; this rework does it
for real.

## Handling class imbalance

Churn is ~10% of customers (10.15% in the actual training split — the
original notebook's own EDA markdown said "12%," but its own preceding
value-counts cell gives 6975/67919 ≈ 10.3% by share; the "12%" figure
doesn't match either that share or the churn:active odds ratio (≈11.4%)
cleanly, and is likely just an eyeball rounding slip in the original
text). Random Forest and Logistic Regression use
`class_weight` to upweight the minority class; XGBoost uses
`scale_pos_weight` (the ratio of negative to positive examples) — the
sklearn-compatible way to do the same thing, since `XGBClassifier` doesn't
accept a `class_weight` argument the way scikit-learn's own estimators do.

## Hyperparameter tuning

All three models are tuned with stratified k-fold cross-validation,
scoring on ROC-AUC (appropriate for an imbalanced binary target).
Random Forest and Logistic Regression use `GridSearchCV`; XGBoost uses
`RandomizedSearchCV` over the same parameter families as the original
notebook, but with a much smaller `n_iter` — the original ran 800
iterations over a grid with only 36 unique combinations, which reruns most
combinations multiple times for no benefit.

## Threshold selection

Each model's submission threshold is chosen by maximizing F1 on the held-
out test split, and — unlike the original notebook's logistic-regression
path, which explored cutoffs of 0.33 and 0.6 but then generated its
actual submission with a plain `.predict()` call (an unexamined, implicit
0.5) — that exact F1-optimal threshold is what's used to generate the
submission file here. (The original's own labeled "final" submission,
the tuned XGBoost model, did use an examined threshold — 0.5, matching
its own cutoff-analysis recommendation — so this inconsistency was
specific to the logistic-regression path, not universal across all three
of the original notebook's submissions.)

F1 weights precision and recall equally, so it doesn't intrinsically
favor catching more churners the way a strict cost-asymmetry argument
("a missed churner costs far more than a false alarm") would suggest.
In this run, the F1-optimal thresholds (0.55/0.60/0.80) all land above
0.5 and reduce recall relative to a naive 0.5 cutoff for all three
models — i.e., F1-maximization here produces a more conservative,
higher-precision model, not the more churner-catching one the cost
argument would call for. If the business really does weigh missed
churners that much more heavily, a recall-weighted objective (e.g. F2)
or an explicit cost-weighted threshold search would match that stated
asymmetry better than F1 does.

## Model comparison and final pick

See the results table in the root `README.md` (sourced from
`outputs/figures/model_comparison_at_submission_threshold.csv`, since
that's the one computed at each model's own tuned threshold — the same
threshold used to produce the predictions in `outputs/submissions/` —
see "Limitations and future work" below for why there are two comparison
files). The best model by test ROC-AUC is
printed in section 10 of the notebook.

## Feature importance and SHAP

The winning tree-based model's built-in `feature_importances_`
(`outputs/figures/feature_importance.png`) is the primary importance
signal — deterministic, always available. A SHAP summary plot
(`outputs/figures/shap_summary.png`, produced when `TreeExplainer` runs
successfully for the installed SHAP/XGBoost/scikit-learn version
combination) adds direction on top of that: whether a high value of a
feature pushes a prediction toward or away from churn, not just how much
it matters on average. If the best model turns out to be Logistic
Regression instead, use the `statsmodels` p-value table from section 8
the same way.

## Business recommendations

1. The dominant churn signal is a month-8 drop in usage and recharge
   activity, not roaming or STD: `total_ic_mou_8` has the strongest
   correlation with churn in the EDA heatmap (notebook section 5) and is
   the single most important feature for the winning model (section
   11) — roughly 3x the importance of the next usage-pattern feature.
   Roaming and STD usage are real but secondary signals (much weaker
   correlation, lower importance); retention effort should lead with
   re-engaging customers whose month-8 usage/recharge is dropping, not
   with roaming/STD pricing.
2. Voice usage predicts retention more than data usage here — more
   competitive data packages are a plausible retention lever the model
   doesn't directly test.
3. Segment-level churn is stark: low-value customers churn at roughly
   double the rate of medium-value customers, and high-value customers
   churn least of all — retention spend aimed at keeping medium-value
   customers from sliding into low-value usage patterns is likely
   higher-leverage than a blanket offer.
4. This model explains *who* is at risk, not *why* — pair its usage-based
   signals with direct customer feedback (support tickets, surveys) before
   committing large-scale retention spend.

## Limitations and future work

- Random Forest and XGBoost both show a real train/test overfitting gap
  here, comparable in size to the original notebook — train ROC-AUC
  (printed in the notebook's section 9) is roughly 0.97 for both, about
  0.03 above their test ROC-AUC (~0.94). Their cross-validated score
  during tuning tracks test ROC-AUC closely (within ~0.005) — a
  reassuring signal that the CV procedure itself estimates held-out
  performance reliably, but not evidence against overfitting, since CV
  and test are both held-out measurements, neither is a training-set
  one. The more actionable caveat: each model's submission threshold is
  chosen by maximizing F1 on the same test split the comparison table
  reports on, so both numbers share the same held-out data — treat them
  as mildly optimistic and re-validate on a genuinely separate month
  before using this to drive spend decisions.
- The equal-footing comparison table
  (`outputs/figures/model_comparison.csv`) uses a uniform 0.5 probability
  cutoff for accuracy/precision/recall/F1, so the three models are
  compared on equal footing — ROC-AUC/PR-AUC are threshold-independent
  either way. Each model's actual submission uses its own tuned
  threshold instead; cite
  `outputs/figures/model_comparison_at_submission_threshold.csv` for what
  each model's metrics look like at the threshold it's actually
  submitted with, not the 0.5-cutoff table, when describing submitted
  performance.
- The column-drop list, median imputer, and HVC/LVC value thresholds
  are computed from the full labeled `train.csv` before the internal
  70/30 split — none of these use the target label, so this doesn't
  leak label information, but it does mean the internal test split
  isn't fully independent of the fitting data for these steps.
  Outlier trimming matters more: it's applied to the full labeled set
  before the split, so the internal holdout ends up outlier-trimmed
  too, while the real unseen Kaggle set is never trimmed — reported
  test metrics come from a slightly cleaner distribution than the
  actual submission faces. The Kaggle-unseen-set handling itself is
  unaffected by this: the same fitted drop list, imputer, scaler,
  PCA, and value thresholds are reused there too, never refit.
- Natural next steps, carried over from the original README's "Looking
  Ahead": survival analysis (to estimate *when* a high-risk customer is
  likely to churn, not just *whether*) and customer lifetime value
  modeling (to prioritize retention spend by expected value, not just
  churn probability).

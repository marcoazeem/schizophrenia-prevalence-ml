# Reanalysis results: schizophrenia prevalence capstone study

Run date: 25 October 2026. Code: `analysis.py`, `figures.py` (Python 3, scikit-learn). Seed 42.

## Data

- Source file: `ISMAIL AZEEM_TP073405_DATASET.csv` (Kaggle; the values are IHME Global Burden of Disease estimates, as republished by Our World in Data).
- 5,516 rows. Dropped 2 regional aggregates (Australasia, Caribbean) and 3 countries with no sex-specific values (Macedonia, Montenegro, Swaziland).
- **Analytic sample: 5,376 country-years, 192 countries, 1990 to 2017.**
- Target: schizophrenia prevalence (%), mean 0.208, SD 0.042, range 0.147 to 0.375.
- The `Male` and `Female` columns are most likely depression prevalence by sex (they correlate 0.77 to 0.79 with overall depression). **Check this against the original Kaggle page before submitting.**
- Features: prevalence of bipolar, eating, anxiety, drug use, depressive and alcohol use disorders; depression by sex; log10 population; year. No country identifier.

## Key structural fact

**99.5% of the variance in schizophrenia prevalence is between countries.** On average a country's value moves only 0.007 percentage points over 28 years. That's why random row splits give near-perfect scores: the test rows are nearly copies of training rows from the same country.

## Model performance (R²)

| Evaluation design | Ridge | Random Forest | Gradient Boosting |
|---|---|---|---|
| A. Random 80/20 split + country ID (original thesis) | 0.637 | 0.992 | 0.997 |
| B. Random 80/20 split, no country ID | 0.636 | 0.992 | 0.997 |
| **C. Unseen countries, 5x5 grouped CV (mean ± SD across folds)** | **0.577 ± 0.160** | **0.579 ± 0.267** | **0.660 ± 0.148** |
| D. Future years, countries seen (train ≤2010, test 2011 to 2017) | 0.660 | 0.960 | 0.971 |
| E. Unseen countries + future years (single split, 38 countries) | 0.847 | 0.919 | 0.934 |

- Mean-only baseline: R² about 0 in every design.
- **Persistence baseline under design D (carry each country's 2010 value forward): R² = 0.998.** That beats every ML model, so forecasting known countries is not a useful task for ML on this data.
- Under design C, gradient boosting: MAE 0.0115 percentage points, MAPE 5.0%. Country-level bootstrap 95% CI for pooled out-of-fold R²: 0.59 to 0.84.
- Design E is one random draw of countries and sits at the optimistic end of design C's fold spread. **Design C is the headline result.**
- The thesis's SVR result (R² = -1.29) came from unscaled inputs. With scaling, SVR gets 0.99 (design B) and 0.46 (design C).
- Full table with MAE, RMSE and MAPE: `results_all_designs.csv`.

## What drives the predictions (permutation importance, unseen countries)

Gradient boosting, drop in R² when each feature is shuffled:
1. Eating disorders: 1.06
2. Bipolar disorder: 0.59
3. Drug use disorders: 0.14
4. Everything else is below 0.03 or indistinguishable from zero (depression, anxiety, year, population).

Random forest and ridge give the same top two. Between-country Spearman correlations with schizophrenia: eating disorders 0.62, drug use 0.51, anxiety 0.39, bipolar 0.23, depression -0.11.

**Interpretation for the discussion:** eating disorder prevalence in GBD tracks national income and region closely. The signal is probably a shared development and modeling gradient, not a clinical link between the disorders. All inputs come from the same GBD modeling framework, so some of the association is built into the estimates. Under-predicted countries (China, Netherlands, Japan) and over-predicted ones are worth a sentence in the paper.

## What this means for the paper

- Drop the "detecting schizophrenia" framing. The honest contribution is (1) how much of cross-national variation in schizophrenia prevalence is explained by co-occurring disorder burden (about two thirds, R² ≈ 0.66), and (2) a methods lesson: random splits on country-year panels overstate accuracy (0.997 vs 0.66).
- Point (2) is a useful contribution in its own right. Many published ML studies on GBD or OWID panels use random splits.

## Figures (300 dpi)

- `fig1_r2_by_design.png`: R² by evaluation design (main figure)
- `fig2_observed_vs_predicted.png`: unseen-country predictions, country means
- `fig3_permutation_importance.png`: feature importance
- `fig4_country_trajectories.png`: between- vs within-country variation

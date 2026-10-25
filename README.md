# Co-occurring mental disorder burden and cross-national schizophrenia prevalence

Code, data and results for the manuscript *Co-occurring mental disorder burden explains two thirds of cross-national variation in schizophrenia prevalence: a machine learning analysis of global panel data, 1990 to 2017* (Ismail Azeem, Asia Pacific University of Technology & Innovation).

The study has two aims: estimate how much of the between-country variation in schizophrenia prevalence is explained by the prevalence of other mental disorders, and show how far that estimate depends on whether the evaluation design lets the model see the test countries during training. Under a random row split gradient boosting reaches R² = 0.997; with test countries withheld it reaches 0.66 (grouped CV) to 0.71 (leave-one-country-out).

## Contents

| File | What it is |
| --- | --- |
| `mental_health_depression_data.csv` | The dataset as analyzed: 5,516 country-years, 14 columns, GBD estimates compiled by Our World in Data and redistributed on Kaggle |
| `analysis.py` | All model fitting and evaluation (five designs, permutation importance, bootstrap, variance decomposition) |
| `figures.py` | Produces the four manuscript figures from the results |
| `results_all_designs.csv` | R², MAE, RMSE and MAPE for every model under every design |
| `permutation_importance.csv` | Permutation importance on held-out countries |
| `summary.json` | Sample sizes, correlations, variance decomposition, bootstrap intervals |
| `fig1_r2_by_design.png` to `fig4_country_trajectories.png` | Figures 1 to 4 at 300 dpi |
| `RESULTS.md` | Plain-language summary of the results and the independent check |

## Reproduce

```bash
pip install pandas numpy scikit-learn matplotlib
mkdir -p out
python analysis.py mental_health_depression_data.csv out   # about 4 minutes
python figures.py out
```

Python 3.12, scikit-learn 1.9.1. All seeds are fixed at 42.

## Data source

Prevalence estimates are from the Institute for Health Metrics and Evaluation's Global Burden of Disease study, as compiled by [Our World in Data](https://ourworldindata.org/mental-health) and redistributed on Kaggle. See the manuscript for the full citation.

## License

Code: MIT. The data file is redistributed under the terms of its original source (Our World in Data, CC BY).

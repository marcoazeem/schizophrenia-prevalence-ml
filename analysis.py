"""Reanalysis of the capstone study with leakage-free evaluation.

Target: national schizophrenia prevalence (%), country-year panel 1990-2017.
Evaluation designs:
  A. Original design reproduced: random 80/20 row split, country label-encoded as a feature.
  B. Random row split without country identifier.
  C. Country-held-out: GroupKFold (5 folds x 5 repeats) by country.
  D. Temporal holdout: train 1990-2010, test 2011-2017 (all countries seen).
  E. Country + time held out: unseen countries, future years.
"""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42
DATA = sys.argv[1]
OUT = sys.argv[2]

raw = pd.read_csv(DATA)
df = raw[raw["Code"].notna()].copy()                  # drop regional aggregates
df = df.rename(columns={"Male": "Depression, male (%)", "Female": "Depression, female (%)"})
df = df.dropna(subset=["Depression, male (%)", "Depression, female (%)"]).reset_index(drop=True)
df["log_population"] = np.log10(df["Population"])

TARGET = "Schizophrenia (%)"
DISORDERS = ["Bipolar disorder (%)", "Eating disorders (%)", "Anxiety disorders (%)",
             "Drug use disorders (%)", "Depression (%)", "Alcohol use disorders (%)"]
SEX = ["Depression, male (%)", "Depression, female (%)"]
FEATURES = DISORDERS + SEX + ["log_population", "Year"]

y = df[TARGET].values
X = df[FEATURES].values
groups = df["Entity"].values

summary = {
    "n_rows_raw": int(len(raw)), "n_rows": int(len(df)), "n_countries": int(df.Entity.nunique()),
    "years": [int(df.Year.min()), int(df.Year.max())],
    "dropped_entities": sorted(set(raw.Entity) - set(df.Entity)),
    "target_mean": float(y.mean()), "target_sd": float(y.std()),
    "target_min": float(y.min()), "target_max": float(y.max()),
}

def models():
    return {
        "Mean baseline": DummyRegressor(),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "k-NN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
        "SVR (RBF, scaled)": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.005, gamma="scale")),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED, min_samples_leaf=5),
        "Random Forest": RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1, min_samples_leaf=2),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=SEED, max_iter=400, learning_rate=0.05),
    }

def metrics(yt, yp):
    return {"R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAPE_%": float(np.mean(np.abs((yt - yp) / yt)) * 100)}

rows = []

# ---- A. original design reproduced (random split + Entity_encoded)
ent_code = LabelEncoder().fit_transform(df["Entity"])
XA = np.column_stack([X, ent_code])
Xtr, Xte, ytr, yte = train_test_split(XA, y, test_size=0.2, random_state=SEED)
for name, m in models().items():
    m.fit(Xtr, ytr); rows.append({"design": "A. Random split + country ID (original)", "model": name, **metrics(yte, m.predict(Xte))})

# ---- B. random split, no country ID
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
for name, m in models().items():
    m.fit(Xtr, ytr); rows.append({"design": "B. Random split, no country ID", "model": name, **metrics(yte, m.predict(Xte))})

# ---- C. country-held-out, repeated GroupKFold
uniq = np.unique(groups)
oof_store = {}
for name in models():
    fold_r2, fold_mae, fold_rmse, fold_mape = [], [], [], []
    oof_rep = []
    for rep in range(5):
        rng = np.random.RandomState(SEED + rep)
        perm = dict(zip(uniq, rng.permutation(len(uniq))))
        g = np.array([perm[x] for x in groups])
        oof = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g % 5 if False else g):
            m = models()[name]; m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
            mt = metrics(y[te], oof[te])
            fold_r2.append(mt["R2"]); fold_mae.append(mt["MAE"]); fold_rmse.append(mt["RMSE"]); fold_mape.append(mt["MAPE_%"])
        oof_rep.append(oof)
    oof_mean = np.mean(oof_rep, axis=0); oof_store[name] = oof_mean
    pooled = metrics(y, oof_rep[0])
    rows.append({"design": "C. Country held out (5x5 GroupKFold)", "model": name,
                 "R2": float(np.mean(fold_r2)), "R2_sd": float(np.std(fold_r2)),
                 "MAE": float(np.mean(fold_mae)), "RMSE": float(np.mean(fold_rmse)), "MAPE_%": float(np.mean(fold_mape)),
                 "R2_pooled_oof": pooled["R2"]})

# ---- D. temporal holdout
tr = df.Year <= 2010; te = df.Year > 2010
for name, m in models().items():
    m.fit(X[tr], y[tr]); rows.append({"design": "D. Temporal holdout (train <=2010, test 2011-17)", "model": name, **metrics(y[te], m.predict(X[te]))})
# persistence baseline: each country's 2010 value carried forward
last = df[df.Year == 2010].set_index("Entity")[TARGET]
pp = df.loc[te, "Entity"].map(last).values
rows.append({"design": "D. Temporal holdout (train <=2010, test 2011-17)", "model": "Persistence (country's 2010 value)", **metrics(y[te], pp)})

# ---- E. unseen countries AND future years
rs = np.random.RandomState(SEED); test_c = set(rs.choice(uniq, size=len(uniq) // 5, replace=False))
trE = (df.Year <= 2010) & (~df.Entity.isin(test_c)); teE = (df.Year > 2010) & (df.Entity.isin(test_c))
for name, m in models().items():
    m.fit(X[trE], y[trE]); rows.append({"design": "E. Unseen countries + future years", "model": name, **metrics(y[teE], m.predict(X[teE]))})

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_all_designs.csv", index=False)

# ---- Bootstrap CI (country-level) for best model under design C
def boot_ci(pred, n=2000):
    rs = np.random.RandomState(SEED); r2s = []
    by = {c: np.where(groups == c)[0] for c in uniq}
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rs.choice(uniq, len(uniq), replace=True)])
        r2s.append(r2_score(y[idx], pred[idx]))
    return [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))]
summary["bootstrap_ci_R2_C"] = {k: boot_ci(v) for k, v in oof_store.items() if k != "Mean baseline"}
np.save(f"{OUT}/oof_rf.npy", oof_store["Random Forest"])
np.save(f"{OUT}/oof_gb.npy", oof_store["Gradient Boosting"])
np.save(f"{OUT}/oof_ridge.npy", oof_store["Ridge"])

# ---- Permutation importance, country-held-out (GB and RF)
imp_rows = []
for name in ["Random Forest", "Gradient Boosting", "Ridge"]:
    for k, (tr_i, te_i) in enumerate(GroupKFold(5).split(X, y, groups)):
        m = models()[name]; m.fit(X[tr_i], y[tr_i])
        pi = permutation_importance(m, X[te_i], y[te_i], n_repeats=10, random_state=SEED, scoring="r2")
        for f, mu in zip(FEATURES, pi.importances_mean):
            imp_rows.append({"model": name, "fold": k, "feature": f, "drop_in_R2": mu})
imp = pd.DataFrame(imp_rows).groupby(["model", "feature"]).drop_in_R2.agg(["mean", "std"]).reset_index()
imp.to_csv(f"{OUT}/permutation_importance.csv", index=False)

# ---- Correlations (country-mean level and within-country)
cm = df.groupby("Entity")[[TARGET] + DISORDERS + SEX + ["log_population"]].mean()
summary["between_country_spearman"] = cm.corr(method="spearman")[TARGET].drop(TARGET).round(3).to_dict()
wc = df.groupby("Entity")[[TARGET] + DISORDERS].transform(lambda s: s - s.mean())
summary["within_country_pearson"] = wc.corr()[TARGET].drop(TARGET).round(3).to_dict()

# variance decomposition: share of target variance that is between-country
between = df.groupby("Entity")[TARGET].transform("mean")
summary["share_variance_between_country"] = float(np.var(between) / np.var(y))
summary["mean_within_country_range"] = float(df.groupby("Entity")[TARGET].agg(lambda s: s.max() - s.min()).mean())

df.assign(oof_rf=oof_store["Random Forest"], oof_gb=oof_store["Gradient Boosting"]).to_csv(f"{OUT}/clean_with_oof.csv", index=False)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
pd.set_option("display.width", 200)
print(res.round(4).to_string())
print(json.dumps(summary, indent=1))
print(imp.sort_values(["model", "mean"], ascending=[True, False]).round(4).to_string())

# ------------------------------------------------------------------
# 0) Imports  – keep everything you already had
# ------------------------------------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time, seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------------
# 1) Settings
# ------------------------------------------------------------------
correlations = np.linspace(-0.9, 0.9, 10)          # F1–F2 correlation
noise_levels = np.linspace(0, 6, 7)                # σ of additive noise
seeds        = range(42, 52)                       # 10 seeds
n_samples    = 2_000

results = []                                       # one dict per run

# ------------------------------------------------------------------
# 2) Simulation loop:  corr ▸ noise ▸ seed
# ------------------------------------------------------------------
for corr in correlations:
    cov = np.array([[1, corr], [corr, 1]])        # 2×2 covariance matrix

    for noise_sd in noise_levels:
        for seed in seeds:
            rng = np.random.default_rng(seed)

            # 2.1) Generate data ------------------------------------
            X  = rng.multivariate_normal([0, 0], cov, size=n_samples)
            eps = rng.normal(0, noise_sd, size=n_samples)          # additive noise
            Y  = (1.5 * np.sin(2 * X[:, 1])**2 +
                  0.5 * X[:, 1]**2 + eps)                          # F2 is causal

            # 2.2) Pre-processing ----------------------------------
            scaler       = StandardScaler()
            X_scaled     = scaler.fit_transform(X)
            X_scaled_cst = sm.add_constant(X_scaled)

            vif_f1      = variance_inflation_factor(X_scaled_cst, 1)
            vif_f2      = variance_inflation_factor(X_scaled_cst, 2)
            cond_number = np.linalg.cond(X_scaled_cst)

            # 2.3) Train/test split --------------------------------
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, Y, test_size=0.30, random_state=seed
            )
            X1_tr, X1_te = X_tr[:, [0]], X_te[:, [0]]
            X2_tr, X2_te = X_tr[:, [1]], X_te[:, [1]]

            # 2.4) MODELS  (same four you already use) --------------
            # ----- LinearReg --------------------------------------
            lin_both = LinearRegression().fit(X_tr,  y_tr)
            lin_f1   = LinearRegression().fit(X1_tr, y_tr)
            lin_f2   = LinearRegression().fit(X2_tr, y_tr)

            rmse_lin_test_both = np.sqrt(np.mean((y_te - lin_both.predict(X_te ))**2))
            rmse_lin_test_f1   = np.sqrt(np.mean((y_te - lin_f1  .predict(X1_te))**2))
            rmse_lin_test_f2   = np.sqrt(np.mean((y_te - lin_f2  .predict(X2_te))**2))

            # ----- SVR -------------------------------------------
            svr_both = SVR().fit(X_tr, y_tr)
            svr_f1   = SVR().fit(X1_tr, y_tr)
            svr_f2   = SVR().fit(X2_tr, y_tr)

            rmse_svr_test_both = np.sqrt(np.mean((y_te - svr_both.predict(X_te ))**2))
            rmse_svr_test_f1   = np.sqrt(np.mean((y_te - svr_f1  .predict(X1_te))**2))
            rmse_svr_test_f2   = np.sqrt(np.mean((y_te - svr_f2  .predict(X2_te))**2))

            # ----- Random Forest ----------------------------------
            rf_both = RandomForestRegressor(random_state=seed, n_jobs=3).fit(X_tr, y_tr)
            rf_f1   = RandomForestRegressor(random_state=seed, n_jobs=3).fit(X1_tr, y_tr)
            rf_f2   = RandomForestRegressor(random_state=seed, n_jobs=3).fit(X2_tr, y_tr)

            rmse_rf_test_both = np.sqrt(np.mean((y_te - rf_both.predict(X_te ))**2))
            rmse_rf_test_f1   = np.sqrt(np.mean((y_te - rf_f1  .predict(X1_te))**2))
            rmse_rf_test_f2   = np.sqrt(np.mean((y_te - rf_f2  .predict(X2_te))**2))

            # ----- MLP -------------------------------------------
            mlp_both = MLPRegressor(random_state=seed).fit(X_tr, y_tr)
            mlp_f1   = MLPRegressor(random_state=seed).fit(X1_tr, y_tr)
            mlp_f2   = MLPRegressor(random_state=seed).fit(X2_tr, y_tr)

            rmse_mlp_test_both = np.sqrt(np.mean((y_te - mlp_both.predict(X_te ))**2))
            rmse_mlp_test_f1   = np.sqrt(np.mean((y_te - mlp_f1  .predict(X1_te))**2))
            rmse_mlp_test_f2   = np.sqrt(np.mean((y_te - mlp_f2  .predict(X2_te))**2))

            # 2.5) Store everything (all your old keys + new noise)
            results.append(dict(
                corr=corr,
                noise_sd=noise_sd,
                seed=seed,
                vif_feature1=vif_f1,
                vif_feature2=vif_f2,
                cond_number=cond_number,
                rmse_lin_test_both=rmse_lin_test_both,
                rmse_lin_test_f1  =rmse_lin_test_f1,
                rmse_lin_test_f2  =rmse_lin_test_f2,
                rmse_svr_test_both=rmse_svr_test_both,
                rmse_svr_test_f1  =rmse_svr_test_f1,
                rmse_svr_test_f2  =rmse_svr_test_f2,
                rmse_rf_test_both =rmse_rf_test_both,
                rmse_rf_test_f1   =rmse_rf_test_f1,
                rmse_rf_test_f2   =rmse_rf_test_f2,
                rmse_mlp_test_both=rmse_mlp_test_both,
                rmse_mlp_test_f1  =rmse_mlp_test_f1,
                rmse_mlp_test_f2  =rmse_mlp_test_f2,
            ))

# ------------------------------------------------------------------
# 3) Dump raw results – needed by the rest of your script
# ------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("causal_vs_noncausal_noise_and_corr.csv", index=False)

# ------------------------------------------------------------------
# 4)  NEW summary & plot:  Δ median RMSE  vs noise  (all models)
# ------------------------------------------------------------------
summary = (
    df.groupby(['corr', 'noise_sd'], as_index=False)
      .median(numeric_only=True)
      .assign(
          diff_lin = lambda d: d.rmse_lin_test_f2 - d.rmse_lin_test_f1,
          diff_svr = lambda d: d.rmse_svr_test_f2 - d.rmse_svr_test_f1,
          diff_rf  = lambda d: d.rmse_rf_test_f2  - d.rmse_rf_test_f1,
          diff_mlp = lambda d: d.rmse_mlp_test_f2 - d.rmse_mlp_test_f1,
      )
)

# tidy → long
diff_long = (
    summary.melt(
        id_vars=['corr', 'noise_sd'],
        value_vars=['diff_lin', 'diff_svr', 'diff_rf', 'diff_mlp'],
        var_name='ModelKey',
        value_name='DiffRMSE'
    )
    .assign(Model=lambda d: d.ModelKey.map({
        'diff_lin':'LinearReg', 'diff_svr':'SVR',
        'diff_rf':'RF',         'diff_mlp':'MLP'
    }))
)

# nice string for facet labels
diff_long['Correlation'] = diff_long['corr'].round(2).astype(str)

sns.set_theme(style="whitegrid")
g = sns.relplot(
        data=diff_long,
        x='noise_sd', y='DiffRMSE',
        hue='Model', kind='line',
        col='Correlation', col_wrap=5, height=3, facet_kws=dict(sharey=True)
)

# add zero-line to every subplot
for ax in g.axes.flatten():
    ax.axhline(0, ls='--', lw=0.8)

g.set_axis_labels("Additive noise σ",
                  "Median RMSE(F2 only) – Median RMSE(F1 only)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Noise amplifies harm of spurious correlation\n(positive Δ ⇒ causal feature wins)")
g.savefig("8_RMSE_diff_vs_noise_all_models.png", dpi=300)
plt.close()

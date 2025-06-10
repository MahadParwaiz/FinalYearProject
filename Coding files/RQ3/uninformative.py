import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns  # for catplots

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------
# Settings & Initialization
# ----------------------------
correlations = np.linspace(-0.9, 0.9, 10)  # from -0.9 to 0.9
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
n_samples = 2000
results = []

# ----------------------------
# Loop over Correlations and Seeds
# ----------------------------
for corr in correlations:
    for seed in seeds:
        np.random.seed(seed)
        # 1) Generate Correlated Data & Target
        cov_matrix = [[1, corr], [corr, 1]]
        X = np.random.multivariate_normal(
            mean=[0, 0], cov=cov_matrix, size=n_samples
        )

        noise = np.random.normal(0,4, size=n_samples)
        Y = (
            1.5 * np.sin(2 * X[:, 1])**2 + 0.5 * X[:, 1]**2     # true driver is Feature‑2 only
        )

        # 2) Standard Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # VIF & condition number
        X_scaled_const = sm.add_constant(X_scaled)
        vif_f1 = variance_inflation_factor(X_scaled_const, 1)
        vif_f2 = variance_inflation_factor(X_scaled_const, 2)
        cond_number = np.linalg.cond(X_scaled_const)

        # 3) Train/Test split
        X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=0.3, random_state=seed
        )

        # ============ A. full‑feature models ============
        # Linear regression
        t0 = time.time()
        lr = LinearRegression().fit(X_train_scaled, Y_train)
        time_lr_both = time.time() - t0
        coef_lin_f1, coef_lin_f2 = lr.coef_
        rmse_lin_train_both = np.sqrt(
            np.mean((Y_train - lr.predict(X_train_scaled)) ** 2)
        )
        rmse_lin_test_both = np.sqrt(
            np.mean((Y_test - lr.predict(X_test_scaled)) ** 2)
        )

        # SVR
        t0 = time.time()
        svr = SVR().fit(X_train_scaled, Y_train)
        time_svr_both = time.time() - t0
        rmse_svr_train_both = np.sqrt(
            np.mean((Y_train - svr.predict(X_train_scaled)) ** 2)
        )
        rmse_svr_test_both = np.sqrt(
            np.mean((Y_test - svr.predict(X_test_scaled)) ** 2)
        )
        perm_svr = permutation_importance(
            svr, X_test_scaled, Y_test, n_repeats=10, random_state=seed, n_jobs=3
        )
        svr_imp_f1, svr_imp_f2 = perm_svr.importances_mean

        # Random forest
        t0 = time.time()
        rf = RandomForestRegressor(random_state=seed, n_jobs=3).fit(
            X_train_scaled, Y_train
        )
        time_rf_both = time.time() - t0
        rmse_rf_train_both = np.sqrt(
            np.mean((Y_train - rf.predict(X_train_scaled)) ** 2)
        )
        rmse_rf_test_both = np.sqrt(
            np.mean((Y_test - rf.predict(X_test_scaled)) ** 2)
        )
        perm_rf = permutation_importance(
            rf, X_test_scaled, Y_test, n_repeats=10, random_state=seed, n_jobs=3
        )
        rf_imp_f1, rf_imp_f2 = perm_rf.importances_mean

        # MLP
        t0 = time.time()
        mlp = MLPRegressor(random_state=seed).fit(X_train_scaled, Y_train)
        time_mlp_both = time.time() - t0
        rmse_mlp_train_both = np.sqrt(
            np.mean((Y_train - mlp.predict(X_train_scaled)) ** 2)
        )
        rmse_mlp_test_both = np.sqrt(
            np.mean((Y_test - mlp.predict(X_test_scaled)) ** 2)
        )
        perm_mlp = permutation_importance(
            mlp, X_test_scaled, Y_test, n_repeats=10, random_state=seed, n_jobs=3
        )
        mlp_imp_f1, mlp_imp_f2 = perm_mlp.importances_mean

        # ============ B. models on Feature‑1 only ============
        X1_train, X1_test = X_train_scaled[:, [0]], X_test_scaled[:, [0]]

        lr_f1 = LinearRegression().fit(X1_train, Y_train)
        rmse_lin_test_f1 = np.sqrt(
            np.mean((Y_test - lr_f1.predict(X1_test)) ** 2)
        )

        svr_f1 = SVR().fit(X1_train, Y_train)
        rmse_svr_test_f1 = np.sqrt(
            np.mean((Y_test - svr_f1.predict(X1_test)) ** 2)
        )

        rf_f1 = RandomForestRegressor(random_state=seed).fit(X1_train, Y_train)
        rmse_rf_test_f1 = np.sqrt(
            np.mean((Y_test - rf_f1.predict(X1_test)) ** 2)
        )

        mlp_f1 = MLPRegressor(random_state=seed).fit(X1_train, Y_train)
        rmse_mlp_test_f1 = np.sqrt(
            np.mean((Y_test - mlp_f1.predict(X1_test)) ** 2)
        )

        # ============ C. models on Feature‑2 only ============
        X2_train, X2_test = X_train_scaled[:, [1]], X_test_scaled[:, [1]]

        lr_f2 = LinearRegression().fit(X2_train, Y_train)
        rmse_lin_test_f2 = np.sqrt(
            np.mean((Y_test - lr_f2.predict(X2_test)) ** 2)
        )

        svr_f2 = SVR().fit(X2_train, Y_train)
        rmse_svr_test_f2 = np.sqrt(
            np.mean((Y_test - svr_f2.predict(X2_test)) ** 2)
        )

        rf_f2 = RandomForestRegressor(random_state=seed).fit(X2_train, Y_train)
        rmse_rf_test_f2 = np.sqrt(
            np.mean((Y_test - rf_f2.predict(X2_test)) ** 2)
        )

        mlp_f2 = MLPRegressor(random_state=seed).fit(X2_train, Y_train)
        rmse_mlp_test_f2 = np.sqrt(
            np.mean((Y_test - mlp_f2.predict(X2_test)) ** 2)
        )

        # ---------- store everything (SHAP columns removed) ----------
        results.append(
            dict(
                corr=corr,
                seed=seed,
                vif_feature1=vif_f1,
                vif_feature2=vif_f2,
                cond_number=cond_number,
                time_lr_both=time_lr_both,
                time_svr_both=time_svr_both,
                time_rf_both=time_rf_both,
                time_mlp_both=time_mlp_both,
                coef_lin_f1=coef_lin_f1,
                coef_lin_f2=coef_lin_f2,
                rmse_lin_test_both=rmse_lin_test_both,
                rmse_svr_test_both=rmse_svr_test_both,
                rmse_rf_test_both=rmse_rf_test_both,
                rmse_mlp_test_both=rmse_mlp_test_both,
                svr_imp_f1_both=svr_imp_f1,
                svr_imp_f2_both=svr_imp_f2,
                rf_imp_f1_both=rf_imp_f1,
                rf_imp_f2_both=rf_imp_f2,
                mlp_imp_f1_both=mlp_imp_f1,
                mlp_imp_f2_both=mlp_imp_f2,
                rmse_lin_test_f1=rmse_lin_test_f1,
                rmse_svr_test_f1=rmse_svr_test_f1,
                rmse_rf_test_f1=rmse_rf_test_f1,
                rmse_mlp_test_f1=rmse_mlp_test_f1,
                rmse_lin_test_f2=rmse_lin_test_f2,
                rmse_svr_test_f2=rmse_svr_test_f2,
                rmse_rf_test_f2=rmse_rf_test_f2,
                rmse_mlp_test_f2=rmse_mlp_test_f2,
            )
        )

# ----------------------------
# Convert to DataFrame
# ----------------------------
df = pd.DataFrame(results)
df.to_csv("causal_noncausal_correlation.csv", index=False)

sns.set_theme(style="whitegrid")

# ====================================================================
# 1) Permutation‑importance plot  (unchanged)
# ====================================================================
perm_data = []
for _, row in df.iterrows():
    for mdl, f1_key, f2_key in [
        ("SVR", "svr_imp_f1_both", "svr_imp_f2_both"),
        ("RF", "rf_imp_f1_both", "rf_imp_f2_both"),
        ("MLP", "mlp_imp_f1_both", "mlp_imp_f2_both"),
    ]:
        perm_data.append(
            dict(
                Correlation=row["corr"],
                Model=mdl,
                Feature="F1",
                PermImportance=row[f1_key],
            )
        )
        perm_data.append(
            dict(
                Correlation=row["corr"],
                Model=mdl,
                Feature="F2",
                PermImportance=row[f2_key],
            )
        )

df_perm = pd.DataFrame(perm_data)
df_perm["Correlation"] = df_perm["Correlation"].round(2).astype(str)

g = sns.catplot(
    data=df_perm,
    x="Correlation",
    y="PermImportance",
    col="Model",
    hue="Feature",
    kind="box",
    showfliers=False,
    sharey=True,
    sharex=False,
    height=4,
)
g.fig.subplots_adjust(top=0.85)
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Permutation Importance (Both Features)")
g.set_axis_labels("Correlation", "PermImportance")
g.fig.savefig("1_PermutationImportance_uninformative_script2.png")
plt.close()

# ====================================================================
# 2) RMSE plot  (unchanged)
# ====================================================================
rmse_long = []
model_keys = {
    "lin": "LinearReg",
    "svr": "SVR",
    "rf": "RF",
    "mlp": "MLP",
}
for _, row in df.iterrows():
    for short, long_name in model_keys.items():
        rmse_long.extend(
            [
                dict(
                    Correlation=row["corr"],
                    Model=long_name,
                    FeatureSet="Both",
                    RMSE=row[f"rmse_{short}_test_both"],
                ),
                dict(
                    Correlation=row["corr"],
                    Model=long_name,
                    FeatureSet="OnlyF1",
                    RMSE=row[f"rmse_{short}_test_f1"],
                ),
                dict(
                    Correlation=row["corr"],
                    Model=long_name,
                    FeatureSet="OnlyF2",
                    RMSE=row[f"rmse_{short}_test_f2"],
                ),
            ]
        )

df_rmse = pd.DataFrame(rmse_long)
df_rmse["Correlation"] = df_rmse["Correlation"].round(2).astype(str)

g = sns.catplot(
    data=df_rmse,
    x="Correlation",
    y="RMSE",
    col="Model",
    hue="FeatureSet",
    kind="box",
    showfliers=False,
    sharey=True,
    legend_out=True,
    height=4,
)
g.fig.subplots_adjust(top=0.85)
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Test RMSE vs. Correlation")
g.fig.savefig("3_RMSE_uninformative.png")
plt.close()

# ====================================================================
# 3) Training‑time plot  (unchanged)
# ====================================================================
time_long = []
for _, row in df.iterrows():
    for mdl, colnames in {
        "LinearReg": ["time_lr_both"],
        "SVR": ["time_svr_both"],
        "RF": ["time_rf_both"],
        "MLP": ["time_mlp_both"],
    }.items():
        time_long.append(
            dict(
                Correlation=row["corr"],
                Model=mdl,
                FeatureSet="Both",
                TrainTime=row[colnames[0]],
            )
        )

df_time = pd.DataFrame(time_long)
df_time["Correlation"] = df_time["Correlation"].round(2).astype(str)

g = sns.catplot(
    data=df_time,
    x="Correlation",
    y="TrainTime",
    col="Model",
    kind="box",
    showfliers=False,
    sharey=True,
    height=4,
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Training Times vs. Correlation")
g.fig.savefig("4_TrainTimes_uninformative.png")
plt.close()

# ====================================================================
# 4) VIF & condition‑number plots  (unchanged)
# ====================================================================
vif_long = []
for _, row in df.iterrows():
    vif_long.extend(
        [
            dict(Correlation=row["corr"], Metric="VIF F1", Value=row["vif_feature1"]),
            dict(Correlation=row["corr"], Metric="VIF F2", Value=row["vif_feature2"]),
        ]
    )
df_vif = pd.DataFrame(vif_long)
df_vif["Correlation"] = df_vif["Correlation"].round(2).astype(str)

g = sns.catplot(
    data=df_vif,
    x="Correlation",
    y="Value",
    col="Metric",
    kind="box",
    showfliers=False,
    sharey=True,
    height=4,
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("VIF vs. Correlation")
g.fig.savefig("5_VIF_uninformative.png")
plt.close()

cond_long = [
    dict(Correlation=r["corr"], Value=r["cond_number"]) for _, r in df.iterrows()
]
df_cond = pd.DataFrame(cond_long)
df_cond["Correlation"] = df_cond["Correlation"].round(2).astype(str)

g = sns.catplot(
    data=df_cond,
    x="Correlation",
    y="Value",
    kind="box",
    showfliers=False,
    height=4,
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Condition Number vs. Correlation")
g.fig.savefig("6_ConditionNumber_uninformative.png")
plt.close()

# ====================================================================
# 5) Linear‑coefficient plot  (unchanged)
# ====================================================================
coef_long = []
for _, row in df.iterrows():
    coef_long.extend(
        [
            dict(Correlation=row["corr"], Feature="F1", Coefficient=row["coef_lin_f1"]),
            dict(Correlation=row["corr"], Feature="F2", Coefficient=row["coef_lin_f2"]),
        ]
    )
df_coef = pd.DataFrame(coef_long)
df_coef["Correlation"] = df_coef["Correlation"].round(2).astype(str)

g = sns.catplot(
    data=df_coef,
    x="Correlation",
    y="Coefficient",
    col="Feature",
    kind="box",
    showfliers=False,
    sharey=True,
    height=4,
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Linear Model Coefficients vs. Correlation")
g.fig.savefig("7_LinearCoefficients_uninformative.png")
plt.close()

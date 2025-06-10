import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import shap  # Ensure you have installed shap (pip install shap)
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
seeds = [42,43,44,45,46,47,48,49,50,51]
n_samples = 2000
results = []

# ----------------------------
# Loop over Correlations and Seeds
# ----------------------------f
for corr in correlations:
    for seed in seeds:
        np.random.seed(seed)
        # ---------------------------------------------------
        # 1) Generate Correlated Data & Target
        # ---------------------------------------------------
        cov_matrix = [[1, corr], [corr, 1]]
        X = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n_samples)
        
        # Non-linear target + noise
        noise = np.random.normal(0, 1, size=n_samples)
        Y = (
            1.5 * (np.sin(2 * X[:, 1]))             # sine term on feature 2
          + 0.5 * X[:, 1]**3)+noise                   # quadratic on feature 2
        
        # ---------------------------------------------------
        # 2) Standard Scaling
        # ---------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # shape: (n_samples, 2)
        
        # For VIF & condition number, add constant to data
        X_scaled_const = sm.add_constant(X_scaled)
        vif_f1 = variance_inflation_factor(X_scaled_const, 1)
        vif_f2 = variance_inflation_factor(X_scaled_const, 2)
        cond_number = np.linalg.cond(X_scaled_const)
        
        # ---------------------------------------------------
        # 3) Train/Test Split (Both features)
        # ---------------------------------------------------
        X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=0.3, random_state=seed
        )
        
        # =====================================================
        # (A) MODELS TRAINED ON BOTH FEATURES
        # =====================================================
        
        # 3.A.1) Linear Regression (both features)
        start_lr = time.time()
        lr = LinearRegression()
        lr.fit(X_train_scaled, Y_train)
        train_time_lr_both = time.time() - start_lr

        # Coefficients for reference
        coef_lin_f1 = lr.coef_[0]
        coef_lin_f2 = lr.coef_[1]

        # RMSE for Linear Regression (both features)
        y_pred_train_lr = lr.predict(X_train_scaled)
        y_pred_test_lr = lr.predict(X_test_scaled)
        rmse_lin_train_both = np.sqrt(np.mean((Y_train - y_pred_train_lr)**2))
        rmse_lin_test_both = np.sqrt(np.mean((Y_test - y_pred_test_lr)**2))
        
        # 3.A.2) SVR (both features)
        start_svr = time.time()
        svr = SVR()
        svr.fit(X_train_scaled, Y_train)
        train_time_svr_both = time.time() - start_svr
        
        y_pred_train_svr = svr.predict(X_train_scaled)
        y_pred_test_svr = svr.predict(X_test_scaled)
        rmse_svr_train_both = np.sqrt(np.mean((Y_train - y_pred_train_svr)**2))
        rmse_svr_test_both = np.sqrt(np.mean((Y_test - y_pred_test_svr)**2))
        
        perm_res_svr = permutation_importance(svr, X_test_scaled, Y_test, n_repeats=10, random_state=seed, n_jobs=3)
        svr_imp_f1 = perm_res_svr.importances_mean[0]
        svr_imp_f2 = perm_res_svr.importances_mean[1]
        
        # 3.A.3) Random Forest (both features)
        start_rf = time.time()
        rf = RandomForestRegressor(random_state=seed, n_jobs=3)
        rf.fit(X_train_scaled, Y_train)
        train_time_rf_both = time.time() - start_rf
        
        y_pred_train_rf = rf.predict(X_train_scaled)
        y_pred_test_rf = rf.predict(X_test_scaled)
        rmse_rf_train_both = np.sqrt(np.mean((Y_train - y_pred_train_rf)**2))
        rmse_rf_test_both = np.sqrt(np.mean((Y_test - y_pred_test_rf)**2))
        
        perm_res_rf = permutation_importance(rf, X_test_scaled, Y_test, n_repeats=10, random_state=seed, n_jobs=3)
        rf_imp_f1 = perm_res_rf.importances_mean[0]
        rf_imp_f2 = perm_res_rf.importances_mean[1]
        
        # 3.A.4) MLP (both features)
        start_mlp = time.time()
        mlp = MLPRegressor(random_state=seed)
        mlp.fit(X_train_scaled, Y_train)
        train_time_mlp_both = time.time() - start_mlp
        
        y_pred_train_mlp = mlp.predict(X_train_scaled)
        y_pred_test_mlp = mlp.predict(X_test_scaled)
        rmse_mlp_train_both = np.sqrt(np.mean((Y_train - y_pred_train_mlp)**2))
        rmse_mlp_test_both = np.sqrt(np.mean((Y_test - y_pred_test_mlp)**2))
        
        perm_res_mlp = permutation_importance(mlp, X_test_scaled, Y_test, n_repeats=10, random_state=seed, n_jobs=3)
        mlp_imp_f1 = perm_res_mlp.importances_mean[0]
        mlp_imp_f2 = perm_res_mlp.importances_mean[1]
        
        # --- NEW: Compute SHAP Values for "Both Features" models ---
        background = X_train_scaled[np.random.choice(len(X_train_scaled), 100, replace=False)]
        
        # SVR SHAP (via KernelExplainer)
        explainer_svr = shap.KernelExplainer(svr.predict, background)
        shap_values_svr = explainer_svr.shap_values(X_test_scaled, nsamples=100)
        svr_shap_f1 = np.mean(np.abs(shap_values_svr[:, 0]))
        svr_shap_f2 = np.mean(np.abs(shap_values_svr[:, 1]))
        
        # RF SHAP (via KernelExplainer -- note: in practice you'd use TreeExplainer for speed)
        explainer_rf = shap.TreeExplainer(rf)  # Not typical usage, but we keep your original approach
        shap_values_rf = explainer_rf.shap_values(X_test_scaled)
        rf_shap_f1 = np.mean(np.abs(shap_values_rf[:, 0]))
        rf_shap_f2 = np.mean(np.abs(shap_values_rf[:, 1]))
        
        # MLP SHAP (via KernelExplainer)
        explainer_mlp = shap.KernelExplainer(mlp.predict, background)
        shap_values_mlp = explainer_mlp.shap_values(X_test_scaled, nsamples=100)
        mlp_shap_f1 = np.mean(np.abs(shap_values_mlp[:, 0]))
        mlp_shap_f2 = np.mean(np.abs(shap_values_mlp[:, 1]))
        
        # =====================================================
        # (B) MODELS TRAINED ON ONLY FEATURE 1
        # =====================================================
        X_f1_train = X_train_scaled[:, [0]]
        X_f1_test = X_test_scaled[:, [0]]
        
        # 3.B.1) Linear Regression (only feature 1)
        start_lr_f1 = time.time()
        lr_f1 = LinearRegression()
        lr_f1.fit(X_f1_train, Y_train)
        train_time_lr_f1 = time.time() - start_lr_f1

        y_pred_train_lr_f1 = lr_f1.predict(X_f1_train)
        y_pred_test_lr_f1  = lr_f1.predict(X_f1_test)
        rmse_lin_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_lr_f1)**2))
        rmse_lin_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_lr_f1)**2))

        # 3.B.2) SVR (only feature 1)
        start_svr_f1 = time.time()
        svr_f1 = SVR()
        svr_f1.fit(X_f1_train, Y_train)
        train_time_svr_f1 = time.time() - start_svr_f1
        
        y_pred_train_svr_f1 = svr_f1.predict(X_f1_train)
        y_pred_test_svr_f1  = svr_f1.predict(X_f1_test)
        rmse_svr_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_svr_f1)**2))
        rmse_svr_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_svr_f1)**2))
        
        # 3.B.3) RF (only feature 1)
        start_rf_f1 = time.time()
        rf_f1 = RandomForestRegressor(random_state=seed)
        rf_f1.fit(X_f1_train, Y_train)
        train_time_rf_f1 = time.time() - start_rf_f1
        
        y_pred_train_rf_f1 = rf_f1.predict(X_f1_train)
        y_pred_test_rf_f1  = rf_f1.predict(X_f1_test)
        rmse_rf_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_rf_f1)**2))
        rmse_rf_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_rf_f1)**2))
        
        # 3.B.4) MLP (only feature 1)
        start_mlp_f1 = time.time()
        mlp_f1 = MLPRegressor(random_state=seed)
        mlp_f1.fit(X_f1_train, Y_train)
        train_time_mlp_f1 = time.time() - start_mlp_f1
        
        y_pred_train_mlp_f1 = mlp_f1.predict(X_f1_train)
        y_pred_test_mlp_f1  = mlp_f1.predict(X_f1_test)
        rmse_mlp_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_mlp_f1)**2))
        rmse_mlp_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_mlp_f1)**2))
        
        # =====================================================
        # (C) MODELS TRAINED ON ONLY FEATURE 2
        # =====================================================
        X_f2_train = X_train_scaled[:, [1]]
        X_f2_test  = X_test_scaled[:, [1]]
        
        # 3.C.1) Linear Regression (only feature 2)
        start_lr_f2 = time.time()
        lr_f2 = LinearRegression()
        lr_f2.fit(X_f2_train, Y_train)
        train_time_lr_f2 = time.time() - start_lr_f2

        y_pred_train_lr_f2 = lr_f2.predict(X_f2_train)
        y_pred_test_lr_f2  = lr_f2.predict(X_f2_test)
        rmse_lin_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_lr_f2)**2))
        rmse_lin_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_lr_f2)**2))
        
        # 3.C.2) SVR (only feature 2)
        start_svr_f2 = time.time()
        svr_f2 = SVR()
        svr_f2.fit(X_f2_train, Y_train)
        train_time_svr_f2 = time.time() - start_svr_f2
        
        y_pred_train_svr_f2 = svr_f2.predict(X_f2_train)
        y_pred_test_svr_f2  = svr_f2.predict(X_f2_test)
        rmse_svr_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_svr_f2)**2))
        rmse_svr_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_svr_f2)**2))
        
        # 3.C.3) RF (only feature 2)
        start_rf_f2 = time.time()
        rf_f2 = RandomForestRegressor(random_state=seed)
        rf_f2.fit(X_f2_train, Y_train)
        train_time_rf_f2 = time.time() - start_rf_f2
        
        y_pred_train_rf_f2 = rf_f2.predict(X_f2_train)
        y_pred_test_rf_f2  = rf_f2.predict(X_f2_test)
        rmse_rf_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_rf_f2)**2))
        rmse_rf_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_rf_f2)**2))
        
        # 3.C.4) MLP (only feature 2)
        start_mlp_f2 = time.time()
        mlp_f2 = MLPRegressor(random_state=seed)
        mlp_f2.fit(X_f2_train, Y_train)
        train_time_mlp_f2 = time.time() - start_mlp_f2
        
        y_pred_train_mlp_f2 = mlp_f2.predict(X_f2_train)
        y_pred_test_mlp_f2  = mlp_f2.predict(X_f2_test)
        rmse_mlp_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_mlp_f2)**2))
        rmse_mlp_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_mlp_f2)**2))
        
        # ---------------------------------------------------
        # Store all outcomes in a single dictionary
        # ---------------------------------------------------
        results.append({
            'corr': corr,
            'seed': seed,
            
            # VIF & Condition Number
            'vif_feature1': vif_f1,
            'vif_feature2': vif_f2,
            'cond_number': cond_number,
            
            # Times (Both Features)
            'time_lr_both': train_time_lr_both,
            'time_svr_both': train_time_svr_both,
            'time_rf_both':  train_time_rf_both,
            'time_mlp_both': train_time_mlp_both,
            
            # LR Coeffs (Both)
            'coef_lin_f1': coef_lin_f1,
            'coef_lin_f2': coef_lin_f2,
            
            # RMSE with Both Features
            'rmse_lin_train_both': rmse_lin_train_both,
            'rmse_lin_test_both':  rmse_lin_test_both,
            'rmse_svr_train_both': rmse_svr_train_both,
            'rmse_svr_test_both':  rmse_svr_test_both,
            'rmse_rf_train_both':  rmse_rf_train_both,
            'rmse_rf_test_both':   rmse_rf_test_both,
            'rmse_mlp_train_both': rmse_mlp_train_both,
            'rmse_mlp_test_both':  rmse_mlp_test_both,
            
            # Permutation Importances (Both)
            'svr_imp_f1_both': svr_imp_f1,
            'svr_imp_f2_both': svr_imp_f2,
            'rf_imp_f1_both':  rf_imp_f1,
            'rf_imp_f2_both':  rf_imp_f2,
            'mlp_imp_f1_both': mlp_imp_f1,
            'mlp_imp_f2_both': mlp_imp_f2,
            
            # SHAP Values (Both)
            'svr_shap_f1_both': svr_shap_f1,
            'svr_shap_f2_both': svr_shap_f2,
            'rf_shap_f1_both':  rf_shap_f1,
            'rf_shap_f2_both':  rf_shap_f2,
            'mlp_shap_f1_both': mlp_shap_f1,
            'mlp_shap_f2_both': mlp_shap_f2,
            
            # RMSE with Only Feature 1
            'rmse_lin_train_f1': rmse_lin_train_f1,
            'rmse_lin_test_f1':  rmse_lin_test_f1,
            'rmse_svr_train_f1': rmse_svr_train_f1,
            'rmse_svr_test_f1':  rmse_svr_test_f1,
            'rmse_rf_train_f1':  rmse_rf_train_f1,
            'rmse_rf_test_f1':   rmse_rf_test_f1,
            'rmse_mlp_train_f1': rmse_mlp_train_f1,
            'rmse_mlp_test_f1':  rmse_mlp_test_f1,
            'time_lr_f1': train_time_lr_f1,
            'time_svr_f1': train_time_svr_f1,
            'time_rf_f1':  train_time_rf_f1,
            'time_mlp_f1': train_time_mlp_f1,
            
            # RMSE with Only Feature 2
            'rmse_lin_train_f2': rmse_lin_train_f2,
            'rmse_lin_test_f2':  rmse_lin_test_f2,
            'rmse_svr_train_f2': rmse_svr_train_f2,
            'rmse_svr_test_f2':  rmse_svr_test_f2,
            'rmse_rf_train_f2':  rmse_rf_train_f2,
            'rmse_rf_test_f2':   rmse_rf_test_f2,
            'rmse_mlp_train_f2': rmse_mlp_train_f2,
            'rmse_mlp_test_f2':  rmse_mlp_test_f2,
            'time_lr_f2': train_time_lr_f2,
            'time_svr_f2': train_time_svr_f2,
            'time_rf_f2':  train_time_rf_f2,
            'time_mlp_f2': train_time_mlp_f2,
        })

# ----------------------------
# Convert results to DataFrame
# ----------------------------
df = pd.DataFrame(results)
df.to_csv("Uninformative-informative_correlation.csv", index=False)

# --------------------------------------------------------------------
# Seaborn Settings
# --------------------------------------------------------------------
sns.set_theme(style="whitegrid")  # White grid background

# ====================================================================
# 1) PERMUTATION IMPORTANCE (Both Features Only)
#    We have: svr_imp_f1_both, svr_imp_f2_both, rf_imp_f1_both, ...
#    We'll reshape so that each row has columns:
#      - Correlation
#      - Model  (SVR, RF, MLP)
#      - Feature (F1, F2)
#      - PermImportance (float)
# ====================================================================

perm_data = []
for i, row in df.iterrows():
    # SVR

    perm_data.append({
        'Correlation': row['corr'],
        'Model': 'SVR',
        'Feature': 'F1',
        'PermImportance': row['svr_imp_f1_both']
    })
    perm_data.append({
        'Correlation': row['corr'],
        'Model': 'SVR',
        'Feature': 'F2',
        'PermImportance': row['svr_imp_f2_both']
    })
    # RF
    perm_data.append({
        'Correlation': row['corr'],
        'Model': 'RF',
        'Feature': 'F1',
        'PermImportance': row['rf_imp_f1_both']
    })
    perm_data.append({
        'Correlation': row['corr'],
        'Model': 'RF',
        'Feature': 'F2',
        'PermImportance': row['rf_imp_f2_both']
    })
    # MLP
    perm_data.append({
        'Correlation': row['corr'],
        'Model': 'MLP',
        'Feature': 'F1',
        'PermImportance': row['mlp_imp_f1_both']
    })
    perm_data.append({
        'Correlation': row['corr'],
        'Model': 'MLP',
        'Feature': 'F2',
        'PermImportance': row['mlp_imp_f2_both']
    })

df_perm = pd.DataFrame(perm_data)
df_perm['Correlation'] = df_perm['Correlation'].round(2).astype(str)
g = sns.catplot(
    data=df_perm, x='Correlation', y='PermImportance',
    col='Model', hue='Feature',
    kind='box', showfliers=False,
    sharey=True,
    sharex=False,
    height=4
)
g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Permutation Importance (Both Features)")
g.set_axis_labels("Correlation", "PermImportance")  # Optional: set axis labels if needed
g.fig.savefig("1_PermutationImportance_uninformative.png")
plt.close()

# ====================================================================
# 2) SHAP VALUES (Both Features Only)
#    Columns: svr_shap_f1_both, svr_shap_f2_both, etc.
# ====================================================================
shap_data = []
for i, row in df.iterrows():

    # SVR
    shap_data.append({
        'Correlation': row['corr'],
        'Model': 'SVR',
        'Feature': 'F1',
        'SHAP': row['svr_shap_f1_both']
    })
    shap_data.append({
        'Correlation': row['corr'],
        'Model': 'SVR',
        'Feature': 'F2',
        'SHAP': row['svr_shap_f2_both']
    })
    # RF
    shap_data.append({
        'Correlation': row['corr'],
        'Model': 'RF',
        'Feature': 'F1',
        'SHAP': row['rf_shap_f1_both']
    })
    shap_data.append({
        'Correlation': row['corr'],
        'Model': 'RF',
        'Feature': 'F2',
        'SHAP': row['rf_shap_f2_both']
    })
    # MLP
    shap_data.append({
        'Correlation': row['corr'],
        'Model': 'MLP',
        'Feature': 'F1',
        'SHAP': row['mlp_shap_f1_both']
    })
    shap_data.append({
        'Correlation': row['corr'],
        'Model': 'MLP',
        'Feature': 'F2',
        'SHAP': row['mlp_shap_f2_both']
    })

df_shap = pd.DataFrame(shap_data)
df_shap['Correlation'] = df_shap['Correlation'].round(2).astype(str)

g = sns.catplot(
    data=df_shap, x='Correlation', y='SHAP',
    col='Model', hue='Feature',
    kind='box', showfliers=False,
    sharey=True,
    height=4
)
g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Mean Absolute SHAP Values (Both Features)")
plt.tight_layout()
plt.savefig("2_SHAP_Values_uninformative.png")
plt.close()


# ====================================================================
# 3) TEST RMSE (Comparing "Both", "OnlyF1", "OnlyF2") for each Model
#    We'll gather: rmse_{model}_test_both, rmse_{model}_test_f1, rmse_{model}_test_f2
# ====================================================================
rmse_data = []
models = ['lin', 'svr', 'rf', 'mlp']
model_map = {'lin': 'LinearReg', 'svr': 'SVR', 'rf': 'RF', 'mlp': 'MLP'}

for i, row in df.iterrows():
    corr_val = row['corr']
    # For each model, gather test RMSE for (both, f1, f2)
    for m in models:
        model_label = model_map[m]
        rmse_both = row[f'rmse_{m}_test_both']
        rmse_f1   = row[f'rmse_{m}_test_f1']
        rmse_f2   = row[f'rmse_{m}_test_f2']
        
        rmse_data.append({
            'Correlation': corr_val,
            'Model': model_label,
            'FeatureSet': 'Both',
            'RMSE': rmse_both
        })
        rmse_data.append({
            'Correlation': corr_val,
            'Model': model_label,
            'FeatureSet': 'OnlyF1',
            'RMSE': rmse_f1
        })
        rmse_data.append({
            'Correlation': corr_val,
            'Model': model_label,
            'FeatureSet': 'OnlyF2',
            'RMSE': rmse_f2
        })

df_rmse = pd.DataFrame(rmse_data)
df_rmse['Correlation'] = df_rmse['Correlation'].round(2).astype(str)
g = sns.catplot(
    data=df_rmse, x='Correlation', y='RMSE',
    col='Model', hue='FeatureSet',
    kind='box', showfliers=False,
    sharey=True,
    height=4
)




g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Test RMSE vs. Correlation")
plt.tight_layout()
plt.savefig("3_RMSE_uninformative.png")
plt.close()

# ====================================================================
# 4) TRAINING TIMES
#    We have times for each model (LR, SVR, RF, MLP) and each feature set (both, f1, f2).
# ====================================================================
time_data = []
model_time_cols = {
    'LinearReg': ['time_lr_both', 'time_lr_f1', 'time_lr_f2'],
    'SVR':       ['time_svr_both', 'time_svr_f1', 'time_svr_f2'],
    'RF':        ['time_rf_both', 'time_rf_f1', 'time_rf_f2'],
    'MLP':       ['time_mlp_both', 'time_mlp_f1', 'time_mlp_f2']
}

feature_map = {0: 'Both', 1: 'OnlyF1', 2: 'OnlyF2'}

for i, row in df.iterrows():
    for model_label, col_names in model_time_cols.items():
        corr_val = row['corr']
        for idx, col_name in enumerate(col_names):
            time_data.append({
                'Correlation': corr_val,
                'Model': model_label,
                'FeatureSet': feature_map[idx],
                'TrainTime': row[col_name]
            })

df_time = pd.DataFrame(time_data)
df_time['Correlation'] = df_time['Correlation'].round(2).astype(str)
g = sns.catplot(
    data=df_time, x='Correlation', y='TrainTime',
    col='Model', hue='FeatureSet',
    kind='box', showfliers=False,
    sharey=True,
    height=4
)
g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Training Times vs. Correlation")
plt.tight_layout()
plt.savefig("4_TrainTimes_uninformative.png")
plt.close()

# --------------------------------------
# VIF & Condition Number (Separate Plots)
# --------------------------------------

# 1) Build a DataFrame that has only VIF rows
vif_data = []
for i, row in df.iterrows():
    vif_data.append({
        'Correlation': row['corr'],
        'Metric': 'VIF Feature1',
        'Value': row['vif_feature1']
    })
    vif_data.append({
        'Correlation': row['corr'],
        'Metric': 'VIF Feature2',
        'Value': row['vif_feature2']
    })

df_vif = pd.DataFrame(vif_data)
df_vif['Correlation'] = df_vif['Correlation'].round(2).astype(str)
# 2) Build a DataFrame that has only Condition Number
cond_data = []
for i, row in df.iterrows():
    cond_data.append({
        'Correlation': row['corr'],
        'Metric': 'Condition Number',
        'Value': row['cond_number']
    })

df_cond = pd.DataFrame(cond_data)
df_cond['Correlation'] = df_cond['Correlation'].round(2).astype(str)
# --- Now Plot VIF (two facets: VIF Feature1, VIF Feature2) ---
g = sns.catplot(
    data=df_vif,
    x='Correlation', y='Value',
    col='Metric',      # separate plots for VIF Feature1, VIF Feature2
    kind='box',
    showfliers=False,
    sharey=True,       # share the y-axis for the two VIF facets
    height=4
)



g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.fig.suptitle("VIF vs. Correlation")
plt.tight_layout()
plt.savefig("5_VIF_uninformative.png")
plt.close()


# --- Now Plot Condition Number (only one facet) ---
g = sns.catplot(
    data=df_cond,
    x='Correlation', y='Value',
    kind='box',
    showfliers=False,
    height=4
)

g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.fig.suptitle("Condition Number vs. Correlation")
plt.tight_layout()
plt.savefig("6_ConditionNumber_uninformative.png")
plt.close()
# --------------------------------------
# Linear Model Coefficients
# --------------------------------------
coef_data = []
for i, row in df.iterrows():
    coef_data.append({
        'Correlation': row['corr'],
        'Feature': 'F1',
        'Coefficient': row['coef_lin_f1']
    })
    coef_data.append({
        'Correlation': row['corr'],
        'Feature': 'F2',
        'Coefficient': row['coef_lin_f2']
    })

df_coef = pd.DataFrame(coef_data)
df_coef['Correlation'] = df_coef['Correlation'].round(2).astype(str)
g = sns.catplot(
    data=df_coef,
    x='Correlation', y='Coefficient',
    col='Feature',      # separate subplots for F1 and F2
    kind='box',
    showfliers=False,
    sharey=True,
    height=4
)

g.fig.subplots_adjust(top=0.85)  # Reserve top space for the title
g.fig.suptitle("Linear Model Coefficients vs. Correlation")
plt.tight_layout()
plt.savefig("7_LinearCoefficients_uninformative.png")
plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import shap 

from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Settings
correlations = np.linspace(-0.9, 0.9, 10)  # from -0.9 to 0.9
seeds = [42, 43]
n_samples = 2000

results = []

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
        Y = 2 * (X[:, 0])**3 + 2* (X[:, 1]) **3
        
        # ---------------------------------------------------
        # 2) Standard Scaling
        # ---------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # shape: (n_samples, 2)
        
        # For VIF & condition number, add constant
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
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (A) MODELS TRAINED ON BOTH FEATURES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # ---------------------------
        # 3.A.1) Linear Regression (both features)
        # ---------------------------
        start_lr = time.time()
        lr = LinearRegression()
        lr.fit(X_train_scaled, Y_train)
        train_time_lr_both = time.time() - start_lr

        # Coefficients for reference
        # Note: lr.coef_ returns an array corresponding to your features.
        coef_lin_f1 = lr.coef_[0]
        coef_lin_f2 = lr.coef_[1]

        # RMSE: train/test
        y_pred_train_lr = lr.predict(X_train_scaled)
        y_pred_test_lr = lr.predict(X_test_scaled)
        rmse_lin_train_both = np.sqrt(np.mean((Y_train - y_pred_train_lr)**2))
        rmse_lin_test_both = np.sqrt(np.mean((Y_test - y_pred_test_lr)**2))
        
        # ---------------------------
        # 3.A.2) SVR
        # ---------------------------
        start_svr = time.time()
        svr = SVR()
        svr.fit(X_train_scaled, Y_train)
        train_time_svr_both = time.time() - start_svr
        
        y_pred_train_svr = svr.predict(X_train_scaled)
        y_pred_test_svr = svr.predict(X_test_scaled)
        rmse_svr_train_both = np.sqrt(np.mean((Y_train - y_pred_train_svr)**2))
        rmse_svr_test_both = np.sqrt(np.mean((Y_test - y_pred_test_svr)**2))
        
        perm_res_svr = permutation_importance(svr, X_test_scaled, Y_test, n_repeats=10, random_state=seed)
        svr_imp_f1 = perm_res_svr.importances_mean[0]
        svr_imp_f2 = perm_res_svr.importances_mean[1]
        
        # ---------------------------
        # 3.A.3) Random Forest
        # ---------------------------
        start_rf = time.time()
        rf = RandomForestRegressor(random_state=seed)
        rf.fit(X_train_scaled, Y_train)
        train_time_rf_both = time.time() - start_rf
        
        y_pred_train_rf = rf.predict(X_train_scaled)
        y_pred_test_rf = rf.predict(X_test_scaled)
        rmse_rf_train_both = np.sqrt(np.mean((Y_train - y_pred_train_rf)**2))
        rmse_rf_test_both = np.sqrt(np.mean((Y_test - y_pred_test_rf)**2))
        
        perm_res_rf = permutation_importance(rf, X_test_scaled, Y_test, n_repeats=10, random_state=seed)
        rf_imp_f1 = perm_res_rf.importances_mean[0]
        rf_imp_f2 = perm_res_rf.importances_mean[1]
        
        # ---------------------------
        # 3.A.4) MLP
        # ---------------------------
        start_mlp = time.time()
        mlp = MLPRegressor(random_state=seed, max_iter=1000)
        mlp.fit(X_train_scaled, Y_train)
        train_time_mlp_both = time.time() - start_mlp
        
        y_pred_train_mlp = mlp.predict(X_train_scaled)
        y_pred_test_mlp = mlp.predict(X_test_scaled)
        rmse_mlp_train_both = np.sqrt(np.mean((Y_train - y_pred_train_mlp)**2))
        rmse_mlp_test_both = np.sqrt(np.mean((Y_test - y_pred_test_mlp)**2))
        
        perm_res_mlp = permutation_importance(mlp, X_test_scaled, Y_test, n_repeats=10, random_state=seed)
        mlp_imp_f1 = perm_res_mlp.importances_mean[0]
        mlp_imp_f2 = perm_res_mlp.importances_mean[1]
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (B) MODELS TRAINED ON ONLY FEATURE 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Keep only col0
        X_f1_train = X_train_scaled[:, [0]]
        X_f1_test = X_test_scaled[:, [0]]
        
        # ---------------------------
        # 3.B.1) Linear Regression (only feature 1)
        # ---------------------------
        start_lr_f1 = time.time()
        lr_f1 = LinearRegression()
        lr_f1.fit(X_f1_train, Y_train)
        train_time_lr_f1 = time.time() - start_lr_f1

        y_pred_train_lr_f1 = lr_f1.predict(X_f1_train)
        y_pred_test_lr_f1  = lr_f1.predict(X_f1_test)
        rmse_lin_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_lr_f1)**2))
        rmse_lin_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_lr_f1)**2))

        
        # 3.B.2) SVR
        start_svr_f1 = time.time()
        svr_f1 = SVR()
        svr_f1.fit(X_f1_train, Y_train)
        train_time_svr_f1 = time.time() - start_svr_f1
        
        y_pred_train_svr_f1 = svr_f1.predict(X_f1_train)
        y_pred_test_svr_f1  = svr_f1.predict(X_f1_test)
        rmse_svr_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_svr_f1)**2))
        rmse_svr_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_svr_f1)**2))
        
        # 3.B.3) RF
        start_rf_f1 = time.time()
        rf_f1 = RandomForestRegressor(random_state=seed)
        rf_f1.fit(X_f1_train, Y_train)
        train_time_rf_f1 = time.time() - start_rf_f1
        
        y_pred_train_rf_f1 = rf_f1.predict(X_f1_train)
        y_pred_test_rf_f1  = rf_f1.predict(X_f1_test)
        rmse_rf_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_rf_f1)**2))
        rmse_rf_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_rf_f1)**2))
        
        # 3.B.4) MLP
        start_mlp_f1 = time.time()
        mlp_f1 = MLPRegressor(random_state=seed, max_iter=1000)
        mlp_f1.fit(X_f1_train, Y_train)
        train_time_mlp_f1 = time.time() - start_mlp_f1
        
        y_pred_train_mlp_f1 = mlp_f1.predict(X_f1_train)
        y_pred_test_mlp_f1  = mlp_f1.predict(X_f1_test)
        rmse_mlp_train_f1 = np.sqrt(np.mean((Y_train - y_pred_train_mlp_f1)**2))
        rmse_mlp_test_f1  = np.sqrt(np.mean((Y_test - y_pred_test_mlp_f1)**2))
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (C) MODELS TRAINED ON ONLY FEATURE 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Keep only col1
        X_f2_train = X_train_scaled[:, [1]]
        X_f2_test  = X_test_scaled[:, [1]]
        
        # ---------------------------
        # 3.C.1) Linear Regression (only feature 2)
        # ---------------------------
        start_lr_f2 = time.time()
        lr_f2 = LinearRegression()
        lr_f2.fit(X_f2_train, Y_train)
        train_time_lr_f2 = time.time() - start_lr_f2

        y_pred_train_lr_f2 = lr_f2.predict(X_f2_train)
        y_pred_test_lr_f2  = lr_f2.predict(X_f2_test)
        rmse_lin_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_lr_f2)**2))
        rmse_lin_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_lr_f2)**2))

        
        # 3.C.2) SVR
        start_svr_f2 = time.time()
        svr_f2 = SVR()
        svr_f2.fit(X_f2_train, Y_train)
        train_time_svr_f2 = time.time() - start_svr_f2
        
        y_pred_train_svr_f2 = svr_f2.predict(X_f2_train)
        y_pred_test_svr_f2  = svr_f2.predict(X_f2_test)
        rmse_svr_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_svr_f2)**2))
        rmse_svr_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_svr_f2)**2))
        
        # 3.C.3) RF
        start_rf_f2 = time.time()
        rf_f2 = RandomForestRegressor(random_state=seed)
        rf_f2.fit(X_f2_train, Y_train)
        train_time_rf_f2 = time.time() - start_rf_f2
        
        y_pred_train_rf_f2 = rf_f2.predict(X_f2_train)
        y_pred_test_rf_f2  = rf_f2.predict(X_f2_test)
        rmse_rf_train_f2 = np.sqrt(np.mean((Y_train - y_pred_train_rf_f2)**2))
        rmse_rf_test_f2  = np.sqrt(np.mean((Y_test - y_pred_test_rf_f2)**2))
        
        # 3.C.4) MLP
        start_mlp_f2 = time.time()
        mlp_f2 = MLPRegressor(random_state=seed, max_iter=1000)
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
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            # Times (Both Features)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            'time_lr_both': train_time_lr_both,
            'time_svr_both': train_time_svr_both,
            'time_rf_both':  train_time_rf_both,
            'time_mlp_both': train_time_mlp_both,
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            # OLS Coeffs (Both)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            'coef_lin_f1': coef_lin_f1,
            'coef_lin_f2': coef_lin_f2,
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            # RMSE with BOTH Features
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            'rmse_lin_train_both': rmse_lin_train_both,
            'rmse_lin_test_both':  rmse_lin_test_both,
            'rmse_svr_train_both': rmse_svr_train_both,
            'rmse_svr_test_both':  rmse_svr_test_both,
            'rmse_rf_train_both':  rmse_rf_train_both,
            'rmse_rf_test_both':   rmse_rf_test_both,
            'rmse_mlp_train_both': rmse_mlp_train_both,
            'rmse_mlp_test_both':  rmse_mlp_test_both,
            
            # Feature Importances (Both)
            'svr_imp_f1_both': svr_imp_f1,
            'svr_imp_f2_both': svr_imp_f2,
            'rf_imp_f1_both':  rf_imp_f1,
            'rf_imp_f2_both':  rf_imp_f2,
            'mlp_imp_f1_both': mlp_imp_f1,
            'mlp_imp_f2_both': mlp_imp_f2,
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            # RMSE with ONLY Feature 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            'rmse_lin_train_f1': rmse_lin_train_f1,
            'rmse_lin_test_f1':  rmse_lin_test_f1,
            'rmse_svr_train_f1': rmse_svr_train_f1,
            'rmse_svr_test_f1':  rmse_svr_test_f1,
            'rmse_rf_train_f1':  rmse_rf_train_f1,
            'rmse_rf_test_f1':   rmse_rf_test_f1,
            'rmse_mlp_train_f1': rmse_mlp_train_f1,
            'rmse_mlp_test_f1':  rmse_mlp_test_f1,
            
            # Train Times (Only F1)
            'time_lr_f1': train_time_lr_f1,
            'time_svr_f1': train_time_svr_f1,
            'time_rf_f1':  train_time_rf_f1,
            'time_mlp_f1': train_time_mlp_f1,
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            # RMSE with ONLY Feature 2
            # ~~~~~~~~~~~~~~~~~~~~~~~~~
            'rmse_lin_train_f2': rmse_lin_train_f2,
            'rmse_lin_test_f2':  rmse_lin_test_f2,
            'rmse_svr_train_f2': rmse_svr_train_f2,
            'rmse_svr_test_f2':  rmse_svr_test_f2,
            'rmse_rf_train_f2':  rmse_rf_train_f2,
            'rmse_rf_test_f2':   rmse_rf_test_f2,
            'rmse_mlp_train_f2': rmse_mlp_train_f2,
            'rmse_mlp_test_f2':  rmse_mlp_test_f2,
            
            # Train Times (Only F2)
            'time_lr_f2': train_time_lr_f2,
            'time_svr_f2': train_time_svr_f2,
            'time_rf_f2':  train_time_rf_f2,
            'time_mlp_f2': train_time_mlp_f2,
        })

# Convert to DataFrame
df = pd.DataFrame(results)

# ------------------------------------------------------------------------------
# Plots for VIF, Condition #, Coeffs, Perm. Importances remain the same,
# except now some column names have changed (like rmse_lin_test_both) etc.
# ------------------------------------------------------------------------------
unique_corrs = sorted(df['corr'].unique())
offset = 0.05

# 1) VIF & Condition Number
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
vif1_data, vif2_data = [], []
positions1, positions2 = [], []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    vif1_data.append(slice_df['vif_feature1'].tolist())
    vif2_data.append(slice_df['vif_feature2'].tolist())
    positions1.append(c - offset)
    positions2.append(c + offset)
bp1 = ax.boxplot(vif1_data, positions=positions1, widths=0.03, patch_artist=True)
bp2 = ax.boxplot(vif2_data, positions=positions2, widths=0.03, patch_artist=True)
for box in bp1['boxes']:
    box.set(facecolor='red')
for box in bp2['boxes']:
    box.set(facecolor='blue')
ax.set_xlabel('Correlation')
ax.set_ylabel('VIF')
ax.set_title('VIF vs. Correlation')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax2 = axes[1]
cond_data = []
pos_cond = []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    cond_data.append(slice_df['cond_number'].tolist())
    pos_cond.append(c)
bp_cond = ax2.boxplot(cond_data, positions=pos_cond, widths=0.05, patch_artist=True)
for box in bp_cond['boxes']:
    box.set(facecolor='green')
ax2.set_xlabel('Correlation')
ax2.set_ylabel('Condition Number')
ax2.set_title('Condition Number vs. Correlation')
ax2.set_xticks(unique_corrs)
ax2.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 2) Coefficients of Linear Model
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
ax1 = axs[0]
coef_f1_data = []
pos_f1 = []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    coef_f1_data.append(slice_df['coef_lin_f1'].tolist())
    pos_f1.append(c)
bp_f1 = ax1.boxplot(coef_f1_data, positions=pos_f1, widths=0.03, patch_artist=True)
for box in bp_f1['boxes']:
    box.set(facecolor='red')
ax1.set_xlabel('Correlation')
ax1.set_ylabel('Coefficient (Feature 1)')
ax1.set_title('Linear Model: Coef(Feature 1) vs. Correlation')
ax1.set_xticks(unique_corrs)
ax1.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2 = axs[1]
coef_f2_data = []
pos_f2 = []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    coef_f2_data.append(slice_df['coef_lin_f2'].tolist())
    pos_f2.append(c)
bp_f2 = ax2.boxplot(coef_f2_data, positions=pos_f2, widths=0.03, patch_artist=True)
for box in bp_f2['boxes']:
    box.set(facecolor='blue')
ax2.set_xlabel('Correlation')
ax2.set_ylabel('Coefficient (Feature 2)')
ax2.set_title('Linear Model: Coef(Feature 2) vs. Correlation')
ax2.set_xticks(unique_corrs)
ax2.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 3) Permutation Importances for (Both-Features) Models
#    We'll do subplots for SVR, RF, MLP as an example
fig, axs = plt.subplots(1, 3, figsize=(21, 6))
offset = 0.03

# 3A) SVR
ax = axs[0]
svr_f1_data, svr_f2_data = [], []
pos_svr_f1, pos_svr_f2 = [], []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    svr_f1_data.append(slice_df['svr_imp_f1_both'].tolist())
    svr_f2_data.append(slice_df['svr_imp_f2_both'].tolist())
    pos_svr_f1.append(c - offset)
    pos_svr_f2.append(c + offset)
bp_svr_f1 = ax.boxplot(svr_f1_data, positions=pos_svr_f1, widths=0.02, patch_artist=True)
bp_svr_f2 = ax.boxplot(svr_f2_data, positions=pos_svr_f2, widths=0.02, patch_artist=True)
for box in bp_svr_f1['boxes']:
    box.set(facecolor='red')
for box in bp_svr_f2['boxes']:
    box.set(facecolor='blue')
ax.set_xlabel('Correlation')
ax.set_ylabel('Permutation Importance')
ax.set_title('SVR Importances (Both Features)')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 3B) RF
ax = axs[1]
rf_f1_data, rf_f2_data = [], []
pos_rf_f1, pos_rf_f2 = [], []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    rf_f1_data.append(slice_df['rf_imp_f1_both'].tolist())
    rf_f2_data.append(slice_df['rf_imp_f2_both'].tolist())
    pos_rf_f1.append(c - offset)
    pos_rf_f2.append(c + offset)
bp_rf_f1 = ax.boxplot(rf_f1_data, positions=pos_rf_f1, widths=0.02, patch_artist=True)
bp_rf_f2 = ax.boxplot(rf_f2_data, positions=pos_rf_f2, widths=0.02, patch_artist=True)
for box in bp_rf_f1['boxes']:
    box.set(facecolor='red')
for box in bp_rf_f2['boxes']:
    box.set(facecolor='blue')
ax.set_xlabel('Correlation')
ax.set_ylabel('Permutation Importance')
ax.set_title('RF Importances (Both Features)')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 3C) MLP
ax = axs[2]
mlp_f1_data, mlp_f2_data = [], []
pos_mlp_f1, pos_mlp_f2 = [], []
for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    mlp_f1_data.append(slice_df['mlp_imp_f1_both'].tolist())
    mlp_f2_data.append(slice_df['mlp_imp_f2_both'].tolist())
    pos_mlp_f1.append(c - offset)
    pos_mlp_f2.append(c + offset)
bp_mlp_f1 = ax.boxplot(mlp_f1_data, positions=pos_mlp_f1, widths=0.02, patch_artist=True)
bp_mlp_f2 = ax.boxplot(mlp_f2_data, positions=pos_mlp_f2, widths=0.02, patch_artist=True)
for box in bp_mlp_f1['boxes']:
    box.set(facecolor='red')
for box in bp_mlp_f2['boxes']:
    box.set(facecolor='blue')
ax.set_xlabel('Correlation')
ax.set_ylabel('Permutation Importance')
ax.set_title('MLP Importances (Both Features)')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 4) RMSE for "Both Features" in a single subplot
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
offset = 0.02

rmse_lin_test_both_data  = []
rmse_svr_test_both_data  = []
rmse_rf_test_both_data   = []
rmse_mlp_test_both_data  = []
pos_lin, pos_svr, pos_rf, pos_mlp = [], [], [], []

for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    rmse_lin_test_both_data.append(slice_df['rmse_lin_test_both'].tolist())
    rmse_svr_test_both_data.append(slice_df['rmse_svr_test_both'].tolist())
    rmse_rf_test_both_data.append(slice_df['rmse_rf_test_both'].tolist())
    rmse_mlp_test_both_data.append(slice_df['rmse_mlp_test_both'].tolist())
    
    pos_lin.append(c - 3*offset)
    pos_svr.append(c - offset)
    pos_rf.append(c + offset)
    pos_mlp.append(c + 3*offset)

bp_lin = ax.boxplot(rmse_lin_test_both_data, positions=pos_lin, widths=0.015, patch_artist=True)
bp_svr = ax.boxplot(rmse_svr_test_both_data, positions=pos_svr, widths=0.015, patch_artist=True)
bp_rf  = ax.boxplot(rmse_rf_test_both_data,  positions=pos_rf,  widths=0.015, patch_artist=True)
bp_mlp = ax.boxplot(rmse_mlp_test_both_data, positions=pos_mlp, widths=0.015, patch_artist=True)

for box in bp_lin['boxes']:
    box.set(facecolor='orange')
for box in bp_svr['boxes']:
    box.set(facecolor='tomato')
for box in bp_rf['boxes']:
    box.set(facecolor='lightblue')
for box in bp_mlp['boxes']:
    box.set(facecolor='lightgreen')

ax.set_title('Test RMSE (Both Features) vs. Correlation')
ax.set_xlabel('Correlation')
ax.set_ylabel('RMSE')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax.legend([bp_lin["boxes"][0], bp_svr["boxes"][0], bp_rf["boxes"][0], bp_mlp["boxes"][0]],
          ['Linear', 'SVR', 'RF', 'MLP'], loc='best')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5) Additional plot: RMSE with "feature removal"
#    We show test RMSE for:
#       - Only F1
#       - Only F2
#       - Both
#    for each correlation, for each model. Box-plot style.

models = ['lin', 'svr', 'rf', 'mlp']  # short names
colors = ['orange', 'tomato', 'lightblue', 'lightgreen']
fig, axs = plt.subplots(1, 4, figsize=(21, 6))
# We'll do one subplot per model for clarity, showing
#  - RMSE with OnlyF1
#  - RMSE with OnlyF2
#  - RMSE with Both
# in separate box groups.

for i, model in enumerate(models):
    ax = axs[i if len(models) > 1 else 0] if len(models)>1 else axs
    data_f1 = []
    data_f2 = []
    data_both = []
    pos_f1, pos_f2, pos_both = [], [], []
    
    for c in unique_corrs:
        slice_df = df[df['corr'] == c]
        # E.g. if model=='lin', we gather slice_df['rmse_lin_test_f1'] ...
        rmse_f1_col  = f'rmse_{model}_test_f1'
        rmse_f2_col  = f'rmse_{model}_test_f2'
        rmse_both_col= f'rmse_{model}_test_both'
        
        data_f1.append(slice_df[rmse_f1_col].tolist())
        data_f2.append(slice_df[rmse_f2_col].tolist())
        data_both.append(slice_df[rmse_both_col].tolist())
        
        # offset them around c
        pos_f1.append(c - 2*offset)
        pos_f2.append(c)
        pos_both.append(c + 2*offset)
    
    bp_f1   = ax.boxplot(data_f1,   positions=pos_f1,   widths=0.015, patch_artist=True)
    bp_f2   = ax.boxplot(data_f2,   positions=pos_f2,   widths=0.015, patch_artist=True)
    bp_both = ax.boxplot(data_both, positions=pos_both, widths=0.015, patch_artist=True)
    
    # color them
    for box in bp_f1['boxes']:
        box.set(facecolor='pink')
    for box in bp_f2['boxes']:
        box.set(facecolor='lightgray')
    for box in bp_both['boxes']:
        box.set(facecolor=colors[i])  # color from above list
    
    ax.set_title(f'{model.upper()} Test RMSE (Both vs. OnlyF1 vs. OnlyF2)')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('RMSE')
    ax.set_xticks(unique_corrs)
    ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.legend([bp_f1["boxes"][0], bp_f2["boxes"][0], bp_both["boxes"][0]],
              ['Only F1', 'Only F2', 'Both'],
              loc='best')

plt.tight_layout()
plt.show()

# 6) Train Times (Both Features)
#    We can do a simple bar or boxplot showing training times vs. correlation
models_time_cols_both = ['time_lr_both', 'time_svr_both', 'time_rf_both', 'time_mlp_both']
model_time_labels = ['LR', 'SVR', 'RF', 'MLP']

fig, ax = plt.subplots(1,1, figsize=(9,6))
offset = 0.03

positions_ols = []
positions_svr = []
positions_rf  = []
positions_mlp = []
ols_data, svr_data, rf_data, mlp_data = [], [], [], []

for c in unique_corrs:
    slice_df = df[df['corr'] == c]
    ols_data.append(slice_df['time_lr_both'].tolist())
    svr_data.append(slice_df['time_svr_both'].tolist())
    rf_data.append(slice_df['time_rf_both'].tolist())
    mlp_data.append(slice_df['time_mlp_both'].tolist())
    
    positions_ols.append(c - 3*offset)
    positions_svr.append(c - offset)
    positions_rf.append(c + offset)
    positions_mlp.append(c + 3*offset)

bp_ols = ax.boxplot(ols_data, positions=positions_ols, widths=0.015, patch_artist=True)
bp_svr = ax.boxplot(svr_data, positions=positions_svr, widths=0.015, patch_artist=True)
bp_rf  = ax.boxplot(rf_data,  positions=positions_rf,  widths=0.015, patch_artist=True)
bp_mlp = ax.boxplot(mlp_data, positions=positions_mlp, widths=0.015, patch_artist=True)

for box in bp_ols['boxes']:
    box.set(facecolor='lightgray')
for box in bp_svr['boxes']:
    box.set(facecolor='orange')
for box in bp_rf['boxes']:
    box.set(facecolor='lightblue')
for box in bp_mlp['boxes']:
    box.set(facecolor='lightgreen')

ax.set_title('Training Times vs. Correlation (Both Features)')
ax.set_xlabel('Correlation')
ax.set_ylabel('Train Time (seconds)')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{x:.2f}" for x in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.legend([bp_ols["boxes"][0], bp_svr["boxes"][0], bp_rf["boxes"][0], bp_mlp["boxes"][0]],
          model_time_labels, loc='best')
plt.tight_layout()
plt.show()

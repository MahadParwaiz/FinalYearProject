import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Settings: correlation values, two random seeds, and number of samples
correlations = np.linspace(-0.9, 0.9, 10)  # from -0.9 to 0.9
seeds = [42, 43]
n_samples = 2000

# List to store results
results = []

for corr in correlations:
    for seed in seeds:
        np.random.seed(seed)
        # Generate two features with a specified correlation
        cov_matrix = [[1, corr], [corr, 1]]
        X = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n_samples)
        
        # Construct target variable Y with a known non-linear relation plus noise
        # (Previously you defined noise but never used it in Y. Let's add it properly.)
        noise = np.random.normal(0, 1, size=n_samples)
        Y = 2 * (X[:, 0])**2 + 2 * (X[:, 1])**2 
        
        # Create a train/test split (70/30)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=seed
        )
        
        # ============= LINEAR MODEL ============
        # Fit linear model on training data
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        model = sm.OLS(Y_train, X_train_const).fit()
        
        # Compute VIF and condition number using the full design matrix
        X_const_full = sm.add_constant(X)
        vif_feature1 = variance_inflation_factor(X_const_full, 1)
        vif_feature2 = variance_inflation_factor(X_const_full, 2)
        cond_number = np.linalg.cond(X_const_full)
        
        # Compute train & test RMSE for linear model
        y_pred_linear_train = model.predict(X_train_const)
        y_pred_linear_test = model.predict(X_test_const)
        rmse_linear_train = np.sqrt(np.mean((Y_train - y_pred_linear_train) ** 2))
        rmse_linear_test = np.sqrt(np.mean((Y_test - y_pred_linear_test) ** 2))
        
        # ============= TREE-BASED MODEL ============
        tree = DecisionTreeRegressor(random_state=seed)
        tree.fit(X_train, Y_train)
        
        # Compute permutation importance on the test set
        perm_results = permutation_importance(
            tree, X_test, Y_test, n_repeats=10, random_state=seed
        )
        perm_imp_feature1 = perm_results.importances_mean[0]
        perm_imp_feature2 = perm_results.importances_mean[1]
        
        # Compute train & test RMSE for the tree-based model
        y_pred_tree_train = tree.predict(X_train)
        y_pred_tree_test = tree.predict(X_test)
        rmse_tree_train = np.sqrt(np.mean((Y_train - y_pred_tree_train) ** 2))
        rmse_tree_test = np.sqrt(np.mean((Y_test - y_pred_tree_test) ** 2))
        
        # Store everything for later plotting
        results.append({
            'corr': corr,
            'seed': seed,
            'vif_feature1': vif_feature1,
            'vif_feature2': vif_feature2,
            'cond_number': cond_number,
            'perm_imp_feature1': perm_imp_feature1,
            'perm_imp_feature2': perm_imp_feature2,
            'rmse_linear_train': rmse_linear_train,
            'rmse_linear_test': rmse_linear_test,
            'rmse_tree_train': rmse_tree_train,
            'rmse_tree_test': rmse_tree_test
        })

# Convert results to DataFrame
df = pd.DataFrame(results)
unique_corrs = sorted(df['corr'].unique())
offset = 0.05  # offset to separate box plots for each metric/model

# -------------------------------------------
# Plotting Linear Model Metrics: VIF and Condition Number
# -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: VIF for both features
ax = axes[0]
vif1_data, vif2_data = [], []
positions1, positions2 = [], []
for corr in unique_corrs:
    data_corr = df[df['corr'] == corr]
    vif1_data.append(data_corr['vif_feature1'].tolist())
    vif2_data.append(data_corr['vif_feature2'].tolist())
    positions1.append(corr - offset)
    positions2.append(corr + offset)
bp1 = ax.boxplot(vif1_data, positions=positions1, widths=0.03, patch_artist=True)
bp2 = ax.boxplot(vif2_data, positions=positions2, widths=0.03, patch_artist=True)
# Set colors: red for Feature 1, blue for Feature 2
for box in bp1['boxes']:
    box.set(facecolor='red')
for box in bp2['boxes']:
    box.set(facecolor='blue')
ax.set_xlabel('Correlation between features')
ax.set_ylabel('VIF')
ax.set_title('VIF for Features vs. Correlation')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{corr:.1f}" for corr in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Feature 1', 'Feature 2'])

# Subplot 2: Condition Number
ax = axes[1]
cond_data, positions = [], []
for corr in unique_corrs:
    data_corr = df[df['corr'] == corr]
    cond_data.append(data_corr['cond_number'].tolist())
    positions.append(corr)
bp = ax.boxplot(cond_data, positions=positions, widths=0.05, patch_artist=True)
for box in bp['boxes']:
    box.set(facecolor='green')
ax.set_xlabel('Correlation between features')
ax.set_ylabel('Condition Number')
ax.set_title('Condition Number vs. Correlation')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{corr:.1f}" for corr in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------------------
# Plotting Tree-Based Model Metrics: Permutation Importance
# -------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
perm1_data, perm2_data = [], []
positions1, positions2 = [], []
for corr in unique_corrs:
    data_corr = df[df['corr'] == corr]
    perm1_data.append(data_corr['perm_imp_feature1'].tolist())
    perm2_data.append(data_corr['perm_imp_feature2'].tolist())
    positions1.append(corr - offset)
    positions2.append(corr + offset)
bp1 = ax.boxplot(perm1_data, positions=positions1, widths=0.03, patch_artist=True)
bp2 = ax.boxplot(perm2_data, positions=positions2, widths=0.03, patch_artist=True)
for box in bp1['boxes']:
    box.set(facecolor='red')
for box in bp2['boxes']:
    box.set(facecolor='blue')
ax.set_xlabel('Correlation between features')
ax.set_ylabel('Permutation Importance')
ax.set_title('Permutation Importance vs. Correlation')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{corr:.1f}" for corr in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Feature 1', 'Feature 2'])
plt.tight_layout()
plt.show()

# -------------------------------------------
# Plotting RMSE: Train and Test for Both Models
# -------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

rmse_linear_train_data = []
rmse_linear_test_data = []
rmse_tree_train_data = []
rmse_tree_test_data = []

positions_lin_train = []
positions_lin_test = []
positions_tree_train = []
positions_tree_test = []

for corr in unique_corrs:
    data_corr = df[df['corr'] == corr]
    
    rmse_linear_train_data.append(data_corr['rmse_linear_train'].tolist())
    rmse_linear_test_data.append(data_corr['rmse_linear_test'].tolist())
    rmse_tree_train_data.append(data_corr['rmse_tree_train'].tolist())
    rmse_tree_test_data.append(data_corr['rmse_tree_test'].tolist())
    
    # Shift positions so they don't overlap
    # We'll place them close together around 'corr'
    positions_lin_train.append(corr - 3*offset)
    positions_lin_test.append(corr - offset)
    positions_tree_train.append(corr + offset)
    positions_tree_test.append(corr + 3*offset)

# Box plot for each
bp_lin_train = ax.boxplot(rmse_linear_train_data, positions=positions_lin_train,
                          widths=0.03, patch_artist=True)
bp_lin_test = ax.boxplot(rmse_linear_test_data, positions=positions_lin_test,
                         widths=0.03, patch_artist=True)
bp_tree_train = ax.boxplot(rmse_tree_train_data, positions=positions_tree_train,
                           widths=0.03, patch_artist=True)
bp_tree_test = ax.boxplot(rmse_tree_test_data, positions=positions_tree_test,
                          widths=0.03, patch_artist=True)

# Set colors for each group
for box in bp_lin_train['boxes']:
    box.set(facecolor='orange')
for box in bp_lin_test['boxes']:
    box.set(facecolor='red')
for box in bp_tree_train['boxes']:
    box.set(facecolor='lightblue')
for box in bp_tree_test['boxes']:
    box.set(facecolor='blue')

ax.set_xlabel('Correlation between features')
ax.set_ylabel('RMSE')
ax.set_title('Train/Test RMSE for Linear & Tree Models vs. Correlation')
ax.set_xticks(unique_corrs)
ax.set_xticklabels([f"{corr:.1f}" for corr in unique_corrs])
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.legend(
    [bp_lin_train["boxes"][0], bp_lin_test["boxes"][0],
     bp_tree_train["boxes"][0], bp_tree_test["boxes"][0]],
    ['Linear-Train', 'Linear-Test', 'Tree-Train', 'Tree-Test'],
    loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2
)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
n_samples = 100
true_coef = 2  # true effect for X1
noise_scale = 1  # noise standard deviation for outcome
n_steps = 20
corr_values = np.linspace(0.0, 0.99, n_steps)  # from no collinearity to high collinearity

# Lists to store results for the original simulation
coef_X1 = []
coef_X2 = []
se_X1 = []
se_X2 = []
rmse_list = []
vif_X1 = []
vif_X2 = []
sample_corr = []  # list to store the computed sample correlation between X1 and X2

# Loop over different correlation levels
for r in corr_values:
    # Generate X1 as a standard normal variable
    X1 = np.random.normal(0, 1, n_samples)
    # Generate X2 such that its correlation with X1 is approximately r
    noise_X2 = np.random.normal(0, 1, n_samples)
    X2 = r * X1 + np.sqrt(1 - r**2) * noise_X2

    # Compute the actual sample correlation between X1 and X2
    computed_corr = np.corrcoef(X1, X2)[0, 1]
    sample_corr.append(computed_corr)

    # Generate outcome variable y, affected only by X1 (plus some noise)
    y = true_coef * X1 + np.random.normal(0, noise_scale, n_samples)
    
    # Create DataFrame for statsmodels
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
    
    # Prepare predictors with constant
    X = sm.add_constant(df[['X1', 'X2']])
    model = sm.OLS(y, X).fit()
    
    # Store coefficient estimates and standard errors
    coef_X1.append(model.params['X1'])
    coef_X2.append(model.params['X2'])
    se_X1.append(model.bse['X1'])
    se_X2.append(model.bse['X2'])
    
    # Calculate RMSE
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    rmse_list.append(rmse)
    
    # Calculate VIF for predictors (without constant)
    X_no_const = df[['X1', 'X2']]
    vif_temp = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]
    vif_X1.append(vif_temp[0])
    vif_X2.append(vif_temp[1])

# Create a DataFrame of the results for convenience
results = pd.DataFrame({
    'Intended Correlation': corr_values,
    'Sample Correlation': sample_corr,
    'Coef_X1': coef_X1,
    'Coef_X2': coef_X2,
    'SE_X1': se_X1,
    'SE_X2': se_X2,
    'RMSE': rmse_list,
    'VIF_X1': vif_X1,
    'VIF_X2': vif_X2
})

# Calculate the absolute change in coefficient for X1 relative to the true coefficient
results['Coef_Change_X1'] = np.abs(results['Coef_X1'] - true_coef)

# New simulation for the extra subplot: effect of random seed on coefficient estimates
seeds = list(range(10))
coef_X1_r0 = []    # X1 coefficient when r = 0
intercept_r0 = []  # Intercept ("X0") when r = 0
coef_X1_r05 = []   # X1 coefficient when r = 0.5
intercept_r05 = [] # Intercept ("Z0") when r = 0.5

for seed in seeds:
    # --- Simulation for r = 0 (zero correlation) ---
    np.random.seed(seed)
    r = 0.0
    X1 = np.random.normal(0, 1, n_samples)
    noise_X2 = np.random.normal(0, 1, n_samples)
    X2 = r * X1 + np.sqrt(1 - r**2) * noise_X2
    y = true_coef * X1 + np.random.normal(0, noise_scale, n_samples)
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
    X = sm.add_constant(df[['X1', 'X2']])
    model = sm.OLS(y, X).fit()
    intercept_r0.append(model.params['const'])
    coef_X1_r0.append(model.params['X1'])
    
    # --- Simulation for r = 0.5 ---
    np.random.seed(seed)  # reset seed to have comparable randomness
    r = 0.8
    X1 = np.random.normal(0, 1, n_samples)
    noise_X2 = np.random.normal(0, 1, n_samples)
    X2 = r * X1 + np.sqrt(1 - r**2) * noise_X2
    y = true_coef * X1 + np.random.normal(0, noise_scale, n_samples)
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
    X = sm.add_constant(df[['X1', 'X2']])
    model = sm.OLS(y, X).fit()
    intercept_r05.append(model.params['const'])
    coef_X1_r05.append(model.params['X1'])

# Plotting the results in a 2x3 grid of subplots
plt.figure(figsize=(18, 10))

# 1. Coefficient estimates vs. Intended Correlation
plt.subplot(2, 3, 1)
plt.plot(results['Intended Correlation'], results['Coef_X1'], marker='o', label='Coef for X1')
plt.plot(results['Intended Correlation'], results['Coef_X2'], marker='s', label='Coef for X2')
plt.xlabel('Intended Correlation between X1 and X2')
plt.ylabel('Coefficient Estimate')
plt.title('Coefficient Estimates')
plt.legend()

# 2. VIF values vs. Intended Correlation
plt.subplot(2, 3, 2)
plt.plot(results['Intended Correlation'], results['VIF_X1'], marker='o', label='VIF for X1')
plt.plot(results['Intended Correlation'], results['VIF_X2'], marker='s', label='VIF for X2')
plt.xlabel('Intended Correlation between X1 and X2')
plt.ylabel('Variance Inflation Factor')
plt.title('VIF vs. Correlation')
plt.legend()

# 3. RMSE vs. Intended Correlation
plt.subplot(2, 3, 3)
plt.plot(results['Intended Correlation'], results['RMSE'], marker='o', color='purple')
plt.xlabel('Intended Correlation between X1 and X2')
plt.ylabel('RMSE')
plt.title('RMSE vs. Correlation')

# 4. Direct Correlation Analysis: Sample vs. Intended Correlation
plt.subplot(2, 3, 4)
plt.plot(results['Intended Correlation'], results['Sample Correlation'], marker='o', color='orange')
plt.xlabel('Intended Correlation')
plt.ylabel('Sample Correlation')
plt.title('Sample vs. Intended Correlation')

# 5. Coefficient Sensitivity: Absolute Change in X1 Coefficient vs. Sample Correlation
plt.subplot(2, 3, 5)
plt.plot(results['Sample Correlation'], results['Coef_Change_X1'], marker='o', color='green')
plt.xlabel('Sample Correlation')
plt.ylabel('Absolute Change in Coef for X1')
plt.title('Coefficient Change vs. Sample Correlation')

# 6. New subplot: Coefficient Estimates vs. Random Seed for two fixed correlation scenarios
plt.subplot(2, 3, 6)
plt.plot(seeds, coef_X1_r0, marker='o', label='X1 (r=0)')
plt.plot(seeds, intercept_r0, marker='s', label='X0 (r=0)')
plt.plot(seeds, coef_X1_r05, marker='o', linestyle='--', label='X1 (r=0.5)')
plt.plot(seeds, intercept_r05, marker='s', linestyle='--', label='Z0 (r=0.5)')
plt.xlabel('Random Seed')
plt.ylabel('Coefficient Estimate')
plt.title('Coefficient Estimates vs. Random Seed')
plt.legend()

plt.tight_layout()
plt.show()

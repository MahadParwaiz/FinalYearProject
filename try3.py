import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error

# ---------------------------
# Parameters for Simulation
# ---------------------------
n_samples = 100
true_coef = 2           # True effect for X1
noise_scale = 1         # Standard deviation for noise in y
n_steps = 20
corr_values = np.linspace(0.0, 0.99, n_steps)  # Intended correlation values between X1 and X2
seeds = list(range(5))  # Use 5 different random seeds for each intended correlation

# ---------------------------
# Simulation over correlations and seeds
# ---------------------------
results_list = []

for r in corr_values:
    for seed in seeds:
        np.random.seed(seed)
        
        # Generate X1 as a standard normal variable
        X1 = np.random.normal(0, 1, n_samples)
        noise_X2 = np.random.normal(0, 1, n_samples)
        # Generate X2 with controlled correlation
        X2 = r * X1 + np.sqrt(1 - r**2) * noise_X2
        
        # Generate outcome y (only X1 influences y)
        y = true_coef * X1 + np.random.normal(0, noise_scale, n_samples)
        
        # Create DataFrame and add constant for intercept
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
        X = sm.add_constant(df[['X1', 'X2']])
        model = sm.OLS(y, X).fit()
        
        # Calculate predictions and RMSE
        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        # Calculate VIF for predictors (without constant)
        X_no_const = df[['X1', 'X2']]
        vif_vals = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]
        
        # Store results for this run
        results_list.append({
            'Intended Correlation': r,
            'Coef_X1': model.params['X1'],
            'Coef_X2': model.params['X2'],
            'RMSE': rmse,
            'VIF_X1': vif_vals[0],
            'VIF_X2': vif_vals[1],
            'Seed': seed
        })

# Convert list of results to a DataFrame
results_df = pd.DataFrame(results_list)

# Create a categorical label for intended correlation (rounded to 2 decimals) for plotting
results_df['Corr_label'] = results_df['Intended Correlation'].round(2).astype(str)

# ---------------------------
# Plot 1: Box Plot of Coefficient Estimates vs. Intended Correlation using Seaborn Catplot
# ---------------------------
melted_results = results_df.melt(id_vars=['Intended Correlation','Corr_label', 'Seed'],
                                 value_vars=['Coef_X1', 'Coef_X2'], 
                                 var_name='Coefficient', value_name='Estimate')

sns.set(style="whitegrid")
catplot_coef = sns.catplot(x='Corr_label', y='Estimate', hue='Coefficient', data=melted_results,
                           kind='box', height=6, aspect=1.5)
catplot_coef.set_axis_labels("Intended Correlation", "Coefficient Estimate")
catplot_coef.fig.suptitle("Coefficient Estimates vs. Intended Correlation", y=1.03)
plt.show()

# ---------------------------
# Plot 2: Box Plot of VIF vs. Intended Correlation using Seaborn Catplot
# ---------------------------
melted_vif = results_df.melt(id_vars=['Intended Correlation', 'Corr_label', 'Seed'], 
                             value_vars=['VIF_X1', 'VIF_X2'],
                             var_name='VIF_Type', value_name='VIF')
catplot_vif = sns.catplot(x="Corr_label", y="VIF", hue="VIF_Type", data=melted_vif,
                          kind="box", height=6, aspect=1.5)
catplot_vif.set_axis_labels("Intended Correlation", "Variance Inflation Factor")
catplot_vif.fig.suptitle("VIF vs. Intended Correlation", y=1.03)
plt.show()

# ---------------------------
# Plot 3: Box Plot of RMSE vs. Intended Correlation using Seaborn Catplot
# ---------------------------
catplot_rmse = sns.catplot(x="Corr_label", y="RMSE", data=results_df,
                           kind="box", height=6, aspect=1.5)
catplot_rmse.set_axis_labels("Intended Correlation", "RMSE")
catplot_rmse.fig.suptitle("RMSE vs. Intended Correlation", y=1.03)
plt.show()

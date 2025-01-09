import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------------------------
# Parameters to control dataset characteristics
# ------------------------------------------------------------
n_samples = 1000        # Number of samples
n_features = 2        # Total number of features
n_informative = 1       # Number of informative features
noise_level = 0.0       # Amount of noise to add
random_state = 42       # Seed for reproducibility

# ------------------------------------------------------------
# Generate synthetic regression dataset
# ------------------------------------------------------------
X, y, true_coefs = make_regression(n_samples=n_samples,
                       n_features=n_features,
                       n_informative=n_informative,
                       noise=noise_level,
                       coef=True,
                       random_state=random_state)

feature_names = [f"Feature_{i}" for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y

# ------------------------------------------------------------
# Initial Exploratory Data Analysis (EDA)
# ------------------------------------------------------------

# 1. Basic statistics
print("Basic Statistics:")
print(df.describe())

# 2. Pairplot of a subset of features
# To keep things manageable, we'll visualize only a few features
subset_features = feature_names[:4] + ["Target"]
sns.pairplot(df[subset_features], diag_kind='kde')
plt.suptitle("Pairplot of Selected Features and Target", y=1.02)
plt.show()

# 3. Correlation Heatmap
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 4. Distribution of the target variable
sns.histplot(df["Target"], kde=True)
plt.title("Distribution of Target")
plt.show()

print("True coefficients used by make_regression:")
for fname, c in zip(feature_names, true_coefs):
    print(f"{fname}: {c:.4f}")

# ------------------------------------------------------------
# Example: Fit a simple regression model and evaluate
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df["Target"], test_size=0.2, random_state=random_state)
model = LinearRegression()
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)
print(f"R^2 score on the test set: {r2:.4f}")
coef = model.coef_
intercept = model.intercept_

# Create a grid to plot the fitted plane
X_grid, Y_grid = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 50),
                             np.linspace(X[:,1].min(), X[:,1].max(), 50))

Z_grid = intercept + coef[0] * X_grid + coef[1] * Y_grid

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Data points
ax.scatter(X[:,0], X[:,1], y, c='b', marker='o', alpha=0.5)

# Fitted surface
ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='r')

ax.set_xlabel('Feature_0')
ax.set_ylabel('Feature_1')
ax.set_zlabel('Target')
plt.title("3D Scatter Plot with Fitted Regression Surface")
plt.show()



# Print the learned coefficients from the model
print("\nLearned coefficients from the model:")
learned_coefs = pd.DataFrame({"Feature": feature_names, "Estimated_Coefficient": model.coef_})
print(learned_coefs)

# Sort by the absolute value of the learned coefficients to see which features the model deems most important
learned_coefs_sorted = learned_coefs.reindex(learned_coefs.Estimated_Coefficient.abs().sort_values(ascending=False).index)
print("\nFeatures sorted by importance (based on absolute value of learned coefficient):")
print(learned_coefs_sorted)

# Compare the top coefficients the model assigns to the known informative features
# Since we know n_informative=5, let's see the top 5 by absolute value
top_5_model_features = learned_coefs_sorted.head(5)
print("\nTop 5 features according to model coefficients:")
print(top_5_model_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot of the data
ax.scatter(X[:,0], X[:,1], y, c='b', marker='o', alpha=0.5)
ax.set_xlabel('Feature_0')
ax.set_ylabel('Feature_1')
ax.set_zlabel('Target')

plt.title("3D Scatter Plot of 2D Features vs Target")
plt.show()
# ------------------------------------------------------------
# Next Steps (To be integrated later)
# ------------------------------------------------------------
# - Control feature correlation: Correlate some features by linear combinations.
# - Introduce nonlinearity: Transform some features by applying nonlinear functions (e.g., X^2, log(X+1)).
# - Add irrelevant/redundant features: Append random noise features or duplicates.
# - Add outliers: Replace some observations with extreme values.
# - For classification: Use make_classification and control class imbalance.
# - Increase dimensionality (HD): Increase n_features and reduce n_informative.
# - Introduce sparsity: Zero out certain values or use sparse feature representations.
# Apply nonlinearity to one feature






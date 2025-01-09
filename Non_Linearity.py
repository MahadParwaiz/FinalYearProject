import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# 1. Generate a non-linear synthetic dataset
# ------------------------------------------------------------
np.random.seed(42)
n_samples = 1000
X_nl = np.random.uniform(-3, 3, size=(n_samples, 2))  # 2 features in range [-3, 3]

# True non-linear relationship:
# y = 3 * x0^2 + 2 * sin(x1) + some_noise
noise = np.random.normal(loc=0, scale=0.3, size=n_samples)  # optional noise
y_nl = 3.0 * (X_nl[:, 0] ** 2) + 2.0 * np.sin(X_nl[:, 1]) + noise

df_nl = pd.DataFrame(X_nl, columns=["Feature_0", "Feature_1"])
df_nl["Target"] = y_nl

print("Head of the non-linear dataset:")
print(df_nl.head())
print()

# ------------------------------------------------------------
# 2. Quick Exploratory Analysis
# ------------------------------------------------------------
sns.pairplot(df_nl, diag_kind='kde')
plt.suptitle("Pairplot of Non-linear Data", y=1.02)
plt.show()

corr_nl = df_nl.corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr_nl, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Non-linear Data)")
plt.show()

plt.figure()
sns.histplot(df_nl["Target"], kde=True)
plt.title("Distribution of Target (Non-linear Data)")
plt.show()

# ------------------------------------------------------------
# 3. Fit LinearRegression on Non-linear Data
# ------------------------------------------------------------
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    df_nl[["Feature_0", "Feature_1"]], df_nl["Target"], test_size=0.2, random_state=42
)

lin_model = LinearRegression()
lin_model.fit(X_train_nl, y_train_nl)
r2_lin = lin_model.score(X_test_nl, y_test_nl)

print(f"Linear Regression R^2 on non-linear data: {r2_lin:.4f}")
print("Learned coefficients:", lin_model.coef_)
print("Intercept:", lin_model.intercept_)

# ------------------------------------------------------------
# 4. Visualize the Linear Fit in 3D
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the actual data
ax.scatter(X_nl[:, 0], X_nl[:, 1], y_nl, c='b', marker='o', alpha=0.5, label='Data')

# Create mesh for the predicted plane
x_surf = np.linspace(X_nl[:, 0].min(), X_nl[:, 0].max(), 50)
y_surf = np.linspace(X_nl[:, 1].min(), X_nl[:, 1].max(), 50)
x_surf_mesh, y_surf_mesh = np.meshgrid(x_surf, y_surf)
z_surf = (lin_model.intercept_ + 
          lin_model.coef_[0] * x_surf_mesh + 
          lin_model.coef_[1] * y_surf_mesh)

ax.plot_surface(x_surf_mesh, y_surf_mesh, z_surf, alpha=0.3, color='r', label='Linear fit')
ax.set_xlabel('Feature_0')
ax.set_ylabel('Feature_1')
ax.set_zlabel('Target')
plt.title("3D Plot of Non-linear Data and Linear Fit")
plt.show()

# ------------------------------------------------------------
# 5. Fit a Model That Can Handle Non-linearity
# ------------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_nl, y_train_nl)
r2_rf = rf_model.score(X_test_nl, y_test_nl)
print(f"Random Forest R^2 on non-linear data: {r2_rf:.4f}")

# ------------------------------------------------------------
# 6. Plot Random Forest Predictions in 3D
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter the actual data
ax.scatter(X_nl[:, 0], X_nl[:, 1], y_nl, c='b', marker='o', alpha=0.5, label='Data')

# Create mesh for the predicted surface
z_surf_rf = rf_model.predict(np.column_stack([x_surf_mesh.ravel(), y_surf_mesh.ravel()]))
z_surf_rf = z_surf_rf.reshape(x_surf_mesh.shape)

ax.plot_surface(x_surf_mesh, y_surf_mesh, z_surf_rf, alpha=0.3, color='g', label='RF fit')
ax.set_xlabel('Feature_0')
ax.set_ylabel('Feature_1')
ax.set_zlabel('Target')
plt.title("3D Plot of Non-linear Data and Random Forest Fit")
plt.show()

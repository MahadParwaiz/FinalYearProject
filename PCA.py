import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##############################################################################
# 1) Generate Friedman #1 data for regression
##############################################################################
N_SAMPLES = 1000
N_FEATURES = 5  # Friedman #1 typically uses 5 relevant features by default
NOISE = 1.0     # Add some noise to make it realistic

X, y_true = make_friedman1(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    noise=NOISE,
    random_state=42
)

##############################################################################
# 2) Train-test split
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.3, random_state=42
)

##############################################################################
# 3) Scale the data
##############################################################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##############################################################################
# 4) Train a simple regression model (RandomForest)
##############################################################################
regr = RandomForestRegressor(n_estimators=100, random_state=42)
regr.fit(X_train_scaled, y_train)

# Obtain predictions for the test set
y_pred = regr.predict(X_test_scaled)

##############################################################################
# 5) PCA on the test set
##############################################################################
# We'll keep 2 components for a 2D visualization
pca = PCA(n_components=2, random_state=42)
X_test_pca = pca.fit_transform(X_test_scaled)

##############################################################################
# 6) Plot in 2D, colored by TRUE vs. PREDICTED
##############################################################################
sns.set(style="white", context="talk")

def plot_pca_scatter(X_2d, color_values, title="PCA Projection"):
    """
    X_2d: 2D array after PCA transform, shape (n_samples, 2)
    color_values: the continuous values we color by (e.g. true y or predicted y)
    """
    plt.figure(figsize=(8,6))
    sc = plt.scatter(
        X_2d[:,0], X_2d[:,1], c=color_values, cmap="viridis", alpha=0.7
    )
    plt.colorbar(sc, label="Value")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.show()

# 6A) Plot colored by TRUE regression values
plot_pca_scatter(X_test_pca, y_test, title="PCA Projection - Colored by TRUE Friedman y")

# 6B) Plot colored by PREDICTED regression values
plot_pca_scatter(X_test_pca, y_pred, title="PCA Projection - Colored by MODEL PREDICTION")

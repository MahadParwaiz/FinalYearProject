import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

###############################################################################
# 1) Generate a 10-feature Friedman #1 dataset
###############################################################################
N_SAMPLES = 1000
N_FEATURES = 10
NOISE = 0.0  # Keep it zero for a clean function

X, y = make_friedman1(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    noise=NOISE,
    random_state=42
)
# X.shape => (1000, 10), y.shape => (1000,)

###############################################################################
# 2) Train-test split and scale
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

###############################################################################
# 3) Fit two regression models: LinearRegression and RandomForest
###############################################################################
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

###############################################################################
# 4) Prepare a 2D "slice" of the input space for visualization
#    We'll vary only feature0 and feature1 from their min to max
#    and fix the others at 0.5
###############################################################################
f0_min, f0_max = X_train[:,0].min(), X_train[:,0].max()
f1_min, f1_max = X_train[:,1].min(), X_train[:,1].max()

N_GRID = 50
f0_range = np.linspace(f0_min, f0_max, N_GRID)
f1_range = np.linspace(f1_min, f1_max, N_GRID)

xx, yy = np.meshgrid(f0_range, f1_range)
X_mesh = np.zeros((N_GRID*N_GRID, N_FEATURES))

# Fill in the two features we're varying
X_mesh[:,0] = xx.ravel()
X_mesh[:,1] = yy.ravel()

# Fix the other 8 features at 0.5
X_mesh[:,2:] = 0.5

# Scale it
X_mesh_scaled = scaler.transform(X_mesh)

###############################################################################
# 5) Get predictions from both models on this 2D "slice"
###############################################################################
Z_lr = lr.predict(X_mesh_scaled).reshape(xx.shape)  # LinearRegression predictions
Z_rf = rf.predict(X_mesh_scaled).reshape(xx.shape)  # RandomForest predictions

###############################################################################
# 6) Compute the TRUE Friedman #1 surface for that slice
#
#   The standard Friedman #1 equation is:
#       y = 10 * sin(pi * X0 * X1) + 20 (X2 - 0.5)^2 + 10*X3 + 5*X4 + noise
#   But when X2..X9 = 0.5, the terms reduce to constants:
#       20*(0.5-0.5)^2 = 0
#       10*0.5 = 5
#       5*0.5 = 2.5
#   So total constant = 7.5
#
#   => y_true = 10 * sin(pi * X0 * X1) + 7.5
###############################################################################
Z_true = 10 * np.sin(np.pi * xx * yy) + 7.5

###############################################################################
# 7) 3D Plot function to overlay true vs. learned surfaces
###############################################################################
def plot_true_and_learned(xx, yy, Z_true, Z_learned, X_train, y_train, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the TRUE surface
    ax.plot_surface(
        xx, yy, Z_true,
        cmap='viridis', edgecolor='none',
        alpha=0.3,  # partially transparent so we can see overlaps
        label='True surface'
    )
    # Plot the LEARNED surface
    ax.plot_surface(
        xx, yy, Z_learned,
        cmap='viridis', edgecolor='none',
        alpha=0.8,  # a bit more opaque
        label='Learned surface'
    )
    
    # Optionally scatter the training data (using only f0, f1, y)
    ax.scatter(
        X_train[:,0], 
        X_train[:,1], 
        y_train,
        c='red', 
        s=20, 
        alpha=0.5
    )

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.show()


###############################################################################
# 8) Plot: True vs. Linear Regression
###############################################################################
plot_true_and_learned(
    xx, yy, Z_true, Z_lr, 
    X_train, y_train,
    title="True Surface vs. LinearRegression"
)

###############################################################################
# 9) Plot: True vs. Random Forest
###############################################################################
plot_true_and_learned(
    xx, yy, Z_true, Z_rf, 
    X_train, y_train,
    title="True Surface vs. RandomForestRegressor"
)

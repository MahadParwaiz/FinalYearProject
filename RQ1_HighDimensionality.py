import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import time
import pandas as pd

# For reproducibility
np.random.seed(42)

def generate_regression_data_friedman(n_samples, n_features,
                                      noise=0.0,
                                      skew_factor=1.0):
    """
    Generates a Friedman-1 style dataset (for regression).
    We always have 5 'true' informative features by definition.
    The rest are non-informative (i.e., random).
    
    :param n_samples: number of samples
    :param n_features: total number of features (>= 5)
    :param noise: float, std of Gaussian noise added to y
    :param skew_factor: float, used to skew the distribution of y
                        (e.g. y = y ** skew_factor).
                        >1 => right-skew, <1 => left-skew, 1 => no skew
    :return: (X, y, imbalance_ratio)
             where imbalance_ratio is a simple measure of
             how "skewed" or "imbalanced" the y distribution is
             for demonstration purposes.
    """
    # Generate data via Friedman-1
    X, y = make_friedman1(n_samples=n_samples,
                          n_features=n_features,
                          noise=noise,
                          random_state=42)
    
    # Apply skew if requested
    if skew_factor != 1.0:
        # Because y can be negative if noise is large,
        # we shift y upward so it's non-negative before powering.
        shift = abs(min(y)) + 1e-3
        y = (y + shift) ** skew_factor
    
    # Calculate a naive "imbalance ratio":
    # ratio of the mean of the top 10% to the mean of the bottom 10%.
    n_top = max(1, int(0.1 * len(y)))
    n_bot = max(1, int(0.1 * len(y)))
    y_sorted = np.sort(y)
    top_mean = np.mean(y_sorted[-n_top:])
    bot_mean = np.mean(y_sorted[:n_bot])
    imbalance_ratio = (top_mean / bot_mean) if bot_mean != 0 else np.inf

    return X, y, imbalance_ratio


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    """
    Trains the model, records training time, and computes
    key regression metrics (MSE, R^2) on both train and test.
    Also does feature-selection analysis using permutation importance:
    - feature_recall: out of the top-5 important features, how many
      are from the "true" feature set [0..4]?
    - feature_precision: proportion of the top-5 selected that are correct.
    Returns a dictionary of results.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # TRAIN predictions (optional, for overfitting measure)
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # TEST predictions
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Permutation importance
    perm_import = permutation_importance(model, X_test, y_test, n_repeats=5,
                                         random_state=42)
    importances = perm_import.importances_mean
    
    # Sort features by importance (descending order)
    sorted_idx = np.argsort(importances)[::-1]
    top_5 = sorted_idx[:5]   # top-5 important features
    true_features = set(range(5))
    selected_features = set(top_5)
    
    correct_selected = len(true_features.intersection(selected_features))
    feature_recall = correct_selected  # out of 5 real features
    feature_precision = correct_selected / 5.0
    
    return {
        'MSE_Train': mse_train,
        'R2_Train': r2_train,
        'MSE_Test': mse_test,
        'R2_Test': r2_test,
        'Train_Time': train_time,
        'Feature_Recall': feature_recall,       # 0 to 5
        'Feature_Precision': feature_precision  # 0.0 to 1.0
    }


def run_experiments(sample_sizes, feature_list,
                    noise_list=[0.0, 1.0],
                    skew_list=[1.0, 2.0]):
    """
    Main loop to run experiments across different sample sizes,
    feature dimensions, noise levels, and skew factors.
    Returns a list of results (each entry is a dict).
    """
    results = []
    
    for n in sample_sizes:
        for p in feature_list:
            for noise in noise_list:
                for skew in skew_list:
                    
                    # Generate data
                    X, y, imbalance_ratio = generate_regression_data_friedman(
                        n_samples=n, 
                        n_features=p, 
                        noise=noise, 
                        skew_factor=skew
                    )
                    
                    # Train/Test Split (80/20)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Define some regression models
                    models = {
                        'LinearRegression': LinearRegression(),
                        'SVR': SVR(),
                        'RandomForest': RandomForestRegressor(),
                        'MLP': MLPRegressor(max_iter=10000)
                    }
                    
                    for model_name, model in models.items():
                        metrics = evaluate_regression_model(model,
                                                             X_train, X_test,
                                                             y_train, y_test)
                        
                        # Store the outcome
                        entry = {
                            'SampleSize': n,
                            'NumFeatures': p,
                            'Noise': noise,
                            'SkewFactor': skew,
                            'ImbalanceRatio': imbalance_ratio,
                            'Model': model_name
                        }
                        entry.update(metrics)
                        results.append(entry)
    
    return results

def plot_results(results_df, metric='R2'):
    """
    Plots the chosen metric vs. NumFeatures for each model,
    facet by SampleSize.
    """
    sns.set(style='whitegrid')
    
    g = sns.FacetGrid(results_df, col='SampleSize', hue='Model', height=4, sharey=False)
    g.map(sns.lineplot, 'NumFeatures', metric, marker='o')
    g.add_legend()
    g.set_axis_labels("Number of Features", metric)
    
    # Adjust y-axis to start at 0 if there are no negative values
    # Otherwise, start at the negative min
    min_val = results_df[metric].min()
    start_val = min(0, min_val)
    for ax in g.axes.flatten():
        ax.set_ylim(start_val, None)
    
    plt.tight_layout()
    plt.show()

def plot_mse(results_df):
    """
    Similar plotting function for MSE.
    """
    sns.set(style='whitegrid')
    g = sns.FacetGrid(results_df, col='SampleSize', hue='Model', height=4, sharey=False)
    g.map(sns.lineplot, 'NumFeatures', 'MSE', marker='o')
    g.add_legend()
    g.set_axis_labels("Number of Features", "MSE")
    
    # Adjust y-axis to start at 0 if there are no negative values
    min_val = results_df['MSE'].min()
    start_val = min(0, min_val)
    for ax in g.axes.flatten():
        ax.set_ylim(start_val, None)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # Example experiment settings
    sample_sizes = [100, 1000]
    feature_list = [10, 50, 100]
    noise_list = [0.0]   # 0 would be no noise 
    skew_list = [1.0]    # 1 would be no skew 
    
    all_runs = []
    n_runs = 5  # We will do multiple runs (e.g. 5) for each parameter combination
    for _ in range(n_runs):
        run_results = run_experiments(
            sample_sizes=sample_sizes,
            feature_list=feature_list,
            noise_list=noise_list,
            skew_list=skew_list
        )
        all_runs.extend(run_results)
    
    # Convert to DataFrame
    df_all = pd.DataFrame(all_runs)
    
    # Average metrics over the 5 runs for each (SampleSize, NumFeatures, Noise, SkewFactor, Model)
    # Note: This will average MSE_Train, R2_Train, MSE_Test, R2_Test, etc.
    df_results = df_all.groupby(
        ['SampleSize','NumFeatures','Noise','SkewFactor','Model'],
        as_index=False
    ).mean()
    
    # Quick preview
    print("Sample of averaged results:\n", df_results.head(), "\n")
    
    # 1) Show only RandomForest rows (example)
    df_random_forest = df_results[df_results["Model"] == "RandomForest"]
    print("RandomForest results:\n", df_random_forest, "\n")

    # 2) Plot MSE across feature dimensions & sample sizes
    #    We'll rename 'MSE_Test' -> 'MSE' so we can reuse plot_mse function
    df_plot_mse = df_results.rename(columns={'MSE_Test':'MSE'})
    plot_mse(df_plot_mse)
    
    # 3) Plot Feature Recall
    plot_results(df_results, metric='Feature_Recall')
    
    # 4) Plot Feature Precision
    plot_results(df_results, metric='Feature_Precision')
    
    # 5) Plot Train_Time
    plot_results(df_results, metric='Train_Time')

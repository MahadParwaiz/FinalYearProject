import numpy as np
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def generate_regression_data_friedman(n_samples, n_features,
                                      noise=0.0,
                                      skew_factor=1.0,
                                      seed=42):
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
    :param seed: random seed for reproducibility
    :return: (X, y, imbalance_ratio)
             where imbalance_ratio is a simple measure of
             how "skewed" or "imbalanced" the y distribution is
             for demonstration purposes.
    """
    # Generate data via Friedman-1 using the provided seed
    X, y = make_friedman1(n_samples=n_samples,
                          n_features=n_features,
                          noise=noise,
                          random_state=seed)

    feature_names = [f"x{i}" for i in range(X.shape[1])]
    
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

    return X, y, imbalance_ratio, feature_names


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, seed=42):
    """
    Trains the model, records training time, and computes
    key regression metrics (MSE, R^2) on both train and test.
    Also does feature-selection analysis using permutation importance:
    - feature_recall: out of the top-5 important features, how many
      are from the "true" feature set [0..4]?
    - feature_precision: proportion of the top-10 selected that are correct.
    Returns a dictionary of results.
    
    :param seed: random seed for permutation importance reproducibility.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # TRAIN predictions (optional, for overfitting measure)
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train) 
    r2_train = r2_score(y_train, y_pred_train)
    
    # TEST predictions
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test=np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Permutation importance using the provided seed
    perm_import = permutation_importance(model, X_test, y_test, n_repeats=5,
                                         random_state=seed, n_jobs=3)
    importances = perm_import.importances_mean
    
    # Sort features by importance (descending order)
    sorted_idx = np.argsort(importances)[::-1]
    top_10 = sorted_idx[:10]   # top 10 important features

    
    true_features = set(range(5))
    selected_features = set(top_10)

    correct_selected = len(true_features.intersection(selected_features))
    recovery_causative = correct_selected  # out of 5 true features

    
    return {
        'MSE_Train': mse_train,
        'RMSE_Train': rmse_train, 
        'R2_Train': r2_train,
        'MSE_Test': mse_test,
        'RMSE_Test': rmse_test,
        'R2_Test': r2_test,
        'Train_Time': train_time,
        'Recovery of Causative Features in Top 5': recovery_causative,
        'Top10_Feature_Indices': top_10, 
        'Top10_Importances': importances[top_10], 
        'AllImportances': importances
    }


def run_experiments(sample_sizes, feature_list,
                    noise_list=[0.0, 1.0],
                    skew_list=[1.0, 2.0],
                    seed=42):
    """
    Main loop to run experiments across different sample sizes,
    feature dimensions, noise levels, and skew factors.
    Returns a list of results (each entry is a dict).
    
    :param seed: random seed to be used in data generation, train/test split, etc.
    """
    results = []
    
    for n in sample_sizes:
        for p in feature_list:
            for noise in noise_list:
                for skew in skew_list:
                    
                    # Generate data
                    X, y, imbalance_ratio, feature_names = generate_regression_data_friedman(
                        n_samples=n, 
                        n_features=p, 
                        noise=noise, 
                        skew_factor=skew,
                        seed=seed
                    )
                    
                    # Train/Test Split (80/20) using provided seed
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=seed
                    )
                    
                    #Scaling
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Define some regression models
                    models = {
                        'LinearRegression': LinearRegression(),
                        'SVR': SVR(),
                        'RandomForest': RandomForestRegressor(),
                        'MLP': MLPRegressor(max_iter=10000)
                    }
                    
                    for model_name, model in models.items():
                        metrics = evaluate_regression_model(model,
                                                             X_train_scaled, X_test_scaled,
                                                             y_train, y_test,
                                                             seed=seed)
                        
                                               # Store config + metrics
                        entry = {
                            'SampleSize': n,
                            'NumFeatures': p,
                            'Noise': noise,
                            'SkewFactor': skew,
                            'ImbalanceRatio': imbalance_ratio,
                            'Model': model_name,
                            'Seed': seed,
                            # Also store feature_names so we can retrieve them later
                            'FeatureNames': feature_names,
                            'Top10_Feature_Indices': metrics['Top10_Feature_Indices'],
                            'Top10_Importances': metrics['Top10_Importances'],
                            'MSE_Train': metrics['MSE_Train'],
                            'RMSE_Train': metrics['RMSE_Train'],
                            'R2_Train': metrics['R2_Train'],
                            'MSE_Test': metrics['MSE_Test'],
                            'RMSE_Test': metrics['RMSE_Test'],
                            'R2_Test': metrics['R2_Test'],
                            'Train_Time': metrics['Train_Time'],
                            'AllImportances': metrics['AllImportances'],
                            'Recovery of Causative Features in Top 5': metrics['Recovery of Causative Features in Top 5']
                        }
                        results.append(entry)

    
    return results




def boxplot_metric(results_df, metric):
    """
    Plots a box plot for the chosen metric across seeds,
    for each configuration. Uses seaborn's catplot to show
    the distribution (median, quartiles, outliers) for each combination
    of SampleSize, NumFeatures, and Model.
    """
    sns.set(style="whitegrid")
    g = sns.catplot(data=results_df, x="NumFeatures", y=metric,
                    hue="Model", col="SampleSize",
                    kind="box", height=4, sharey=True, showfliers=False)
    # If we're plotting time, add units to the label
    if metric == 'Train_Time':
        y_label = "Train Time (seconds)"
    else:
        y_label = metric
    g.set_axis_labels("Number of Features", y_label)
    g.fig.subplots_adjust(top=0.85)
    
    # Define the title and set it
    title = f"Box Plot of {metric} Across Seeds"
    g.fig.suptitle(title)
    
    # Create a filename from the title (replace spaces with underscores)
    filename = title.replace(" ", "_") + ".png"
    
    # Save the figure with the generated filename
    g.fig.savefig(filename)

def plot_top10_importances_from_mean(avg_importances, feature_names, model_name=""):
    """
    Plot a horizontal bar chart of top-10 average importances.
    """
    sorted_idx = np.argsort(avg_importances)[::-1]
    top10_idx = sorted_idx[:10]
    top10_vals = avg_importances[top10_idx]
    top10_names = [feature_names[i] for i in top10_idx]

    plt.figure()
    plt.barh(top10_names, top10_vals)
    plt.xlabel("Avg Permutation Importance (across seeds)")
    plt.title(f"Avg Top-10 Importances {model_name}")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    
    # Example experiment settings
    sample_sizes = [20, 50, 100]
    feature_list = [10, 30, 50, 80, 100]
    noise_list = [0]      # 0 => no noise
    skew_list = [1.0]     # 1 => no skew
    
    all_runs = []
    seeds = [1,2,3,4,5]
    for seed in seeds:
        run_results = run_experiments(
            sample_sizes=sample_sizes,
            feature_list=feature_list,
            noise_list=noise_list,
            skew_list=skew_list,
            seed=seed
        )
        all_runs.extend(run_results)
    
    # Build the DataFrame from all runs
    df_all = pd.DataFrame(all_runs)
    print("Sample of results across seeds:\n", df_all.head(), "\n")
    
    # Create box plots for some metrics
    boxplot_metric(df_all, 'RMSE_Test')
    boxplot_metric(df_all, 'R2_Test')
    boxplot_metric(df_all, 'Train_Time')
    boxplot_metric(df_all, 'Recovery of Causative Features in Top 5')

"""
    # -- Now average over seeds:
    df_all['AllImportances'] = df_all['AllImportances'].apply(np.array)
    group_cols = ['SampleSize', 'NumFeatures', 'Noise', 'SkewFactor', 'Model']
    
    def mean_of_arrays(series_of_arrays):
        stacked = np.vstack(series_of_arrays)
        return stacked.mean(axis=0)

    # Aggregates each group’s 'AllImportances' across seeds
    df_avg = df_all.groupby(group_cols, as_index=False).agg({
        'AllImportances': mean_of_arrays,
        'FeatureNames': 'first'  # pick the first row's FeatureNames
    })

    # Plot the top-10 from the averaged importances
    for _, row in df_avg.iterrows():
        avg_import = row['AllImportances']
        f_names = row['FeatureNames']
        model_name = row['Model']

        # We can incorporate more info in title if desired
        plot_title = (
            f"{model_name} | n={row['SampleSize']} | p={row['NumFeatures']} | Noise={row['Noise']}"
        )

        plot_top10_importances_from_mean(
            avg_importances=avg_import,
            feature_names=f_names,
            model_name=plot_title
        )

"""
import numpy as np
import seaborn as sns
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier  # Make sure you have xgboost installed

from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


def generate_classification_data_friedman(n_samples, n_features,
                                          noise=0.0,
                                          skew_factor=1.0,
                                          seed=42,
                                          threshold="median"):
    """
    Generates a Friedman-1 style dataset and converts it to a binary 
    classification problem by thresholding the target.
    
    We always have 5 'true' informative features by definition.
    The rest are non-informative (i.e., random).
    
    :param n_samples: number of samples
    :param n_features: total number of features (>= 5)
    :param noise: float, std of Gaussian noise added to y
    :param skew_factor: float, used to skew the distribution of y
                        (e.g. y = y ** skew_factor).
                        >1 => right-skew, <1 => left-skew, 1 => no skew
    :param seed: random seed for reproducibility.
    :param threshold: if "median", uses the median of y; otherwise, a float threshold.
    :return: (X, y_bin, imbalance_ratio, feature_names)
             where y_bin is the binary target and imbalance_ratio is computed as 
             (number of positives / number of negatives).
    """
    # Generate continuous data using Friedman-1
    X, y = make_friedman1(n_samples=n_samples,
                          n_features=n_features,
                          noise=noise,
                          random_state=seed)

    # Apply skew if requested
    if skew_factor != 1.0:
        shift = abs(min(y)) + 1e-3  # ensure non-negative before applying power
        y = (y + shift) ** skew_factor

    # Determine the threshold value
    if threshold == "median":
        thresh_val = np.median(y)
    else:
        thresh_val = threshold

    # Convert regression target into binary classification labels
    y_bin = (y > thresh_val).astype(int)

    # Compute a simple imbalance ratio (positives/negatives)
    n_pos = np.sum(y_bin)
    n_neg = len(y_bin) - n_pos
    imbalance_ratio = (n_pos / n_neg) if n_neg != 0 else np.inf

    feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    return X, y_bin, imbalance_ratio, feature_names


def evaluate_classification_model(model, X_train, X_test, y_train, y_test, seed=42):
    """
    Trains the classification model, records training time, and computes
    accuracy metrics on both train and test splits.
    Also performs feature importance analysis using permutation importance.
    
    :param seed: random seed for reproducibility in permutation importance.
    :return: Dictionary of results including training time, accuracies, 
             top feature indices and permutation importances.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Training predictions and accuracy
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    # Test predictions and accuracy
    y_pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    # Permutation importance (using accuracy as the scoring metric)
    perm_import = permutation_importance(model, X_test, y_test,
                                         scoring='accuracy',
                                         n_repeats=5,
                                         random_state=seed, n_jobs=3)
    importances = perm_import.importances_mean
    
    # Sort features by importance (descending order)
    sorted_idx = np.argsort(importances)[::-1]
    top_10 = sorted_idx[:10]   # top 10 important features

    # Calculate recovery of causative features (features 0 to 4 are "true" features)
    true_features = set(range(5))
    selected_features = set(top_10)
    correct_selected = len(true_features.intersection(selected_features))
    recovery_causative = correct_selected  # out of 5 true features
    
    return {
        'Train_Time': train_time,
        'Accuracy_Train': accuracy_train,
        'Accuracy_Test': accuracy_test,
        'Top10_Feature_Indices': top_10,
        'Top10_Importances': importances[top_10],
        'AllImportances': importances,
        'Recovery of Causative Features in Top 5': recovery_causative
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
                    
                    # Generate classification data (Friedman-style)
                    X, y, imbalance_ratio, feature_names = generate_classification_data_friedman(
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
                    
                    # Scaling
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Define classification models
                    models = {
                        'LogisticRegression': LogisticRegression(max_iter=10000),
                        'ElasticNetLogistic': LogisticRegression(penalty='elasticnet', max_iter=10000),
                        'SVC': SVC(),
                        'RandomForest': RandomForestClassifier(),
                        'MLP': MLPClassifier(max_iter=10000),
                        'XGBoost': XGBClassifier()

                    }
                    
                    for model_name, model in models.items():
                        metrics = evaluate_classification_model(model,
                                                                 X_train_scaled, X_test_scaled,
                                                                 y_train, y_test,
                                                                 seed=seed)
                        
                        # Store configuration and metrics
                        entry = {
                            'SampleSize': n,
                            'NumFeatures': p,
                            'Noise': noise,
                            'SkewFactor': skew,
                            'ImbalanceRatio': imbalance_ratio,
                            'Model': model_name,
                            'Seed': seed,
                            'FeatureNames': feature_names,
                            'Top10_Feature_Indices': metrics['Top10_Feature_Indices'],
                            'Top10_Importances': metrics['Top10_Importances'],
                            'Accuracy_Train': metrics['Accuracy_Train'],
                            'Accuracy_Test': metrics['Accuracy_Test'],
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
    if metric == 'Train_Time':
        y_label = "Train Time (seconds)"
    else:
        y_label = metric
    g.set_axis_labels("Number of Features", y_label)
    g.fig.subplots_adjust(top=0.85)
    title = f"Box Plot of {metric} Across Seeds"
    g.fig.suptitle(title)
    filename = title.replace(" ", "_") + ".png"
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
    seeds = [1, 2, 3, 4, 5]
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
    boxplot_metric(df_all, 'Accuracy_Test')
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
        'FeatureNames': 'first'
    })

    # Plot the top-10 from the averaged importances
    for _, row in df_avg.iterrows():
        avg_import = row['AllImportances']
        f_names = row['FeatureNames']
        model_name = row['Model']
        plot_title = (
            f"{model_name} | n={row['SampleSize']} | p={row['NumFeatures']} | Noise={row['Noise']}"
        )
        plot_top10_importances_from_mean(
            avg_importances=avg_import,
            feature_names=f_names,
            model_name=plot_title
        )
"""

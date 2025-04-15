import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_friedman1
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix
)

##############################################################################
# 1) Generate Friedman #1 as a Regression
##############################################################################
def generate_friedman_data(n_samples=2000, n_features=5, noise=1.0, random_state=42):
    """
    Friedman #1 regression:
        y = 10*sin(pi*x1*x2) + 20*(x3 - 0.5)^2 + 10*x4 + 5*x5 + N(0, noise^2).
    Returns X, y_reg.
    """
    X, y_reg = make_friedman1(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y_reg

##############################################################################
# 2) Convert Regression -> Classification with a Desired Fraction of Positives
##############################################################################
def binarize_with_ratio(y_reg, desired_ratio=0.2, random_state=42):
    """
    Finds offset alpha so that the fraction of (y_reg + alpha >= 0) is ~ desired_ratio.
    Returns (y_class, alpha).
    """
    rng = np.random.RandomState(random_state)
    
    def fraction_positive(alpha):
        return np.mean((y_reg + alpha) >= 0.0)
    
    lower, upper = -1000, 1000
    for _ in range(100):
        mid = (lower + upper) / 2
        frac = fraction_positive(mid)
        if frac > desired_ratio:
            upper = mid
        else:
            lower = mid
    alpha = (lower + upper) / 2
    y_class = ((y_reg + alpha) >= 0).astype(int)
    return y_class, alpha

##############################################################################
# 3) Train & Evaluate: SVM, RandomForest, MLP
##############################################################################
def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Trains and evaluates SVM, RandomForest, and MLP.
    Returns a dict: {model_name: {metric: value}} for
             Accuracy, F1_score, Sensitivity, Specificity, ROC_AUC.
    """
    models = {
        "SVM": SVC(probability=True, random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "MLP": MLPClassifier(random_state=random_state, max_iter=10000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Try to obtain probabilities for ROC-AUC calculation
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            try:
                y_decision = model.decision_function(X_test)
                y_proba = y_decision
            except:
                y_proba = None
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = np.nan
        
        results[model_name] = {
            "Accuracy": acc,
            "F1_score": f1,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "ROC_AUC": auc
        }
    
    return results

##############################################################################
# 4) Plotting Function: For each metric, create subplots for sample sizes.
#    X-axis shows imbalance scenarios, and the hue represents the model.
##############################################################################
def plot_metric_subplots(df, metric, sample_sizes, imbalance_order):
    """
    Creates a figure with one subplot per sample_size (sharing the y-axis).
    On the x-axis, we show the different imbalance scenarios.
    The box colors are determined by the model (hue="model").
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(sample_sizes), figsize=(15, 6), sharey=True)
    
    # If there's only one sample size, ensure axes is iterable
    if len(sample_sizes) == 1:
        axes = [axes]
    
    for i, s_size in enumerate(sample_sizes):
        ax = axes[i]
        df_subset = df[(df["metric"] == metric) & (df["sample_size"] == s_size)]
        
        sns.boxplot(
            data=df_subset,
            x="imbalance_scenario",
            y="value",
            hue="model",
            order=imbalance_order,
            ax=ax
        )
        
        sns.stripplot(
            data=df_subset,
            x="imbalance_scenario",
            y="value",
            hue="model",
            order=imbalance_order,
            palette="dark:black",  # All points will be black
            alpha=0.6,
            dodge=True,
            ax=ax,
            legend=False  # Use legend from boxplot only
        )
        
        ax.set_title(f"{metric} for Sample Size = {s_size}", fontsize=15)
        ax.set_xlabel("Imbalance Scenario")
        if i == 0:
            ax.set_ylabel(metric)
        else:
            ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        
        # Show legend on the rightmost subplot only
        if i == len(sample_sizes) - 1:
            ax.legend(loc="best", title="Model")
        else:
            ax.get_legend().remove()
    
    plt.suptitle(f"{metric} Performance Across Imbalance Scenarios", fontsize=18)
    plt.tight_layout()
    filename = f"{metric.replace(' ', '_')}_Performance.png"  # Replace spaces for a safe file name
    plt.savefig(filename)
    plt.close()

##############################################################################
# 5) Main script with Stratified Split
##############################################################################
if __name__ == "__main__":

    # Configuration
    SAMPLE_SIZES = [20,200]   # Two sample sizes
    SEEDS = [0,1]               # Two seeds for demonstration
    TEST_SIZE = 0.3
    N_FEATURES = 5
    NOISE = 1.0

    # Define imbalance scenarios (the ratio represents the fraction of positives)
    imbalance_scenarios = {
        "(50:50)": 0.5,
        "(80:20)": 0.2,
        "(95:5)": 0.05
    }

    # List to store results
    all_records = []

    for imbalance_name, ratio in imbalance_scenarios.items():
        for n_samp in SAMPLE_SIZES:
            for seed in SEEDS:
                # Generate the full dataset
                X, y_reg = generate_friedman_data(
                    n_samples=n_samp,
                    n_features=N_FEATURES,
                    noise=NOISE,
                    random_state=seed
                )
                
                # Binarize the entire dataset to get the desired imbalance ratio
                y_class, _ = binarize_with_ratio(y_reg, desired_ratio=ratio, random_state=seed+100)
                
                # Use StratifiedShuffleSplit to maintain the class ratio in train and test
                sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
                for train_index, test_index in sss.split(X, y_class):
                    X_train_all, X_test_all = X[train_index], X[test_index]
                    y_train_class, y_test_class = y_class[train_index], y_class[test_index]
                
                # Train and evaluate the models
                results_dict = train_and_evaluate(
                    X_train_all, y_train_class, X_test_all, y_test_class, random_state=seed
                )
                
                # Store the results
                for model_name, metrics_map in results_dict.items():
                    for metric_name, metric_val in metrics_map.items():
                        rec = {
                            'imbalance_scenario': imbalance_name,
                            'sample_size': n_samp,
                            'seed': seed,
                            'model': model_name,
                            'metric': metric_name,
                            'value': metric_val
                        }
                        all_records.append(rec)
    
    # Convert results to a DataFrame
    df_results = pd.DataFrame(all_records)
    
    # Set Seaborn style
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("Set2")
    
    # Define the order for the x-axis categories
    imbalance_order = [
        "(50:50)",
        "(80:20)",
        "(95:5)"
    ]
    
    # Define metrics to plot
    metrics_list = ["Accuracy", "F1_score", "Sensitivity", "Specificity", "ROC_AUC"]
    
    # Plot each metric across sample sizes with the stratified imbalance scenarios
    for metric in metrics_list:
        plot_metric_subplots(df_results, metric, SAMPLE_SIZES, imbalance_order)

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                        mutual_info_classif, SelectFromModel)
from sklearn.metrics import balanced_accuracy_score, average_precision_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from lassonet import LassoNetClassifier  # pip install lassonet
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV 

# ----------------------------- constants ------------------------------------
K_FIXED      = 5
CLEAN_P      = 100
HIGHDIM_P    = 1000
N_CLEAN      = 80
N_SCARCE     = 30 
POS_CLEAN    = 0.5   # 50‑50 baseline
POS_IMBAL    = 0.05   # 95:5 imbalance (class «1» is minority)
SEEDS        = [42, 43, 44, 45, 46, 47, 48, 49, 50]
SAMPLE_SIZES = [N_CLEAN, N_SCARCE]

# ------------------------- data generation ----------------------------------

def generate_classification_data_friedman(n_samples, n_features,
                                          positive_ratio=0.5,
                                          noise=0.0,
                                          skew_factor=1.0,
                                          seed=42):
    """Generate a binary‑label Friedman‑1 dataset with controllable class ratio."""
    X, y = make_friedman1(n_samples=n_samples,
                          n_features=n_features,
                          noise=noise,
                          random_state=seed)

    if skew_factor != 1.0:
        shift = abs(min(y)) + 1e-3
        y = (y + shift) ** skew_factor

    alpha = np.quantile(y, 1.0 - positive_ratio)
    y_bin = (y > alpha).astype(int)

    pos, neg = y_bin.sum(), len(y_bin) - y_bin.sum()
    imbalance_ratio = pos / neg if neg else np.inf

    feature_names = [f"x{i}" for i in range(n_features)]
    return X, y_bin, imbalance_ratio, feature_names


def add_label_flip_noise(y, flip_prob=0.1, seed=None):
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_flip = int(flip_prob * len(y))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy

# ------------------- feature perturbation helpers ---------------------------

def add_feature_noise(X, noise_std=0.05, prob=1.0, seed=None):
    rng   = np.random.default_rng(seed)
    mask  = rng.random(X.shape) < prob
    noise = rng.normal(0.0, noise_std, size=X.shape)
    return X + noise * mask


def add_correlations(X, corr_strength=0.92, seed=None):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    n, p = Xc.shape
    if p < 6:
        raise ValueError("Need at least 6 features for correlation routine")

    noise = rng.normal(0, (1 - corr_strength), size=n)
    Xc[:, 1] = corr_strength * Xc[:, 0] + noise  # redundant causal
    noise = rng.normal(0, (1 - corr_strength), size=n)
    Xc[:, -1] = corr_strength * Xc[:, 0] + noise  # spurious non‑causal
    return Xc

# ------------------------- evaluation ---------------------------------------

def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    bal_acc_train = balanced_accuracy_score(y_train, model.predict(X_train))
    bal_acc_test  = balanced_accuracy_score(y_test,  model.predict(X_test))

    selected_k = None
    for step in getattr(model, "named_steps", {}).values():
        if isinstance(step, SelectFromModel):
            selected_k = step.get_support().sum()
        elif isinstance(step, RFECV):
            selected_k = step.n_features_

    # ---------- PRC-AUC (Average Precision) --------------------------
    def _scores(clf, X):
        """Return class-1 scores for any classifier."""
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)[:, 1]
        # LinearSVC & others expose decision_function
        return clf.decision_function(X)

    prc_auc_train = average_precision_score(y_train, _scores(model, X_train))
    prc_auc_test  = average_precision_score(y_test,  _scores(model, X_test))
    # ---------------------------------------------------------------

    return {"BalAcc_Train": bal_acc_train,
            "BalAcc_Test":  bal_acc_test,
            "PRC_AUC_Train": prc_auc_train,   # NEW
            "PRC_AUC_Test":  prc_auc_test,    # NEW
            "Train_Time":   train_time,
            "Selected_k":   selected_k}


# -------------------- FS blocks per model -----------------------------------

def embedded_blocks_for(model_name, seed):
    return {
        "LogisticRegression": {
            "LogReg_L1": lambda m: make_pipeline(
                StandardScaler(),
                SelectFromModel(LogisticRegression(penalty="l1", solver="saga", max_iter=5000)),
                m)
        },
        "SVC": {
            "LinSVC_emb": lambda m: make_pipeline(
                StandardScaler(),
                SelectFromModel(LinearSVC()),
                m)
        },
        "RandomForest": {
            "RF_emb": lambda m: make_pipeline(
                SelectFromModel(RandomForestClassifier(n_estimators=500, random_state=seed),
                                 threshold="median"),
                m)
        },
        "MLP": {
            "LassoNet": lambda m: make_pipeline(
                SelectFromModel(LassoNetClassifier(hidden_dims=(128, 64),
                                                   lambda_start=0.1,
                                                   path_multiplier=2.0)),
                m)
        }
    }[model_name]

# ------------------------- experiment runner --------------------------------
def wrapper_blocks_for(model_name, seed):
    """RFECV-based wrappers that use the *same* classifier as the final model."""
    cv_args = dict(step=0.25, cv=3, scoring="balanced_accuracy", n_jobs=2)

    return {
        "LogisticRegression": {
            "RFECV_wrap": lambda m: make_pipeline(
                StandardScaler(),
                RFECV(estimator=LogisticRegression(max_iter=5000,
                                                   random_state=seed),
                      **cv_args),
                m)
        },
        "SVC": {
            "RFECV_wrap": lambda m: make_pipeline(
                StandardScaler(),
                RFECV(estimator=LinearSVC(random_state=seed),
                      **cv_args),
                m)
        },
        "RandomForest": {
            "RFECV_wrap": lambda m: make_pipeline(
                RFECV(estimator=RandomForestClassifier(n_estimators=500,
                                                       random_state=seed),
                      **cv_args),
                m)
        },
        "MLP": {
            # use a light surrogate (RF) for RFE to keep runtime reasonable
            "RFECV_wrap": lambda m: make_pipeline(
                RFECV(estimator=RandomForestClassifier(n_estimators=250,
                                                       random_state=seed),
                      **cv_args),
                m)
        },
    }[model_name]

def run_experiments(sample_sizes, feature_list,
                    positive_ratio_list=[POS_CLEAN],
                    label_flip_list=[0.0],
                    feat_noise_list=[0.0],
                    correlation_list=[False],
                    seed=42):
    results = []

    for n in sample_sizes:
        for p in feature_list:
            for pos_ratio in positive_ratio_list:
                for flip_prob in label_flip_list:
                    for feat_noise in feat_noise_list:
                        for corr_flag in correlation_list:

                            # --- Data ------------------------------------
                            X, y, _, _ = generate_classification_data_friedman(
                                n_samples=n, n_features=p,
                                positive_ratio=pos_ratio,
                                seed=seed)

                            if flip_prob:
                                y = add_label_flip_noise(y, flip_prob, seed)
                            if corr_flag:
                                X = add_correlations(X, seed=seed)
                            elif feat_noise:
                                X = add_feature_noise(X, noise_std=feat_noise, seed=seed)

                            X_tr, X_te, y_tr, y_te = train_test_split(
                                X, y, test_size=0.3, random_state=seed, stratify=y)

                            scaler = StandardScaler()
                            X_tr = scaler.fit_transform(X_tr)
                            X_te = scaler.transform(X_te)

                            base_models = {
                                "LogisticRegression": LogisticRegression(max_iter=5000,random_state=seed),
                                "SVC":                 SVC(probability=True,random_state=seed),
                                "RandomForest":        RandomForestClassifier(random_state=seed),
                                "MLP":                 MLPClassifier(max_iter=10_000,random_state=seed)
                            }

                            for mdl_name, mdl in base_models.items():
                                fs_blocks = {
                                    "NoFS": lambda m: m,
                                    f"Filter_{K_FIXED}": lambda m: make_pipeline(
                                        VarianceThreshold(threshold=1e-4),
                                        SelectKBest(mutual_info_classif, k=K_FIXED),
                                        m)
                                }
                                fs_blocks.update(embedded_blocks_for(mdl_name, seed))
                                fs_blocks.update(wrapper_blocks_for(mdl_name, seed))  

                                for fs_name, builder in fs_blocks.items():
                                    pipe = builder(mdl)
                                    metrics = evaluate_classification_model(pipe, X_tr, X_te, y_tr, y_te)

                                    # ------------- Condition tag ----------
                                    if p == CLEAN_P and n == N_CLEAN:
                                        if pos_ratio != POS_CLEAN:
                                            condition = "Imbalance"
                                        elif corr_flag:
                                            condition = "Correlation"
                                        elif flip_prob == 0 and feat_noise == 0:
                                            condition = "Clean"
                                        else:
                                            condition = "Noise"
                                    elif p == CLEAN_P and n == N_SCARCE:
                                        condition = "Scarce"
                                    elif p == HIGHDIM_P:
                                        condition = "HighDim"
                                    else:
                                        condition = "Other"

                                    fs_label = "Filter" 
                                    if fs_name == "NoFS":
                                        fs_label = "No FS"
                                    elif fs_name.startswith("Filter"):
                                        fs_label = "Filter"
                                    elif fs_name.endswith("_wrap") or "RFECV" in fs_name:
                                        fs_label = "Wrapper"
                                    else:
                                        fs_label = "Embedded"
                                        
                                    results.append({
                                        "Condition": condition,
                                        "FS":        fs_label,
                                        "Model":     mdl_name,
                                        "BalAcc_Test": metrics["BalAcc_Test"],
                                        "PRC_AUC_Test": metrics["PRC_AUC_Test"], 
                                        "Train_Time": metrics["Train_Time"],
                                        "Seed":      seed
                                    })
    return results

# ----------------------------- plotting -------------------------------------

def plot_slope_chart(df, metric="BalAcc_Test"):
    df = df[df["FS"].isin(["No FS","Filter", "Embedded", "Wrapper"])]
    order = ["Clean", "Noise", "Scarce", "Correlation", "Imbalance", "HighDim"]

    grp = (df.groupby(["Model", "FS", "Condition"])[metric]
             .mean()
             .reset_index())
    pivot = (grp.pivot_table(index=["Model", "FS"],
                             columns="Condition",
                             values=metric)
             .reindex(columns=order))

    models = pivot.index.get_level_values("Model").unique()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    axes = axes.flat

    for ax, mdl in zip(axes, models):
        sub = pivot.loc[mdl]
        if "No FS" in sub.index:
            ax.plot(order,
                    sub.loc["No FS", order],
                    marker="o",
                    linestyle="--",
                    color="black",
                    linewidth=2,
                    label="No FS")

        # 2) then plot Filter, Embedded, Wrapper normally
        for fs in ["Filter", "Embedded", "Wrapper"]:
            if fs in sub.index:
                ax.plot(order,
                        sub.loc[fs, order],
                        marker="o",
                        label=fs)

        ax.set_title(mdl)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.tick_params(axis="x", rotation=30, labelsize=9)

    for ax in axes[len(models):]:
        ax.axis("off")

    fig.suptitle(
    f"{metric}: Clean → Noise → Scarce → Correlation → Imbalance → HighDim\n"
    "(No FS vs Filter vs Embedded vs Wrapper)",
    fontweight="bold")
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"slope_chart_{metric.lower()}.png", dpi=300)
    plt.show()

# ------------------------------ driver --------------------------------------

if __name__ == "__main__":
    all_runs = []
    for seed in SEEDS:
        # main grid (no imbalance, pos_ratio = 0.5)
        all_runs.extend(
            run_experiments(
                sample_sizes=SAMPLE_SIZES,
                feature_list=[CLEAN_P, HIGHDIM_P],
                positive_ratio_list=[POS_CLEAN],
                label_flip_list=[0.0, 0.1],
                feat_noise_list=[0.0, 0.05],
                correlation_list=[False],
                seed=seed
            )
        )
        # isolated correlation (still 50‑50)
        all_runs.extend(
            run_experiments(
                sample_sizes=[N_CLEAN],
                feature_list=[CLEAN_P],
                positive_ratio_list=[POS_CLEAN],
                label_flip_list=[0.0],
                feat_noise_list=[0.0],
                correlation_list=[True],
                seed=seed
            )
        )
        # class imbalance 90‑10, otherwise clean
        all_runs.extend(
            run_experiments(
                sample_sizes=[N_CLEAN],
                feature_list=[CLEAN_P],
                positive_ratio_list=[POS_IMBAL],
                label_flip_list=[0.0],
                feat_noise_list=[0.0],
                correlation_list=[False],
                seed=seed
            )
        )

    df_runs = pd.DataFrame(all_runs)
    grp_prc = (
    df_runs
    .groupby(["Model", "FS", "Condition"])["PRC_AUC_Test"]
    .mean()
    .reset_index()
    )
    pivot_prc = (
        grp_prc
        .pivot(index=["Model", "FS"],
               columns="Condition",
               values="PRC_AUC_Test")
        .reindex(
            columns=["Clean", "Noise", "Scarce", "Correlation", "Imbalance", "HighDim"]
        )
    )
    pivot_prc.to_csv("slope_chart_prcauc.csv", index=True)


    # Balanced-accuracy slope chart (unchanged)
    plot_slope_chart(df_runs, metric="BalAcc_Test")
    
    # NEW: PRC-AUC slope chart
    plot_slope_chart(df_runs, metric="PRC_AUC_Test")
    plot_slope_chart(df_runs, metric="Train_Time")


"""
Synthetic Friedman-1 experiment
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
* sample size         : 2 000
* class imbalance     : 95 : 5
* four models         : LR, SVM, RF, MLP
* four imbalance fixes: None, Over-, Under-sampling, SMOTE
* output              : calibration_2x2.png (one subplot per model, four curves)
"""

# ───────────────────────────────── Imports ────────────────────────────── #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.datasets       import make_friedman1
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm            import SVC
from sklearn.ensemble       import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.calibration    import calibration_curve
from sklearn.metrics        import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)

from imblearn.over_sampling   import RandomOverSampler, SMOTE
from imblearn.under_sampling  import RandomUnderSampler

# ─────────────────────────── Synthetic data ──────────────────────────── #
def generate_friedman_data(n_samples=2_000, n_features=5, noise=1.0, rs=42):
    X, y_reg = make_friedman1(
        n_samples=n_samples, n_features=n_features, noise=noise, random_state=rs
    )
    return X, y_reg


def binarize_with_ratio(y_reg, desired_ratio=0.05):
    """Shift regression target so P(y ≥ 0) ≈ desired_ratio, then binarise."""
    def frac_positive(alpha):          # helper for binary-search
        return np.mean((y_reg + alpha) >= 0.0)

    lo, hi = -1_000.0, 1_000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        hi if frac_positive(mid) > desired_ratio else lo
        if frac_positive(mid) > desired_ratio:
            hi = mid
        else:
            lo = mid
    alpha = (lo + hi) / 2.0
    return ((y_reg + alpha) >= 0.0).astype(int)


# ─────────────────────── Class imbalance “fixes” ─────────────────────── #
def apply_fix(X, y, fix="none", rs=42):
    fix = fix.lower()
    if fix == "none":
        return X, y
    if fix == "over sampling":
        return RandomOverSampler(random_state=rs).fit_resample(X, y)
    if fix == "under sampling":
        return RandomUnderSampler(random_state=rs).fit_resample(X, y)
    if fix == "smote":
        n_min = int(y.sum())
        k = max(min(5, n_min - 1), 1)
        try:
            return SMOTE(random_state=rs, k_neighbors=k).fit_resample(X, y)
        except ValueError:
            return RandomOverSampler(random_state=rs).fit_resample(X, y)
    raise ValueError(f"Unknown fix {fix}")


# ─────────────────────────── Train & score ───────────────────────────── #
def train_once(X_tr, y_tr, X_te, y_te, fix, rs, calib_store):
    """Returns dict of metric-dicts per model & pushes preds into calib_store."""
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    X_bal, y_bal   = apply_fix(X_tr_s, y_tr, fix, rs)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=10_000, random_state=rs),
        "SVM"               : SVC(probability=True, random_state=rs),
        "RandomForest"      : RandomForestClassifier(random_state=rs),
        "MLP"               : MLPClassifier(max_iter=10_000, random_state=rs),
    }

    results = {}
    for name, clf in models.items():
        clf.fit(X_bal, y_bal)
        y_pred  = clf.predict(X_te_s)
        y_score = clf.predict_proba(X_te_s)[:, 1]

        # store for calibration curves
        calib_store[name][fix]["y_true"].append(y_te)
        calib_store[name][fix]["y_prob"].append(y_score)

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        results[name] = {
            "Accuracy"         : accuracy_score(y_te, y_pred),
            "F1_score"         : f1_score(y_te, y_pred, pos_label=1),
            "Sensitivity"      : tp / (tp + fn) if (tp + fn) else 0,
            "Specificity"      : tn / (tn + fp) if (tn + fp) else 0,
            "ROC_AUC"          : roc_auc_score(y_te, y_score),
            "PR_AUC"           : average_precision_score(y_te, y_score),
            "Balanced_Accuracy": balanced_accuracy_score(y_te, y_pred),
            "MCC"              : matthews_corrcoef(y_te, y_pred),
        }
    return results


# ────────────────────────────── Main run ─────────────────────────────── #
if __name__ == "__main__":
    sns.set_style("whitegrid")
    sns.set_context("talk")

    SAMPLE_SIZE  = 2_000
    DESIRED_RATIO = 0.05          # 95 : 5
    TEST_SIZE    = 0.30
    SEEDS        = range(1, 11)   # 1 … 10
    FIXES        = ["None", "Over Sampling", "Under Sampling", "SMOTE"]

    calib_store = defaultdict(lambda: defaultdict(
        lambda: {"y_true": [], "y_prob": []})
    )

    # run experiments
    for seed in SEEDS:
        # synthetic data
        X, y_reg = generate_friedman_data(SAMPLE_SIZE, noise=1.0, rs=seed)
        y_cls    = binarize_with_ratio(y_reg, desired_ratio=DESIRED_RATIO)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr_idx, te_idx = next(sss.split(X, y_cls))
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_cls[tr_idx], y_cls[te_idx]

        for fix in FIXES:
            train_once(X_tr, y_tr, X_te, y_te, fix, seed, calib_store)

    # ─────────────── Calibration curves: 2 × 2, one model per panel ───────────── #
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (ax, model) in enumerate(zip(axes, ["LogisticRegression", "SVM", "RandomForest", "MLP"])):
        for fix in FIXES:
            y_t = np.concatenate(calib_store[model][fix]["y_true"])
            y_p = np.concatenate(calib_store[model][fix]["y_prob"])
            frac_pos, mean_pred = calibration_curve(y_t, y_p, n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", label=fix)

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_title(model, fontsize=12)
        if idx % 2 == 0:          # first column
            ax.set_ylabel("Fraction of positives", fontsize=11)
        if idx >= 2:              # last row
            ax.set_xlabel("Mean predicted probability", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Fix", bbox_to_anchor=(1.02, 0.5),
               loc="center left", frameon=False, fontsize=10, title_fontsize=11)

    fig.suptitle("Calibration curves – Friedman-1, 95 : 5 imbalance, n=2 000", y=0.95)
    fig.tight_layout()
    fig.savefig("calibration_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Done!  Figure saved as calibration_2x2.png")

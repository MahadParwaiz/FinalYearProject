# What Makes Supervised Learning Hard?

This repository contains the code and results for my final year project at Imperial College London. The work investigates why supervised learning sometimes breaks down by creating controlled synthetic benchmarks.

## Project Overview

The project is organised around four research questions (RQs):

1. **High Dimensionality (RQ1)** – How does increasing the number of features affect common regression algorithms?
2. **Class Imbalance (RQ2)** – What impact does skewed class distributions have on classification models and how do common resampling fixes compare?
3. **Feature Correlation & Uninformative Predictors (RQ3)** – How do correlated or irrelevant features influence model interpretation and performance?
4. **Feature Selection Strategies (RQ4)** – Can feature selection mitigate the above challenges without harming predictive accuracy?

Early exploratory scripts such as `Non_Linearity.py`, `LearnedSurface.py`, and `PCA.py` demonstrate baseline experiments in low dimensional settings. The RQ scripts extend these ideas to larger and more complex data.

---

## Repository Structure

- `Coding files/`
  - `RQ1/` – Experiments on high-dimensional Friedman regression datasets.
  - `RQ2/` – Classification experiments with heavy class imbalance and various resampling strategies (none, over-sampling, under-sampling, SMOTE).
  - `RQ3/` – Scripts exploring feature correlation and the effect of adding uninformative variables using permutation importance, SHAP values and VIF calculations.
  - `RQ4/` – Classification pipelines evaluating filter, wrapper and embedded feature-selection methods.
- `Figures/` – Generated figures for each RQ.
- `Raw data (Csv files)/` – CSV outputs summarising metrics and results.
- Baseline scripts in the project root: `LearnedSurface.py`, `Non_Linearity.py`, `PCA.py`.

---

## Dependencies

The project uses Python 3 with the following main packages:

- `numpy` and `pandas`
- `scikit-learn`
- `matplotlib` and `seaborn`
- `shap` and `statsmodels` (for RQ3)

Install all dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap statsmodels
```

---

## How to Run

1. Clone the repository and install the dependencies.
2. Run individual RQ experiments from the `Coding files` directory, for example:

```bash
python 'Coding files/RQ1/RQ1_HighDimensionality2.py'
python 'Coding files/RQ2/RQ2_ClassImbalance_withFix4.py'
python 'Coding files/RQ3/Uninformative_latest.py'
python 'Coding files/RQ4/Final_with_training_time.py'
```

3. Output figures will be saved to the `Figures` folder and tabulated results to `Raw data (Csv files)`.

The simpler baseline scripts in the repository root can be run directly (e.g. `python Non_Linearity.py`) to reproduce the introductory analysis.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Supervisor**: Dr. Pedro Ballester, Imperial College London.
- **Research Student**: Josh Fitch, Imperial College London.
- Many aspects of the benchmark design were inspired by related studies on supervised learning challenges.

---

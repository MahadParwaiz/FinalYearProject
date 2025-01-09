
# What Makes Supervised Learning Hard?

This repository contains the implementation, scripts, and results for my final year project at Imperial College London. The project investigates the factors that make supervised learning challenging, using synthetic benchmarks for controlled experimentation.

## Project Overview

Supervised learning is a core branch of machine learning that builds models to predict labels from features. While effective in many applications, supervised learning faces challenges due to factors such as high dimensionality, feature correlation, noise, sparsity, and class imbalance. The project aims to:
1. Characterize the impact of these challenges on supervised learning algorithms.
2. Evaluate algorithm suitability under controlled synthetic benchmarks.
3. Provide guidelines for algorithm selection based on dataset characteristics.

The work is divided into several phases, starting with foundational experiments in 2D regression problems and scaling up to higher dimensions and more complex scenarios.

---

## Repository Structure

- **`Linearity.py`**: Python script for analyzing the performance of supervised learning algorithms on linear regression benchmarks. Includes visualization of regression surfaces and evaluation of model accuracy using metrics such as R².
- **`Non_Linearity.py`**: Python script for extending the analysis to nonlinear regression problems. Demonstrates the limitations of linear models on complex data and compares their performance with algorithms such as Random Forests.
- **`Planning_Report.docx`**: The official planning report documenting the project's aims, methodology, preliminary results, and evaluation strategy.

---

## Key Experiments

### 1. Regression Benchmark Analysis
- **Objective**: To study the performance of supervised learning algorithms on synthetic regression datasets with varying complexities.
- **Tools**: Utilized `make_regression` from `scikit-learn` to generate data with controlled noise, feature relevance, and dimensionality.
- **Results**:
  - Linear regression performs well in purely linear cases (R² = 1.0).
  - Significant degradation (R² = ~0.047) in performance observed on nonlinear data.

### 2. Nonlinearity Exploration
- **Objective**: Investigate the behavior of linear and nonlinear models on complex feature relationships.
- **Approach**:
  - Simulate nonlinear interactions using polynomial and exponential terms.
  - Compare model performance using visualization and quantitative metrics.
- **Findings**:
  - Random Forests adapt better to nonlinearities compared to linear models.
  - Visualizations highlight gradient differences across models.

---

## Dependencies

The project is implemented in Python. Install the required packages using:
```bash
pip install -r requirements.txt
```

### Key Libraries
- **`scikit-learn`**: For dataset generation and machine learning models.
- **`matplotlib`** & **`seaborn`**: For visualizations.
- **`numpy`** & **`pandas`**: For data processing.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/what-makes-supervised-learning-hard.git
   cd what-makes-supervised-learning-hard
   ```
2. Run experiments:
   - For linear regression: `python Linearity.py`
   - For nonlinear regression: `python Non_Linearity.py`

3. Outputs:
   - Visualizations of regression surfaces.
   - Evaluation metrics displayed in the terminal.

---

## Next Steps

- Extend analysis to higher dimensions and larger datasets.
- Introduce additional complexities such as noise, feature correlation, and sparsity.
- Benchmark classification tasks using synthetic datasets with controlled class imbalance.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Supervisor**: Dr. Pedro Ballester, Imperial College London.
- **Research Student**: Josh Fitch, Imperial College London.
- Contributions to the synthetic benchmark framework inspired by multiple peer-reviewed studies on supervised learning challenges.

---

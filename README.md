# Pulsar Star Classification using SVM

This project presents a comprehensive SVM-based analysis for classifying pulsar stars using a real-world astrophysical dataset. With over 17,900 samples and highly imbalanced classes, the study demonstrates the application of various SVM kernels, hyperparameter tuning, and performance visualization techniques.

## Dataset Overview

- **Source**: [UCI ML Repository – Pulsar Dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2)
- **Total Samples**: 17,901
- **Features**: 8 continuous statistical features derived from radio frequency data
- **Target**: Binary label indicating pulsar (1) or non-pulsar (0)
- **Imbalance**: Only ~9.2% of observations are pulsars

## Objectives

- Clean and preprocess the dataset (handle missing values, encode labels)
- Perform exploratory data analysis (EDA) and visualize feature distributions
- Apply various SVM classifiers:
  - `SVC` (RBF, Linear, Polynomial, Sigmoid)
  - `LinearSVC` (with and without L1 penalty)
  - `NuSVC` (RBF and Linear)
- Tune hyperparameters using `GridSearchCV`
- Evaluate performance with accuracy, F1-score, ROC curves, and confusion matrices
- Visualize insights with matplotlib and seaborn

## Key Results

- **Best Model**: `SVC` with RBF kernel (C=10)
- **Test Accuracy**: 98.13%
- **F1-Score**: 0.9809
- **Training Time**: ~0.43 seconds

| Kernel Type | Accuracy (avg) | Std Dev |
|-------------|----------------|---------|
| RBF         | 97.99%         | ±0.15%  |
| Linear      | 97.87%         | ±0.27%  |
| Polynomial  | 97.85%         | ±0.15%  |
| Sigmoid     | 87.69%         | N/A     |

## Visualizations

- Feature Distributions
- Correlation Heatmap
- ROC Curves
- Confusion Matrices
- Kernel Comparison Heatmaps

All saved in the `plots/` directory.

## How to Run

```bash
git clone https://github.com/Gauravpawar101/svm-eval.git
cd svm-eval
pip install -r requirements.txt
python main.py
```
Tools and Libraries
Python 3.x

pandas, numpy, matplotlib, seaborn

scikit-learn (SVC, GridSearchCV, StandardScaler)

joblib (for model caching)
```
Project Structure
pulsar-svm-analysis/
├── data/
│   └── pulsar_dataset.csv
├── plots/
│   ├── feature_distributions.png
│   ├── confusion_matrix_rbf.png
│   └── ...
├── All_svm.py
└── README.md
```

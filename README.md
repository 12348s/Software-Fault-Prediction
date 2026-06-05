# Software Fault Prediction on NASA MW1 Dataset

Binary defect prediction on the NASA MW1 dataset using a full ML pipeline —
from raw ARFF data to explainable AI with SHAP.

## Problem

Identifying defective software modules from static code metrics on a heavily
imbalanced dataset (226 non-defective vs 27 defective modules in MW1).
Accuracy is a misleading metric here — F1 and ROC-AUC are the primary targets.

## Pipeline Overview

| Step | Description |
|------|-------------|
| 1. EDA | Feature distributions, class imbalance visualization |
| 2. Correlation Analysis | Pairwise correlations, multicollinearity check |
| 3. Advanced Feature Analysis | Mutual Information scores, VIF, variance thresholds |
| 4. Log Transformation | log1p applied to count/volume metrics to reduce skew |
| 5. Outlier Handling | 1st–99th percentile capping (preserves all 253 samples) |
| 6. Feature Selection — VIF | Domain drop + iterative VIF elimination → 13 features |
| 7. Feature Selection — RFECV + MI | Intersection of RFECV and MI top-8 → 5 final features |
| 8. Baseline Evaluation | 5-fold stratified CV across 8 classifiers with SMOTEENN |
| 9. Hyperparameter Tuning | Bayesian search (BayesSearchCV, 30 iterations) |
| 10. Threshold Calibration | Optimal threshold via precision-recall curve (OOF predictions) |
| 11. SHAP Analysis | Bar, beeswarm, waterfall, force, and dependence plots |
| 12. Final Results Summary | 3-stage F1/AUC comparison + ROC curves for top 3 models |

## Key Design Decisions

- **SMOTEENN inside the pipeline** — synthetic samples are generated within
  each CV fold, not before splitting, preventing data leakage
- **Bayesian search over Grid search** — 30 iterations converge faster than
  exhaustive grid search on this search space
- **Threshold calibration** — default 0.5 threshold is suboptimal for
  imbalanced data; optimal threshold is found per model using OOF probabilities
- **VIF elimination before RFECV** — removes multicollinear features first so
  RFECV selects on genuinely independent signals

## Models Evaluated

Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting,
CatBoost, MLP, XGBoost

## Results (MW1, post-tuning and threshold calibration)

| Model | F1 | ROC-AUC |
|-------|----|---------|
| CatBoost | best F1 | — |
| Logistic Regression | — | best AUC / best Recall |

> Full results table and ROC curves are in the final notebook section.

## Dataset

NASA MW1 from the PROMISE repository — publicly available.  
Download the `.arff` file and place it in the working directory before running.

- Source: https://github.com/klainfo/NASADefectDataset
- 253 modules, 38 static code features, binary defect label (Y/N)

## Stack
scikit-learn
imbalanced-learn
xgboost
catboost
shap
scikit-optimize
pandas
numpy
matplotlib
seaborn
statsmodels
liac-arff

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook MW1.ipynb
```

Place `MW1.arff` in the same directory before running cell 1.

## Status

MW1 complete.  
Part of a research internship at SRM University AP (June–July 2025)  

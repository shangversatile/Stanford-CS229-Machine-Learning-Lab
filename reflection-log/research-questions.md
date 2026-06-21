# Research Questions

## Supervised Learning

### 1. Heavy-tailed Noise and Linear Regression

Question: How does linear regression fail under heavy-tailed noise compared with Gaussian noise?

Why it matters: CS229 derives least squares partly through Gaussian noise assumptions, but real data may violate this assumption. This matters for reliable regression because outliers can dominate squared loss.

Possible mini experiment: Generate synthetic linear data with Gaussian noise, Laplace noise, Student-t noise, and injected outliers; compare ordinary least squares, ridge regression, and a simple robust loss approximation.

### 2. Feature Scaling and Gradient Descent Stability

Question: How sensitive is gradient descent for linear or logistic regression to feature scaling?

Why it matters: Optimization behavior can be mistaken for model quality. Poor scaling can slow convergence or create unstable updates even when the model class is appropriate.

Possible mini experiment: Train from-scratch linear regression on the same synthetic dataset with raw, standardized, and badly scaled features; compare loss curves, parameter error, and convergence speed.

## Generative vs Discriminative Modeling

### 3. Small-sample Behavior of GDA vs Logistic Regression

Question: When does Gaussian Discriminant Analysis outperform logistic regression in small-sample classification?

Why it matters: CS229 contrasts generative and discriminative approaches. Generative assumptions may help in small data regimes, but only when assumptions are approximately true.

Possible mini experiment: Generate two-class Gaussian data with varying sample sizes and covariance structures; compare GDA and logistic regression on accuracy, calibration, and boundary stability.

### 4. Naive Bayes Under Dependence Violation

Question: How quickly does Naive Bayes degrade when conditional independence assumptions are violated?

Why it matters: Naive Bayes can perform well despite unrealistic assumptions, but reliability requires knowing when this breaks.

Possible mini experiment: Generate binary features with controllable correlation within each class; compare Naive Bayes and logistic regression as correlation strength increases.

## Generalization and Reliability

### 5. Regularization Under Distribution Shift

Question: Does ridge regularization improve test reliability when train and test distributions differ moderately?

Why it matters: Regularization is often justified by variance control, but reliability requires testing whether it helps under shifted data, not only random train/test splits.

Possible mini experiment: Train polynomial regression on one input range and test on shifted ranges; compare unregularized regression, ridge regression, and model selection by validation error.

### 6. Bias-variance Diagnostics From Learning Curves

Question: Can learning curves reliably distinguish underfitting from overfitting in small CS229-style experiments?

Why it matters: Bias-variance language is useful only if it leads to concrete diagnosis and action.

Possible mini experiment: Fit polynomial models of different degrees to synthetic nonlinear data; plot train/validation error as sample size grows and check whether the curves match expected bias-variance patterns.

### 7. Metric Mismatch in Classification

Question: When does accuracy hide reliability problems in imbalanced logistic regression tasks?

Why it matters: A model can achieve high accuracy by ignoring minority classes. This is directly relevant to trustworthy evaluation.

Possible mini experiment: Create imbalanced binary classification data; compare accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and calibration for logistic regression under different thresholds.

## Representation Analysis

### 8. PCA Stability Under Noise

Question: How stable are PCA directions when the data contains noise, outliers, or nearly equal eigenvalues?

Why it matters: PCA is a foundational representation method, but its components can be unstable when signal-to-noise ratio is low.

Possible mini experiment: Generate low-rank data with controlled Gaussian noise and outliers; measure angle differences between recovered and true principal components.

### 9. Kernel Choice and Representation Geometry

Question: How does kernel choice change the effective representation and decision boundary in SVM-style classification?

Why it matters: Kernels can look like a technical trick, but they encode assumptions about similarity and geometry.

Possible mini experiment: Use toy datasets such as circles, moons, and linearly separable data; compare linear, polynomial, and RBF kernels using decision boundary plots and validation performance.

## Evaluation and Benchmarking

### 10. Sanity Checks for From-scratch Implementations

Question: What minimal sanity checks are sufficient before trusting a from-scratch CS229 implementation?

Why it matters: Passing a single benchmark does not prove correctness. Research code needs small controlled tests before larger experiments.

Possible mini experiment: Build a checklist for linear regression, logistic regression, k-means, PCA, and EM; include known-solution synthetic data, shape checks, monotonic objective checks, and comparison with sklearn where appropriate.

### 11. Validation Leakage in Model Selection

Question: How much can repeated validation-set tuning inflate reported performance?

Why it matters: Portfolio experiments can accidentally overfit the validation set, producing unreliable claims.

Possible mini experiment: Simulate repeated hyperparameter choices on noisy data; compare validation-selected performance with held-out test performance over many random seeds.

## AI for Science / Spatiotemporal Forecasting

### 12. Linear Baselines for Spatiotemporal Forecasting

Question: How strong can simple linear or ridge baselines be for spatiotemporal forecasting before using complex deep models?

Why it matters: Reliable AI for Science needs strong baselines. Without them, deep learning gains may be overstated.

Possible mini experiment: Use a small synthetic spatiotemporal dataset with trend, seasonality, and noise; compare persistence, linear regression with lag features, ridge regression, and a simple nonlinear baseline.

### 13. Distribution Shift in Forecasting Noise Patterns

Question: How do regression models behave when training noise and deployment noise have different structure?

Why it matters: Scientific forecasting often faces nonstationarity, sensor changes, or regime shifts. A model that works under one noise pattern may fail under another.

Possible mini experiment: Train on Gaussian-noise synthetic time series and test on bursty or heavy-tailed noise; compare prediction error, residual distribution, and failure cases.

# Research Questions

## Lecture 1 Research Questions

### Q1. How does biased training data affect learned decision boundaries?

Question: How does sampling bias in the training set change the decision boundary learned by a supervised classifier?

Why it matters: Supervised learning assumes the training set is informative about future data, but real datasets often overrepresent some groups, regions, or regimes. A model may learn the geometry of the sample rather than the geometry of the target population.

Possible mini experiment: Generate two 2D binary classification datasets with the same class-conditional distributions but different group proportions; train logistic regression and compare decision boundaries, subgroup error, and calibration.

### Q2. How robust is supervised learning to label noise?

Question: How does increasing label noise affect classification accuracy, calibration, and learned parameters?

Why it matters: Labels are treated as supervision, but in medical, scientific, and human-annotated datasets they may contain mistakes or subjective bias. Reliable ML requires knowing when noisy labels distort the learned hypothesis.

Possible mini experiment: Create a binary classification dataset and randomly flip 0%, 5%, 10%, 20%, and 40% of labels; compare logistic regression training loss, test accuracy, confidence, and parameter stability.

### Q3. When does regression become classification by thresholding?

Question: What changes when a continuous target is converted into discrete labels by thresholding?

Why it matters: Regression and classification differ not only in output type but also in loss, evaluation, and error interpretation. Thresholding can hide magnitude information and create artificial decision boundaries.

Possible mini experiment: Generate continuous risk scores, train one regression model on the original target and one classifier on thresholded labels; compare ranking quality, threshold sensitivity, and errors near the cutoff.

### Q4. Which evaluation metric best exposes failure under class imbalance?

Question: When classes are imbalanced, which metrics reveal failures that accuracy hides?

Why it matters: A high-accuracy classifier can be unreliable if it ignores rare but important classes. Lecture 1's evaluation framing implies that metric choice is part of the learning problem.

Possible mini experiment: Train logistic regression on imbalanced synthetic data; report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and calibration across different thresholds.

### Q5. How does representation change generalization?

Question: How do different feature representations affect model performance on unseen data?

Why it matters: The same algorithm can succeed or fail depending on whether the representation exposes the relevant structure. This connects Lecture 1's feature intuition to later kernel methods and representation analysis.

Possible mini experiment: Use a nonlinear 2D dataset such as concentric circles; compare logistic regression on raw features, polynomial features, and radial features, then evaluate decision boundaries and held-out error.

### Q6. Can low training error coexist with poor reliability?

Question: Under what conditions does a model achieve low training error but fail on shifted test data?

Why it matters: Generalization is the central tension of ML. Training performance alone cannot justify reliability claims, especially when deployment data differs from training data.

Possible mini experiment: Train polynomial regression on one input interval and test on a shifted interval; compare training error, validation error, shifted test error, and curve behavior for different polynomial degrees.

### Q7. How does objective mismatch create systematic failure?

Question: What happens when the training loss does not match the real-world deployment objective?

Why it matters: Loss defines what the model treats as good. A model can optimize the specified loss while failing the actual task, especially when errors have asymmetric costs.

Possible mini experiment: Simulate a medical screening task with different costs for false negatives and false positives; train classifiers using standard logistic loss and compare threshold choices under cost-sensitive evaluation.

### Q8. What is the minimum reliable supervised learning pipeline?

Question: What components must be specified before a supervised learning result is meaningful?

Why it matters: Lecture 1 frames ML as a pipeline, not a single algorithm. Data source, representation, hypothesis class, objective, optimization, evaluation, and generalization target must all be explicit.

Possible mini experiment: Build a tiny supervised learning checklist and apply it to housing-price regression, spam classification, and a toy medical diagnosis task; identify which reliability assumptions are missing in each case.

### Q9. How should spatiotemporal forecasting be evaluated under distribution shift?

Question: How should forecasting models be evaluated when future time periods, spatial regions, or sensor conditions differ from training data?

Why it matters: AI for Science and spatiotemporal forecasting often involve missing data, noisy measurements, and nonstationary distributions. Average error on random splits can overstate deployment reliability.

Possible mini experiment: Generate synthetic spatiotemporal data with trend, seasonality, missing values, and a regime shift; compare random split, time-based split, and region-held-out evaluation for linear baselines.

## Lecture 1 Preparation

### 0. Learning Setup and Reliability

Question: What has to be specified before a supervised learning result can be called reliable?

Why it matters: Lecture 1 introduces the basic learning setup. For research use, I need to make explicit the data distribution, features, labels, hypothesis space, learning algorithm, evaluation protocol, and generalization target before trusting any result.

Possible mini experiment: Use a tiny binary classification toy dataset and evaluate the same model under different train/test splits, metrics, and class balances; document how the reliability claim changes.

### 0.1 Evaluation Design as Part of Learning

Question: How can evaluation design change the apparent quality of the same hypothesis?

Why it matters: A model is not reliable just because it performs well on one benchmark. Lecture 1 should be read with the expectation that evaluation is part of the learning problem, not a separate reporting step.

Possible mini experiment: Train one simple classifier and report accuracy, balanced accuracy, precision/recall, and calibration under class imbalance; compare which claims each metric supports.

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

# PS1 Implementation Workplan

| Implementation | Related Theory | Minimal Test | Stress Test | Status |
| -------------- | -------------- | ------------ | ----------- | ------ |
| LinearRegressionGD | least squares gradient descent | Recover parameters on simple synthetic linear data. | Feature scaling and learning-rate sensitivity. | Pending |
| NormalEquationSolver | normal equation | Match gradient descent on a well-conditioned dataset. | Collinearity and near-singular design matrix. | Pending |
| LogisticRegressionGD | logistic likelihood and gradient | Fit a linearly separable two-class toy dataset. | Imbalanced labels and poorly scaled features. | Pending |
| LogisticRegressionNewton | logistic gradient and Hessian | Match gradient descent predictions on a small convex case. | Ill-conditioned Hessian and step-size damping. | Pending |
| LocallyWeightedRegression or LocallyWeightedLogisticRegression | local weighting and query-dependent objectives | Show different predictions as query location changes. | Tau sensitivity and sparse local neighborhoods. | Pending |
| DecisionBoundaryPlotter | logistic classification geometry | Plot a stable boundary for a two-feature dataset. | Boundary behavior under outliers. | Pending |
| TauSensitivityExperiment | locality weighting | Compare predictions across small, medium, and large tau. | Overfitting at very small tau and underfitting at very large tau. | Pending |

## Implementation Rule

Implement only after derivation is understood. Do not copy public solutions.

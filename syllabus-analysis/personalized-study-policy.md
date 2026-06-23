# Personalized CS229 Study Policy

## 1. Why This Course Should Not Be Studied as a Beginner Course

This repository should not treat CS229 as a beginner survey course. The learner already has partial machine learning background, practical exposure to Python, sklearn, and PyTorch, and prior experience with a robust spatiotemporal forecasting project.

Therefore, the goal is not to memorize basic machine learning categories or produce lecture summaries for every familiar concept. CS229 should be used to rebuild the mathematical, algorithmic, and experimental foundations needed for research-level work. Familiar overview material should be compressed, while derivations, assumptions, implementations, tests, and failure modes should be expanded.

## 2. Core Learning Objective

Use CS229 to build a mathematically grounded, implementation-verified, and reliability-aware foundation for later work in Trustworthy ML, Reliable AI Systems, LLM evaluation, Representation Analysis, Causal Reliability, AI for Science, and PhD preparation.

The repository should answer not only "what is the algorithm?" but also:

* What objective is being optimized?
* What assumptions make the objective meaningful?
* How is the update rule derived?
* How can the implementation be checked?
* Where can the model fail?
* What does the result imply for reliability and generalization?

## 3. Depth Allocation Rule

| Content Type | Treatment |
| ------------ | --------- |
| Familiar overview concepts | Compressed summary focused on terminology and placement in the course map. |
| Core mathematical derivations | Full derivation with intermediate steps and notation checks. |
| Probabilistic assumptions | Explicit analysis of what is assumed, why it leads to the objective, and how violations affect reliability. |
| Optimization methods | Derivation plus implementation when central to the algorithm. |
| Algorithms | From-scratch implementation when the algorithm is central to CS229 foundations. |
| Experiments | Sanity checks plus stress tests for robustness, generalization, and failure modes. |
| Research connections | Focused and necessary only; connect to reliability or representation when it clarifies the CS229 concept. |
| Broad DL/LLM/system extensions | Avoid over-expansion unless they directly clarify a CS229 foundation. |

## 4. Per-Lecture Study Protocol

For each lecture, use the following protocol:

1. **Core question**: State the central learning problem the lecture is trying to solve.
2. **Objective function**: Identify the loss, likelihood, margin objective, or optimization target.
3. **Modeling assumption**: Make explicit the data, noise, distributional, independence, or geometry assumptions.
4. **Mathematical derivation**: Derive the key equations rather than only quoting final formulas.
5. **Implementation target**: Decide whether this lecture requires a from-scratch implementation, a utility function, a test, or no code.
6. **Sanity check**: Define the smallest controlled example that should pass if the derivation and implementation are correct.
7. **Failure mode**: Record how the method can fail under assumption violation, optimization issues, data problems, metric mismatch, or distribution shift.
8. **Research connection**: Add only the research connection that directly strengthens understanding of the CS229 concept.
9. **Commit output**: Each commit should leave a concrete artifact: a note, derivation, implementation, test, experiment report, or critical question.

This protocol is meant to prevent passive watching. Every important lecture should become either a mathematical artifact, an implementation artifact, or a reliability-oriented analysis artifact.

## 5. High-Priority Modules

* **Linear Regression**: The first full bridge from supervised learning setup to objective, least squares, gradient descent, normal equation, and probabilistic interpretation.
* **Logistic Regression**: The basic classification model connecting linear decision boundaries, probabilities, likelihood, cross-entropy, and calibration concerns.
* **GLM**: The conceptual generalization that links exponential family assumptions to model form and learning objectives.
* **Generative Learning Algorithms**: Essential for understanding modeling data generation, assumptions, Bayes rule, and discriminative versus generative tradeoffs.
* **SVM and Kernels**: Core for margin geometry, convex optimization, duality, and feature-space reasoning.
* **Bias-Variance and Regularization**: Central to model complexity, generalization, overfitting, and reliability under limited data.
* **Learning Theory**: Necessary for distinguishing empirical performance from generalization claims.
* **EM / GMM**: Important for latent-variable modeling, likelihood lower bounds, and iterative optimization.
* **PCA / ICA**: Foundational for representation analysis, dimensionality reduction, latent structure, and later interpretability work.
* **Basic Reinforcement Learning**: Worth studying for value functions, Bellman equations, sequential decision-making, and the limits of supervised learning framing.

## 6. Low-Priority or Compressed Modules

The following should be compressed unless they become directly relevant to a derivation, implementation, or research question:

* Basic overview of ML subfields.
* Broad deep learning survey material.
* Broad reinforcement learning applications.
* Non-core examples that do not affect the main mathematical arc.
* Overly large final projects that distract from foundational mastery.
* Unrelated LLM/system expansions.

These topics are not unimportant. They are compressed because the repository's current role is to strengthen CS229 foundations, not to expand every topic into a separate research program.

## 7. Research-Oriented Output Standard

Every important module should produce at least one of:

* lecture note
* math derivation
* from-scratch implementation
* unit test
* stress test
* short experiment report
* critical question
* mini research idea

The output should be verifiable. A polished note without derivation, implementation, or evaluation value is not enough for a high-priority module.

## 8. Relationship to Existing Repositories

This CS229 repository remains independent during course study, but it can inform later work in related repositories:

* **ML-DL-NLP-Lab**: Later reusable implementation foundations may be generalized there after the CS229 versions are derived and tested.
* **Paper-Reading-Notes**: Important classical papers or theory notes can be synced later when they directly support a CS229 module.
* **Reliable-AI-Research-Lab**: Reliability experiments and final project ideas may migrate later after the CS229 foundations are complete.
* **Representation-Analysis-Lab**: PCA, ICA, GMM, and representation analysis modules can connect later when the mathematical foundation is mature.

During CS229 study, this repository should not depend on those repositories. Cross-repository migration should happen only after a concept is independently understood and documented here.

## 9. Anti-Patterns

Avoid:

* writing beautiful notes without derivations
* watching lectures without experiments
* copying public solutions
* expanding every topic into LLM research
* implementing before understanding assumptions
* treating benchmark performance as reliability
* confusing familiarity with mastery

The strongest risk is mistaking recognition for understanding. If a concept feels familiar, it should still be tested by derivation, implementation, or a controlled example before being marked as mastered.

## 10. Immediate Next Step

The next learning target is Lecture 2: Supervised Learning Setup and Linear Regression.

Lecture 2 must produce:

* lecture note
* least squares derivation
* Gaussian noise / MLE interpretation
* normal equation derivation
* gradient descent derivation
* failure mode analysis
* preparation for NumPy from-scratch implementation

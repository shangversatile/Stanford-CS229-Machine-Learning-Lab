# Stanford CS229 Machine Learning Lab

This repository is a research-oriented learning lab for Stanford CS229: Machine Learning, based primarily on the Autumn 2018 offering taught by Andrew Ng.

The goal is not to passively watch lectures or collect notes. The goal is to rebuild the core mathematical, algorithmic, and experimental foundations of machine learning through derivations, from-scratch implementations, tests, benchmark-style experiments, and research-style reflections.

## Personalized Study Strategy

This repository follows a depth-weighted CS229 strategy built around prior ML study, practical implementation experience, and research preparation. Basic concepts are compressed, while mathematical derivations, assumptions, implementations, tests, and reliability-oriented experiments are expanded.

See: [syllabus-analysis/personalized-study-policy.md](syllabus-analysis/personalized-study-policy.md)

## 1. Course Positioning

CS229 is treated here as a foundational machine learning course for advanced AI study and research preparation. It provides the mathematical and algorithmic base for later work in trustworthy machine learning, reliable AI systems, representation analysis, causal reliability, LLM evaluation, and AI for Science.

## 2. Learning Principles

1. Derivation before implementation.
2. Implementation before library abstraction.
3. Tests before claims.
4. Error analysis before model improvement.
5. Critical reflection before portfolio presentation.

## 3. Core Modules

1. Mathematical and Python review
2. Supervised learning and linear models
3. Logistic regression, Newton's method, and GLMs
4. Generative learning algorithms
5. SVMs and kernel methods
6. Bias-variance, regularization, and model selection
7. Tree ensembles and neural network basics
8. Unsupervised learning: k-means, GMM, EM, PCA, ICA
9. Reinforcement learning and control
10. Final research-style mini project

## 4. Repository Structure

* `syllabus-analysis/`: Course positioning, official resource mapping, and module dependency planning.
* `lecture-notes/`: Lecture-level notes focused on core questions, derivations, implementation insights, and research connections.
* `math-derivations/`: Standalone derivation records for core algorithms and probabilistic interpretations.
* `assignments/`: Independent assignment workspaces and the academic integrity policy for public repository use.
* `labs/`: From-scratch implementation labs that connect CS229 theory with reproducible code.
* `src/`: Reusable Python package for implementations, optimization utilities, metrics, data helpers, and visualization code.
* `tests/`: Unit tests and future gradient checks for implementation correctness.
* `experiments/`: Experiment configs, reproducible run notes, and generated outputs.
* `paper-notes/`: Reading templates and selected paper notes connected to CS229 topics.
* `reports/`: Weekly reports, final project report, and one-page project summary.
* `concept-maps/`: Global and module-specific concept maps for tracking dependencies across ideas.
* `assets/`: Figures and diagrams used in notes, reports, and summaries.
* `reflection-log/`: Weekly reflections, open research questions, and learning decisions.

## 5. Academic Integrity Statement

This repository is for independent learning and research preparation. Official problem statements, solution keys, private course materials, and copied student solutions should not be publicly committed. Public solution repositories may be used only for debugging or conceptual comparison after independent attempts.

All code and notes in this repository should be written independently unless explicitly marked otherwise.

## Assignment Gates

This repository now follows an assignment-gated learning process. Notes and derivations are necessary but not sufficient. Each lecture cluster must be linked to public-safe problem set work.

Current gate:

PS1 Supervised Learning Gate

Links:

* [assignments/problem-set-map.md](assignments/problem-set-map.md)
* [assignments/ps1-supervised-learning/README.md](assignments/ps1-supervised-learning/README.md)
* [assignments/ps1-supervised-learning/ps1-gate-checklist.md](assignments/ps1-supervised-learning/ps1-gate-checklist.md)

## 6. Final Deliverables

By the end of this course lab, the repository should contain:

* 8 to 15 high-quality lecture notes
* Mathematical derivations for core algorithms
* From-scratch implementations of major CS229 algorithms
* Unit tests and gradient checks
* Reproducible experiments and visualizations
* A final research-style mini project
* A 5-page final report
* A 1-page project summary
* A resume-ready project description
* An English abstract suitable for contacting professors or research mentors

## 7. Current Progress

* [x] Repository initialized
* [x] Official resource map created
* [x] Course positioning note written
* [x] Module dependency graph created
* [x] Personalized study policy created
* [x] Lecture 1 note completed
* [x] Lecture 2 linear regression derivation completed
* [x] Lecture 3 note completed
* [x] ML vs probabilistic modeling perspective added
* [x] Frequentist vs Bayesian contrast added
* [x] Markdown math rendering audited
* [x] PS1 supervised learning gate initialized
* [x] Locally weighted regression derivation completed
* [x] Logistic regression gradient derivation completed
* [x] Newton method derivation completed
* [ ] Linear regression from scratch implemented
* [ ] First tests completed

Implementation intentionally follows derivation: the next stage will implement gradient descent and the normal-equation baseline, then verify both with tests and stress cases.

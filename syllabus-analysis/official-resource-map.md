# Official Resource Map

| Resource | Source | Role in Learning | Priority | GitHub Usage | Integrity Risk |
| -------- | ------ | ---------------- | -------- | ------------ | -------------- |
| CS229 Autumn 2018 Syllabus | [Stanford syllabus and schedule](https://cs229.stanford.edu/syllabus-autumn2018.html) | 确定 Autumn 2018 的课程顺序、主题边界和学习节奏；当前阶段用于规划，不下载材料。 | Core / 当前阶段必读索引 | 只记录主题顺序、个人学习计划和非侵权 summary。 | Medium：避免复制 schedule 大段原文或受限作业信息。 |
| CS229 Official Materials Page | [Official materials](https://cs229.stanford.edu/materials.html-withcomments) | 官方 notes 和 review material 的入口；用于判断哪些内容应优先推导。 | Core / 当前阶段资源地图 | 只做链接索引和个人阅读顺序；后续 notes 用自己的语言重建。 | Medium：official notes 可参考但不应 verbatim copy。 |
| CS229 Main Course Page | [Main course page](https://cs229.stanford.edu/) | 确认课程身份、官方入口和资源来源，避免依赖随机二手材料。 | Core / 当前阶段索引 | README 或 resource map 中保留官方入口链接。 | Low：主要是入口链接；仍需避免复制页面文本。 |
| Stanford Online CS229 Autumn 2018 YouTube Playlist | [YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) | 课堂讲解来源，用于配合 lecture notes 建立 core question 和 derivation path。 | Core / Lecture 学习阶段 | 记录个人 lecture notes、timestamp-level personal questions 和反思。 | Low to Medium：不要转录大段字幕或照搬讲解。 |
| Lecture Notes 1: Supervised Learning, Discriminative Algorithms | Official materials page | supervised learning, linear regression, logistic regression, GLM 的核心阅读入口。 | Core / Lecture 1-3 | 用于指导个人推导：loss, likelihood, gradient, Newton method。 | Medium：公式可独立重写，文字解释不能复制。 |
| Lecture Notes 2: Generative Algorithms | Official materials page | Naive Bayes, GDA 等 generative learning 的基础；连接 probabilistic assumptions 和 classifier behavior。 | Core / Generative module | 写个人比较笔记：generative vs discriminative assumptions, sample efficiency, failure modes。 | Medium：避免复制 official derivation prose。 |
| Lecture Notes 3: Support Vector Machines | Official materials page | SVM margin, duality, kernels 的主线材料；也是 optimization bottleneck。 | Core / SVM module | 建立 KKT, dual objective, kernel trick 的个人推导和 sanity examples。 | Medium to High：作业或解答相关内容必须避开。 |
| Lecture Notes 4: Learning Theory | Official materials page | generalization, sample complexity, bias/variance 之外的理论视角；连接 reliability。 | Recommended / 中后期精读 | 写 non-verbatim summary，聚焦概念依赖和研究问题。 | Medium：理论表述需用自己的结构重写。 |
| Lecture Notes 5: Regularization and Model Selection | Official materials page | regularization, cross-validation, model selection 的核心材料；直接服务实验纪律。 | Core / 实验设计阶段 | 用于设计 learning curves, validation split, model comparison。 | Medium：不要复制 notes 文本或官方例题。 |
| Notes on K-means | Official materials page | unsupervised learning 的入口；理解 objective、cluster assumptions 和 local minima。 | Core / Unsupervised module | 用合成数据做 k-means failure mode 实验。 | Medium：只保留个人解释和实验。 |
| Notes on GMM | Official materials page | mixture modeling 和 latent variable thinking 的基础。 | Core / Unsupervised module | 比较 k-means 与 GMM 在 cluster covariance 上的差异。 | Medium：避免复制完整推导文字。 |
| Notes on EM | Official materials page | 训练 latent variable models 的关键 algorithmic pattern。 | Core / 重点难点 | 写 E-step/M-step 的个人推导和数值 sanity check。 | Medium：不能复制官方解题过程。 |
| Notes on PCA | Official materials page | linear algebra, dimensionality reduction, representation analysis 的基础。 | Core / Representation module | 用 reconstruction error 和 variance explained 做实验。 | Low to Medium：公式需自己推导和表述。 |
| Notes on ICA | Official materials page | source separation 和 independence assumption 的扩展材料。 | Recommended / 可后置 | 只做概念 map 和小实验，避免过度展开。 | Low to Medium：不要复制 notes prose。 |
| Notes on Reinforcement Learning | Official materials page | MDP, value function, Bellman equation, policy/value iteration 的基础入口。 | Recommended / 后期 core | 建立 gridworld RL toy experiment 和 Bellman update notes。 | Medium：避开 assignment solution。 |
| Linear Algebra Review | Official materials page | 支撑 vectors, matrices, projections, eigendecomposition, PCA, SVM geometry。 | Core / Day 1 后快速复盘 | 列出个人薄弱概念和最小练习，不复制 review material。 | Low to Medium：review 可参考，笔记必须原创。 |
| Probability Review | Official materials page | 支撑 MLE, MAP, generative models, EM, RL uncertainty。 | Core / Day 1 后快速复盘 | 形成 probability checklist：conditional probability, expectation, variance, Bayes rule。 | Low to Medium：避免复制原文解释。 |
| Convex Optimization Review | Official materials page | 支撑 gradient methods, Newton method, SVM duality, KKT。 | Core / SVM 前复盘 | 写个人 optimization map 和 derivation checks。 | Medium：推导必须自己完成。 |
| GitHub mirror maxim5/cs229-2018-autumn | [Community mirror](https://github.com/maxim5/cs229-2018-autumn) | 只作为目录对照、资源定位和 independent attempt 后的 debugging reference。 | Optional / 当前阶段仅索引 | 不复制代码、solution 或作业答案；不把 mirror 作为主要学习源。 | High：可能包含作业材料或 solution-like 内容，必须严格隔离。 |
| Shervine / Afshine CS229 Cheatsheets | [CS229 cheatsheets](https://stanford.edu/~shervine/teaching/cs-229) | 高质量复习索引，用于快速查漏补缺和期末回顾。 | Recommended / 复盘阶段 | 作为 checklist，不替代个人推导。引用时标明来源。 | Medium：cheatsheet 内容不能整段搬运到仓库。 |

## Notes

当前阶段只建立 resource map，不下载官方讲义、作业文件或 mirror 内容。后续每个 lecture note 都应从 core question 开始，用个人语言重建 assumptions, objective, derivation, implementation insight 和 failure modes。

## Academic Integrity Rules for Resource Usage

Official notes may guide personal derivations, but should not be copied verbatim.

Official assignments should not be publicly reproduced if restricted.

Public solutions should not be copied.

GitHub mirror is for indexing and debugging after independent attempts only.

This repository should contain original notes, original code, experiments, reflections, and non-infringing summaries.

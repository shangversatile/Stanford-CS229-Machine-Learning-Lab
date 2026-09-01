# CMU 10-601 Spring 2023 Selected Supplements

Primary course: CMU 10-601 Introduction to Machine Learning, Spring 2023.

官方课程页：<https://www.cs.cmu.edu/~mgormley/courses/10601-s23/>

官方 schedule：<https://www.cs.cmu.edu/~mgormley/courses/10601-s23/schedule.html>

官方 coursework 页：<https://www.cs.cmu.edu/~mgormley/courses/10601-s23/coursework.html>

## Role / 定位

CMU 10-601 不是第二条 ML 主线。

它在这个仓库中的角色是有选择地补充 Stanford CS229，主要补充：

* alternative derivations：从另一套课程语言重新推导同一核心对象；
* implementation-oriented viewpoints：把估计式转成可实现的计数、数组、log-space 预测和复杂度判断；
* experimental methodology：补充模型选择、实验比较、误差分析等 CS229 需要落实到代码的部分；
* selected written/programming homework mapping：只建立练习映射，不提前解题；
* topics that CS229 treats more briefly：补足 CS229 讲得较快但后续实现需要清楚的主题。

Stanford CS229 lecture sequence 仍然是仓库主课程骨架。CMU 模块按主题挂到 CS229 节点上，而不是按 CMU lecture 编号重建一套平行课程。

## Relationship to CS229 / 与 CS229 的关系

| CS229 主线 | CMU 补充 | 状态 | 主要目的 |
| --- | --- | --- | --- |
| L2-L5 | L16 MLE segment + L17 MAP/NB | current | MLE -> MAP -> NB |
| L6-L7 | L10 regularization | planned | regularization / feature practice |
| L8 | L5 model selection | planned | experimental design |
| Logistic / GLM | L8-L9 | planned | optimization / implementation |
| L13 EM/GMM | L18-L19 | later | latent-state / HMM connection |
| RL | L21-L23 | later | MDP / Q-learning |

只学习 selected modules。没有进入当前计划的 CMU lectures 不会被列成 required，也不会占据和 CS229 lecture 同等的导航权重。

## 结构选择

当前仓库结构已经很清楚：

* `lecture-notes/`：Stanford CS229 lecture sequence。
* `math-derivations/`：按 CS229 lecture 组织的核心推导。
* `assignments/`：CS229 assignments 和 integrity boundary。
* `syllabus-analysis/`：CS229 官方资源、路线和学习策略。

因此 CMU 内容放在 `course-supplements/` 最合适。这样它可以引用 CS229 L4/L5/L6 等节点，但不会进入 `lecture-notes/` 伪装成主线 lecture，也不会把 CMU 推导塞进全局 `math-derivations/` 干扰 CS229 的 lecture-ordered 结构。

每个 CMU module 自己保留 `derivations/`、`figures/`、`scripts/`。这些是局部补充材料，不是全局 CS229 推导树的一部分。

## CMU 历史材料定位

本层主源是 Spring 2023 的 CMU 10-601，但会用可验证的 CMU 历史公开材料帮助判断课程长期强调什么。

| 来源 | 可验证信息 | 在本仓库中的用途 |
| --- | --- | --- |
| Spring 2023, Matt Gormley | Lecture 16 覆盖 PAC Learning / MLE + MAP；Lecture 17 覆盖 MLE/MAP + Naive Bayes；HW6 标为 Generative Models written | Module 01 的主要课程来源 |
| Fall 2013, William Cohen / Eric Xing | CMU previous courses 页面列出 10-601 Fall 2013；Internet Archive 可访问课程页和 syllabus；课程描述强调 mathematical, statistical, computational foundations，评分包含 written assignments、programming assignments、project | 证明 10-601 的计算/实现/实验视角不是本模块臆造 |
| Fall 2013/2014 Cohen Naive Bayes 页面 | Naive Bayes 页列出 slides、Matlab examples、vectorized code，并要求掌握 multinomial NB implementation 和 prediction interpretation | 支撑 Module 01 的编程视角：计数、平滑、文本 multinomial、Gaussian NB、NB linearity |
| Tom Mitchell readings | CMU assigned readings 包含 MLE/MAP、Naive Bayes 和 Logistic Regression 的概念解释 | 作为 supporting reading 融合进本模块，不单独建立 Mitchell 笔记线 |

可访问的 Fall 2013 入口：

* CMU previous offerings: <https://www.cs.cmu.edu/~mgormley/courses/10601-f24/previous.html>
* Fall 2013 course page archive: <https://web.archive.org/web/20131202063739/http://curtis.ml.cmu.edu:80/w/courses/index.php/Machine_Learning_10-601_in_Fall_2013>
* Fall 2013 syllabus archive: <https://web.archive.org/web/20140815014801/http://curtis.ml.cmu.edu/w/courses/index.php/Syllabus_for_Machine_Learning_10-601>
* Cohen Naive Bayes page archive: <https://web.archive.org/web/20141017053234/http://curtis.ml.cmu.edu/w/courses/index.php/10-601_Naive_Bayes>

## Current Modules / 当前模块

| Module | Topic | CS229 connection | Status |
| --- | --- | --- | --- |
| [01-mle-map-naive-bayes](01-mle-map-naive-bayes/README.md) | MLE、MAP、Beta-Bernoulli、Bernoulli/Gaussian/Multinomial Naive Bayes | CS229 L4 sufficient statistics；CS229 L5 generative learning | ready for interactive study |

## Coursework Mapping / 练习映射

| CMU coursework | Official source | Current treatment |
| --- | --- | --- |
| HW6 - Generative Models (written) | Spring 2023 `coursework.html` lists Homework 6 as Generative Models (written)；`hw6.zip` was temporarily inspected outside the repo | parallel with CS229 PS1 / not yet started by Codex |

临时检查 HW6 handout 后，确认其标题包含 Learning Theory and Generative Models，组成包含 MLE/MAP 与 Naive Bayes，也包含 learning theory 等非当前主题内容。因此当前处理为：

* generative subset: current；
* MLE/MAP subset: current；
* Naive Bayes subset: current；
* PAC / learning-theory subset: deferred；
* CNN/RNN parts: outside this CS229 supplement node。

不解 HW6，不写 HW6 答案，不把作业题目变成正文练习。

## Source Policy / 来源边界

官方 CMU slides、schedule、coursework 和 assigned readings 只作为来源依据；仓库正文是我们自己的中文综合笔记和独立推导。

Tom Mitchell readings 只作为对应 module 的 supporting reading，不建立第三套 Mitchell course notes。

公开实现资源，如 scikit-learn Naive Bayes 文档，只用于帮助整理工程实现注意事项，不作为数学推导主源。

## Progress State / 当前状态

| Item | Status |
| --- | --- |
| CS229 Lecture 5 | complete |
| CS229 PS1 | in progress independently |
| CMU Supplement 01: MLE / MAP / Naive Bayes | ready for interactive study |
| CS229 Lecture 6 | not started |

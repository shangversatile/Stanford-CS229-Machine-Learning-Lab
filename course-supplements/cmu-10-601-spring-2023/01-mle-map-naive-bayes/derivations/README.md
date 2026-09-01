# Module 01 Derivations

返回 [Module 01](../README.md)。

这些推导属于 CMU 10-601 supplement 的局部材料。它们不放进全局 CS229 `math-derivations/`，因为全局目录按 CS229 lecture 主线组织；这里的文件服务于专题补充和引用关系。

## Files

1. [MLE、MAP 与 Beta-Bernoulli](01-mle-map-beta-bernoulli.md)
2. [Bernoulli Naive Bayes 参数估计与 logistic posterior](02-bernoulli-nb-estimation-logistic-posterior.md)
3. [Gaussian 与 Multinomial Naive Bayes](03-gaussian-multinomial-nb.md)

## Cross-References

| 局部推导 | CS229 anchor | 关系 |
| --- | --- | --- |
| MLE、MAP 与 Beta-Bernoulli | [CS229 Lecture 4 sufficient statistics](../../../../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-a-what-information-about-a-parameter-is-actually-in-the-data) | 把 likelihood compression 转成可执行的参数估计 recipe |
| Bernoulli NB 参数估计与 logistic posterior | [CS229 Lecture 5 Naive Bayes](../../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#15-naive-bayes-for-discrete-features) | 补全 NB 的计数估计和 posterior log-odds 推导 |
| Gaussian 与 Multinomial NB | [CS229 Lecture 5 GDA/QDA](../../../../lecture-notes/lecture-05-generative-learning-gda-naive-bayes/note.md#13-qda-unequal-covariance-and-quadratic-boundary) | 把 covariance assumptions 和 text count event model 接到 CS229 L5 |

## Language Policy / 语言策略

正文中文为主，保留必要英文术语用于和 CMU/CS229/scikit-learn 文档对齐。公式统一使用 inline `$...$` 和 display `math` fence。

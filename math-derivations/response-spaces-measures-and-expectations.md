# Response Spaces, Measures, and Expectations

Cross-link: see [Lecture 4 Conceptual Interlude E: What Does a Response Value Mean?](../lecture-notes/lecture-04-perceptron-exponential-family-glm/note.md#conceptual-interlude-e-what-does-a-response-value-mean), [Softmax GLM and Cross-Entropy](softmax-glm-cross-entropy.md), [GLM Response and Distribution Map](glm-response-distribution-map.md), [Exponential Family Anatomy](exponential-family-anatomy.md), and [Log-Partition Mean, Variance, and Convexity](log-partition-mean-variance-convexity.md).

## 1. Why E[T(Y)] raises a deeper question

The softmax derivation contains the identity:

```math
\mathbb E[(T(Y))_i]
=
P(Y=i)
=
\phi_i
```

Here:

```math
T_i(Y)=\mathbf1\{Y=i\}
```

so:

```math
\mathbb E[T_i(Y)]
=
\mathbb E[\mathbf1\{Y=i\}]
=
P(Y=i)
```

This is not a mysterious new expectation rule. It is the ordinary fact that the expectation of an event indicator is the probability of that event.

The deeper question is why the categorical derivation looks at $`E[T(Y)]`$ instead of simply writing $`E[Y]`$. In Bernoulli GLMs, $`Y\in\{0,1\}`$ and $`\mathbb E[Y]=P(Y=1)`$ is meaningful. In multiclass classification, a label may be written as $`1,2,\ldots,K`$, but those symbols usually name classes rather than numerical magnitudes. That is the point where the question becomes:

```text
What is the response space of Y?
Which functions of Y have numerical meaning?
Where is probability defined?
What does expectation integrate?
```

The path from $`\mathbb E[T_i(Y)]=\phi_i`$ to response spaces and measures is therefore not a detour. It is the formal version of asking what kind of object a supervised-learning response actually is.

## 2. Probability space and measurable response space

Start one level below the label itself. A probability model begins with a probability space:

```math
(\Omega,\mathcal F,P)
```

The symbols mean:

* $`\Omega`$: the set of elementary outcomes of the underlying random experiment;
* $`\mathcal F`$: the collection of events whose probabilities are allowed to be discussed;
* $`P`$: the probability measure assigning probabilities to events in $`\mathcal F`$.

The response values live in a response space:

```math
(\mathcal Y,\mathcal A)
```

Here $`\mathcal Y`$ is the set of possible response values, and $`\mathcal A`$ is the collection of measurable subsets of $`\mathcal Y`$.

The response space is needed because supervised learning does not always predict a real number. It may predict a binary event, a nominal category, a count, a positive duration, or a probability vector. Those response types have different legal values and different meaningful operations.

Concrete examples:

In the table, $`\Delta^{K-1}`$ denotes the $`K`$-class probability simplex:

```math
\Delta^{K-1}
=
\left\{
p\in\mathbb R^K:
p_i\geq0,
\sum_{i=1}^{K}p_i=1
\right\}
```

| Task | Response space $`\mathcal Y`$ | Natural reading |
| ---- | ----------------------------- | --------------- |
| Temperature regression | subset of $`\mathbb R`$ | numerical quantity |
| Binary click prediction | $`\{0,1\}`$ | event indicator |
| Image class label | $`\{\mathrm{cat},\mathrm{dog},\mathrm{car}\}`$ | nominal category |
| Count prediction | $`\mathbb N_0`$ | nonnegative integer count |
| Topic mixture | $`\Delta^{K-1}`$ | probability vector |

This is why response semantics come before the loss function in a GLM.

## 3. Random variables as measurable maps

A random variable is formally a measurable map:

```math
Y:(\Omega,\mathcal F)\to(\mathcal Y,\mathcal A)
```

Plainly: an underlying outcome $`\omega\in\Omega`$ happens, and $`Y(\omega)`$ reports the response value in $`\mathcal Y`$.

The word "measurable" means that every response event $`A\in\mathcal A`$ pulls back to an event in the original probability space:

```math
Y^{-1}(A)=\{\omega\in\Omega:Y(\omega)\in A\}\in\mathcal F
```

This condition is needed so that $`P(Y\in A)`$ is actually defined.

The important conceptual correction is:

```text
A random variable does not fundamentally mean a random real number.
It means a measurable map from an underlying probability space into a chosen
measurable response space.
```

Only when $`\mathcal Y\subseteq\mathbb R`$ do we automatically have a real-valued random variable with ordinary arithmetic available on its values.

## 4. Pushforward distribution of a response

Once $`Y`$ maps outcomes into the response space, it induces a probability measure on that response space:

```math
P_Y(A)
=
P(Y\in A)
=
P(Y^{-1}(A)),
\qquad A\in\mathcal A
```

This induced distribution is the pushforward measure:

```math
P_Y=P\circ Y^{-1}
```

Plainly: $`P`$ lives on the underlying sample space, while $`P_Y`$ lives on the response space. The random variable $`Y`$ transfers probability mass from one space to the other.

For a categorical response:

```math
\mathcal Y=\{c_1,\ldots,c_K\}
```

the probability of class $`c_i`$ is:

```math
P_Y(\{c_i\})=\phi_i
```

For a continuous real response:

```math
\mathcal Y=\mathbb R
```

one may ask:

```math
P_Y([a,b])
```

The same formal object, a probability measure on measurable sets, covers both cases.

## 5. Expectation as integration of an observable

Expectation is not restricted to "averaging the raw label." More generally, choose a measurable numerical observable:

```math
f:\mathcal Y\to\mathbb R
```

Then $`f(Y)`$ is a real-valued random variable, and its expectation is:

```math
\mathbb E[f(Y)]
=
\int_\Omega f(Y(\omega))\,dP(\omega)
```

Equivalently, integrate over the response space using the pushforward distribution:

```math
\mathbb E[f(Y)]
=
\int_{\mathcal Y}f(y)\,dP_Y(y)
```

These two integrals are the same because $`P_Y`$ was defined by pushing $`P`$ through $`Y`$. The left integral averages over underlying elementary outcomes. The right integral averages over response values, weighted by the distribution of the response.

For categorical outcomes, choose the event observable:

```math
f_i(y)=\mathbf1\{y=c_i\}
```

Then:

```math
\mathbb E[f_i(Y)]
=
\int_{\mathcal Y}\mathbf1\{y=c_i\}\,dP_Y(y)
=
P_Y(\{c_i\})
=
\phi_i
```

Thus $`T(Y)`$ is a vector of chosen observables:

```math
T(Y)
=
\begin{bmatrix}
f_1(Y)\\
\vdots\\
f_K(Y)
\end{bmatrix}
```

and:

```math
\mathbb E[T(Y)]=\phi
```

This is the bridge back to exponential families. The canonical statistic is a selected set of parameter-relevant observables, and the log-partition gradient returns their expectation coordinates:

```math
\nabla a(\eta)=\mathbb E_\eta[T(Y)]
```

## 6. Real-valued regression responses

In regression, the task often tells us before probability enters that the response is a numerical quantity: temperature, concentration, income, velocity, PM2.5 concentration, or another measurement with meaningful arithmetic.

Then it is natural to choose:

```math
\mathcal Y\subseteq\mathbb R
```

The identity observable:

```math
\mathrm{id}(y)=y
```

is meaningful because $`y`$ itself is a numerical magnitude. The response mean:

```math
\mathbb E[Y]
=
\int_{\mathcal Y}y\,dP_Y(y)
```

then has task meaning.

Variance also uses real-number structure:

```math
\mathrm{Var}(Y)
=
\int_{\mathcal Y}
\left(y-\mathbb E[Y]\right)^2
\,dP_Y(y)
```

The measure supplies probability weights. The numerical response space supplies the arithmetic meaning of $`y`$, subtraction, squaring, averaging, and residuals. These are different roles.

## 7. Nominal categorical responses

For nominal classification:

```math
\mathcal Y=\{\mathrm{cat},\mathrm{dog},\mathrm{car}\}
```

The probability model can be completely valid without defining any of:

```text
cat + dog
dog - car
distance(cat, dog)
the midpoint between dog and car
```

For example:

```math
P_Y(\{\mathrm{cat}\})=0.2
```

```math
P_Y(\{\mathrm{dog}\})=0.5
```

```math
P_Y(\{\mathrm{car}\})=0.3
```

This defines a probability distribution on the finite response space. No numerical geometry is required.

For convenience, we may rename the classes:

```math
\{\mathrm{cat},\mathrm{dog},\mathrm{car}\}
\leftrightarrow
\{1,2,3\}
```

But $`1,2,3`$ are first category identifiers. Writing the classes as $`1,2,\ldots,K`$ is a coordinate or naming convention. It does not by itself turn the categorical response space into a numerical interval with meaningful order, addition, distance, midpoint, or ratio.

## 8. Numeric coding is not intrinsic

Suppose the encoding is:

```text
cat = 1
dog = 2
car = 3
```

If these labels are treated as real numbers, then a scalar-coded expectation can be computed:

```math
\mathbb E[Y]
=
\sum_{i=1}^{K}iP(Y=i)
=
\sum_{i=1}^{K}i\phi_i
```

But for nominal categories this is encoding-dependent, not intrinsic.

Change the coding:

```text
cat = 100
dog = -4
car = 7
```

The real classification problem has not changed. The class probabilities may still be $`0.2,0.5,0.3`$. But the scalar-coded mean changes because the chosen numbers changed.

Therefore:

```text
E[Y] may be mathematically computable after a numerical encoding is chosen,
but for nominal categories it is encoding-dependent rather than an intrinsic
semantic quantity.
```

The issue is not that categorical expectations are forbidden. The issue is that the raw label must already be a meaningful numerical object, or we must choose a meaningful numerical observable of it.

## 9. Indicator coordinates and E[T(Y)]

Instead of treating arbitrary label codes as magnitudes, define indicator observables:

```math
T_i(Y)=\mathbf1\{Y=c_i\}
```

Each component is a binary view of one class event:

```text
Y = c_i versus Y != c_i
```

Therefore:

```math
T_i(Y)\in\{0,1\}
```

and marginally:

```math
T_i(Y)\sim\mathrm{Bernoulli}(\phi_i)
```

so:

```math
\mathbb E[T_i(Y)]=\phi_i
```

This preserves the useful intuition that each class coordinate asks one binary question. But the $`K`$ coordinates are not $`K`$ independent Bernoulli response variables. They are tied by:

```math
\sum_{i=1}^{K}T_i(Y)=1
```

almost surely. Exactly one class happens in a one-trial categorical outcome.

Thus "many binary views" means projection or coordinate intuition. It does not mean that softmax is a collection of independent one-vs-rest binary classifiers.

If the random response variable itself is defined as a one-hot random vector:

```math
Y\in\{e_1,\ldots,e_K\}\subset\mathbb R^K
```

then writing:

```math
\mathbb E[Y]=\phi
```

is perfectly valid. CS229's notation instead keeps the original categorical outcome $`Y`$ separate from the numerical statistic $`T(Y)`$. That separation makes clear which observable is being averaged.

For the reference-class representation:

```math
T(Y)=(T_1(Y),\ldots,T_{K-1}(Y))
```

class $`K`$ maps to the zero vector. This removes redundant coordinates. It does not mean the categorical distribution is intrinsically $`K-1`$ independent Bernoulli variables.

## 10. Counting measure as reference measure

It is tempting to say that categorical probability is counting measure. That is not correct.

On a finite response space, the counting measure $`\nu`$ is:

```math
\nu(A)=\#A
```

where $`\#A`$ is the number of elements in $`A`$. This measure assigns size to sets by counting elements.

The categorical probability measure is different:

```math
P_Y(\{c_i\})=\phi_i
```

If counting measure is used as a reference measure, the probability mass function is the Radon-Nikodym derivative:

```math
p(y)=\frac{dP_Y}{d\nu}(y)
```

For a class value:

```math
p(c_i)=\phi_i
```

Counting measure is therefore a background reference measure for writing mass functions. It is not the learned categorical distribution unless the class probabilities happen to equal normalized counting mass.

## 11. Lebesgue measure as reference measure

For many continuous real-valued models, the reference measure is Lebesgue measure $`\lambda`$ on $`\mathbb R`$. If:

```math
P_Y\ll\lambda
```

then $`P_Y`$ is absolutely continuous with respect to $`\lambda`$, and the density is:

```math
f(y)=\frac{dP_Y}{d\lambda}(y)
```

The probability of an interval is:

```math
P_Y([a,b])
=
\int_a^b f(y)\,d\lambda(y)
```

For ordinary real integration this is usually written:

```math
P_Y([a,b])
=
\int_a^b f(y)\,dy
```

Lebesgue measure is a reference measure for density. It is not itself the probability distribution, except after normalization and restriction in special cases such as the uniform distribution on a bounded interval.

Counting measure and Lebesgue measure are common reference measures. They are not themselves the learned probability distribution.

## 12. Measure, metric, topology, and algebra are different structures

Several structures can live on the same set, but they do different jobs.

Set and measurable structure tell us what outcomes exist and which subsets are measurable events:

```math
(\mathcal Y,\mathcal A)
```

A metric tells us distances:

```math
d(y_1,y_2)
```

Probability theory does not require every response space to carry a metric.

A topology tells us open sets, neighborhoods, continuity, and limits. It may be induced by a metric, but it need not be the same object as a measure.

Order, vector, or algebraic structure allows operations such as:

```text
comparison
addition
scalar multiplication
midpoints
differences
ratios
```

A general measure assigns size to measurable sets:

```math
\mu(A)
```

A probability measure is a measure with total mass one:

```math
P(\Omega)=1
```

The key separation is:

```text
A probability measure does not become valid merely because the underlying
space has a meaningful metric, and a probability space does not become invalid
because no metric exists.
```

Likewise, equal distance is not automatically equal probability, and lack of distance is not lack of probability.

## 13. Relabeling invariance of nominal categories

Nominal class labels are names. If:

```text
cat, dog, car
```

are reordered as:

```text
car, cat, dog
```

the classification problem has not changed. A nominal classifier should transform consistently under this relabeling.

Let a permutation $`\pi`$ relabel classes:

```math
c_i\mapsto c_{\pi(i)}
```

Then the probability vector changes by the same permutation:

```math
\phi\mapsto P_\pi\phi
```

where $`P_\pi`$ is the permutation matrix associated with $`\pi`$.

Indicator coordinates respect this symmetry. They do not require class $`2`$ to be twice class $`1`$, or class $`3`$ to be farther from class $`1`$ than class $`2`$ is.

The scalar-coded mean:

```math
\sum_{i=1}^{K}i\phi_i
```

does not generally respect relabeling. Changing names can change the numerical mean even though the semantic classification problem is the same.

This gives a more precise statement than "classes have no distance":

```text
Nominal category labels should be interpreted up to relabeling, while scalar
arithmetic on arbitrary codes is not relabeling-invariant.
```

## 14. Ordinal responses as an intermediate case

Not every classification task is purely nominal. Some labels have order:

```text
low < medium < high
```

This is an ordinal response. The order may be meaningful, but the distances may still not be:

```text
distance(low, medium) = distance(medium, high)
```

is an extra assumption, not a consequence of the labels.

Thus:

| Response type | Structure |
| ------------- | --------- |
| Nominal category | no intrinsic order or arithmetic |
| Ordinal category | order, but not necessarily equal spacing |
| Regression quantity | numerical magnitude with richer arithmetic |

This caveat matters because it prevents an over-simple "classification versus regression" split. The correct question is what structure the response values really carry.

## 15. Semantic equivalence vs geometric invariance vs equiprobability

"Same meaning" and "same probability" are different claims.

Semantic equivalence means the task says two representations describe the same real situation. For example, renaming classes should not change a nominal classification problem.

Geometric invariance means a transformation group $`G`$ acts on a space while preserving a structure. Examples include translations preserving Euclidean distances, rotations preserving angles and lengths, and permutations preserving the nominal nature of class labels.

Equal probability means:

```math
P(A)=P(B)
```

for two measurable events $`A`$ and $`B`$.

The implications do not hold automatically:

```math
\text{semantic equivalence}
\nRightarrow
\text{equal probability}
```

and:

```math
\text{same geometric size}
\nRightarrow
\text{same probability}
```

unless the probability model explicitly has the corresponding symmetry or invariance.

For example, two income intervals can have the same length but very different probabilities under a skewed income distribution. Two classes can be equally valid semantic labels but have very different base rates in data. Equivalence of representation is not the same as equiprobability of outcomes.

## 16. Coordinate dependence of "uniform"

Symmetry can help choose a natural measure, but it does not make "uniform" a coordinate-free word in every problem.

If a problem specifies a transformation group $`G`$, one may want a reference measure $`\mu`$ satisfying:

```math
\mu(gA)=\mu(A),
\qquad g\in G
```

Examples:

* translation invariance motivates Lebesgue measure on Euclidean spaces;
* rotation invariance motivates surface measure on a sphere;
* locally compact groups have Haar measures under suitable conditions.

But this comes with caveats:

* invariance may define only a measure up to scale, not a probability law;
* noncompact spaces often have invariant measures that cannot be normalized to total mass one;
* different sampling mechanisms can preserve different symmetries;
* deciding which symmetry is physically or semantically appropriate is part of the modeling problem.

Therefore "uniformly random" is incomplete until the space, reference measure, and sampling mechanism are specified.

## 17. Change-of-variables derivation

Coordinate choices change density formulas. Suppose:

```math
Z=g(Y)
```

where $`g`$ is monotone and differentiable. Then:

```math
f_Z(z)
=
f_Y(g^{-1}(z))
\left|
\frac{d}{dz}g^{-1}(z)
\right|
```

The derivative factor appears because equal coordinate intervals in $`z`$ may correspond to unequal intervals in $`y`$.

A simple example makes this concrete. Let:

```math
U\sim\mathrm{Uniform}(0,1)
```

and define:

```math
Z=U^2
```

Then for $`0<z<1`$:

```math
P(Z\leq z)
=
P(U^2\leq z)
=
P(U\leq\sqrt z)
=
\sqrt z
```

So:

```math
f_Z(z)
=
\frac{1}{2\sqrt z},
\qquad 0<z<1
```

Thus $`Z`$ is not uniform even though it is a one-to-one transformation of the same underlying outcomes. The lesson is:

```text
Uniform in one coordinate is usually not uniform in a nonlinear coordinate.
```

Saying "uniform in a parameter" already selects a coordinate structure and a reference measure.

## 18. Random chords and underspecified probability measures

The phrase "choose a random chord of a circle" is not yet a complete probability model.

One must specify:

1. the chord space $`\mathcal C`$;
2. a parameterization or sampling mechanism;
3. the probability measure induced on chords.

Different protocols include:

* choose two endpoints independently and uniformly on the circumference;
* choose the perpendicular distance from the center uniformly;
* choose the chord midpoint uniformly in the disk.

These procedures induce different probability measures on the same geometric chord space. Therefore the probability of an event such as "the chord is longer than the side of the inscribed equilateral triangle" can differ. Classical Bertrand-type constructions give different values, commonly stated as:

```text
1/3, 1/2, 1/4
```

This is not a contradiction in probability theory. It shows that the original phrase "random chord" did not uniquely specify a probability measure.

## 19. Pushforward view of sampling mechanisms

The chord example is another pushforward story.

Let the sampling protocol live on a parameter space:

```math
(\mathcal Z,\mathcal G,Q)
```

Here $`\mathcal Z`$ is the sampling parameter space, $`\mathcal G`$ is its event collection, and $`Q`$ is the probability law used by the protocol.

Let:

```math
F:\mathcal Z\to\mathcal C
```

map each sampled parameter value to a chord in chord space $`\mathcal C`$.

The induced chord distribution is:

```math
P_{\mathcal C}
=
Q\circ F^{-1}
```

Different triples:

```math
(\mathcal Z,Q,F)
```

can induce different measures $`P_{\mathcal C}`$ on the same chord space. Therefore "the same set of possible chords" is not enough to define "a random chord." The probability measure must be specified.

This mirrors the supervised-learning response setup:

```math
P_Y=P\circ Y^{-1}
```

The response space tells us what values can occur. The random variable and the underlying probability law tell us how likely measurable response events are.

## 20. Probability simplex and categorical mean parameters

For a $`K`$-class categorical distribution:

```math
\phi_i\geq0,
\qquad
\sum_{i=1}^{K}\phi_i=1
```

Therefore:

```math
\phi\in\Delta^{K-1}
```

where the probability simplex is:

```math
\Delta^{K-1}
=
\left\{
p\in\mathbb R^K:
p_i\geq0,
\sum_{i=1}^{K}p_i=1
\right\}
```

The one-hot vectors are the vertices:

```math
e_1,\ldots,e_K
```

An observed categorical outcome gives one vertex. The expectation of the one-hot statistic is:

```math
\mathbb E[T(Y)]=\phi
```

which lies in the simplex. It may be on the boundary if some class probabilities are zero, or in the interior if all probabilities are positive.

For softmax, logits or natural parameters live in an unconstrained coordinate space, while softmax maps them to the simplex:

```math
\eta
\mapsto
\phi(\eta)\in\Delta^{K-1}
```

This separates the roles:

| Object | Space | Role |
| ------ | ----- | ---- |
| class label $`Y`$ | nominal response space | observed category |
| one-hot statistic $`T(Y)`$ | vertices of simplex | numerical event coordinates |
| mean parameter $`\phi`$ | simplex | class-probability vector |
| natural parameter $`\eta`$ | unconstrained log-odds coordinates | distribution coordinate |

## 21. Return to exponential families and GLMs

The categorical exponential family chooses:

```math
T_i(Y)=\mathbf1\{Y=i\},
\qquad i=1,\ldots,K-1
```

These are parameter-relevant observables on the response space. Their expectations are class probabilities:

```math
\mathbb E[T_i(Y)]
=
P(Y=i)
```

The log-partition identity:

```math
\nabla a(\eta)
=
\mathbb E_\eta[T(Y)]
```

therefore maps natural coordinates to expected sufficient-statistic coordinates.

In a canonical softmax GLM, the full chain is:

```text
x
-> linear predictor / logits
-> natural parameters
-> softmax
-> probability simplex
-> E[T(Y) | x]
-> categorical response distribution
```

This is why the small-looking identity $`\mathbb E[(T(Y))_i]=\phi_i`$ naturally leads to response spaces, observables, measures, and expectation. It asks what kind of object $`Y`$ is, which numerical functions of $`Y`$ are meaningful, and how probability is assigned to events in the response space.

For a complete one-hot random vector $`Z=T(Y)\in\mathbb R^K`$:

```math
\mathbf1^TZ=1
```

and:

```math
\mathrm{Cov}(Z)
=
\mathrm{Diag}(\phi)-\phi\phi^T
```

This covariance is singular because:

```math
\mathrm{Cov}(Z)\mathbf1=0
```

The singularity does not make $`\mathbb E[Z]=\phi`$ meaningless. It only reflects the one-dimensional affine constraint. Reference-class coordinates remove redundancy and improve identifiability; they do not make expectation valid for the first time.

## 22. Common misconceptions

**Misconception 1: A categorical random variable cannot have an expectation.**

Correction: an expectation requires a chosen real-valued or vector-valued representation or observable. A scalar label encoding can have a numerical expectation, but it is generally not intrinsic under relabeling.

**Misconception 2: Categorical probability is the counting measure.**

Correction: counting measure is a common reference measure. The categorical probability measure has masses $`\phi_i`$.

**Misconception 3: Continuous probability is attached to intervals, while discrete probability is attached to points.**

Correction: both are probability measures on sigma-algebras of measurable sets. Singletons and intervals are just different measurable sets.

**Misconception 4: A meaningful metric uniquely determines a valid probability distribution.**

Correction: metric and probability measure are separate structures.

**Misconception 5: Symmetry always uniquely determines the probability measure.**

Correction: invariance can constrain or motivate a measure, but it does not universally select a unique normalized probability law.

**Misconception 6: The $`K`$ indicator coordinates are $`K`$ independent Bernoulli variables.**

Correction: each coordinate has a Bernoulli marginal, but the coordinates are constrained by $`\sum_iT_i(Y)=1`$.

**Misconception 7: One-hot $`\mathbb E[Y]=\phi`$ is invalid.**

Correction: if $`Y`$ itself is defined as a one-hot random vector, $`\mathbb E[Y]=\phi`$ is perfectly valid. CS229 instead separates the categorical outcome $`Y`$ from its vector statistic $`T(Y)`$.

**Misconception 8: The singular covariance of one-hot variables makes their expectation meaningless.**

Correction: the expectation is well-defined. Singularity reflects redundancy, not invalidity.

## 23. Fast review

* A categorical label written as $`1,2,\ldots,K`$ is usually a name, not a magnitude.
* $`\mathbb E[Y]`$ is intrinsic only when $`Y`$ has a meaningful real or vector representation.
* $`T_i(Y)=\mathbf1\{Y=c_i\}`$ is an event observable, so $`\mathbb E[T_i(Y)]=P(Y=c_i)`$.
* The $`K`$ one-hot coordinates are Bernoulli marginals but not independent Bernoulli responses.
* Probability is assigned to measurable sets through a probability measure.
* $`P_Y=P\circ Y^{-1}`$ is the response distribution induced by the random variable $`Y`$.
* Counting and Lebesgue measures are reference measures, not automatically probability laws.
* Metric, topology, algebra, measure, and probability are separate structures.
* Semantic equivalence and equal probability are different claims.
* "Uniform" depends on a specified coordinate, reference measure, or sampling protocol.
* Bertrand-type chord ambiguity is caused by an underspecified probability measure, not by inconsistent probability axioms.
* Softmax maps logits to the probability simplex, which is $`\mathbb E[T(Y)\mid x]`$ for a categorical response.

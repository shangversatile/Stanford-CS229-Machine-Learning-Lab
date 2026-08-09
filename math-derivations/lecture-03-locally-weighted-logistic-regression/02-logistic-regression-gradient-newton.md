# Logistic Regression: Gradient, Hessian, and Newton Method

## 1. Sigmoid Function and Derivative

The sigmoid function is:

$$g(z)=\frac{1}{1+e^{-z}}.$$

It maps real-valued logits into probabilities:

$$0<g(z)<1.$$

Differentiate:

$$g'(z)=\frac{e^{-z}}{(1+e^{-z})^2}.$$

Because:

$$g(z)=\frac{1}{1+e^{-z}}.$$

And:

$$1-g(z)=\frac{e^{-z}}{1+e^{-z}}.$$

Therefore:

$$g'(z)=g(z)(1-g(z)).$$

Logistic regression uses:

$$h_\theta(x)=g(\theta^Tx).$$

## 2. Bernoulli Likelihood

For binary labels $y\in\{0,1\}$:

$$P(y=1|x;\theta)=h_\theta(x).$$

$$P(y=0|x;\theta)=1-h_\theta(x).$$

The unified Bernoulli form is:

$$P(y|x;\theta)=h_\theta(x)^y\left(1-h_\theta(x)\right)^{1-y}.$$

For independent training examples:

$$L(\theta)=\prod_{i=1}^{m}h_\theta(x^{(i)})^{y^{(i)}}\left(1-h_\theta(x^{(i)})\right)^{1-y^{(i)}}.$$

## 3. Log Likelihood

Take the logarithm:

$$\ell(\theta)=\log L(\theta).$$

Then:

$$\ell(\theta)=\sum_{i=1}^{m}\left[y^{(i)}\log h_\theta(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)})\right)\right].$$

This is the objective maximized by maximum likelihood estimation.

## 4. Negative Log Likelihood

For minimization, define:

$$J(\theta)=-\ell(\theta).$$

Thus:

$$J(\theta)=-\sum_{i=1}^{m}\left[y^{(i)}\log h_\theta(x^{(i)})+\left(1-y^{(i)}\right)\log\left(1-h_\theta(x^{(i)})\right)\right].$$

This is binary cross-entropy loss.

## 5. Step-by-step Gradient Derivation

Let:

$$z_i=\theta^Tx^{(i)}.$$

Then:

$$h_i=h_\theta(x^{(i)})=g(z_i).$$

The per-example loss is:

$$J_i(\theta)=-y^{(i)}\log h_i-\left(1-y^{(i)}\right)\log(1-h_i).$$

Differentiate with respect to $h_i$:

$$\frac{\partial J_i}{\partial h_i}=-\frac{y^{(i)}}{h_i}+\frac{1-y^{(i)}}{1-h_i}.$$

Combine the fraction:

$$\frac{\partial J_i}{\partial h_i}=\frac{h_i-y^{(i)}}{h_i(1-h_i)}.$$

Differentiate $h_i$ with respect to $z_i$:

$$\frac{\partial h_i}{\partial z_i}=h_i(1-h_i).$$

Differentiate $z_i$ with respect to $\theta$:

$$\nabla_\theta z_i=x^{(i)}.$$

Apply chain rule:

$$\nabla_\theta J_i(\theta)=\frac{\partial J_i}{\partial h_i}\frac{\partial h_i}{\partial z_i}\nabla_\theta z_i.$$

Substitute:

$$\nabla_\theta J_i(\theta)=\frac{h_i-y^{(i)}}{h_i(1-h_i)}h_i(1-h_i)x^{(i)}.$$

Cancel:

$$\nabla_\theta J_i(\theta)=\left(h_i-y^{(i)}\right)x^{(i)}.$$

Summing over examples:

$$\nabla_{\theta}J(\theta)=\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}.$$

In matrix notation, with $h=g(X\theta)$ applied elementwise:

$$\nabla_\theta J(\theta)=X^T(h-y).$$

## 6. Hessian Derivation

The gradient is:

$$\nabla_\theta J(\theta)=\sum_{i=1}^{m}(h_i-y^{(i)})x^{(i)}.$$

Only $h_i$ depends on $\theta$. Since:

$$\nabla_\theta h_i=h_i(1-h_i)x^{(i)}.$$

The Hessian is:

$$\nabla_\theta^2J(\theta)=\sum_{i=1}^{m}h_i(1-h_i)x^{(i)}(x^{(i)})^T.$$

Let $S$ be diagonal with entries:

$$S_{ii}=h_{\theta}(x^{(i)})(1-h_{\theta}(x^{(i)})).$$

Then:

$$S=\mathrm{diag}(S_{11},S_{22},\ldots,S_{mm}).$$

The matrix Hessian is:

$$H=X^TSX.$$

## 7. Convexity Intuition

For any vector $v$:

$$v^THv=v^TX^TSXv.$$

Rewrite:

$$v^THv=(Xv)^TS(Xv).$$

Since $0<h_i(1-h_i)\leq\frac{1}{4}$, every diagonal entry of $S$ is nonnegative:

$$S\succeq0.$$

Therefore:

$$v^THv\geq0.$$

So the negative log likelihood is convex. If the data are not degenerate and the design has enough rank in the weighted directions, the Hessian can be positive definite locally. With complete separation, however, the optimum may not exist as a finite vector unless regularization is added.

## 8. Newton Update for Logistic Regression

Newton method for minimizing $J(\theta)$ uses:

$$\theta_{t+1}=\theta_t-H^{-1}\nabla_\theta J(\theta_t).$$

For logistic regression:

$$\nabla_\theta J(\theta_t)=X^T(h_t-y).$$

And:

$$H_t=X^TS_tX.$$

Thus the Newton update is:

$$\theta_{t+1}=\theta_t-(X^TS_tX)^{-1}X^T(h_t-y).$$

In practice, one usually solves the linear system instead of explicitly forming an inverse:

$$X^TS_tX\Delta_t=X^T(h_t-y).$$

Then:

$$\theta_{t+1}=\theta_t-\Delta_t.$$

## 9. Why No Normal Equation Exists

Linear regression has a normal equation because the squared-error objective is quadratic:

$$J_{\mathrm{lin}}(\theta)=\frac{1}{2}\left\|X\theta-y\right\|_2^2.$$

Its gradient is linear in $\theta$:

$$\nabla_\theta J_{\mathrm{lin}}(\theta)=X^TX\theta-X^Ty.$$

Setting this to zero gives:

$$X^TX\theta=X^Ty.$$

Logistic regression has sigmoid predictions:

$$h_\theta(x)=g(\theta^Tx).$$

Its stationary condition is:

$$\sum_{i=1}^{m}\left(g(\theta^Tx^{(i)})-y^{(i)}\right)x^{(i)}=0.$$

Because $\theta$ is inside a nonlinear sigmoid, this equation generally cannot be rearranged into $A\theta=b$. Iterative optimization is required.

## 10. Linear Regression vs Logistic Regression Gradient Comparison

Linear regression gradient:

$$\nabla_\theta J_{\mathrm{lin}}(\theta)=\sum_{i=1}^{m}\left(\theta^Tx^{(i)}-y^{(i)}\right)x^{(i)}.$$

Logistic regression gradient:

$$\nabla_\theta J_{\mathrm{log}}(\theta)=\sum_{i=1}^{m}\left(g(\theta^Tx^{(i)})-y^{(i)}\right)x^{(i)}.$$

They share the form:

$$\text{gradient}=\sum_{i=1}^{m}\text{prediction error}\times\text{feature vector}.$$

But the prediction functions and statistical assumptions are different:

| Aspect | Linear Regression | Logistic Regression |
| ------ | ----------------- | ------------------- |
| Output | real-valued $\theta^Tx$ | probability $g(\theta^Tx)$ |
| Data model | Gaussian conditional noise | Bernoulli conditional label |
| Loss | squared error | cross-entropy |
| Objective shape | quadratic | convex but not quadratic |
| Closed form | normal equation when invertible | generally no normal equation |
| Optimizer | closed form, gradient descent, SGD | gradient descent, Newton, quasi-Newton |

Similar gradients do not imply identical models. They indicate a shared supervised-learning pattern: residual information flows through features into parameter updates.
# Newton Method and Quadratic Convergence

## 1. Newton as Root Finding

Newton method can be introduced as a root-finding algorithm. Given a differentiable function $F(\theta)$, the goal is:

$$F(\theta^\star)=0.$$

At iterate $\theta_t$, approximate $F$ by its tangent line:

$$F(\theta)\approx F(\theta_t)+F'(\theta_t)(\theta-\theta_t).$$

Set the approximation to zero:

$$0=F(\theta_t)+F'(\theta_t)(\theta-\theta_t).$$

Solving for $\theta$ gives:

$$\theta_{t+1}=\theta_t-\frac{F(\theta_t)}{F'(\theta_t)}.$$

## 2. Newton as Second-order Optimization

For optimization, the goal is to find a stationary point:

$$f'(\theta^\star)=0.$$

Set:

$$F(\theta)=f'(\theta).$$

Then:

$$F'(\theta)=f''(\theta).$$

Newton root finding becomes Newton optimization:

$$\theta_{t+1}=\theta_t-\frac{f'(\theta_t)}{f''(\theta_t)}.$$

## 3. One-dimensional Derivation

Use the second-order Taylor approximation of $f$ around $\theta_t$:

$$f(\theta)\approx f(\theta_t)+f'(\theta_t)(\theta-\theta_t)+\frac{1}{2}f''(\theta_t)(\theta-\theta_t)^2.$$

Differentiate the quadratic approximation with respect to $\theta$:

$$\frac{d}{d\theta}\left[f(\theta_t)+f'(\theta_t)(\theta-\theta_t)+\frac{1}{2}f''(\theta_t)(\theta-\theta_t)^2\right]=f'(\theta_t)+f''(\theta_t)(\theta-\theta_t).$$

Set the derivative to zero:

$$0=f'(\theta_t)+f''(\theta_t)(\theta-\theta_t).$$

Solve:

$$\theta=\theta_t-\frac{f'(\theta_t)}{f''(\theta_t)}.$$

Thus:

$$\theta_{t+1}=\theta_t-\frac{f'(\theta_t)}{f''(\theta_t)}.$$

## 4. Multivariate Update

For $f:\mathbb{R}^d\to\mathbb{R}$, the second-order approximation is:

$$f(\theta)\approx f(\theta_t)+\nabla f(\theta_t)^T(\theta-\theta_t)+\frac{1}{2}(\theta-\theta_t)^TH_t(\theta-\theta_t).$$

Here:

$$H_t=\nabla^2 f(\theta_t).$$

Differentiate the approximation:

$$0=\nabla f(\theta_t)+H_t(\theta-\theta_t).$$

Solve the linear system:

$$H_t(\theta-\theta_t)=-\nabla f(\theta_t).$$

Therefore:

$$\theta_{t+1}=\theta_t-H_t^{-1}\nabla f(\theta_t).$$

In implementation, $H_t^{-1}$ should usually not be formed explicitly. One solves:

$$H_t\Delta_t=\nabla f(\theta_t).$$

Then:

$$\theta_{t+1}=\theta_t-\Delta_t.$$

## 5. Geometric Intuition

Gradient descent uses a first-order local model. It sees the slope and moves in the negative-gradient direction:

$$\theta_{t+1}=\theta_t-\alpha\nabla f(\theta_t).$$

Newton method uses a second-order local model. It sees both slope and curvature, then jumps to the minimizer of the local quadratic approximation.

When curvature information is accurate and the iterate is close to the optimum, Newton steps can become extremely fast. When curvature is wrong, singular, indefinite, or poorly conditioned, Newton steps can be unstable.

## 6. Quadratic Convergence Theorem

Let $F$ be twice continuously differentiable near $\theta^\star$, with:

$$F(\theta^\star)=0.$$

Assume:

$$F'(\theta^\star)\neq0.$$

Also assume $F''$ is bounded in a local neighborhood and the starting point is sufficiently close to $\theta^\star$. Then Newton iterates satisfy:

$$|\theta_{t+1}-\theta^\star|\leq C|\theta_t-\theta^\star|^2.$$

This is quadratic convergence: once close enough, the number of correct digits can roughly double at each step.

## 7. Proof Sketch with Taylor Expansion

Let:

$$e_t=\theta_t-\theta^\star.$$

Newton update is:

$$\theta_{t+1}=\theta_t-\frac{F(\theta_t)}{F'(\theta_t)}.$$

Subtract $\theta^\star$:

$$e_{t+1}=e_t-\frac{F(\theta_t)}{F'(\theta_t)}.$$

Apply Taylor expansion of $F(\theta^\star)$ around $\theta_t$:

$$0=F(\theta^\star)=F(\theta_t)+F'(\theta_t)(\theta^\star-\theta_t)+\frac{1}{2}F''(\xi_t)(\theta^\star-\theta_t)^2.$$

Since $\theta^\star-\theta_t=-e_t$:

$$0=F(\theta_t)-F'(\theta_t)e_t+\frac{1}{2}F''(\xi_t)e_t^2.$$

Rearrange:

$$F(\theta_t)=F'(\theta_t)e_t-\frac{1}{2}F''(\xi_t)e_t^2.$$

Substitute into the error recursion:

$$e_{t+1}=e_t-\frac{F'(\theta_t)e_t-\frac{1}{2}F''(\xi_t)e_t^2}{F'(\theta_t)}.$$

Cancel the first-order term:

$$e_{t+1}=\frac{F''(\xi_t)}{2F'(\theta_t)}e_t^2.$$

If $|F''(\xi_t)|\leq M$ and $|F'(\theta_t)|\geq m_0>0$ locally, then:

$$|e_{t+1}|\leq\frac{M}{2m_0}|e_t|^2.$$

Let:

$$C=\frac{M}{2m_0}.$$

Then:

$$|e_{t+1}|\leq C|e_t|^2.$$

## 8. When Newton Fails

Newton method can fail or behave poorly when:

* the initial point is far from the optimum;
* $F'(\theta_t)$ is near zero in root finding;
* the Hessian is singular or ill-conditioned;
* the Hessian is indefinite for minimization;
* the objective is not well approximated by a local quadratic;
* the full step is too aggressive and requires damping or line search;
* the data create separation or near-separation in logistic regression.

These failure modes explain why practical Newton methods are often damped, regularized, or replaced by quasi-Newton methods.

## 9. Connection to Logistic Regression and CS229

For logistic regression, the negative log likelihood has gradient:

$$\nabla_\theta J(\theta)=X^T(h-y).$$

Its Hessian is:

$$H=X^TSX.$$

Newton update is:

$$\theta_{t+1}=\theta_t-H^{-1}\nabla_\theta J(\theta_t).$$

Because logistic regression has no normal equation, Newton method gives a principled second-order iterative route. Lecture 3 uses it to connect likelihood-based classification with numerical optimization and prepares the path toward generalized linear models.

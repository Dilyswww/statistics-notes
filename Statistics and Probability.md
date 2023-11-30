> This is the class notes from UIUC STAT 410: Statistics and Probability II taught by Professor Alexey Stephanov and Professor Douglas Jeffery. 

# Textbook & References
Introduction to Mathematical Statistics, 8th ( or 7th, or 6th ) edition by Robert V. Hogg, Joseph W. McKean, Allen T. Craig.

# Random Variables 
(from STAT 400) (1.6, 1.7, 1.8, 1.9)
### Discrete Random Variable
Probability **mass** function, p.m.f.
$$
\begin{align*}
p(x) &= P(X=x),\\

\forall x,\:&0 \leq p(x) \leq 1,\\ 

\: \sum_{all\:x}&p(x) = 1
\end{align*}$$

Cumulative distribution function, c.d.f.
$$F(x) = p(X \leq x) = \sum_{y\leq x} p(y)$$
Expected value, $E(x) = \mu_x$

If $\sum_{all\:x}|x|\:p(x) < \infty$, then
$$E(x) = \sum_{all\:x}x \cdot p(x)$$
If $\sum_{all\:x}|g(x)|\:p(x) < \infty$, then
$$E(x) = \sum_{all\:x}g(x) \cdot p(x)$$
Variance, $Var(x) = \sigma_X^2 = E([X-\mu_X]^2) = E(X^2) - [E(X)]^2$

Moment-generating function (MGF), $M_X(t) = E(e^{tX})$, and the power of $t$ is called the $t^{th}$ moment of $X$. 
$$M_X(t) = \sum_{all\:x}e^{tx} \cdot p(x)$$
>[!note] Moment and Expected Value
>We have $$E(X^k) = \frac{d^k}{dt^k}M_X(t)|_{t=0}$$

### Continuous Random Variable
Probability **mass** function, p.m.f.
$$
\begin{align*}
f(x),\\

\forall x,\:p(x)>0,\\ 

\: \int_{-\infty}^{\infty}f(x)dx = 1
\end{align*}$$


Cumulative distribution function, c.d.f.
$$F(x) = p(X \leq x) = \int_{-\infty}^{x}f(t) dt$$

Expected value, $E(x) = \mu_x$

If $\int_{-\infty}^{\infty}|x|\cdot f(x) < \infty$, then
$$E(x) = \int_{-\infty}^{\infty}x \cdot f(x)dx$$
If $\sum_{all\:x}|g(x)|\:p(x) < \infty$, then
$$E(x) = \int_{-\infty}^{\infty} g(x) \cdot f(x) dx$$
Moment-generating function
$$M_X(t) = \int_{-\infty}^{\infty}e^{tx} \cdot f(x)dx$$

>[!note] Expected Value
>There are two common ways to solve for expected value.
>1) Directly compute from p.m.f, $E(x) = \int_{-\infty}^{\infty}x\cdot f(x) dx$ (continuous)
>2) Take the derivative of first moment of X, i.e., $E(X) = M_X(t)|_{t=0}$

# Functions of One Random Variable, 1 –> 1 
(1.6.1, 1.7.2)
Let $X$ be a random variable, $Y=g(X)$, find the probability distribution of $Y$.
>[!important] Theorem 1.7.1
>1) Cumulative Distribution Approach
>	$$F_Y(y) = P(Y \leq y) = P(g(X) \leq y) =\int_{\{x:g(x) < y\}}f_X(x)dx$$
>2) MGF Approach 
>   $$M_Y(t) = E(e^{tY}) = E(e^{tg(X)}) = \int_{-\infty}^{\infty} e^{tg(X)} f_X(x) dx$$
>4) Change of Variable Approach (Thm 1.7.1)
>	Let $X$ be a continuous random variable, $Y = g(x)$ and $g(x)$ is one-to-one and differentiable, then 
>	$$f_Y(y) = f_X(g^{-1}(y))\left|\frac{dx}{dy}\right|$$

Remark: Method 1 will always give out CDF while Method 3 will always give out PDF.

%%# Mixed Random Variables 
(1.9) %%
# Joint Probability Distributions
(from STAT 400) (2.1)
Let $X$ and $Y$ be two discrete random variables. The joint probability mass function $p ( x, y )$ is defined for each pair of numbers $( x, y )$ by  
$$p ( x, y ) = P ( X = x\: and\: Y = y ) .  $$
Let $A$ be any set consisting of pairs of $( x, y )$ values. Then  
$$P ( ( X, Y ) ∈ A ) = 
∑ ∑  
_{x,y\in A}  
p(x,y)  
$$
Let $X$ and $Y$ be two continuous random variables. Then $f ( x, y )$ is the joint probability density function for $X$ and $Y$ if for any two-dimensional set $A$  
$$P ( ( X, Y ) ∈ A ) = ∫∫_{A}f(x,y)dxdy$$
# Independent Random Variables 
(from STAT 400) (2.5)
>[!note] Def. Independent random variables
>Two random variables X and Y are independent if and only if for all $x,y$:
> discrete:     $p ( x, y ) = p_X ( x ) \cdot p_Y ( y )$.
>continuous: $f ( x, y ) = f_X ( x ) \cdot f_Y ( y )$.  
>OR: $F(x,y) = F_X(x) \cdot F_Y(y)$.
> where $F ( x, y ) = P ( X \leq x, Y \leq y )$. $f ( x, y ) = \partial^2 F ( x, y )/\partial x \partial y .$

# Covariance and Correlation 
(from STAT 400) (2.4)
>[!note] Def. Covariance
>Covariance is **a measure of the relationship between two random variables**. The metric evaluates to what extent the variables change together.
>$$\sigma_{XY} = Cov(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]=E(XY) - \mu_X\mu_Y$$

Remark.
$Cov(X,X) = Var(X)$
$Cov(X,Y) = Cov(Y,X)$
$Cov(aX+Y) = a\:Cov(X,Y)$
$Cov(X+Y,W) = Cov(X,W)+Cov(Y,W)$
$Cov(aX+bY,cX+dY) = ac\:Var(X)+(ad+bc)\:Cov(X,Y) + bd\:Var(Y)$
$Var(aX+bY) = a^2Var(X)+2ab\:Cov(X,Y)+b^2Var(Y)$


>[!note] Def. Correlation coefficient
>The correlation coefficient of two random variables $X,Y$ is given by
>	$$\rho_{X,Y} = \frac{\sigma_{XY}}{\sigma_X \sigma_Y} = \frac{Cov(X,Y)}{\sqrt {Var(X)} \sqrt {Var(Y)}} = E\left[\left(\frac{X-\mu_X}{\sigma_X})(\frac{Y-\mu_Y}{\sigma_Y}\right)\right]$$
>	a) $-1 \leq \rho_{XY} \leq 1$
>	b) $\rho_{XY} = \pm 1 \iff$ $X$ and $Y$ are linear functions of one another.
>	c) If $X,Y$ are independent, then $\rho_{XY} = 0$

# Conditional Distributions and Expected Values 
(2.3)
>[!note] Def. Conditional Distributions
>Given two random variable $X,Y$, we have 
>$$f_{X|Y}(x|y) = \frac{f_{XY}(x,y)}{f_Y(y)}$$
>$$f_{Y|X}(x|y) = \frac{f_{XY}(x,y)}{f_x(x)}$$

>[!note] Def. Conditional Expected Values
>Given two random variables $X,Y$,
>Discrete: $E(X|Y=y) = \sum_{x}xP(X=x\:|Y=y) = \sum_{x}x\:p_{X|Y}(x|y)$ 
>Continuous: $E(X|Y=y) = \int_{-\infty}^{\infty} x\cdot f_{X|Y}\:(x|y)\:dx$ 

A form is commonly used to find the conditional expected value of **discrete random vars**.

Remark.
$E( a_1 X_1 + a_2 X_2 | Y ) = a_1 E ( X_1 | Y ) + a_2 E ( X_2 | Y )$
$E [ g ( Y ) | Y ] = g ( Y )$
$E [ E ( X | Y ) ] = E ( X )$  
$E [ E ( X | Y ) | Y ] = E ( X | Y )$
$E [ g ( Y ) X | Y ] = g ( Y ) E ( X | Y )$

> [!note] Def. Conditional Variance
> $$Var ( X | Y ) = E [ ( X – E ( X | Y ) )^2 | Y ] = E ( X^2 | Y ) – [ E ( X | Y ) ]^2$$

> [!important] Theorem
$E ( E ( X | Y ) ) = E ( X )$
>$Var ( E ( X | Y ) ) \leq Var ( X )$
>Furthermore, $Var ( X ) = Var ( E ( X | Y ) ) + E [ Var ( X | Y ) ]$

If $X$ is a function of $Y$, then  
$E ( X | Y ) = X$ and $Var( E ( X | Y ) ) = Var ( X )$,  
Var $( X | Y ) = 0$ and $E [ Var ( X | Y ) ] = 0$.  

If $X$ and $Y$ are independent, then  
$E ( X | Y ) = E ( X )$ and $Var ( E ( X | Y ) ) = 0$ since $E ( X )$ is a constant,  
$Var ( X | Y ) = Var ( X )$ and $E [ Var ( X | Y ) ] = Var ( X )$.
# Functions of Two Random Variables, 2 –> 1 
(2.2)

# Sum of Two Random Variables, Convolution  
(2.2)
Let X and Y be continuous random variables with joint p.d.f. $f_{XY}(x,y)$, then
$$\begin{align*}
	f_{X+Y}(w) = \int_{-\infty}^{\infty}f(x,w-x)dx\\
	f_{X+Y}(w) = \int_{-\infty}^{\infty}f(w-y,y)dy
\end{align*}$$

>[!note] Convolution
>Let X and Y be  **independent** continuous random variables with joint p.d.f. $f_{XY}(x,y)$, then>$$\begin{align*}	
>f_{X+Y}(w) = \int_{-\infty}^{\infty}f_X(x) \cdot f_Y(w-x)dx\\
>f_{X+Y}(w) = \int_{-\infty}^{\infty}f_X(w-y) \cdot f_Y(y)dy
>\end{align*}$$

Given, $X,Y$ are independent.
If $X$ is Bernoulli $( p )$, $Y$ is Bernoulli $( p )$ , $X + Y$ is Binomial $( n = 2, p )$;  
If $X$ is Binomial $( n_1 , p )$, $Y$ is Binomial $( n_2 , p )$ ,  $X + Y$ is Binomial $( n_1 + n_2 , p )$;  
If $X$ is Geometric $( p )$, $Y$ is Geometric $( p )$,  $X + Y$ is Neg. Binomial $( r = 2, p )$;  
If $X$ is Neg. Binomial $( r_1 , p )$, $Y$ is Neg. Binomial $( r_2 , p )$, $X + Y$ is Neg. Binomial $( r_1 + r_2 , p )$; 
If $X$ is Poisson $( \lambda_1 )$, $Y$ is Poisson $( \lambda_2 )$, $X + Y$ is Poisson $( \lambda_1 + \lambda_2 )$;  
If $X$ is Exponential $( \theta )$, $Y$ is Exponential $( \theta )$, $X + Y$ is Gamma $( n = 2, \theta)$;  
If $X$ is Gamma $( \alpha_1 , \theta )$, $Y$ is Gamma $( \alpha_2 , \theta )$, $X + Y$ is Gamma $( \alpha_1 + \alpha_2 , \theta )$;  
If $X$ is Normal $( \mu_1 , \sigma_1^2 )$, $Y$ is Normal $( \mu_2 , \sigma_2^2 )$, then $X + Y$ is Normal $(\mu_1+\mu_2 , \sigma_1^2+\sigma_2^2)$

# Transformations of Two Random Variables, 2 –> 2 
(2.2)
# Order Statistics 
(4.4)
# Bivariate Normal Distribution 
(3.5)
# Random Vectors, Variance-Covariance Matrix 
(2.6.1)
# Multivariate Normal Distribution, _n_ > 2 
(3.5)
# The _t_ Distribution 
(3.6)
# Functions of One Random Variable


# Order Statistics


# Gamma, Chi-Square, Poisson


# Point Estimation


# Method of Moments
>[!important] Def. Method of Moments Estimator
>Given $f(x;\theta)$, and $X_1,X_2,...,X_n$, i.i.d., we can construct a **Method of Moments** Estimator (MoM) of $\tilde{\theta}$ by
>1) Take the mean of $\{X_i\}$s and express it as a function of $\theta$, $E(X) = g(\theta)$
>2) Express $\tilde{\theta}$ as a function of $\bar{X}$.

# Maximum Likelihood Estimator
> [!important] Def. Maximum Likelihood Estimator
> Given $f(x;\theta)$, and $X_1,X_2,...,X_n$, i.i.d., we can construct a Maximum Likelihood Estimator (MLE) of $\hat{\theta}$ by
> 1) Take $L(\theta) = \prod_{i=1}^nf(x_i;\theta)$ 
> 2) Take $lnL(\theta)$. 
> 3) Take $\frac{\partial lnL(\theta)}{\partial \theta}$ and set it to $0$. 
> 4) Then we have $\hat{\theta}$.


# Unbiased Estimators

# Consistent Estimators

# Jensen's Inequality

# Mean Square Error

# Markov's Inequality

# Chebyshev's Inequality

# Convergence in Probability

# Weak Law of Large Numbers

# Consistent Estimators

# Convergence in Distribution

# Central Limit Theorem

# Delta Method

# Confidence Intervals

# Sufficient Statistics

# Fisher Information

# Efficiency of Estimators


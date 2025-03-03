---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import matplotlib.pyplot as plt
```

+++ {"slideshow": {"slide_type": "slide"}}

## Discrete random variables

+++ {"slideshow": {"slide_type": "skip"}}

The notion of an unpredictable process is too general and in the following we will restrict ourself to outcome sets that are subsets of $\mathbb{R}^M$. We will call such a process a _random variable_.

+++ {"slideshow": {"slide_type": "skip"}}

If the outcome set is countable, in particular if it is finite, then we call such random variable _discrete_. As shown in the `probability` notebook,  to characterise such a variable it is enough to assign  the probability to each of the elements of the outcome set. This is called _probability mass function_ (pmf).

+++ {"slideshow": {"slide_type": "skip"}}

We will denote the probability of random variable $X$  taking a value $x$ by

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(X=x)\equiv P_X(\{x\})$$

+++ {"slideshow": {"slide_type": "skip"}}

However we will often abreviate it further to

+++ {"slideshow": {"slide_type": "fragment"}}

$$ P_X(x) \equiv P_X(\{x\})$$

+++ {"slideshow": {"slide_type": "skip"}}

I will omit the subscript $X$ when it's clear from the context  which random variable I have in mind.

+++ {"slideshow": {"slide_type": "slide"}}

### Join probability distribution

+++ {"slideshow": {"slide_type": "skip"}}

When we have two random variables $X$ and $Y$ with outcome sets $S_X$  and $S_Y$ we can treat them  together as one random variable with outcome set  $S_{X\times  Y}$ being the _cartesian product_  of sets $S_X$ and $S_Y$

+++ {"slideshow": {"slide_type": "skip"}}

$$S_{X\times Y}=S_{X}\times S_Y$$

+++ {"slideshow": {"slide_type": "skip"}}

and joint probability mass function

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{X\times Y}(x,y) \equiv P_{X\times Y}(X=x, Y=y)$$

+++

#### Marginal distribution

+++ {"slideshow": {"slide_type": "skip"}}

If we are interested in only one of the variables we can calculate its probability mass function as _marginal_ pmf

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_X(x)= \sum_y P_{X\times Y}(x, y)\qquad P_Y(Y=y)= \sum_x P_{X\times Y}(x,y)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Independent random variables

+++ {"slideshow": {"slide_type": "skip"}}

The concept of independence applies also to random variables. We say that two random variables $X$ and $Y$ are independent iff (if and only if)

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(X=x|  Y=y)= P(X=x)\quad\text{for all }x,y$$

+++ {"slideshow": {"slide_type": "skip"}}

or equivalently

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{X\times Y}(x, y)= P_X(x)\cdot P_Y(y) \quad\text{for all }x,y$$

+++ {"slideshow": {"slide_type": "skip"}}

For example when $X$ and $Y$ represents a first and second toss of a coin they are independent random variables.

+++ {"slideshow": {"slide_type": "slide"}}

### Expectation value

+++ {"slideshow": {"slide_type": "skip"}}

Expectation value of a function with respect to a random variable $X$ is defined as

+++ {"slideshow": {"slide_type": "fragment"}}

$$E_X[f(X)] \equiv \sum_i f(x_i)P(X=x_i),\quad x_i\in S_X$$

+++ {"slideshow": {"slide_type": "skip"}}

In particular the expectation value of the random variable _i.e._ its _mean_ or _average_ is

+++ {"slideshow": {"slide_type": "fragment"}}

$$E_X[X] \equiv \sum_i x_i P_X(x_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

and the variance

+++ {"slideshow": {"slide_type": "fragment"}}

$$\operatorname{var}(X)=\sigma_X^2 = E[(X-E[X])^2]$$

+++ {"slideshow": {"slide_type": "skip"}}

The square root of variance $\sigma_X$  is called _standard deviation_.

+++ {"slideshow": {"slide_type": "slide"}}

#### Problem: linearity of expectation value

+++ {"slideshow": {"slide_type": "-"}}

Show that

+++ {"slideshow": {"slide_type": "-"}}

$$E_{X\times Y}[a X + b  Y]= a E_X[X] + b E_Y[Y]\quad\text{where }a,b\text{ are constants}$$

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}, "jupyter": {"source_hidden": true}}

$$E_{X\times Y}[a X + b  Y]=\sum_{x,y}\left(a x + b y\right) P(X=x, Y=y) = a \sum_{x,y} x  P(X=x, Y=y)+b \sum_{x,y}  y P(X=x, Y=y) $$

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}, "jupyter": {"source_hidden": true}}

$$a \sum_{x,y} x  P(X=x, Y=y) = a\sum_x  x \sum_y P(X=x, Y=y)= a\sum_x x P(X=x) = a E[X]$$

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}, "jupyter": {"source_hidden": true}}

and same for other term.

+++ {"tags": ["problem"], "slideshow": {"slide_type": "slide"}}

__Problem:__ Variance

+++ {"tags": ["problem"], "slideshow": {"slide_type": "-"}}

Show that

+++ {"tags": ["problem"], "slideshow": {"slide_type": "-"}}

$$\operatorname{var}(X) = E[X^2]-E[X]^2$$

+++ {"slideshow": {"slide_type": "skip"}}

__Answer__

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}, "jupyter": {"source_hidden": true}}

$$\operatorname{var}(X) = E[(X-E[X])^2] = E\left[X^2-2 E[X]+E[X]^2\right]$$

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}, "jupyter": {"source_hidden": true}}

$E[X]$ is a constant so using the linearity of expectation value we obtain

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}, "jupyter": {"source_hidden": true}}

$$E\left[X^2-2 E[X]+E[X]^2\right]=E[X^2]-2E[X]E[X]+E[X]^2$$

+++ {"slideshow": {"slide_type": "slide"}}

### Covariance and correlation

+++ {"slideshow": {"slide_type": "skip"}}

The expectation value of a product of two random variables is given by

+++ {"slideshow": {"slide_type": "fragment"}}

$$E_{X\times Y}[X\cdot Y]=\sum_{x,y} x\cdot y\, P(X=x , Y=y)$$

+++ {"slideshow": {"slide_type": "skip"}}

If the two random variables are independent then

+++ {"slideshow": {"slide_type": "fragment"}}

$$E_{X\times Y}[X\cdot Y]=\sum_{x,y} x y P(X=x , Y=y)
=\sum_{x,y} x y P(X=x)P(Y=y)=
\left(\sum_{x} x P(X=x)\right)
\left(\sum_{y} y P(Y=y)\right)
$$

+++ {"slideshow": {"slide_type": "skip"}}

leading to the familiar result that the expectation value of independent random variables factorises

+++ {"slideshow": {"slide_type": "fragment"}}

$$E_{X\times Y}[X\cdot Y]=E_X[X] E_Y[Y].$$

+++ {"slideshow": {"slide_type": "skip"}}

The quantity

+++ {"slideshow": {"slide_type": "slide"}}

$$\operatorname{cov}(X,Y)=E_{X\times Y}[X\cdot Y]-E_X[X] E_Y[Y]=E[(X-E[X])(Y-E[Y])]$$

+++ {"slideshow": {"slide_type": "skip"}}

is called a _covariance_ and when variables $X$ and $Y$ are independent then it is equal to zero. Please take note however that zero covariance does not imply indpendence.

+++ {"slideshow": {"slide_type": "skip"}}

Magnitude of the covariance depeds on the magnitude of random variables e.g. scaling one variable by $a$ will also scale the covariance by $a$. That is why often a normalised version called _correlation_ coeficient is used:

+++ {"slideshow": {"slide_type": "fragment"}}

$$\operatorname{corr}(X,Y)=\frac{E\left[(X-E[X])(Y-E[Y])\right]}{\sqrt{E\left[(X-E[X])^2\right]E\left[(Y-E[Y])^2\right]}}
=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y}$$

+++ {"tags": ["problem"], "slideshow": {"slide_type": "slide"}}

#### Problem: Linear dependence

+++ {"tags": ["problem"], "slideshow": {"slide_type": "-"}}

Please check that when variables $X$ and $Y$ are linearly dependent _i.e._  $Y =a \cdot X + b$ correlation between them is 1 or -1.

+++

### Examples

+++ {"slideshow": {"slide_type": "skip"}}

Let's illustrate this with some Python code. We start by generating values of two random variables from uniform distribution

```{code-cell}
---
slideshow:
  slide_type: slide
---
xs = np.random.uniform(size=10000)
ys = np.random.uniform(size=10000)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
#covariance 
np.mean( (xs-xs.mean())*(ys-ys.mean() ))
```

+++ {"slideshow": {"slide_type": "skip"}}

We get same result using build in function

```{code-cell}
---
slideshow:
  slide_type: fragment
---
np.cov(xs,ys)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
#correlation
np.mean( (xs-xs.mean())*(ys-ys.mean() ))/np.sqrt(np.mean( (xs-xs.mean())**2)*np.mean((ys-ys.mean() )**2))
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
np.corrcoef(xs,ys)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
zs = xs + ys 
np.corrcoef((xs,ys,zs))
```

+++ {"tags": ["problem"], "slideshow": {"slide_type": "slide"}}

__Problem:__ Variance of sum  of independent random variables

+++ {"tags": ["problem"], "slideshow": {"slide_type": "-"}}

Show that if random variables $X$ and $Y$ are independent then

+++ {"tags": ["problem"]}

$$\operatorname{var}(X+Y) = \operatorname{var}(X) +  \operatorname{var}(Y)$$

+++ {"slideshow": {"slide_type": "skip"}}

Some other characteristics of the random variables include

+++ {"slideshow": {"slide_type": "slide"}}

### Median

+++ {"slideshow": {"slide_type": "skip"}}

Median $m$ is a number that divides the values of the random variable into two  sets as equiprobable as possible

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(X\le m) \ge \frac{1}{2}\text{ and } P(X\ge m) \ge \frac{1}{2}$$

+++ {"tags": ["problem"], "slideshow": {"slide_type": "slide"}}

#### Problem: Median for coin toss

+++ {"tags": ["problem"]}

What is a median for coin toss if you assign value $1$ to heads and $0$ to tails?

+++ {"slideshow": {"slide_type": "slide"}}

### Mode

+++ {"slideshow": {"slide_type": "skip"}}

The mode is the value for which the probability mass function has its maximum. That's an element most likely to be sampled.

+++ {"slideshow": {"slide_type": "-"}}

$$\operatorname{mode}(X)=\underset{x_k}{\operatorname{argmax}} P(X=x_k)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Entropy

+++ {"slideshow": {"slide_type": "skip"}}

Last characteristic of an distribution that I would like to introduce is the _entropy_

+++ {"slideshow": {"slide_type": "fragment"}}

$$H[X] \equiv -\sum_i P_X(x_i) \log P_X(x_i)=-E[\log X] $$

+++ {"slideshow": {"slide_type": "skip"}}

Entropy is a "measure of randomnes", the greater entropy, the greater randomness or harder to predict outcome.

+++ {"slideshow": {"slide_type": "skip"}}

Take for example a coin toss with unfair coin with probability $p$ of comming up heads. The entropy is

+++ {"slideshow": {"slide_type": "fragment"}}

$$-p\log p - (1-p)\log(1-p)$$

```{code-cell}
---
slideshow:
  slide_type: fragment
---
ps = np.linspace(0,1,100)[1:-1] # we reject 0 and 1
plt.plot(ps, -ps *np.log(ps)-(1-ps)*np.log(1-ps));
```

+++ {"slideshow": {"slide_type": "skip"}}

We can see that the entropy is maximum when $p=1/2$ and zero when $p=0$ or $p=1$ that is when the outcome is certain.

+++

## Some interesting discrete distributions

+++

### Bernouli distribution

+++

### Binomial distribution

+++ {"jp-MarkdownHeadingCollapsed": true}

### Categorical distribution

+++

### Multinomial distribution

+++

### Poisson distribution

```{code-cell}

```

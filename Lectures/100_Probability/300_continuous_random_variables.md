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
import scipy.stats as st
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Continuous random variables

+++ {"slideshow": {"slide_type": "fragment"}}

### Univariate  (one dimensional)

+++ {"slideshow": {"slide_type": "skip"}}

By continous random variables we will understand variables with have a connected subset of $S\in \mathbb{R}$ e.g. an interval as the outcome set.

+++ {"slideshow": {"slide_type": "skip"}}

When the set of the outcomes is not countable _i.e._ we cannot enumerate them, we cannot  specify probability of the event by adding probabilities of elementary events it contains.  Actually for most of the interesting continous random variables the probability of a single outcome is zero

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(X=x) = 0.$$

+++ {"slideshow": {"slide_type": "slide"}}

### Cummulative distribution function

+++ {"slideshow": {"slide_type": "skip"}}

However we can ask for the probability that the outcome is smaller then some number:

+++ {"slideshow": {"slide_type": "fragment"}}

$$F_X(x) = P(X\le x)$$

+++ {"slideshow": {"slide_type": "skip"}}

This is called a cummulative distribution function (cdf) or _cummulant_.

+++ {"slideshow": {"slide_type": "slide"}}

#### Problem

+++ {"slideshow": {"slide_type": "-"}}

Let $X$  be an random variable taking values in the  interval $[a,b)$. Please show that

+++ {"slideshow": {"slide_type": "-"}}

$$
F_X(a) = 0 \quad\text{and}\quad  \lim_{x\rightarrow b}F_X(x) = 1
$$

+++ {"slideshow": {"slide_type": "-"}}

Is this a really true?

+++ {"slideshow": {"slide_type": "slide"}}

### Probability density function

+++ {"slideshow": {"slide_type": "skip"}}

We can also ask for the probability that the outcome lies in a small interval $\Delta x$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(x<X\le x+\Delta x)$$

+++ {"slideshow": {"slide_type": "skip"}}

For small intervals and "well behaved" random variables we expect that this probability will be proportional to $\Delta x$

+++

$$P(x<X\le x+\Delta x)\approx f(x) \Delta x.$$

+++ {"slideshow": {"slide_type": "skip"}}

So let's take the ratio and go to the limit $\Delta x\rightarrow 0$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{P(x<X<x+\Delta x)}{\Delta x}\underset{\Delta x\rightarrow 0}{\longrightarrow} f(x) \equiv P_X(x)$$

+++ {"slideshow": {"slide_type": "skip"}}

If this limit exists $P_X(x)$ is called a _probability density function_ (pdf).

+++ {"slideshow": {"slide_type": "skip"}}

There is a relation between cdf and pdf

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$ P_X(x) =\frac{\text{d}}{\text{d}x}F_X(x)\qquad F_X(x) = \int\limits_{-\infty}^x P_X(x')\text{d}x'$$

+++ {"slideshow": {"slide_type": "skip"}}

Most of the definitions and properties of the probability mass function apply to probability density function with summation changed to integral _e.g._

+++ {"slideshow": {"slide_type": "slide"}}

$$E_X[f(X)]\equiv \int\text{d}x f(x) P(x)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Multivariate

+++ {"slideshow": {"slide_type": "skip"}}

When the outcome set of the random variable is some connected subset of $\mathbb{R}^n$ we are talking about _multivariate_ random variables. The probability density function is defined in the same way:

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\begin{split}
P_X(\mathbf{x}) &\equiv \\
&\lim_{\Delta x_{i}\rightarrow 0}\frac{P(x_1<X_1<x_1+\Delta x_1,\ldots,x_n<X_n<x_n+\Delta x_n )}{\Delta x_1\cdots \Delta x_n}
\end{split}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

One can generalize the cummulative distribution function in the same way, but it is less commonly used.

+++ {"slideshow": {"slide_type": "slide"}}

## Some useful continuous random variables

+++ {"slideshow": {"slide_type": "slide"}}

### Normal distribution

+++ {"slideshow": {"slide_type": "skip"}}

Probably  the most known continuous distribution is the _normal_ or Gaussian distribution. It is characterised by its mean $\mu$  and  standard deviation $\sigma$. Its probability density function is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{\displaystyle -\frac{(x-\mu)^2}{2\sigma^2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

and it has a characteristic bell-like shape

```{code-cell}
---
slideshow:
  slide_type: fragment
---
xs = np.linspace(-5,7,500)
for s in [0.25, 0.5, 1,2]:
    plt.plot(xs,st.norm.pdf(xs, loc=1, scale=s), label="$\\sigma = {:4.2f}$".format(s));
plt.axvline(1, c='grey', linewidth=1);
plt.legend();
```

As you can see this distribution has a form

+++

$$\frac{1}{\sigma}f\left(\frac{x-\mu}{\sigma}\right)$$

+++

When we encounter such distributions we will also call $\mu$ a _location_ paramter and $\sigma$ a _scale_ parameter.

+++ {"slideshow": {"slide_type": "skip"}}

The prevalence of this  random variable can be attributed to central limit theorem that states that, under some mild assumptions,  the sum of independent random variables  approaches the normal random variable as the number of variables tends to infinity.

+++ {"slideshow": {"slide_type": "skip"}}

Another feature  of the normal distribution is that it is the distribution with highest entropy with given mean and variance.

+++ {"slideshow": {"slide_type": "skip"}}

As you can see on the probability density function $P_X(x)$ is not restricted to be less then one. That's because this is a _density_. We  can meaningfully only ask about probability of $X$ having an outcome in an interval  which is given by the area under a fragment of the curve

```{code-cell}
---
slideshow:
  slide_type: slide
---
distrib  = st.norm(loc=1, scale=0.25)
a = 0.75
b = 0.90
xs = np.linspace(0,2,500)
ab = np.linspace(a,b,100)
plt.axhline(0, linewidth=0.5, color = 'grey')
plt.plot(xs,distrib.pdf(xs));
plt.fill_between(ab,distrib.pdf(ab), alpha=0.5 )
plt.axvline(1, c='grey', linewidth=0.5);
area = distrib.cdf(b)-distrib.cdf(a)
plt.text(0.2, 1.4, "$P(a<X<b) = {:2f}$".format(area), fontsize=14);
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
n=100000
sample = distrib.rvs(size=n)
( (a<sample) & (sample<b)).sum()/n
```

+++ {"slideshow": {"slide_type": "skip"}}

The area was calculated using the cumulative distribution function

+++ {"slideshow": {"slide_type": "slide"}}

$$P(a<X<b)=F_X(b)-F_X(a)$$

```{code-cell}
xs = np.linspace(0,2,500)
plt.plot(xs,distrib.cdf(xs));
plt.plot([a,a,0],[0,distrib.cdf(a), distrib.cdf(a)], c='grey')
plt.plot([b,b,0],[0,distrib.cdf(b),distrib.cdf(b)], c='grey');
plt.axhline(0, linewidth=0.5, color = 'grey');
plt.axhline(1, linewidth=0.5, color = 'grey');
plt.axvline(0, linewidth=0.5, color = 'grey');
plt.axvline(2, linewidth=0.5, color = 'grey');
```

+++ {"slideshow": {"slide_type": "skip"}}

The normal distribution generalizes easilly to more dimesions

+++ {"slideshow": {"slide_type": "slide"}}

$$\newcommand{\b}[1]{\mathbf{#1}}$$
$$P(\b{x}|\pmb\mu, \pmb\Sigma) = 
\frac{1}{(2\pi)^{\frac{n}{2}}}
\frac{1}{\sqrt{\det \pmb{\Sigma}}}
e^{\displaystyle -\frac{1}{2}(\b{x}-\pmb\mu)^T \pmb\Sigma^{-1}(\b{x}-\pmb{\mu})}$$

+++ {"slideshow": {"slide_type": "skip"}}

where $\pmb\Sigma$ is some _positive definite_ covariance matrix matrix and $\pmb\mu$ is a vector of mean values.

+++ {"slideshow": {"slide_type": "skip"}}

Show that when matrix $\pmb\Sigma$ is

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

### Beta distribution

+++ {"slideshow": {"slide_type": "skip"}}

The  [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) has two parameters  $\alpha$ and $\beta$  and its probability density function is

+++ {"slideshow": {"slide_type": "-", "slideshow": {"slide_type": "fragment"}}}

$$P(x|\alpha,\beta) =  \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
x^{\alpha-1}(1-x)^{\beta-1},\quad 0\leq x\leq 1
$$

+++ {"slideshow": {"slide_type": "skip"}}

Its importance stems from  the fact that it is a _conjugate_ prior to Bernoulli distribution so it is used to set the "probability on probability". You will learn more about this  in bayesian_analysis notebook.

+++ {"slideshow": {"slide_type": "skip"}}

Here are plots of the probability density function for some values of $\alpha=\beta$

```{code-cell}
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,2,5,10]:
    ys = st.beta(a,a).pdf(xs)
    plt.plot(xs,ys, label='%4.2f' %(a,))
plt.legend(loc='best', title='$\\alpha=\\beta$');
```

+++ {"slideshow": {"slide_type": "skip"}}

And here for some values of $\alpha\neq\beta$

```{code-cell}
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,5]:
    ys = st.beta(a,2.0).pdf(xs)
    plt.plot(xs,ys, label='%4.2f' %(a,))
plt.legend(loc=1, title='$\\alpha$');
```

+++ {"slideshow": {"slide_type": "skip"}}

It can be more convenient to parametrise  Beta distrubution by its mean and variance. The mean and variance of Beta distribution are

+++ {"slideshow": {"slide_type": "slide"}}

$$\mu = \frac{\alpha}{\alpha+\beta}\quad\text{and}\quad \sigma^2=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

+++ {"slideshow": {"slide_type": "skip"}}

Introducing a new auxiliary variable

+++ {"slideshow": {"slide_type": "fragment"}}

$$\nu = \alpha+\beta$$

+++ {"slideshow": {"slide_type": "skip"}}

we have

+++ {"slideshow": {"slide_type": "fragment"}}

$$\alpha = \mu \nu,\quad \beta = (1-\mu)\nu,\quad \sigma^2=\frac{\mu(1-\mu)}{\nu +1} $$

+++ {"slideshow": {"slide_type": "skip"}}

so

+++ {"slideshow": {"slide_type": "fragment"}}

$$\nu=\frac{\mu(1-\mu)}{\sigma^2}-1$$

+++ {"slideshow": {"slide_type": "skip"}}

and finally

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\alpha = \mu \left(\frac{\mu(1-\mu)}{\sigma^2}-1\right)\quad\text{and}\quad\beta = (1-\mu) \left(\frac{\mu(1-\mu)}{\sigma^2}-1\right)$$

+++

### Dirichlet distribution

+++

### Exponential distribution

+++

### Gamma distribution

+++

### Student's t distribution.

+++

### $\chi^2$ distribution

```{code-cell}

```

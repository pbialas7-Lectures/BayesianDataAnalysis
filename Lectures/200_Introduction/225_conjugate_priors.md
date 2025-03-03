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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Conjugate priors

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
import numpy as np
import scipy
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Categorical variables

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

A natural generalization of the Bernoulli distribution is the multinouilli or categorical distribution and the generalization of the binomial distribution is the _multinomial_ distribution.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Let's say we have $m$ categories with probability $p_k$ for each category. Then after $n$ trials the probability that we get $n_k$ results in category $k$ is:

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(n_1,\ldots, n_{m}|p_1,\ldots, p_{m}) = \frac{n!}{n_1!\cdots n_{m}!}p_1^{n_1}\cdots p_{m}^{n_{m}},\qquad n = n_1+n_2+\cdots+n_m$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Example: Dice

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$m=6, \quad p_i=\frac{1}{6}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

 $$ P(n_1,\ldots, n_{6}) = \frac{n!}{n_1!\cdots n_{m}!}\frac{1}{6^m}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

#### Problem

We are rolling four dices. What is the probability that we roll four ones?  What is the probability that we roll numbers from one to four?

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$\frac{4!}{4!0!0!0!0!0!}\frac{1}{6^4}=\frac{1}{1296}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$\frac{4!}{1!1!1!1!0!0!}\frac{1}{6^4}=\frac{1}{54}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Dirichlet distribution

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Conjugate prior  to this distribution is the Dirichlet distribution which is a generalization of the Beta distribution. It has $m$ parameters $\alpha_k$ and its probability mass function is

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P_{Dir}(p_1,\ldots,p_{m}|\alpha_1,\ldots,\alpha_{m}) = \frac{\Gamma\left(\sum\limits_{i=1}^{m} \alpha_i\right)}{\prod\limits_{i=1}^{m}\Gamma(\alpha_i)}
\prod\limits_{i=1}^{m}p_i^{\alpha_i-1}\qquad \sum_{i=1}^m p_i =1$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

It is easy to check that the posterior probability density for $P(p_1,\ldots, p_{m}|n_1,\ldots, n_{m})$ with prior given by Dirichlet distribution with parameters $\alpha_k$  is again given with by the  Dirichlet distribution with parameters $\alpha_1+n_1,\ldots, \alpha_{m}+n_{m}$.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p_1,\ldots, p_{m}|n_1,\ldots, n_{m})=P_{Dir}(p_1,\ldots,p_{m}|\alpha_1+n_1,\ldots,\alpha_{m}+n_m)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### MAP

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The maximal a posteriori estimate is:

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{MAP\,k} = \frac{n_k+\alpha_k-1}{n + \sum_i \alpha_k-m}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Mean

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and the mean

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\langle p_k\rangle = \frac{n_k+\alpha_k}{\sum_{k=1}^m n_k  + \sum_{k=1}^m \alpha_k}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Poisson distribution

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Poisson distribution appears when  some events occur with uniform probability in time. For example there are $\lambda$ customers per hour on average walking into a store and probability of this happening is same all the time, then number of customers that visit the store in time interval $t$ (measured in hours)  is a discrete random variable with probability mass function

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(k|t, \lambda) = e^{-t\lambda}\frac{(t\lambda)^k}{k!}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.stats import poisson
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
ns=np.arange(30)
for l in [1,2,5,10]:
  plt.scatter(ns, poisson(l).pmf(ns),s=10, label=f"$\\lambda={l}$");
plt.legend();  
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For large $\lambda$ Poisson distribution approaches the normal distribution with $\mu=\lambda$ and $\sigma^2=\lambda$.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
ns=np.arange(200)
plt.scatter(ns, poisson(100).pmf(ns),s=10, label=f"$\\lambda={l}$");
xs = np.linspace(0,200,1000);
plt.plot(xs, scipy.stats.norm(100,np.sqrt(100)).pdf(xs));
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Assuming uniform prior on the parameter $\lambda$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(\lambda)=1$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

(this is an _improper_ prior, as this is not a normalizable probability distribution) we obtain for posterior after  observing a series of counts $\{k_i\}$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(\{k_i\}|t,\lambda) = \prod_{i=1}^n e^{-t\lambda}\frac{(t\lambda)^{k_i}}{k_i!}\propto (t\lambda)^{\displaystyle n \bar{k}}e^{\displaystyle -n t\lambda}$$

+++ {"slideshow": {"slide_type": ""}, "editable": true}

 $$ \bar{k}=\frac{1}{n}\sum_{i=1}^n k_i$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

From this formula we can infer that distribution with a pdf of the form

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(\lambda)\propto e^{-\beta\lambda}\lambda^{\alpha-1}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

would be a conjugate prior to Poisson distribution, as

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$(t\lambda)^{n\bar k}e^{- tn\lambda}\cdot e^{-\beta\lambda}\lambda^{\alpha-1} = e^{-(\beta+n t)\lambda}t^{n\bar k}\lambda^{n\bar k+\alpha-1}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The distribution of this form is called [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\operatorname{PDF}[Gamma(\alpha,\beta),x]=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and  posterior distribution for Poisson distribution after observing $\{k_i\}$ counts in time $t$, with prior $\Gamma(\alpha,\beta)$ is

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(\lambda|t,\{k_i\}) =PDF[Gamma(\alpha+n \bar k,\beta+n t)]$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.stats import gamma
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
xs = np.linspace(1,20,500)
fig, ax = plt.subplots()
b=0.75
ax.set_title(f"Gamma distribution $\\beta={b:.2f}$")
for a in [0.5,0.75,1,2,5,7]:
    pdf = gamma(a=a,scale=1/b).pdf
    ax.plot(xs, pdf(xs), label=f"$\\alpha={a:.2f}$");
ax.legend();    
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
xs = np.linspace(1,20,500)
fig, ax = plt.subplots()
b=2
ax.set_title(f"Gamma distribution $\\beta={b:.2f}$")
for a in [0.5,0.75,1,2,5,7]:
    pdf = gamma(a=a,scale=1/b).pdf
    ax.plot(xs, pdf(xs), label=f"$\\alpha={a:.2f}$");
ax.legend();    
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Mode

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\lambda_{MAP} =\frac{\alpha-1+n \bar k}{\beta + n t}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For uniform prior $\alpha = 1$, $\beta=0$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\lambda_{MAP} = \frac{\bar k}{t}$$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Mean

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\lambda_{MAP} =\frac{\alpha+n \bar k}{\beta + n t}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For uniform prior $\alpha = 1$, $\beta=0$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\lambda_{MAP} = \frac{1+ n \bar k}{n t}=\frac{\bar k}{t}+\frac{1}{n t}$$

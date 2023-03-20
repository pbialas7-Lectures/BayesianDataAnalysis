---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import scipy as sp
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}}

# Bayesian data analysis -- Normal model

+++ {"slideshow": {"slide_type": "skip"}}

Let's recall the most important formula of the Bayesian data analysis:

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\theta|y) \propto P(y|\theta) P(\theta)$$

+++ {"slideshow": {"slide_type": "skip"}}

The quantity $P(y|\theta)$ on the right side is called the _sampling distribution_ when viewed as the function of $y$. It describes how data $y$ is generated given the (unknown) parameters $\theta$. When viewed as a function of $\theta$ with $y$ fixed it is called the _likelihood_. Please note that likelihood  in general is **not** a probability distribution for parameters $\theta$.

+++ {"slideshow": {"slide_type": "skip"}}

Probability distribution $P(\theta)$ is the _prior_. It represents our knowledge, or lack of it, of parameters $\theta$ before we collected any data.

+++ {"slideshow": {"slide_type": "skip"}}

And finally the quantity on the left is the _posterior_  probability distribution which represents our new knowledge of parameters $\theta$ after having collected data $y$. The formula above is valid only up to a normlizing constant $Z^{-1}$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\theta|y) = \frac{ P(y|\theta) P(\theta)}{Z}\qquad Z=\int\text{d}\theta P(y|\theta) P(\theta)$$

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model with know variance

+++ {"slideshow": {"slide_type": "skip"}}

Let's assume that our sampling distribution is Gaussian with know variance $\sigma^2$ and unknown mean $\mu$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\{y_k\}|\mu,\sigma) =\prod_k \frac{1}{\sqrt{2\pi}\sigma} 
e^{-\displaystyle\frac{1}{2\sigma^2}\left(y_k-\mu\right)^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

To get the posterior we need a prior on $\mu$.

+++ {"slideshow": {"slide_type": "slide"}}

###  Uninformative (improper) prior on location parameter

+++ {"slideshow": {"slide_type": "skip"}}

Our sampling distribution has this property that it depends only on the differences of $y_i$ and $\mu$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\{y_k\}|\mu,\sigma)=f(\{y_k-\mu\},\sigma)$$

+++ {"slideshow": {"slide_type": "skip"}}

It is reasonable to  assume same relation for the posterior

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|\{y_k\},\sigma)=g(\{y_k-\mu\},\sigma)$$

+++ {"slideshow": {"slide_type": "skip"}}

Inserting this into the formula for posterior we obtain the relation

+++ {"slideshow": {"slide_type": "fragment"}}

$$g(\{y_k-\mu\},\sigma) \propto f(\{y_k-\mu\},\sigma) P(\mu)$$

+++ {"slideshow": {"slide_type": "skip"}}

that can be satisfied only when the prior does not depend on $\mu$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu)\propto 1$$

+++ {"slideshow": {"slide_type": "skip"}}

Please note that this is __not__ a probability distribution as it is not normalizable. We call it an _improper_ prior. Nevertheless we can still use it  as long as posterior is a proper distribution.  So the posterior is proportional to the likelihood

+++ {"slideshow": {"slide_type": "slide"}}

$\newcommand{\b}[1]{\mathbf{#1}}$
$$P(\mu|\b y,\sigma) \propto \prod_{k=1}^n \frac{1}{\sqrt{2\pi}\sigma} 
e^{-\frac{1}{2\sigma^2}\left(y_k-\mu\right)^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

I have changed the notation and now bold letter indicate vectors. In this case $\b y \equiv \{y_k\}$.

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu|\b y,\sigma) \propto  \left(\sqrt{2\pi}\sigma\right)^{-n} 
e^{-\frac{1}{2\sigma^2}\sum_{k=1}^n\left(y_k-\mu\right)^2}
$$

+++ {"slideshow": {"slide_type": "slide"}}

$$\sum_{k=1}^n\left(y_k-\mu\right)^2=\sum_{k=1}^n\left(y_k^2-2y_k \mu +\mu^2\right)=\sum_{k=1}^n y_k^2-2\sum_{k=1}^ny_k \mu +n\mu^2$$

+++ {"slideshow": {"slide_type": "skip"}}

denoting the averages by bar

+++ {"slideshow": {"slide_type": "slide"}}

$$
\overline{y^2}\equiv\frac{1}{n}\sum_{k=1}^n y_k^2,\qquad 
\bar{y}\equiv\frac{1}{n}\sum_{k=1}^n y_k
$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$
\begin{split}
\sum_{k=1}^n\left(y_k-\mu\right)^2&= n \left(\bar{y^2}-2\bar{y}\mu+\mu^2\right)\\
&= n \left(\bar{y}^2-2\bar{y}\mu+\mu^2  +\bar{y^2}-\bar{y}^2\right)\\
&=n\left(\bar{y}-\mu\right)^2+n\left(\bar{y^2}-\bar{y}^2\right)
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}}

leading finally to

+++ {"slideshow": {"slide_type": "slide"}}

$$
P(\mu|y,\sigma) \propto  \sigma^{-n} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n}{2\sigma^2}\left(\overline{y^2} -{\bar y }^2\right)}
$$

+++ {"slideshow": {"slide_type": "skip"}}

The above relation means that random variable $\mu$ with fixed $\b y$ and $\sigma$ is distributed according to normal distribution with mean $\bar y$ and variance $\frac{\sigma}{\sqrt{n}}$ which we will denote by the notation below

+++ {"slideshow": {"slide_type": "fragment"}}

$$\mu|y,\sigma \sim \operatorname{Norm}\left(\bar y,\frac{\sigma}{\sqrt{n}}\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

That also means the the MAP estimate of $\mu$ is  the average of $\b y$

+++ {"slideshow": {"slide_type": "skip"}}

$$\mu_{MAP}= \bar{\b y}$$

+++ {"slideshow": {"slide_type": "skip"}}

as expected.

+++ {"slideshow": {"slide_type": "slide"}}

## Nuisance parameters

+++ {"slideshow": {"slide_type": "skip"}}

More often that not, our model will have more then one parameter

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1,\theta_2|y) \propto P(y|\theta_1,\theta_2) P(\theta_1,\theta_2)$$

+++ {"slideshow": {"slide_type": "skip"}}

However we may be not interested in all of them. The uninteresting parameters are so called _nuisance_ parameters. In this example let's assume that we are only interested in parameter $\theta_1$ while $\theta_2$ is the nuisance parameter. We can obtain  the posterior for $\theta_1$ by integrating  over the nuisance parameter to obtain the _marginal_ distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1|y) = \int\text{d}{\theta_2}P(\theta_1,\theta_2|y) $$

+++ {"slideshow": {"slide_type": "skip"}}

This can be also rewritten using the  product rule

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1|y) = \int\text{d}\theta_2 P(\theta_1|y,\theta_2)P(\theta_2|y)$$

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model with unknown variance

+++ {"slideshow": {"slide_type": "slide"}}

### Prior on scale parameter

+++ {"slideshow": {"slide_type": "skip"}}

Now we will consider  again Gaussian sampling distribution, but this time variance  will be also an unknown parameter. To find the prior on $\sigma$ we will  use similar reasoning as in case of $\mu$. The sampling distribution does depend on $\b y$, $\mu$ and $\sigma$ in a specific way

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y|\mu,\sigma)=\frac{1}{\sigma}\cdot f\left(\frac{y-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

We say that $\sigma$ is a _scale_ parameter. The $\sigma^{-1}$ factor assures the proper normalization: the integral over $y$ does not depend on $\sigma$. We will assume similar relation for posterior

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu,\sigma|y) = \frac{y}{\sigma^2} g\left(\frac{y-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

again the factor in front stems from the normalization. It ensures that the integral over $\mu$ and $\sigma$ is independent of $y$.

+++ {"slideshow": {"slide_type": "skip"}}

This leads to the relation

+++ {"slideshow": {"slide_type": "slide"}}

$$\frac{1}{\sigma}f\left(\frac{y-\mu}{\sigma}\right)P(\mu,\sigma)\propto \frac{y}{\sigma^2}g\left(\frac{y-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

which is satisfied only if

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_\sigma(\sigma)\propto \frac{1}{\sigma}$$

+++ {"slideshow": {"slide_type": "skip"}}

If we want to use the variance $\sigma^2$ instead of standard deviation $\sigma$ the the distribution is

+++ {"slideshow": {"slide_type": "slide"}}

$$P_{\sigma^2}(\sigma^2)\propto \frac{1}{\sigma^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

This can be checked by calculating the cumulative distribution function for this distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$CDF_{P_{\sigma^2}}(z)=P(\sigma^2<z)=P(\sigma<\sqrt{z})=CDF_{P_\sigma}(\sqrt{z})$$

+++ {"slideshow": {"slide_type": "skip"}}

and differentiating it with respect to $z$.

+++ {"slideshow": {"slide_type": "skip"}}

Please note that those are also improper priors.

+++ {"slideshow": {"slide_type": "skip"}}

Sometimes is useful to use $x=\log \sigma$ as our variable. We can derive the distribution for $x$ in the same way

+++ {"slideshow": {"slide_type": "slide"}}

$$P(x<z)=P(\log \sigma< z)=P(\sigma<e^z)=CDF_{P_\sigma}(e^z)$$

+++ {"slideshow": {"slide_type": "skip"}}

Differentiating with respect to $z$ we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{\log\sigma}(z)=e^z \text{CDF}^\prime_{P_\sigma}(e^z)\propto e^z \frac{1}{e^z} = 1$$

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model --  posterior for variance $\sigma^2$

+++ {"slideshow": {"slide_type": "skip"}}

Using variance as our parameter we obtain for the posterior

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu,\sigma^2|y) \propto  \left(\sigma^2\right)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n}{2\sigma^2}\left(\overline{y^2} -{\bar y }^2\right)}
$$

+++ {"slideshow": {"slide_type": "skip"}}

Introducing the unbiased variance estimator $s^2$

+++ {"slideshow": {"slide_type": "fragment"}}

$$ s^2=\frac{n}{n-1}\left(\overline{y^2} -{\bar y }^2\right)
$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu,\sigma^2|y) \propto  (\sigma^2)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

We will start with obtaining the posterior distribution for $\sigma^2$ by integrating  the joint posterior over $\mu$ which is just a Gaussian integral

+++ {"slideshow": {"slide_type": "slide"}}

$$
\begin{split}
P(\sigma^2|y) &\propto  \int\text{d}\mu\,\sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}\\
&\propto
\sigma^{-n-2} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}\sqrt{2\pi\frac{\sigma^2}{n}}
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}}

leading to

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\sigma^2|y) \propto\left(\sigma^2\right)^{-\frac{n+1}{2}} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

This is a [_inverse Gamma_ distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)

+++ {"slideshow": {"slide_type": "slide"}}

$$\sigma^2|y \sim \operatorname{Inv-}\Gamma\left(\alpha=\frac{n-1}{2},\beta=\frac{1}{2}(n-1)s^2\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

Mode of this distribution is

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{\beta}{\alpha+1}=\frac{n-1}{n+1}s^2$$

+++ {"slideshow": {"slide_type": "skip"}}

leading to slightly biased MAP estimator of $\sigma^2$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
from scipy.stats import norm
sigma=0.7
var = sigma**2
mu=1.5
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
np.random.seed(86864755)
n=20
y = norm(mu, sigma).rvs(n)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
s2 = y.var(ddof=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import invgamma
alpha = (n-1)/2
beta  = ((n-1)*s2)/2
post_var = invgamma(a=alpha, scale=beta)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
vars = np.linspace(1e-6,1.5,500)
plt.plot(vars,post_var.pdf(vars));
plt.axvline(var);
```

```{code-cell} ipython3
vars[np.argmax(post_var.pdf(vars))]
```

```{code-cell} ipython3
var_map = s2*(n-1)/(n+1)
var_map
```

+++ {"slideshow": {"slide_type": "skip"}}

The join posterior can be rewritten as

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu,\sigma^2|y) = P(\mu|\sigma^2,y)P(\sigma^2|y)$$

+++ {"slideshow": {"slide_type": "skip"}}

with

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|\sigma^2,y) = \frac{1}{\sqrt{2\pi \frac{\sigma^2}{n}}}
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

which is the  Gaussian distribution $Norm(\bar y,\frac{\sigma^2}{\sqrt{n}})$

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model -- posterior distribution for the mean $\mu$

+++ {"slideshow": {"slide_type": "skip"}}

To obtain the posterior for $\mu$ we have to integrate out the $\sigma^2$ parameter

+++ {"slideshow": {"slide_type": "slide"}}

$$
P(\mu|y)=\int\text{d}\sigma^2 P(\mu,\sigma^2|y) \propto \int_0^\infty\text{d}\sigma^2 \sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

Introducing a new variable

+++ {"slideshow": {"slide_type": "fragment"}}

$$z=\frac{A}{2\sigma^2},\quad A=n(\bar y-\mu)^2+(n-1)s^2$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\text{d}z=-\frac{A}{2\sigma^4}\text{d}\sigma^2$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu|y)\propto A^{-\frac{n}{2}}\int_0^\infty\text{d}z\,z^{\frac{n-1}{2}}e^{-z}$$

+++ {"slideshow": {"slide_type": "skip"}}

leading to

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu, y)\propto \left(n(\bar y-\mu)^2+(n-1)s^2\right)^{-\frac{n}{2}}=\left((n-1)s^2\right)^{-\frac{n}{2}}\left(\frac{n}{n-1}\frac{(\bar y-\mu)^2}{s^2}+1\right)^{-\frac{n}{2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

Putting $\nu=n-1$ we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|y)\propto\left(\frac{n}{\nu}\frac{(\bar y-\mu)^2}{s^2}+1\right)^{-\frac{\nu+1}{2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

which can be identified as [Student's _t_-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)  with $\nu$ degrees of freedom for variable

+++ {"slideshow": {"slide_type": "slide"}}

$$x=
\frac{\mu-\bar y}
{\sqrt{\frac{s^2}{n}}}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\left.
\frac{\mu-\bar y}
{\sqrt{\frac{s^2}{n}}}
\right|\sim t_{n-1}$$

+++ {"slideshow": {"slide_type": "skip"}}

giving the expected MAP estimator

+++ {"slideshow": {"slide_type": "fragment"}}

$$\mu_{MAP}=\bar y$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
mu_map= y.mean()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import t
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
post_mu = t(df=n-1, loc=y.mean(), scale=np.sqrt(s2/n))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
mus = np.linspace(0.5,2,500)
plt.plot(mus, post_mu.pdf(mus) );
plt.axvline(mu);
```

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model -- Joint distribution for $\mu$ and $\sigma^2$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu,\sigma^2|y) = P(\mu|\sigma^2,y)P(\sigma^2|y)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def post_mu_cond_var_log_pdf(mu, var):
    return norm(y.mean(),np.sqrt(var/n)).logpdf(mu)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def post_mu_cond_var_pdf(mu, var):
    return norm(y.mean(),np.sqrt(var/n)).pdf(mu)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
xs,ys = np.meshgrid(mus, vars)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def post_joined_pdf(mu,var):
    return post_var.pdf(var)*post_mu_cond_var_pdf(mu, var)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def post_joined_log_pdf(mu,var):
    return post_var.logpdf(var)+post_mu_cond_var_log_pdf(mu, var)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
joined = post_joined_pdf(xs,ys)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
log_joined = post_joined_log_pdf(xs,ys)
log_joined-=np.max(log_joined)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots()
ax.set_aspect(1)
cax=ax.contourf(xs,ys,log_joined, levels=np.log(np.array([0.0001,0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])))
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$\\sigma^2$")
ax.scatter([mu],[var], color='red', label='true')
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Marginal distributions

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
var_dist = joined.sum(1) 
var_dist/=np.trapz(var_dist, vars)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(vars, var_dist)
plt.plot(vars,post_var.pdf(vars));
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
mu_dist = joined.sum(0) 
mu_dist/=np.trapz(mu_dist, mus)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(mus, mu_dist)
plt.plot(mus, post_mu.pdf(mus));
```

+++ {"slideshow": {"slide_type": "slide"}}

### MAP

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
i_joined_map = np.unravel_index(np.argmax(joined), joined.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
vars[i_joined_map[0]]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
mus[i_joined_map[1]]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
ax.scatter([mu_map],[var_map], color='orange',label='marginal MAP');
ax.scatter([mus[i_joined_map[1]]],[vars[i_joined_map[0]]], color='blue', label='joined MAP');
ax.legend();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3

```

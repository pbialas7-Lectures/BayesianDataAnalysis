---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
editable: true
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

+++ {"slideshow": {"slide_type": "skip"}}

$$\renewcommand{\b}[1]{\boldsymbol{#1}}$$

+++ {"slideshow": {"slide_type": "slide"}}

# Bayesian data analysis -- Normal model

+++ {"slideshow": {"slide_type": "skip"}}

Let's recall the most important formula of the Bayesian data analysis:

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\b\theta|\b y) \propto P(\b y|\b\theta) P(\b\theta)$$

+++ {"slideshow": {"slide_type": "skip"}}

In this formula $\b y$ stands for the data we have collected and $\b\theta$ for the parameters we want to estimate. Bold letters denote vectors or more generally tensors

+++

$$\b y = \{y_k\} = \{y_1,y_2,\ldots,y_N\}$$

+++ {"slideshow": {"slide_type": "skip"}}

The quantity $P(\b y|\b\theta)$ on the right side is called the _sampling distribution_ when viewed as the function of $\b y$. It describes how data $\b y$ is generated given the (unknown) parameters $\b\theta$. When viewed as a function of $\b\theta$ with $y$ fixed it is called the _likelihood_. Please note that likelihood  in general is **not** a probability distribution for parameters $\theta$.

Probability distribution $P(\b\theta)$ is the _prior_. It represents our knowledge, or lack of it, of parameters $\b\theta$ before we collected any data.

+++ {"slideshow": {"slide_type": "skip"}}

And finally the quantity on the left is the _posterior_  probability distribution which represents our new knowledge of parameters $\theta$ after having collected data $y$. The formula above is valid only up to a normalizing constant $Z^{-1}$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\theta|\b y) = \frac{ P(\b y|\b\theta) P(\b\theta)}{Z}\qquad Z=\int\text{d}\b\theta P(\b y|\b\theta) P(\b\theta)$$

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model with know variance

+++ {"slideshow": {"slide_type": "skip"}}

Let's assume that our sampling distribution is Gaussian with know variance $\sigma^2$ and unknown mean $\mu$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\{y_k\}|\mu,\sigma) =\prod_{k=1}^N \frac{1}{\sqrt{2\pi}\sigma} 
e^{-\displaystyle\frac{1}{2\sigma^2}\left(y_k-\mu\right)^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

To get the posterior we need a prior on $\mu$.

+++ {"slideshow": {"slide_type": "slide"}}

###  Uninformative prior on location parameter

+++ {"slideshow": {"slide_type": "skip"}}

Our sampling distribution has a property that it depends only on the differences of $y_k$ and $\mu$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\{y_k\}|\mu,\sigma^2)=f(\{y_k-\mu\},\sigma^2)$$

+++ {"slideshow": {"slide_type": "skip"}}

It is reasonable to  assume same relation for the posterior

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|\{y_k\},\sigma^2)=g(\{y_k-\mu\},\sigma^2)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Inserting this into the formula for posterior we obtain the relation

+++ {"slideshow": {"slide_type": "fragment"}}

$$g(\{y_k-\mu\},\sigma^2) \propto f(\{y_k-\mu\},\sigma^2) P(\mu)$$

+++ {"slideshow": {"slide_type": "skip"}}

that can be satisfied only when the prior does not depend on $\mu$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu)\propto 1$$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Improper prior

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Please note that this is __not__ a probability distribution as it is not normalizable. We call it an _improper_ prior. Nevertheless we can still use it  as long as posterior is a proper distribution. Because of this prior the posterior is proportional to the likelihood

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(\mu|\b y,\sigma^2) \propto \prod_{k=1}^N \frac{1}{\sqrt{2\pi}\sigma} 
e^{-\frac{1}{2\sigma^2}\left(y_k-\mu\right)^2}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu|\b y,\sigma^2) \propto  \left(\sqrt{2\pi}\sigma\right)^{-N} 
e^{\displaystyle -\frac{1}{2\sigma^2}\sum_{k=1}^N\left(y_k-\mu\right)^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

We can expand the sum in the exponent

+++ {"slideshow": {"slide_type": "slide"}}

$$\sum_{k=1}^N\left(y_k-\mu\right)^2=\sum_{k=1}^N\left(y_k^2-2y_k\, \mu +\mu^2\right)=\sum_{k=1}^N y_k^2-2\sum_{k=1}^N y_k \,\mu +N\mu^2$$

+++ {"slideshow": {"slide_type": "skip"}}

Denoting the averages by bar

+++ {"slideshow": {"slide_type": "slide"}}

$$
\overline{y^2}\equiv\frac{1}{N}\sum_{k=1}^N y_k^2,\qquad 
\bar{y}\equiv\frac{1}{N}\sum_{k=1}^N y_k
$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$
\begin{split}
\sum_{k=1}^N\left(y_k-\mu\right)^2&= N \left(\bar{y^2}-2\bar{y}\mu+\mu^2\right)\\
&= N \left(\bar{y}^2-2\bar{y}\mu+\mu^2  +\bar{y^2}-\bar{y}^2\right)\\
&=N\left(\bar{y}-\mu\right)^2+N\left(\bar{y^2}-\bar{y}^2\right)
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}}

leading finally to

+++ {"slideshow": {"slide_type": "slide"}}

$$
P(\mu|\b y,\sigma^2) \propto  
\sigma^{-N} 
e^{\displaystyle -\frac{N}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{N}{2\sigma^2}\left(\overline{y^2} -{\bar y }^2\right)}
$$

+++ {"slideshow": {"slide_type": "skip"}}

The above relation means that random variable $\mu$ with fixed $\b y$ and $\sigma^2$ is distributed according to normal distribution with mean $\bar y$ and variance $\frac{\sigma^2}{N}$ which we will denote by the notation below

+++ {"slideshow": {"slide_type": "fragment"}}

$$\mu|\b y,\sigma \sim \operatorname{Norm}\left(\bar y,\frac{\sigma}{\sqrt{n}}\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

That also means the the MAP estimate of $\mu$ is  the average of $\b y$

+++ {"slideshow": {"slide_type": "skip"}}

$$\mu_{MAP}= \bar y$$

+++ {"slideshow": {"slide_type": "skip"}}

as expected.

+++ {"slideshow": {"slide_type": "slide"}}

## Nuisance parameters

+++ {"slideshow": {"slide_type": "skip"}}

More often that not, our model will have more then one parameter

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1,\theta_2|y) \propto P(y|\theta_1,\theta_2) P(\theta_1,\theta_2)$$

+++ {"slideshow": {"slide_type": "skip"}}

However we may be not interested in all of them. The uninteresting parameters are so called _nuisance_ parameters. In this example let's assume that we are only interested in parameter $\theta_1$ while $\theta_2$ is the nuisance parameter. We can obtain  the posterior for $\theta_1$ by integrating  over the nuisance parameter to obtain the _marginal_ distribution for $\theta_1$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1|y) = \int\text{d}{\theta_2}P(\theta_1,\theta_2|y) $$

+++ {"slideshow": {"slide_type": "skip"}}

This can be also rewritten using the  product rule

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\theta_1|y) = \int\text{d}\theta_2 P(\theta_1|y,\theta_2)P(\theta_2|y)$$

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model with unknown variance

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

As an example we will take the normal model with unknown mean and variance. We will treat the variance as nuisance parameter

+++ {"slideshow": {"slide_type": "skip"}}

### Prior on scale parameter

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

To find the prior on $\sigma$ we will  use similar reasoning as in case of $\mu$. The sampling distribution does depend on $\b y$, $\mu$ and $\sigma$ in a specific way

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(\b y|\mu,\sigma^2)=\prod_{k=1}^NP(y_k|\mu,\sigma^2)=\prod_{k=1}^N \frac{1}{\sigma} f\left(\frac{y_k-\mu}{\sigma}\right)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We say that $\sigma$ is a _scale_ parameter as it sets the scale for $\b y$ and $\mu$. The $\sigma^{-1}$ factor assures the proper normalization: the integral over $\b y$ does not depend on $\sigma$.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$
\int_{-\infty}^\infty\text{d}y
f\left(\frac{y-\mu}{\sigma}\right) = 
\sigma\int_{-\infty}^\infty\text{d}z
f\left(z\right)\qquad z= \frac{y-\mu}{\sigma}\
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

If we now set define a new variable

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$u_k  = \ln (y_k-\mu)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

 we will obtain that

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$P_{u_k}(u_k) = e^{u_k-\log \sigma}f(e^{u_k-\log\sigma}) $$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Using same resoning as in case of $\mu$ we obtain that the prior on $\log \sigma$ is uniform 

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$P(\log\sigma)\propto 1 $$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which translates into

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$P_{\sigma}(\sigma)\propto \frac{1}{\sigma}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

If we want to use the variance $\sigma^2$ instead of standard deviation $\sigma$ the the distribution is

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P_{\sigma^2}(\sigma^2)\propto \frac{1}{\sigma^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

This can be checked by calculating the cumulative distribution function for this distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$CDF_{P_{\sigma^2}}(z)=P(\sigma^2<z)=P(\sigma<\sqrt{z})=CDF_{P_\sigma}(\sqrt{z})$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

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

$$P_{\log\sigma}(\log \sigma)=e^z \text{CDF}^\prime_{P_\sigma}(e^z)\propto e^z \frac{1}{e^z} = 1$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Normal model --  posterior for variance $\sigma^2$

+++ {"slideshow": {"slide_type": "skip"}}

Using variance as our parameter with prior

+++

$$P(\sigma^2)\propto \frac{1}{\sigma^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

 we obtain for the posterior

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
P(\sigma^2|y) &\propto  \int\text{d}\mu\,(\sigma^2)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}\\
&\propto
(\sigma^2)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}\sqrt{2\pi\frac{\sigma^2}{n}}
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

leading to

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\sigma^2|y) \propto\left(\sigma^2\right)^{-\frac{n+1}{2}} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

This is a [_inverse Gamma_ distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) with propbability density function

+++

$$f(x;\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{-(\alpha+1)}e^{-\frac{\beta}{x}}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This leads finally to

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\sigma^2|y \sim \operatorname{Inv-}\Gamma\left(\alpha=\frac{n-1}{2},\beta=\frac{1}{2}(n-1)s^2\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

Mode of this distribution is

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{\beta}{\alpha+1}=\frac{n-1}{n+1}s^2$$

+++ {"slideshow": {"slide_type": "skip"}}

leading to slightly biased MAP estimator of $\sigma^2$. This is due to our choice of prior which favors small $\sigma^2$.

+++ {"slideshow": {"slide_type": "skip"}}

Let's give an example. We start by generating 20 random number from normal distribution

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
y_bar = y.mean()
s2 = y.var(ddof=1)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Parameter `ddof` (delta degrees of freedom) determines what divisor is to be used when calculating variance. In this case value one means that we divide by $n-1$ thus obtaining the unbiased estimate (see [documentation](https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy.var)).

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
plt.xlabel("$\sigma^2$")
plt.plot(vars,post_var.pdf(vars), label='posterior');
plt.axvline(var, color='orange',label='true value');
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can find the MAP estimate of $\sigma^2$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
vars[np.argmax(post_var.pdf(vars))]
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and check if it agrees with the formula we have derived

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
var_map = s2*(n-1)/(n+1)
var_map
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The join posterior can be rewritten as

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(\mu,\sigma^2|y) = P(\mu|\sigma^2,y)P(\sigma^2|y)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

with

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|\sigma^2,y) = \frac{1}{\sqrt{2\pi \frac{\sigma^2}{n}}}
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

which is the  Gaussian distribution $\operatorname{Norm}\left(\bar y,\sqrt{\frac{\sigma^2}{n}}\right)$

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model - posterior distribution for the mean $\mu$

+++ {"slideshow": {"slide_type": "skip"}}

To obtain the posterior for $\mu$ we have to integrate out the $\sigma^2$ parameter

+++ {"slideshow": {"slide_type": "slide"}}

$$
P(\mu|y)=\int\text{d}\sigma^2 P(\mu,\sigma^2|y) \propto \int_0^\infty\text{d}\sigma^2 \sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

We introduce a new variable

+++ {"slideshow": {"slide_type": "fragment"}}

$$z=\frac{A}{2\sigma^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

with

+++ {"slideshow": {"slide_type": "fragment"}}

$$A=n(\bar y-\mu)^2+(n-1)s^2$$

+++ {"slideshow": {"slide_type": "skip"}}

Differentiating $z$ with respect to $\sigma^2$ we obtain that

+++ {"slideshow": {"slide_type": "fragment"}}

$$\text{d}z=-\frac{A}{2\sigma^4}\text{d}\sigma^2$$

+++ {"slideshow": {"slide_type": "skip"}}

so finally

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu|y)\propto A^{-\frac{n}{2}}\int_0^\infty\text{d}z\,z^{\frac{n-1}{2}}e^{-z}$$

+++ {"slideshow": {"slide_type": "skip"}}

The integral over $z$ does not depend on $\mu$ so

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu, y)\propto A^{-\frac{n}{2}}=\left(n(\bar y-\mu)^2+(n-1)s^2\right)^{-\frac{n}{2}}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Pulling out the $\left((n-1)s^2\right)^{-\frac{n}{2}}$ term we obtain

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\left((n-1)s^2\right)^{-\frac{n}{2}}\left(\frac{n}{n-1}\frac{(\bar y-\mu)^2}{s^2}+1\right)^{-\frac{n}{2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

Putting $\nu=n-1$ we finally get

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

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Again we will use the `scipy.stats` module

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.stats import t
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The student's $t$ distribution depends only on the number of degrees of freedom $\nu$ but `scipy.stats` permits us to provide `location` and `scale` parameter that allow us to obtain the distribution directly for $\mu$ instead of $x$

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
plt.xlabel("$\mu$")
plt.plot(mus, post_mu.pdf(mus),label='posterior' );
plt.axvline(mu, color='orange', label='true value');
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model - Joint distribution for $\mu$ and $\sigma^2$

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
xs,ys = np.meshgrid(mus, vars)
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
  slide_type: skip
---
joined = np.exp(log_joined)
```

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
from scipy.special import logsumexp
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
var_dist = np.exp(logsumexp(log_joined, axis=1) )
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
mu_dist = np.exp(logsumexp(log_joined, axis=0)) 
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
i_joined_map = np.unravel_index(np.argmax(log_joined), log_joined.shape)
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

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model -  posterior for variance $\sigma^2$ - flat prior on $\sigma^2$

+++ {"slideshow": {"slide_type": "skip"}}

 Analogously to previously considered case the posterior in this case is

+++ {"slideshow": {"slide_type": "fragment"}}

$$
P(\mu,\sigma^2|y) \propto  (\sigma^2)^{-\frac{n}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

the only difference being the missing factor $\sigma^{-2}$.

+++ {"slideshow": {"slide_type": "skip"}}

Integrating  the joint posterior over $\mu$ we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\sigma^2|y) \propto\left(\sigma^2\right)^{-\frac{n-1}{2}} 
e^{\displaystyle -\frac{n-1}{2\sigma^2}s^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

This agian is a [_inverse Gamma_ distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) with propbability density function

+++

$$f(x;\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{-(\alpha+1)}e^{-\frac{\beta}{x}}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

but now we obtain slightly different values for parameters $\alpha$ and $\beta$

+++ {"slideshow": {"slide_type": "slide"}}

$$\sigma^2|y \sim \operatorname{Inv-}\Gamma\left(\alpha=\frac{n-3}{2},\beta=\frac{1}{2}(n-1)s^2\right)$$

+++ {"slideshow": {"slide_type": "skip"}}

Mode of this distribution is

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sigma^2_{MAP}=\frac{\beta}{\alpha+1}=s^2$$

+++ {"slideshow": {"slide_type": "skip"}}

giving an  unbiased  estimator of $\sigma^2$.

+++ {"slideshow": {"slide_type": "skip"}}

Let's give an example. We start by generating 3 random number from normal distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import invgamma
alpha = (n-3)/2
beta  = ((n-1)*s2)/2
post_var = invgamma(a=alpha, scale=beta)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
vars = np.linspace(1e-6,1.5,500)
plt.xlabel("$\sigma^2$")
plt.plot(vars,post_var.pdf(vars), label='posterior');
plt.axvline(var, color='orange',label='true value');
plt.legend();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
vars[np.argmax(post_var.pdf(vars))]
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and check if it agrees with the formula we have derived

```{code-cell} ipython3
var_map = s2
var_map
```

+++ {"slideshow": {"slide_type": "skip"}}

The join posterior can be rewritten as

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu,\sigma^2|y) = P(\mu|\sigma^2,y)P(\sigma^2|y)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

with

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\mu|\sigma^2,y) = \frac{1}{\sqrt{2\pi \frac{\sigma^2}{n}}}
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

which is the  Gaussian distribution $\operatorname{Norm}\left(\bar y,\sqrt{\frac{\sigma^2}{n}}\right)$

+++ {"slideshow": {"slide_type": "slide"}}

### Normal model - posterior distribution for the mean $\mu$

+++ {"slideshow": {"slide_type": "skip"}}

To obtain the posterior for $\mu$ we have to integrate out the $\sigma^2$ parameter

+++ {"slideshow": {"slide_type": "slide"}}

$$
P(\mu|y)=\int\text{d}\sigma^2 P(\mu,\sigma^2|y) \propto \int_0^\infty\text{d}\sigma^2 \sigma^{-n-2} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2}
$$

+++ {"slideshow": {"slide_type": "skip"}}

We introduce a new variable

+++ {"slideshow": {"slide_type": "fragment"}}

$$z=\frac{A}{2\sigma^2}$$

+++ {"slideshow": {"slide_type": "skip"}}

with

+++ {"slideshow": {"slide_type": "fragment"}}

$$A=n(\bar y-\mu)^2+(n-1)s^2$$

+++ {"slideshow": {"slide_type": "skip"}}

Differentiating $z$ with respect to $\sigma^2$ we obtain that

+++ {"slideshow": {"slide_type": "fragment"}}

$$\text{d}z=-\frac{A}{2\sigma^4}\text{d}\sigma^2$$

+++ {"slideshow": {"slide_type": "skip"}}

so finally

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu|y)\propto A^{-\frac{n}{2}}\int_0^\infty\text{d}z\,z^{\frac{n-1}{2}}e^{-z}$$

+++ {"slideshow": {"slide_type": "skip"}}

The integral over $z$ does not depend on $\mu$ so

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\mu, y)\propto A^{-\frac{n}{2}}=\left(n(\bar y-\mu)^2+(n-1)s^2\right)^{-\frac{n}{2}}$$

+++

Pulling out the $\left((n-1)s^2\right)^{-\frac{n}{2}}$ term we obtain

+++ {"slideshow": {"slide_type": "slide"}}

$$\left((n-1)s^2\right)^{-\frac{n}{2}}\left(\frac{n}{n-1}\frac{(\bar y-\mu)^2}{s^2}+1\right)^{-\frac{n}{2}}$$

+++ {"slideshow": {"slide_type": "skip"}}

Putting $\nu=n-1$ we finally get

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

Again we will use the `scipy.stats` module

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import t
```

The student's $t$ distribution depends only on the number of degrees of freedom $\nu$ but `scipy.stats` permits us to provide `location` and `scale` parameter that allow us to obtain the distribution directly for $\mu$ instead of $x$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
post_mu = t(df=n-1, loc=y.mean(), scale=np.sqrt(s2/n))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
mus = np.linspace(0.5,2,500)
plt.xlabel("$\mu$")
plt.plot(mus, post_mu.pdf(mus),label='posterior' );
plt.axvline(mu, color='orange', label='true value');
plt.legend();
```

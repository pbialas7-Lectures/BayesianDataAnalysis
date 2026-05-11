---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
%load_ext autoreload
%autoreload 2
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Hierarchical models

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import pymc as pm
import arviz_plots as azp
import arviz_stats as azs
import arviz_base as azb
from scipy.stats import poisson, beta, binom
import scipy
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
figsize = (8, 6)
plt.rcParams["figure.figsize"] = figsize
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's suppose that we have a lousy blacksmith that is making medieval coins. Because he is lousy each coin is different from another, so the probability of heads $p$ is different for each coin. We would like to estimate the parameters of the distribution of $p$. 
To this end we distribute seven coins to different persons and as them to flip the coin approximately $100$ times.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's assume that the true distribution is a Beta distribution with mean equal to $0.51$ and standard deviation equal to $0.05$.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

<img src="../figures/true_coin.png" style="display: block; margin:0 auto; width:70%;"/>

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
n_coins = 7
mu_true = 0.51
sigma_true =0.05
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Recalling that for Beta distribution

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\alpha = \mu \nu,\quad \beta = (1-\mu)\nu,\quad \sigma^2=\frac{\mu(1-\mu)}{\nu +1},\qquad \nu=\alpha+\beta $$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

we obtain

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\nu=\left(\frac{\mu(1-\mu)}{\sigma^2}-1\right),\qquad \alpha= \mu\nu, \qquad\beta = (1-\mu)\nu$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
nu_true = mu_true*(1-mu_true)/(sigma_true**2)-1
alph_true = mu_true * nu_true
beta_true = (1-mu_true)*nu_true
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's check our calculations

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
beta(alph_true, beta_true).mean()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
beta(alph_true, beta_true).std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Now we will simulate the values of the $p_j$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
np.random.seed(432431)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
p = beta(alph_true, beta_true).rvs(size=n_coins) 
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
p
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The simplest estimates of the $p$ distributions would be

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
p.mean()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
p.std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

However we do not know the values of $p$ only the results of the coin tosses, so let's simulate that. First we simulate the number of tosses for each person assuming that it is binomial distribution with mean 100 and stadard deviation $5$.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
mu_t = 100
sigma_t = 5
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For Binomial distribution

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\mu_t = p_t n_t,\qquad \sigma_t^2=p_t(1-p_t) n_t$$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_t = 1-\frac{\sigma_t^2}{\mu_t},\qquad n_t = \frac{\mu_t^2}{\mu_t-\sigma_t^2}$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
p_t = 1-sigma_t**2/mu_t
p_t
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The $n_t$ nas to be an integer so we round it

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
n_t = int(np.round(mu_t**2/(mu_t-sigma_t**2)))
n_t
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
binom(p=p_t, n=n_t).mean()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
binom(p=p_t, n=n_t).std()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
n_toss = binom(p=p_t, n=n_t).rvs(size=n_coins)
n_toss
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

And finally let's throw the coins

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
heads = binom.rvs(p=p, n=n_toss)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plt.bar(np.arange(1,n_coins+1)-0.2,heads/n_toss, width=0.3, label='obs');
plt.bar(np.arange(1,n_coins+1)+0.2,p, width=0.3,label='p');
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Hierarchical model

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

<img src="../figures/hierarchical_coin.png" style="display: block; margin:0 auto; width:70%;"/>

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The values of $p$ for each coin are drawn from the Beta distribution adnwe would like to estimate the parameters of this distribution, which means that we need a prior on those parameters. This is sometimes called a _hyper-prior_. 
Instead of putting prior on the parameters $\alpha$ and $\beta$ directly we will parametrize them

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$\alpha = \mu \nu,\quad \beta = (1-\mu)\nu $$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and use following priors on $\mu$ and $\nu$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\mu \sim Beta(\alpha=2,\beta=2),\qquad \nu\sim \exp\left(-\frac{\nu}{20}\right) $$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots(1,2,figsize=(12,5))
mus = np.linspace(0,1,500); nus = np.linspace(0,100,500);
ax[0].grid();ax[0].plot(mus,beta(2,2).pdf(mus));ax[0].set_xlabel('$\\mu$')
ax[1].grid();ax[1].plot(nus, scipy.stats.expon(scale=20.0).pdf(nus)); ax[1].set_xlabel('$\\nu$');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
h_model = pm.Model()

with h_model:
  mu = pm.Beta('mu', alpha=2, beta=2)
  nu = pm.Exponential('nu',lam=0.05)
  std = pm.Deterministic('std',mu*(1-mu)/(nu+1))
  p = pm.Beta('p', mu=mu, sigma=std, size=n_coins)

  obs = pm.Binomial('obs', n=n_toss, p=p, observed=heads)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
with h_model:
  h_trace = pm.sample(tune=2000, draws=10000, target_accept=0.999, chains=4);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_trace_dist(h_trace, var_names=['mu','nu']);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_dist(h_trace, var_names=['mu','std']);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
with h_model:
  pm.sample_posterior_predictive(h_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
print(h_trace.posterior_predictive)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(
  (azb.extract(h_trace.posterior_predictive).transpose()/n_toss).min('obs_dim_0'), bins=50);
plt.axvline(np.min(heads/n_toss), c='red');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(
  (azb.extract(h_trace.posterior_predictive).transpose()/n_toss).max('obs_dim_0'), bins=50);
plt.axvline(np.max(heads/n_toss), c='red');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(
  (azb.extract(h_trace.posterior_predictive).transpose()/n_toss).std('obs_dim_0'), bins=50);
plt.axvline(np.std(heads/n_toss), c='red');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
if 'log_likelihood' in h_trace:
    delattr(h_trace, 'log_likelihood')
with h_model:
  pm.compute_log_likelihood(h_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
azs.loo(h_trace)
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Separate coins

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Now we will treat each person separately and assume that each $p$ was draw from a prior distribution $Beta(\alpha=2, \beta=2)$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

<img src="../figures/separate_coin.png" style="display: block; margin:0 auto; width:70%;"/>

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
s_alpha=2
s_beta=2
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
s_model = pm.Model() 

with s_model:
    p = pm.Beta(f"p", alpha=s_alpha, beta = s_beta, size=n_coins)
    obs = pm.Binomial(f"obs", p=p, n=n_toss, observed = heads) 
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
with s_model:
  s_trace = pm.sample(tune=2000, draws=32000)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_trace_dist(s_trace);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
with s_model:
  pm.sample_posterior_predictive(s_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(
  (azb.extract(s_trace.posterior_predictive).transpose()/n_toss).min('obs_dim_0'), bins=50);
plt.axvline(np.min(heads/n_toss), c='red');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(
  (azb.extract(s_trace.posterior_predictive).transpose()/n_toss).max('obs_dim_0'), bins=50);
plt.axvline(np.max(heads/n_toss), c='red');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(
  (azb.extract(s_trace.posterior_predictive).transpose()/n_toss).std('obs_dim_0'), bins=50);
plt.axvline(np.std(heads/n_toss), c='red');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
if 'log_likelihood' in s_trace:
    delattr(s_trace, 'log_likelihood')
with s_model:
  pm.compute_log_likelihood(s_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azs.loo(s_trace)
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## All-in  1

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

<img src="../figures/all_coin1.png" style="display: block; margin:0 auto; width:70%;"/>

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
a_alpha=2
a_beta=2
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
a_model = pm.Model() 

with a_model:
    p = pm.Beta(f"p", alpha=s_alpha, beta = s_beta)
    obs = pm.Binomial(f"obs", p=p, n=n_toss.sum(), observed = heads.sum()) 
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
with a_model:
  a_trace = pm.sample(tune=2000, draws=16000)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_trace_dist(a_trace);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_dist(a_trace);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
if 'log_likelihood' in a_trace:
    delattr(a_trace, 'log_likelihood')
with a_model:
  pm.compute_log_likelihood(a_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azs.loo(a_trace)
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## All-in  2

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

<img src="../figures/all_coin2.png" style="display: block; margin:0 auto; width:70%;"/>

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
a_alpha=2
a_beta=2
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
a_model = pm.Model() 

with a_model:
    p = pm.Beta(f"p", alpha=s_alpha, beta = s_beta)
    obs = pm.Binomial(f"obs", p=p, n=n_toss, observed = heads )
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
with a_model:
  a_trace = pm.sample(tune=2000, draws=16000)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_trace_dist(a_trace);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_dist(a_trace);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
if 'log_likelihood' in a_trace:
    delattr(a_trace, 'log_likelihood')
with a_model:
  pm.compute_log_likelihood(a_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
azs.loo(a_trace)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
comp = azs.compare({'hierarchical': h_trace, 'separate': s_trace, 'all-in 2': a_trace}, round_to=1)
comp
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
azp.plot_compare(comp);
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

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
%load_ext autoreload
%autoreload 2
```

# Monte-Carlo methods

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model and PyMC3

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
from scipy.stats import norm
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
np.__version__
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [8,6]
```

+++ {"slideshow": {"slide_type": "slide"}}

## Data

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
n = 100
mu_true = 1.0
sigma_true = 2.0

np.random.seed(1212331)
y = norm(mu_true, sigma_true).rvs(12)
y_bar = y.mean()
s2 = y.var(ddof=1)
print(y_bar, s2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.scatter(np.arange(len(y)), y);
ax.set_ylabel('$y_i$',rotation='horizontal', fontsize=16);ax.set_xlabel('$i$', fontsize=16)
ax.axhline(mu_true);ax.axhline(mu_true-sigma_true, linestyle='--');ax.axhline(mu_true+sigma_true, linestyle='--');
```

+++ {"slideshow": {"slide_type": "slide"}}

## PyMC3 model

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import pymc3 as pm
print(f"Running on PyMC3 v{pm.__version__}")
import arviz as az
import xarray as xray
```

+++ {"slideshow": {"slide_type": "slide"}}

## Known variance

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
normal_model1 = pm.Model()

with normal_model1:
    mu = pm.Flat('mu')   # Prior
    y_obs=pm.Normal('y_obs', mu=mu, sigma=sigma_true, observed=y) #likelihood
```

+++ {"slideshow": {"slide_type": "slide"}}

### Maximal a Posteriori

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
MAP1 =  pm.find_MAP(model=normal_model1)
print(MAP1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_bar
```

+++ {"slideshow": {"slide_type": "slide"}}

### Sampling from posterior distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
trace1 = pm.sample(model=normal_model1,tune=1000, draws=32000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model1:
    az.plot_trace(trace1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model1:
    ax=az.plot_posterior(trace1, hdi_prob=0.9)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.stats import norm
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
posterior1 = norm(loc=y_bar, scale=sigma_true/np.sqrt(len(y)) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
xs=np.linspace(-1, 4,250)
ax.plot(xs, posterior1.pdf(xs), color='orange')
ax.figure
```

+++ {"slideshow": {"slide_type": "slide"}}

## Inference Data

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
trace1
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
trace1.posterior
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
mu = trace1.posterior.mu.stack(z=('chain','draw'))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
mu
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model1:
    summary1 = az.summary(trace1, hdi_prob=.99)
summary1
```

+++ {"slideshow": {"slide_type": "slide"}}

## Unknown variance and flat prior on sigma

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
normal_model2 = pm.Model()

with normal_model2:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    y_obs=pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
pm.find_MAP(model=normal_model2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_bar
```

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sigma^2_{MAP}=\frac{n-1}{n}s^2\qquad s^2=\frac{n}{n-1}\left(\bar{y^2}-\bar{y}^2\right)\qquad \sigma^2_{MAP}=\left(\bar{y^2}-\bar{y}^2\right)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
np.std(y,ddof=0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model2:
    trace2=pm.sample(model=normal_model2,tune=1000, draws=32000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model2:
    trace2=pm.sample(model=normal_model2,tune=1000, draws=32000, return_inferencedata=True, target_accept=0.95)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model2:
    az.plot_trace(trace2)
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
  slide_type: slide
---
with normal_model2:
    ax = az.plot_posterior(trace2, var_names=['mu', 'sigma'])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model2:
    print(az.summary(trace2))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Unknow variance and $\sigma^{-2}$ prior on $\sigma^{2}$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_model4 = pm.Model()

with normal_model4:
    mu = pm.Flat('mu')
    var = pm.HalfFlat('var')
    var_prior = pm.Potential('var_prior', -np.log(var))
    
    y_obs=pm.Normal('y_obs', mu=mu, sigma=np.sqrt(var), observed=y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
map4 = pm.find_MAP(model=normal_model4)
print(map4)
print(np.sqrt(map4['var']))
```

+++ {"slideshow": {"slide_type": "skip"}}

$$\sigma_{MAP}=\frac{n-1}{n+1}s^2\qquad s^2=\frac{n}{n-1}\left(\bar{y^2}-\bar{y}^2\right)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print((n)/(n+1)*s2 )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model4:
    trace4=pm.sample(model=normal_model4,tune=1000, draws=32000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model4:
    az.plot_trace(trace4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model4:
    ax4=az.plot_posterior(trace4, var_names='mu')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
from scipy.stats import t
posterior4 = t(df=len(y)-1, loc = y_bar, scale = np.sqrt(s2/len(y)) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs = np.linspace(-4,6,250)
ax4.plot(xs, posterior4.pdf(xs), color='orange');
ax4.figure
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model4:
    print(az.summary(trace4))
```

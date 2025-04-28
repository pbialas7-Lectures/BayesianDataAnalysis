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

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Normal model and PyMC

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
%load_ext autoreload
%autoreload 2
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
from scipy.stats import norm
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [6, 4]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import sys
from bda.autocorr import ac_and_tau_int
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The Monte-Carlo lecture introduced the basic concepts of how to generate samples from the posterior distribution. In practice however writing the code "by-hand" for each problem would be unwieldy. Besides for more complicated distributions writing  the code that can sample it in an efficient way may be tough. That's why we use probabilistic programming languages (PPL) like PyMC. PyMC is a Python library that allows you to write down the model and then sample from the posterior distribution. In this notebook I will show how to use PyMC to sample from the posterior distribution of the same  normal model as in the Monte-Carlo lecture. I will lasom introduce `ArviZ` a library for analysis of the bayesian models.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Both packages can be installed using conda/mamba

```bash
mamab install -c conda-forge pymc arviz
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Data

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Again we will start by generating some artificial data.

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
n = 20
mu_true = 1.0
sigma_true = 2.0

np.random.seed(1212331)
y = norm(mu_true, sigma_true).rvs(20)
y_bar = y.mean()
s2 = y.var(ddof=1)
print(y_bar, s2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.scatter(np.arange(len(y)), y);
ax.set_ylabel('$y_i$', rotation='horizontal', fontsize=16);
ax.set_xlabel('$i$', fontsize=16)
ax.axhline(mu_true);
ax.axhline(mu_true - sigma_true, linestyle='--');
ax.axhline(mu_true + sigma_true, linestyle='--');
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## PyMC model

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
import pymc as pm
import arviz as az

print(f"Running on PyMC v{pm.__version__} and ArviZ v{az.__version__}")
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Known variance

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

After importing the PyMC package we are ready to define our first model. We start with know variance case and uninformative improper prior on $\sigma^2$. Whe defining the model we start with defining the prior on $\sigma^2$ using one of many [distributions](https://www.pymc.io/projects/docs/en/stable/api/distributions.html) available in PyMC. In this case we use `pm.Flat` which is a flat prior. Next we define the likelihood using `pm.Normal` which is a normal distribution. The `observed` argument is used to pass the data. Notice how the `mu` prior variable is passed to the `pm.Normal` as the mean of the distribution.

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
normal_model1 = pm.Model()

with normal_model1:
    mu = pm.Flat('mu')  # Prior
    pm.Normal('y_obs', mu=mu, sigma=sigma_true, observed=y)  #likelihood
```

+++ {"slideshow": {"slide_type": "slide"}}

### Maximal a Posteriori

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Once we have the model defined we can find the Maximum a Posteriori (MAP) estimate. This is the value of the parameters that maximizes the posterior distribution. In this case we are looking for the value of $\mu$ that maximizes the posterior distribution.

```{code-cell}
---
slideshow:
  slide_type: fragment
---
MAP1 = pm.find_MAP(model=normal_model1)
print(MAP1)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In this case the MAP  estimate is know analytically and is just the mean of the data

```{code-cell}
---
slideshow:
  slide_type: fragment
---
y_bar
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Sampling from posterior distribution

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The model also allows us to sample from the posterior distribution. The `tune` parameter is the number of samples used for tuning the sampler. Those samples will be discarded. The  `draws` parameter is the number of samples to draw. We are not specifying the  algorithm to be used for sampling, so the default [No-U-Turn Sampler](https://arxiv.org/abs/1111.4246) (NUTS) is used. By default the samplers in PyMC run in parallel using available cores. The `cores` parameter can be used to specify the number of cores to use.

```{code-cell}
n_samples = 8000
n_tune = 1000
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
with normal_model1:
    trace1 = pm.sample(tune=n_tune, draws=n_samples, cores=4)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The first to notice is that this sampling is much slower than the one we did in the Monte-Carlo lecture even with acounting that we are sampling four (in my case) chains in parallel. This is the price to pay for the flexibility of the PyMC

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Once the samples are generated a quick way to view the results is the `plot_trace` function from `ArviZ`. For each parameter (in this case just $\mu$) it shows the resulting marginal distribution and the trace of the samples for each chain.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model1:
    az.plot_trace(trace1)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In this case everything looks good.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### ArviZ Inference data

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The `trace1` object returned from the ` pm.sample` function is a `InferenceData` object from the [`ArviZ` library](https://python.arviz.org/en/latest/index.html#). This can be viewed as a collection of the generated samples and some additional data and metadata.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In this case it contains three  groups: `posterior`, `samples_stats` and `observed_data`.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Each of those groups is in turn a `DataSet` object from the [`xarray` library](https://docs.xarray.dev/en/stable/index.html). Let's look at the `posterior` group.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1.posterior
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This is an `DataSet` object from `xarray` which in turn is a collection of [`DataArray` objects](https://docs.xarray.dev/en/stable/api.html#dataarray) from the same library. A `DataArray` is essentially a wrapper around the numpy array with some additional metadata. In this case we have only one variable `mu` which is the posterior samples of the $\mu$ parameter.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1.posterior.mu
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This DataArray consists of an two-dimensional numpy array which holds the samples from posterior for each chain  separately. We can access this data using the `values` attribute

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
trace1.posterior.mu.values.shape
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Besides the data a DataArray also holds the information about the dimensions.  The attribute `dims` holds the names of the dimensions. In this case we have two dimensions `chain` and `draw`.

```{code-cell}
trace1.posterior.mu.dims
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

 You can think of the dimensions as the axis labels. The "chain" dimension corresponds to axis 0, and the "draw" dimension corresponds to axis 1.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

You can access the data either using the normal numpy indexing

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1.posterior.mu[0,:100]
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

 or using the `sel` method. The `sel` method allows you to select the data using the dimension names.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1.posterior.mu.sel(chain=0, draw=slice(0, 100))
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The `coords` attribute holds the values of the dimensions.  You can think of them as  tick marks on the axes.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1.posterior.mu.coords
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Often we would like to combine the chains into a single dimension. We can do it directly on the values

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace1.posterior.mu.values.ravel()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

or using the `stack` method.

```{code-cell}
mu_samples = trace1.posterior.mu.stack(z=('chain', 'draw'))
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

 The `z` argument is the name of the new dimension that combines `chain` and `draw` dimensions.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
mu_samples
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can still use the `sel` method to access the data `chain` and `draw` dimensions.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
mu_samples.sel(chain=2, draw=slice(0, 100))
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Autocorrelation

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

After this short introduction to the ArviZ data type we return to our samples. As explained previously  the qualioty of thise samples depends on the autocorrelation time. We can use the `ac_and_tau_int` function from the `bda` module to calculate the autocorrelation and the integrated autocorrelation time.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
for i in range(4):
    tau1, ac1 = ac_and_tau_int(trace1.posterior.mu[i].values)
    plt.plot(ac1, '.');
    print(f"{tau1:.4f}")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

ArviZ does not have a built-in function to calculate the autocorrelation and the integrated autocorrelation time. However it has a function to calculate the effective sample size which is the number of independent samples that would have the same variance as the samples generated by the MCMC.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.ess ( mu_samples.sel(chain=0).values)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This compares well with the integrated autocorrelation time

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
len(mu_samples.sel(chain=0))/(2*1.1908)
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Posterior distribution

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

An usefull function is the `plot_posterior` function from the `ArviZ` library. It plots the posterior distribution of the parameter together with the HDI and some point estimate which is mean by the default.  The `hdi_prob` argument specifies the probability of the highest density interval. The default value is 0.94.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model1:
    ax = az.plot_posterior(trace1, hdi_prob=0.95)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

It this case we know the true posterior distribution which is a normal distribution with mean `y_bar` and $n$ times smaller variance, and we can compare it with the one generated by the PyMC.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.stats import norm
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
posterior1 = norm(loc=y_bar, scale=sigma_true / np.sqrt(len(y)))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: '-'
---
xs = np.linspace(-1, 4, 250)
ax.plot(xs, posterior1.pdf(xs), color='orange')
ax.figure
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can require  that this functions also plots the mode of the distribution instead of the mean.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model1:
    ax = az.plot_posterior(trace1, hdi_prob=0.95, point_estimate='mode')
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

However we have little control on how this mode will be calculated. I have provided a function `mode_rvs` that calculates the mode of the distribution using the samples.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
from bda.stats import mode_rvs
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
mode_rvs(trace1.posterior.mu)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
y_bar
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

And finally the `summary` function provides a summary of the posterior distribution.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model1:
    summary1 = az.summary(trace1, hdi_prob=.99)
summary1
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Unknown variance and flat prior on sigma

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Next example would be same model but with unknown variance. We will use a flat prior on $\sigma$. The only difference in the model definition is the prior on the sigma. We use `pm.HalfFlat` which is a flat prior on the positive half-line. After defining this variable we pass it later to the likelihood  functions.

```{code-cell}
---
slideshow:
  slide_type: fragment
---
normal_model2 = pm.Model()

with normal_model2:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
pm.find_MAP(model=normal_model2)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
y_bar
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

$$\sigma^2_{MAP}=\frac{n-1}{n}s^2\qquad s^2=\frac{n}{n-1}\left(\bar{y^2}-\bar{y}^2\right)\qquad \sigma^2_{MAP}=\left(\bar{y^2}-\bar{y}^2\right)$$

```{code-cell}
---
slideshow:
  slide_type: fragment
---
np.std(y, ddof=0)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model2:
    trace2 = pm.sample(model=normal_model2, tune=n_tune, draws=n_samples)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
with normal_model2:
    az.plot_trace(trace2)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
with normal_model2:
    ax = az.plot_posterior(trace2, var_names=['mu', 'sigma'])
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
with normal_model2:
    print(az.summary(trace2))
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Unknow variance and $\sigma^{-2}$ prior on $\sigma^{2}$

```{code-cell}
---
slideshow:
  slide_type: slide
---
normal_model3 = pm.Model()

with normal_model3:
    mu = pm.Flat('mu')
    var = pm.HalfFlat('var')
    var_prior = pm.Potential('var_prior', -pm.math.log(var))

    y_obs = pm.Normal('y_obs', mu=mu, sigma=np.sqrt(var), observed=y)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
map3 = pm.find_MAP(model=normal_model3)
print(map3)
print(np.sqrt(map3['var']))
```

+++ {"slideshow": {"slide_type": "skip"}}

$$\sigma_{MAP}^2=\frac{n-1}{n+1}s^2\qquad s^2=\frac{n}{n-1}\left(\bar{y^2}-\bar{y}^2\right)$$

```{code-cell}
---
slideshow:
  slide_type: skip
---
print((n-1) / (n + 1) * s2)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
with normal_model3:
    trace3 = pm.sample(tune=n_tune, draws=n_samples)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
with normal_model3:
    az.plot_trace(trace3)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model3:
    print(az.summary(trace3))
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
with normal_model3:
    ax3 = az.plot_posterior(trace3, var_names='mu')
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
from scipy.stats import t
posterior3 = t(df=len(y) - 1, loc=y_bar, scale=np.sqrt(s2 / len(y)))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs = np.linspace(-4, 6, 250)
ax3.plot(xs, posterior3.pdf(xs), color='orange');
ax3.figure
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model3:
    az.plot_posterior(trace3, var_names='var', point_estimate='mean')
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model3:
    az.plot_posterior(trace3, var_names='var', point_estimate='mode')
```

```{code-cell}
mode_rvs(trace3.posterior['var'], grid_len  = 1000)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

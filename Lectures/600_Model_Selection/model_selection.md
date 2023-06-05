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

+++ {"slideshow": {"slide_type": "slide"}}

# Model testing and selection

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import numpy  as np
import scipy as sp
import scipy.stats as st
from scipy.special import gamma

import pymc as pm
import arviz as az
print(f"Running on PyMC v{pm.__version__} and ArviZ v{az.__version__}") 
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,6)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Students distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs = np.linspace(-5,5,1000)
fig, ax = plt.subplots()
for nu in (1,2,5,10,25,50):
    ax.plot(xs, st.t.pdf(xs, df=nu), label=f"$\\nu={nu}$");
ax.plot(xs, st.norm.pdf(xs), color = 'black', label="$\mathcal{{N}}(0,1)$")    
ax.legend();    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
mu = 1 
xs = np.linspace(-5,5,1000)
fig, ax = plt.subplots()
for s in (0.2,0.5, 1,2,5):
    ax.plot(xs, st.t.pdf(xs, df=1, loc=mu, scale=s), label=f"$s={s}$");
ax.legend();    
```

+++ {"slideshow": {"slide_type": "slide"}}

### True model

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
true_nu = 4
true_mu = 1
true_scale = 2
true_dist = st.t(loc=1, scale=2, df=4)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
true_mean, true_std = true_dist.mean(), true_dist.std()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(true_mean, true_std)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sample = true_dist.rvs(size=10000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample, bins = 100, density=True, histtype='step');
ts = np.linspace(-15,15,200)
plt.plot(ts, true_dist.pdf(ts), color='red', label='true')
plt.plot(ts, st.norm(true_mu, scale=true_scale).pdf(ts), color='gray', label=f"$\mathcal{{N}}(1,{st.norm(true_mu, scale=true_scale).std()})$")
plt.plot(ts, st.norm(true_mu, true_std).pdf(ts), color='gray', label=f"$\mathcal{{N}}(1,{true_std:.2f})$")
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide"}}

### Data

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_size = 129
y = true_dist.rvs(size=y_size, random_state=14543)
y_mean = y.mean()
y_s = y.std(ddof=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(y,'.');
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(y, bins=20);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
print("nu = {} mu = {} scale = {}".format(true_nu, true_mu, true_scale))
print("nu = {:.2f} mu = {:.2f} scale = {:.2f}".format(*st.t.fit(y)))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Normal model

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with pm.Model() as normal_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    MAP = pm.find_MAP()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
MAP
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    normal_trace = pm.sample(draws=4000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    az.plot_trace(normal_trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    az.plot_posterior(normal_trace, hdi_prob=0.95);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Posterior Predictive Check

+++ {"slideshow": {"slide_type": "skip"}}

The idea of posterior predictive check (PPC) is to generate a new synthetic data set(s) and compare it with original data.

+++ {"slideshow": {"slide_type": "slide"}}

### Posterior predictive distribution

+++ {"slideshow": {"slide_type": "skip"}}

The distribution which we will use to generate the new data is called _posterior predictive distribution_. Formally it's pdf function is defined by

+++ {"slideshow": {"slide_type": "fragment"}}

$$\newcommand{\b}[1]{\mathbf{#1}}$$
$$p(\b{\hat y}| \b y) = \int p(\b{\hat y}|\b \theta) p_{post}(\b\theta|\b y)\text{d}\b\theta$$

+++ {"slideshow": {"slide_type": "skip"}}

$\b y$ in this formula represents the data that was observed and used to fit the model and $\b {\hat y}$ denotes the  generated data. In the following I will sometimes omit the $\b y$. Sampling from this distribution is easy

+++ {"slideshow": {"slide_type": "fragment"}}

* First sample the parameters $\b\theta$ according to posterior $p_{post}(\theta|\b y)$
* The sample the $\b{\hat y}$ according to sampling distribution  $p(\b{\hat y}|\theta)$

+++ {"slideshow": {"slide_type": "skip"}}

In our case sampling distribution is the Gaussian distribution

+++ {"slideshow": {"slide_type": "slide"}}

$$p(\hat y) = \int p_{\mathcal{N}}(\hat y|\mu,\sigma) p_{post}(\mu,\sigma)\text{d}\mu\text{d}\sigma$$

+++ {"slideshow": {"slide_type": "skip"}}

and sampling from it proceeds as follows

+++ {"slideshow": {"slide_type": "fragment"}}

* Sample $\mu$ and $\sigma$ from posterior
* Sample $\hat y$ from $\mathcal{N}(\mu,\sigma)$

+++ {"slideshow": {"slide_type": "skip"}}

The samples of $\mu$ and $\sigma$ from the posterior are already calculated and are stored in the `normal_trace` variable

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_trace.posterior['mu']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_trace.posterior['sigma']
```

+++ {"slideshow": {"slide_type": "skip"}}

Each of the the parameters was sampled 4 x 4000 times. We will combine both parameters into single array using `numpy.stack` function

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
posterior_sample = np.stack( (normal_trace.posterior['mu'],normal_trace.posterior['sigma']),-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
posterior_sample.shape
```

+++ {"slideshow": {"slide_type": "skip"}}

and sample `y_size` number for for each pair using the [`np.apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html) function

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
posterior_predictive_sample = np.apply_along_axis(lambda a: st.norm.rvs(*a, size=y_size),2,posterior_sample)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
posterior_predictive_sample.shape
```

+++ {"slideshow": {"slide_type": "skip"}}

Finally we will combine all four chains into single one

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
posterior_predictive_sample = posterior_predictive_sample.reshape(-1,y_size)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
posterior_predictive_sample.shape
```

+++ {"slideshow": {"slide_type": "skip"}}

We have 16000 synthetic data sets to compare with our original `y` data set. We will start by drawing few examples together with the original data

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig,ax = plt.subplots(1,3, figsize=(20,6))
for i in range(3):
    ax[i].plot(y,'.',label="$y$")
    ax[i].plot(posterior_predictive_sample[i],'.',label=f"$\\hat y_{i}$")
    ax[i].legend(loc='upper left');
```

+++ {"slideshow": {"slide_type": "skip"}}

The look reasonably similar by one may notice the absence of  such pronounced outliers in case of the synthetic data sets. We can look at this using histograms but first generate the posterior predictive sample again this time using a PyMC function `sample_posterior_predictive`.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_trace
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with normal_model:
    normal_trace = pm.sample_posterior_predictive(trace=normal_trace, return_inferencedata=True, extend_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_trace
```

+++ {"slideshow": {"slide_type": "skip"}}

```python 
with pm.Model() as normal_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.Normal('y_obs', mu = mu, sd = sigma, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_pps = normal_trace.posterior_predictive['y_obs'].stack({'sample':['chain','draw']}).transpose()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
normal_pps
```

We can check that we obtain identical results as before.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(normal_pps.data.ravel(), bins=100, histtype='step',density=True, range=(-15,15));
plt.hist(posterior_predictive_sample.ravel(), bins=100, histtype='step', density=True, range=(-15,15));
```

+++ {"slideshow": {"slide_type": "skip"}}

Looking at the three first synthetic data sets we observe again that while they resemble the histogram of the original data, the outlying values are missing

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig,ax = plt.subplots(1,3, figsize=(20,6))
for i in range(3):
    ax[i].hist(y,label="$y$", bins=20, density=True, histtype='step')
    ax[i].hist(normal_pps[i],label=f"$\\hat y_{i}$", bins=20, density=True, histtype='step')
    ax[i].legend(loc='upper left');
```

+++ {"slideshow": {"slide_type": "skip"}}

To see this in more detailed we will plot the histogram of the minimum values of each synthetic data set and compare it with the minimum of the original data

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(np.min(normal_pps,axis=1) , histtype='step', density=True, bins=50);
plt.axvline(y.min(), color='black');
```

+++ {"slideshow": {"slide_type": "skip"}}

From this picture we see that it is almost impossible to obtain such a value of minimum from our model. The model does not explain such extreme value.

+++ {"slideshow": {"slide_type": "skip"}}

We do the same thing for maximum

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(np.max(normal_pps,axis=1) , histtype='step', density=True, bins=50);
plt.axvline(y.max(), color='black');
```

+++ {"slideshow": {"slide_type": "skip"}}

This time this value does not seem  impossible to obtain from our model.

+++ {"slideshow": {"slide_type": "skip"}}

Finally we repeat same analysis for so called `kurtosis`, a quantity designed specifically to test for "normalness" of a distribution.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(st.kurtosis(normal_pps,bias=False, axis=1) , histtype='step', density=True, bins=50);
plt.axvline(st.kurtosis(y, bias=False), color='black');
```

+++ {"slideshow": {"slide_type": "skip"}}

Again we see that our model does not explain the value of this variable.

+++ {"slideshow": {"slide_type": "slide"}}

## A better  model

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with pm.Model() as students5_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.StudentT('y_obs', mu = mu, sigma = sigma, nu=5, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students5_model:
    MAP = pm.find_MAP()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
MAP
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students5_model:
    students5_trace = pm.sample(draws=4000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students5_model:
    az.plot_trace(students5_trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students5_model:
    az.plot_posterior(students5_trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students5_model:
    students5_trace = pm.sample_posterior_predictive(trace=students5_trace,  extend_inferencedata=True, return_inferencedata=True)
```

+++ {"slideshow": {"slide_type": "slide"}}

##  Comparing models (model selection)

+++ {"slideshow": {"slide_type": "fragment"}}

## Out of sample predictive fit

+++ {"slideshow": {"slide_type": "skip"}}

If we have a new  sample $\b{\tilde y}$ from the true distribution $f$ then we can judge the quality of our model by calculating the log-likelihood of the new data with respect to posterior predictive density of our model

+++ {"slideshow": {"slide_type": "slide"}}

### log  predictive density (lpd)

+++ {"slideshow": {"slide_type": "-"}}

$$\log p_{post}(\b{\tilde{y}}|\b y)  = \log \int p(\b{\tilde{y}}|\theta)p_{post}(\theta|\b y)\text{d}\theta$$

+++ {"slideshow": {"slide_type": "slide"}}

### Expected log predictive density (ELPD)

+++ {"slideshow": {"slide_type": "skip"}}

The overall measure of the quality of our model would be the log-likelihood averaged over all possible out of sample data

+++ {"slideshow": {"slide_type": "-"}}

$$E_f(\log p_{post}(\b{\tilde{y}})|\b{y}) = \int \log p_{post}(\b{\tilde{y}}|\b y) f(\b{\tilde{y}})\text{d}\b{\tilde{y}} $$

+++ {"slideshow": {"slide_type": "slide"}}

### Expected log pointwise predictive density (ELPPD)

+++ {"slideshow": {"slide_type": "skip"}}

If $\b{\tilde y}$ can be split into individual  data  points we can use the expected log pointwise predictive density defined as

+++ {"slideshow": {"slide_type": "-"}}

$$\sum_i E_f(\log p_{post}(\tilde{y}_i)|\b y) = \sum_i \int \log p_{post}(\tilde{y}_i|\b y) f(\tilde{y_i})\text{d}\tilde{y}_i $$

+++ {"slideshow": {"slide_type": "skip"}}

Of course when $\tilde y_i$ are independent  the two formulas give  same results.

+++ {"slideshow": {"slide_type": "slide"}}

### Expected computed predictive log distribution

```{raw-cell}
---
slideshow:
  slide_type: skip
---
Usually we do not know the true model $f$ and so we cannot realy calculate ELPD or ELPPD. However in this artificial case we can simulate as many new samples as we want and use Monte-Carlo integration to estimate ELPPD
```

+++ {"slideshow": {"slide_type": "skip"}}

To calculate the posterior predictive density we will also use the Monte-Carlo  techniques using the  posterior samples that we have already collected

+++ {"slideshow": {"slide_type": "-"}}

$$\sum_{i=1}^n \frac{1}{N}\sum_{j=1}^N\log\left(\frac{1}{S}\sum_{s=1}^S p(\tilde y_i^j|\theta^s)\right),\qquad \theta^s \sim p_{post}(\theta|\b y),\quad \tilde y^j_i\sim f$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
mu_sigma_posterior_sample = np.stack(
    (normal_trace.posterior['mu'].to_numpy().ravel(), 
     normal_trace.posterior['sigma'].to_numpy().ravel()) , axis=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
def normal_likelihood(mu, sigma,y):
        return st.norm(loc=np.expand_dims(mu,1), scale=np.expand_dims(sigma,1)).pdf(y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
elppds = np.array(
    [y_size*np.log(normal_likelihood(mu_sigma_posterior_sample[:,0],
                   mu_sigma_posterior_sample[:,1],true_dist.rvs(4000)).mean(0)).mean() for i in range(10)])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
print(f"elppd = {elppds.mean():.1f} +-/ {elppds.std()/np.sqrt(10.0):.2f}")
```

+++ {"slideshow": {"slide_type": "slide"}}

### In-sample lppd

+++ {"slideshow": {"slide_type": "skip"}}

If we try to estimate the lppd using the sample we have used to fit the model very likely our lppd will be  bigger

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_size*np.log(normal_likelihood(mu_sigma_posterior_sample[:,0], mu_sigma_posterior_sample[:,1],y).mean(0)).mean()
```

+++ {"slideshow": {"slide_type": "slide"}}

##  Leave-one-out cross validation

+++ {"slideshow": {"slide_type": "skip"}}

One way to proceed in  such a case is to use _k-fold cross validation_. To this end we divide our data into $k$  folds. We use $k-1$ folds to train the model and we test it (estimate the elppd) on the remaining one. And we repeat this procedure $k$ times each time leaving out another fold. We the average the results. When each fold consist of a single data point this is so called _leave one out_ (loo) cross validation.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
with normal_model:
    pm.compute_log_likelihood(normal_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
loo=az.loo(normal_trace,  pointwise=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
loo
```

+++ {"slideshow": {"slide_type": "slide"}}

## Information criteria

+++ {"slideshow": {"slide_type": "skip"}}

Another way to proceed is to use the in-sample lppd but somehow compensate for overfitting. There were several proposed ways to do it, called _information criterias_.

+++ {"slideshow": {"slide_type": "slide"}}

## Widely  applicable information criterion

+++ {"slideshow": {"slide_type": "skip"}}

One of them is so called Widely  applicable information criterion (waic) defined as

+++ {"slideshow": {"slide_type": "fragment"}}

$$-2(\text{lppd}-p_{WAIC})$$

+++ {"slideshow": {"slide_type": "skip"}}

where

+++ {"slideshow": {"slide_type": "fragment"}}

$$p_{WAIC} =\sum_{i=1}^n Var_{s}[\log p(y_i|\theta^s)]$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.waic(normal_trace)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students5_model:
    pm.compute_log_likelihood(students5_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.loo(students5_trace, pointwise=False)    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model_compare = az.compare({'normal': normal_trace, 'student5':students5_trace})
model_compare
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.plot_compare(model_compare);
```

+++ {"slideshow": {"slide_type": "slide"}}

## A yet  better  model?

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
with pm.Model() as students4_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.StudentT('y_obs', mu = mu, sigma = sigma, nu=4, observed = y )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students4_model:
    MAP = pm.find_MAP()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
MAP
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students4_model:
    students4_trace = pm.sample(draws=4000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students4_model:
    az.plot_trace(students4_trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students4_model:
    az.plot_posterior(students4_trace);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students4_model:
    students4_trace = pm.sample_posterior_predictive(trace=students4_trace,  extend_inferencedata=True, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with students4_model:
    pm.compute_log_likelihood(students4_trace, extend_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.loo(students4_trace)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model_compare = az.compare({'normal': normal_trace, 'student5':students5_trace, 'students4':students4_trace})
model_compare
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.plot_compare(model_compare)
```

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
%load_ext autoreload
%autoreload 2 
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Model testing and selection

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy  as np
import scipy as sp
import scipy.stats as st
from scipy.special import gamma

import pymc as pm
import arviz as az
print(f"Running on PyMC v{pm.__version__} and ArviZ v{az.__version__}") 
import matplotlib.pyplot as plt
figsize = (8,6)
plt.rcParams["figure.figsize"] = figsize
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Students distribution

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Our "true" model will be Student's t-distribution. This distribution is similar to the normal distribution but it has long-tails. It's probability density function is

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\frac{\Gamma\left(\frac{\nu+1}{2}\right)}
{\sqrt{\pi\nu}\Gamma\left(\frac{\nu}{2}\right)}
\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

So it falls off like

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$x^{-\nu-1}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

 for large $x$. When $\nu\rightarrow\infty$ this distributions approaches normal distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs = np.linspace(-5,5,1000)
fig, ax = plt.subplots(figsize=figsize)
for nu in (1,2,5,10,25,50):
    ax.plot(xs, st.t.pdf(xs, df=nu), label=f"$\\nu={nu}$", linewidth=1);
ax.plot(xs, st.norm.pdf(xs), color = 'black', label=r"$\mathcal{{N}}(0,1)$")    
ax.legend();    
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Similarly to normal distribution, we can move and scale it

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
mu = 1 
xs = np.linspace(-5,5,1000)
fig, ax = plt.subplots(figsize=figsize)
for s in (0.2,0.5, 1,2,5):
    ax.plot(xs, st.t.pdf(xs, df=1, loc=mu, scale=s), label=f"$s={s}$");
ax.legend();    
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## True model

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The data we will be using as example will be draw from t-distribution with $\nu=4$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
true_nu = 4
true_mu = 1
true_scale = 2
true_dist = st.t(loc=1, scale=2, df=4)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
true_mean, true_std = true_dist.mean(), true_dist.std()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
print(true_mean, true_std)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Data

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
y_size = 129
y = true_dist.rvs(size=y_size, random_state=14543)
y_mean = y.mean()
y_s = y.std(ddof=1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.scatter(np.arange(len(y)), y);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(y, bins=20);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
print("nu = {} mu = {} scale = {}".format(true_nu, true_mu, true_scale))
print("nu = {:.2f} mu = {:.2f} scale = {:.2f}".format(*st.t.fit(y)))
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Normal model

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We will start with fitting normal model

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with pm.Model() as normal_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = y )
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model:
    MAP = pm.find_MAP()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: '-'
---
MAP
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(y, bins=20, density=True);
xs = -np.linspace(-15,10,500)
plt.plot(xs,st.norm.pdf(xs, loc=MAP['mu'],scale=MAP['sigma']));
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model:
    normal_trace = pm.sample(draws=8000, return_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model:
    az.plot_trace(normal_trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model:
    az.plot_posterior(normal_trace, hdi_prob=0.95);
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Posterior Predictive Check

+++ {"slideshow": {"slide_type": "skip"}}

The idea of posterior predictive check (PPC) is to generate a new synthetic data set(s) and compare it with original data.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior predictive distribution

+++ {"slideshow": {"slide_type": "skip"}}

The distribution which we will use to generate the new data is called _posterior predictive distribution_. Formally it's pdf function is defined by

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\newcommand{\b}[1]{\mathbf{#1}}$$
$$p(\b{\tilde y}| \b y) = \int p(\b{\tilde y}|\b \theta) p_{post}(\b\theta|\b y)\text{d}\b\theta$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

$\b y$ in this formula represents the data that was observed and used to fit the model and $\b {\hat y}$ denotes the  generated data. In the following I will sometimes omit the $\b y$. Sampling from this distribution is easy

+++ {"slideshow": {"slide_type": "fragment"}}

* First sample the parameters $\b\theta$ according to posterior $p_{post}(\theta|\b y)$
* The sample the $\b{\hat y}$ according to sampling distribution  $p(\b{\tilde y}|\theta)$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

In our case sampling distribution is the Gaussian distribution and the full posterior predictive distribution is

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$p(\tilde y|\b y) = \int p_{\mathcal{N}}(\tilde y|\mu,\sigma) p_{post}(\mu,\sigma|\b y)\text{d}\mu\text{d}\sigma$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Sampling from it proceeds as follows

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

* Sample $\mu$ and $\sigma$ from posterior
* Sample $\hat y$ from $\mathcal{N}(\mu,\sigma)$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The samples of $\mu$ and $\sigma$ from the posterior are already calculated and are stored in the `normal_trace` variable

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Each of the the parameters was sampled 4 x 8000 times. We will combine both parameters into single array using `numpy.stack` function

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
posterior_sample = np.stack( (normal_trace.posterior['mu'],normal_trace.posterior['sigma']),-1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
posterior_sample.shape
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and sample `y_size` number for for each pair using the [`np.apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html) function

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
posterior_predictive_sample = np.apply_along_axis(lambda a: st.norm.rvs(*a, size=y_size),2,posterior_sample)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
posterior_predictive_sample.shape
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Finally we will combine all four chains into single one

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
posterior_predictive_sample = posterior_predictive_sample.reshape(-1,y_size)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
posterior_predictive_sample.shape
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We have 32000 synthetic data sets to compare with our original `y` data set. We will start by drawing few examples together with the original data

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax = plt.subplots(1,3, figsize=(18,6))
for i in range(3):
    ax[i].scatter(np.arange(len(y)), y,label="$y$")
    ax[i].scatter(np.arange(len(posterior_predictive_sample[i])), posterior_predictive_sample[i],label=f"$\\hat y_{i}$")
    ax[i].legend(loc='upper left');
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The look reasonably similar by one may notice the absence of  such pronounced outliers in case of the synthetic data sets. We can look at this using histograms but first generate the posterior predictive sample again this time using a PyMC function `sample_posterior_predictive`.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with normal_model:
    normal_trace = pm.sample_posterior_predictive(trace=normal_trace, extend_inferencedata=True)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This is confusing but we may ignore this warning in this case. The potential term involves only the `sigma` variable  which is taken from the trace.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The `normal_trace` InferenceData variable is now extended with another group `posterior_predictive`.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
normal_trace
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
print(normal_trace.posterior_predictive)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
normal_pps = normal_trace.posterior_predictive['y_obs'].stack({'z':['chain','draw']}).transpose()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
print(normal_pps)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can check that we obtain identical results as before.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(normal_pps.data.ravel(), bins=100, histtype='step',density=True, range=(-15,15));
plt.hist(posterior_predictive_sample.ravel(), bins=100, histtype='step', density=True, range=(-15,15));
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Looking at the three first synthetic data sets we observe again that while they resemble the histogram of the original data, the outlying values are missing

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax = plt.subplots(1,3, figsize=(18,6))
for i in range(3):
    ax[i].hist(y,label="$y$", bins=20, density=True, histtype='step')
    ax[i].hist(normal_pps[i],label=f"$\\hat y_{i}$", bins=20, density=True, histtype='step')
    ax[i].legend(loc='upper left');
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

To see this in more detailed we will plot the histogram of the minimum values of each synthetic data set and compare it with the minimum of the original data

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(np.min(normal_pps,axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("min(y)")
plt.axvline(y.min(), color='black');
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

From this picture we see that it is almost impossible to obtain such a value of minimum from our model. The model does not explain such extreme value. We do the same thing for maximum

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(np.max(normal_pps,axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("max(y)")
plt.axvline(y.max(), color='black');
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This time this value does not seem  impossible to obtain from our model.
Finally we repeat same analysis for so called `kurtosis`, a quantity designed specifically to test for "normalness" of a distribution.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(st.kurtosis(normal_pps,bias=False, axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("kurtosis(y)")
plt.axvline(st.kurtosis(y, bias=False), color='black');
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Again we see that our model does not explain the value of this variable.

+++ {"slideshow": {"slide_type": "slide"}}

## A better  model

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Now we will look at the  t-distribution model, but with different $\nu$ which we somewhat arbitrarily set to twelve

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with pm.Model() as T_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.StudentT('y_obs', mu = mu, sigma = sigma, nu=12, observed = y )
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with T_model:
    MAP = pm.find_MAP()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
MAP
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(y, bins=20, density=True);
xs = -np.linspace(-15,10,500)
plt.plot(xs,st.t.pdf(xs, loc=MAP['mu'],scale=MAP['sigma']/st.t(df=12).std(),df=12));
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Seems to fit well but still does not explain the outlier

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with T_model:
    T_trace = pm.sample(draws=8000, return_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with T_model:
    az.plot_trace(T_trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with T_model:
    az.plot_posterior(T_trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with T_model:
    T_trace = pm.sample_posterior_predictive(trace=T_trace,  extend_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
T_pps = T_trace.posterior_predictive['y_obs'].stack(z=('chain', 'draw')).transpose()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can again check how this model predict the actual observed features of the data

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(np.min(T_pps,axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("min(y)")
plt.axvline(y.min(), color='black');
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(st.kurtosis(T_pps,bias=False, axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("kurtosis(y)")
plt.axvline(st.kurtosis(y, bias=False), color='black');
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

It is better than the normal model but still rather improbable.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

##  Comparing models (model selection)

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Out of sample predictive fit

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

If we have a new  sample $\b{\tilde y}$ from the true distribution $f$ then we can judge the quality of our model by calculating the log-likelihood of the new data with respect to posterior predictive density of our model

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### log  predictive density (lpd)

+++ {"slideshow": {"slide_type": "-"}, "editable": true}

$$\log p_{post}(\b{\tilde{y}}|\b y)  = \log \int p(\b{\tilde{y}}|\theta)p_{post}(\theta|\b y)\text{d}\theta$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Expected log predictive density (ELPD)

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The overall measure of the quality of our model would be the log-likelihood averaged over all possible out of sample data

+++ {"slideshow": {"slide_type": "-"}, "editable": true}

$$E_f[\log p_{post}(\b{\tilde{y}}|\b{y})] = \int \log p_{post}(\b{\tilde{y}}|\b y) f(\b{\tilde{y}})\text{d}\b{\tilde{y}} $$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Expected log pointwise predictive density (ELPPD)

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

If $\b{\tilde y}$ can be split into individual  data  points we can use the expected log pointwise predictive density defined as

+++ {"slideshow": {"slide_type": "-"}, "editable": true}

$$\sum_i E_f[\log p_{post}(\tilde{y}_i|\b y)] = \sum_i \int \log p_{post}(\tilde{y}_i|\b y) f(\tilde{y_i})\text{d}\tilde{y}_i $$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Of course when $\tilde y_i$ are independent  the two formulas give  same results.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Expected computed predictive log distribution

```{raw-cell}
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: skip
---
Usually we do not know the true model $f$ and so we cannot really calculate ELPD or ELPPD. However in this artificial case we can simulate as many new samples as we want and use Monte-Carlo integration to estimate ELPPD
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

To calculate the posterior predictive density we will also use the Monte-Carlo  techniques using the  posterior samples that we have already collected

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\sum_i E_f[\log p_{post}(\tilde{y}_i|\b y)] \approx \sum_{i=1}^n \frac{1}{N}\sum_{j=1}^N\log\left( p_{post}(\tilde y_i^j|\b{y})\right),\qquad \tilde y^j_i\sim f$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p_{post}(\tilde y_i^j|\b{y})= 
\int p(y_j|\b \theta) p_{post}(\b\theta|\b y)\text{d}\b\theta 
\approx\frac{1}{S}\sum_{s=1}^S p(\tilde y_i^j|\theta^s) \qquad \theta^s \sim p_{post}(\theta|\b y) $$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\sum_i E_f(\log p_{post}(\tilde{y}_i)|\b y) \approx \sum_{i=1}^n \frac{1}{N}\sum_{j=1}^N\log\left(\frac{1}{S}\sum_{s=1}^S p(\tilde y_i^j|\theta^s)\right),\qquad \theta^s \sim p_{post}(\theta|\b y),\quad \tilde y^j_i\sim f$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
mu_sigma_normal_pps = np.stack(
    (normal_trace.posterior['mu'].to_numpy().ravel(), 
     normal_trace.posterior['sigma'].to_numpy().ravel()) , axis=1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
def normal_likelihood(mu, sigma,y):
        return st.norm(loc=np.expand_dims(mu,1), scale=np.expand_dims(sigma,1)).pdf(y)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
%%time
elppds = np.array(
    [(print(f"Sample[{i}] = ",end=''),
      ellpd := y_size*np.log(
          normal_likelihood(mu_sigma_normal_pps[:,0],
                            mu_sigma_normal_pps[:,1],
                            true_dist.rvs(4000)).mean(0)
                            ).mean()
     ,print(ellpd))[1] for i in range(10)])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: '-'
---
print(f"Normal model elppd = {elppds.mean():.1f} +-/ {elppds.std()/np.sqrt(10.0):.2f}")
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
mu_sigma_T_pps = np.stack(
    (T_trace.posterior['mu'].to_numpy().ravel(), 
     T_trace.posterior['sigma'].to_numpy().ravel()) , axis=1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: '-'
---
def T_likelihood(mu, sigma,y):
        return st.t(loc=np.expand_dims(mu,1), scale=np.expand_dims(sigma,1), df=12).pdf(y)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: '-'
---
def T_loglikelihood(mu, sigma,y):
        return st.t(loc=np.expand_dims(mu,1), scale=np.expand_dims(sigma,1), df=12).logpdf(y)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
%%time
elppds = np.array(
    [(print(f"Sample[{i}] = ",end=''),
      ellpd := y_size*np.log(T_likelihood(mu_sigma_T_pps[:,0],
                   mu_sigma_T_pps[:,1],true_dist.rvs(4000)).mean(0)).mean(),
     print(ellpd))[1] for i in range(10)])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: '-'
---
print(f"T model elppd = {elppds.mean():.1f} +-/ {elppds.std()/np.sqrt(10.0):.2f}")
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### In-sample lppd

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

If we try to estimate the lppd using the sample we have used to fit the model very likely our lppd will be  bigger

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
y_size*np.log(normal_likelihood(mu_sigma_normal_pps[:,0], mu_sigma_normal_pps[:,1],y).mean(0)).mean()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
y_size*np.log(T_likelihood(mu_sigma_T_pps[:,0], mu_sigma_T_pps[:,1],y).mean(0)).mean()
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

##  Leave-one-out cross validation

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

One way to proceed in  such a case is to use _k-fold cross validation_. To this end we divide our data into $k$  folds. We use $k-1$ folds to train the model and we test it (estimate the elppd) on the remaining one. And we repeat this procedure $k$ times each time leaving out another fold. We the average the results. When each fold consist of a single data point this is so called _leave one out_ (loo) cross validation.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

To use this functionality in ArviZ we need first to calculate the log-likelighood of the observed data for each posterior sample

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
if 'log_likelihood' in normal_trace:
    delattr(normal_trace, 'log_likelihood')
with normal_model:
    pm.compute_log_likelihood(normal_trace, extend_inferencedata=True)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This  for every $\theta_i$ in the posterior sample calculates the $\log p(y_j|\theta_i)$ for every $y_j$ in the observed data set

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
st.norm.logpdf(loc=normal_trace.posterior['mu'][0,0], scale=normal_trace.posterior['sigma'][0,0], x = y[:20])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
print(normal_trace.log_likelihood['y_obs'][0,0,:20])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
loo=az.loo(normal_trace,  pointwise=True)
print(loo)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Information criteria

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Another way to proceed is to use the in-sample lppd but somehow compensate for overfitting. There were several proposed ways to do it, called _information criterias_.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Widely  applicable information criterion

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

One of them is so called widely  applicable information criterion (waic) defined as

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$-2(\text{lppd}-p_{WAIC})$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

where

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{WAIC} =\sum_{i=1}^n Var_{s}[\log p(y_i|\theta^s)]$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.waic(normal_trace)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
if 'log_likelihood' in T_trace:
    delattr(T_trace, 'log_likelihood')
with T_model:
    pm.compute_log_likelihood(T_trace, extend_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.loo(T_trace, pointwise=False)    
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
az.waic(T_trace)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
model_compare = az.compare({'normal': normal_trace, 'T12':T_trace})
model_compare
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.plot_compare(model_compare);
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## A yet  better  model?

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
with pm.Model() as TT_model:
    mu = pm.Flat('mu')
    sigma = pm.HalfFlat('sigma')
    pm.Potential('sigma_pot', -np.log(sigma))
    y_obs = pm.StudentT('y_obs', mu = mu, sigma = sigma, nu=4, observed = y )
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with TT_model:
    MAP = pm.find_MAP()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
MAP
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with TT_model:
    TT_trace = pm.sample(draws=8000, return_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with TT_model:
    az.plot_trace(TT_trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with TT_model:
    az.plot_posterior(TT_trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with TT_model:
    TT_trace = pm.sample_posterior_predictive(trace=TT_trace,  extend_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
TT_pps = TT_trace.posterior_predictive['y_obs'].stack(z=('chain', 'draw')).transpose()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(np.min(TT_pps,axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("min(y)")
plt.axvline(y.min(), color='black');
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(st.kurtosis(TT_pps,bias=False, axis=1) , histtype='step', density=True, bins=50);
plt.xlabel("kurtosis(y)")
plt.axvline(st.kurtosis(y, bias=False), color='black');
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with TT_model:
    pm.compute_log_likelihood(TT_trace, extend_inferencedata=True)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.loo(TT_trace)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.waic(TT_trace)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
az.rcParams["stats.information_criterion"]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
model_compare = az.compare({'normal': normal_trace, 'T12':T_trace, 'T4':TT_trace})
model_compare
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.plot_compare(model_compare);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
model_compare_waic = az.compare({'normal': normal_trace, 'T12':T_trace, 'T4':TT_trace}, ic='waic')
model_compare_waic
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.plot_compare(model_compare_waic);
```

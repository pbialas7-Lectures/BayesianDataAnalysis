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

+++ {"slideshow": {"slide_type": "slide"}}

# Regression

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import pymc as pm
import arviz as az
print(f"Running PyMC {pm.__version__} and ArviZ {az.__version__}")
from copy import copy 

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams['figure.figsize']=(12,8)
```

+++ {"slideshow": {"slide_type": "slide"}}

## A Motivating Example: Linear Regression

From [Getting started with PyMC3](https://docs.pymc.io/notebooks/getting_started.html)

+++ {"slideshow": {"slide_type": "fragment"}}

$$y_i \sim N(\vec{\beta} \cdot \vec{x_i}+\alpha,\sigma)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')

np.random.seed(123)

# True parameter values
alpha_t, sigma_t = 1, 1
beta_t = [1, 2.5]

size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha_t + beta_t[0]*X1 + beta_t[1]*X2 + np.random.randn(size)*sigma_t
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
basic_model = pm.Model()

with basic_model:

    pred1 = pm.MutableData("pred1",X1,dims="obs_id")
    pred2 = pm.MutableData("pred2",X2,dims="obs_id")
    
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu =  pm.Deterministic('mu', alpha + beta[0]*pred1 + beta[1]*pred2)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
map_estimate = pm.find_MAP(model=basic_model, vars=[alpha, beta])
map_estimate
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with basic_model:
    lr_trace = pm.sample(tune=1000, draws=20000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with basic_model:
    az.plot_trace(lr_trace, var_names=['alpha', 'beta', 'sigma']);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.summary(lr_trace, var_names=['alpha', 'beta', 'sigma']).round(2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
betas=lr_trace.posterior['beta'].values.reshape(-1,2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---
alphas = lr_trace.posterior['alpha'].values.reshape(-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(betas[:,0], betas[:,1]);
plt.scatter(beta_t[:1], beta_t[1:],color='red',s=50);
plt.scatter(betas[:,0].mean(), betas[:,1].mean(),edgecolor='red',facecolor='none', s=50);
plt.colorbar();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
np.corrcoef(betas, rowvar=False )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(alphas, betas[:,0]);
plt.colorbar();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(alphas, betas[:,1]);
plt.colorbar();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
lr_trace=pm.compute_log_likelihood(lr_trace, model = basic_model, extend_inferencedata=True)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Errors

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Size of dataset
valid_size = 50

# Predictor variable
X1_valid = np.random.randn(valid_size)
X2_valid = np.random.randn(valid_size) * 0.2

# Simulate outcome variable
Y_valid = alpha_t + beta_t[0]*X1_valid + beta_t[1]*X2_valid + np.random.randn(valid_size)*sigma_t
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
with basic_model:
    pm.set_data({"pred1": X1_valid, "pred2": X2_valid})
    lr_trace = pm.sample_posterior_predictive(lr_trace, extend_inferencedata=True, var_names=['mu'], predictions=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
lr_trace
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
predictions = lr_trace.predictions
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.scatter(predictions['mu_dim_2'], predictions['mu'].mean(("chain","draw")) , color='orange');
ax.vlines(predictions['mu_dim_2'], *az.hdi(predictions)['mu'].transpose("hdi", ...) , color='orange');
```

+++ {"slideshow": {"slide_type": "slide"}}

# Data analysis recipes: Fitting Model to data, David W. Hong.

+++ {"slideshow": {"slide_type": ""}}

From [Data analysis recipes: Fitting Model to data, David W. Hong](https://arxiv.org/abs/1008.4686)

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
data = np.loadtxt("linear_regression.txt")
clean_data = data[5:]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o');
```

+++ {"slideshow": {"slide_type": "slide"}}

$$y_i \sim N(\beta \cdot x+\alpha,\sigma_i),\quad \sigma_i \text{ known}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_model = pm.Model()

with y_model:
    pred = pm.MutableData("pred",clean_data[:,0])
    sigma = pm.MutableData("sigma",clean_data[:,2])                        
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    # Expected value of outcome
    mu = pm.Deterministic("mu",alpha + beta*pred)
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=clean_data[:,1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
map_estimate = pm.find_MAP(model=y_model)
map_estimate
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ys = map_estimate['alpha']+map_estimate['beta']*xs
plt.plot(xs,ys);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with y_model:
    y_trace = pm.sample(tune=1000, draws=10000, return_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with y_model:
    az.plot_trace(y_trace, var_names=['alpha', 'beta']);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with y_model:
    az.plot_posterior(y_trace, var_names=['alpha', 'beta']);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.summary(y_trace, var_names=['alpha', 'beta']).round(2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
post=y_trace.posterior
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(y_trace.posterior['alpha'].data, y_trace.posterior['beta'].data);
plt.xlabel('$\\alpha$', fontsize=20);
plt.ylabel('$\\beta$', fontsize=20);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ys = map_estimate['alpha']+map_estimate['beta']*xs
plt.plot(xs,ys,'orange');
ys = post['alpha'].mean(('chain', 'draw')).item()+post['beta'].mean(('chain', 'draw')).item()*xs
plt.plot(xs,ys,'red');
ys = post['alpha'].median().item()+ post['beta'].median().item()*xs
plt.plot(xs,ys,'green');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
for i in range(8):
    k  =np.random.randint(0, len(post['draw']))
    ys = post['alpha'][0][k].data+post['beta'][0][k].data*xs
    plt.plot(xs,ys,'grey', alpha=0.25);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
y_trace= pm.sample_posterior_predictive(y_trace, extend_inferencedata=True, 
                                        var_names=['mu','Y_obs'], model=y_model)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ax.vlines(y_trace.constant_data['pred'], 
          *az.hdi(y_trace.posterior_predictive)['mu'].transpose("hdi", ...) , color='orange');
ax.scatter(y_trace.constant_data['pred'], 
           y_trace.posterior_predictive['mu'].mean(('draw', 'chain')),s=5, color='orange');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ax.vlines(y_trace.constant_data['pred'], 
          *az.hdi(y_trace.posterior_predictive)['Y_obs'].transpose("hdi", ...) , color='orange');
ax.scatter(y_trace.constant_data['pred'], 
           y_trace.posterior_predictive['Y_obs'].mean(('draw', 'chain')),s=5, color='orange');
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
y_trace=pm.compute_log_likelihood(y_trace, model=y_model, extend_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---
az.loo(y_trace)
```

+++ {"slideshow": {"slide_type": "slide"}}

# Higher order

+++ {"slideshow": {"slide_type": "fragment"}}

$$\mu_i = \sum_{i=0}^{n-1} x^i \alpha_{n-1-i}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\mu_i = \alpha_0 x_i^4 + \alpha_1 x^{3} + \alpha_2 x^2 +\alpha_3 x +\alpha_4 $$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def poly(x,alpha,n):
    y=alpha[0]
    for i in range(1,n):
        y=y*x+alpha[i]
    return y    
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
y4_model = pm.Model()
n = 5
with y4_model:
    pred = pm.MutableData("pred",clean_data[:,0])
    sigma = pm.MutableData("sigma",clean_data[:,2])
                           
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10, size=(n,))
   
    # Expected value of outcome
    mu = pm.Deterministic("mu",poly(pred,alpha, n))

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=clean_data[:,1])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
from numpy import polyfit
```

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---
a_start = polyfit(clean_data[:,0],clean_data[:,1],n-1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
with y4_model:
    y4_map = pm.find_MAP(model=y4_model, start={'alpha':a_start})
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y4_map
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from numpy import polyval
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,230,100);ys = polyval(a_start, xs)
plt.plot(xs,ys);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
precalculated = True
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
if not precalculated:
    with y4_model:
        y4_trace=pm.sample(tune=1000, draws=25000, return_inferencedata=True, nuts_sampler_kwargs={'target_accept':.995, 'max_tree_depth':20})
    
    y4_trace.to_netcdf("y4_trace.nc", engine='h5netcdf')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y4_trace_disk = az.from_netcdf('y4_trace.nc')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with y4_model:
    az.plot_trace(y4_trace_disk, var_names=['alpha'])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---
az.summary(y4_trace_disk, var_names=['alpha'])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
with y4_model:
    az.plot_posterior(y4_trace_disk, var_names=['alpha'], grid=(3,2))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
alphas4 = y4_trace_disk.posterior['alpha'].data.reshape(-1,5)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(y4_trace_disk.posterior['alpha'][:,:,3], y4_trace_disk.posterior['alpha'][:,:,4]);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hexbin(y4_trace_disk.posterior['alpha'][:,:,2], y4_trace_disk.posterior['alpha'][:,:,3]);
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,230,100)
ys = poly(xs, y4_trace_disk.posterior['alpha'].mean(('chain','draw')).data,n)
plt.plot(xs,ys);
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with y4_model:
    y4_trace_disk=pm.sample_posterior_predictive(y4_trace_disk, extend_inferencedata=True, return_inferencedata=True, var_names=['mu','Y_obs'])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ax.vlines(y4_trace_disk.constant_data['pred'], 
          *az.hdi(y4_trace_disk.posterior_predictive)['mu'].transpose("hdi", ...) , color='orange');
ax.scatter(y4_trace_disk.constant_data['pred'], 
           y4_trace_disk.posterior_predictive['mu'].mean(('draw', 'chain')),s=5, color='orange');
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ax.vlines(y4_trace_disk.constant_data['pred'], 
          *az.hdi(y4_trace_disk.posterior_predictive)['Y_obs'].transpose("hdi", ...) , color='orange');
ax.scatter(y4_trace_disk.constant_data['pred'], 
           y4_trace_disk.posterior_predictive['Y_obs'].mean(('draw', 'chain')),s=20, color='orange');
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
with y4_model:
    y4_trace_disk=pm.compute_log_likelihood(y4_trace_disk, extend_inferencedata=True)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.loo(y4_trace_disk)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
model_comp = az.compare({'y_model':y_trace, 'y4_model':y4_trace_disk})
model_comp
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
az.plot_compare(model_comp);
```

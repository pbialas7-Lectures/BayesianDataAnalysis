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

# Regression

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import pymc as pm
import arviz as az
from copy import copy 

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams['figure.figsize']=(8,6)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
degree = 180.0/np.pi
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## A Motivating Example: Linear Regression

From [Getting started with PyMC3](https://www.pymc.io/projects/examples/en/2021.11.0/getting_started.html)

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$y_i \sim N(\vec{\beta} \cdot \vec{x_i}+\alpha,\sigma)$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid')

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha_t =1
beta_t = [1, 2.5]
sigma_t = 1

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha_t + beta_t[0]*X1 + beta_t[1]*X2 + np.random.randn(size)*sigma_t
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
cs=plt.scatter(X1,X2, c=Y);
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.colorbar(cs);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
map_estimate = pm.find_MAP(model=basic_model)
map_estimate
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
beta_t
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with basic_model:
    trace = pm.sample(tune=1000, draws=8000, cores=4, nuts_sampler_kwargs={'num_blas_workers':4})
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with basic_model:
    az.plot_trace(trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.summary(trace).round(2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
betas=trace.posterior['beta'].values.reshape(-1,2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
alphas = trace.posterior['alpha'].values.reshape(-1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
cs = plt.hexbin(betas[:,0], betas[:,1]);
plt.scatter(beta_t[:1], beta_t[1:],color='red',s=50, label=r"$\vec\beta_{true}$");
plt.scatter(betas[:,0].mean(), betas[:,1].mean(),edgecolor='red',facecolor='none', s=50);
plt.xlabel(r"$\beta_0$");plt.ylabel(r"$\beta_1$");
plt.colorbar(cs);plt.legend();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
np.corrcoef(betas,rowvar=False )
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hexbin(alphas, betas[:,0]);
plt.colorbar();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hexbin(alphas, betas[:,1]);
plt.colorbar();
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Errors

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(az.hdi(trace.posterior['beta'],hdi_prob=.975)['beta'].values)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(az.hdi(trace.posterior['alpha'], hdi_prob=0.975)['alpha'].values)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(az.hdi(trace.posterior['sigma'], hdi_prob=0.975)['sigma'].values)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Data analysis recipes: Fitting Model to data, David W. Hong.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

From [Data analysis recipes: Fitting Model to data, David W. Hong](https://arxiv.org/abs/1008.4686). See also [GLM: Linear regression](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM_linear.html#glm-linear)

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
data = np.loadtxt("linear_regression.txt")
clean_data = data[5:]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
fig, ax = plt.subplots()
ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o');
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$y_i \sim N(\beta x_i+\alpha,\sigma_i),\quad \sigma_i \text{ known}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
y_model = pm.Model()

with y_model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
   
    # Expected value of outcome
    mu = alpha + beta*clean_data[:,0]

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=clean_data[:,2], observed=clean_data[:,1])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
map_estimate = pm.find_MAP(model=y_model)
map_estimate
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ys = map_estimate['alpha']+map_estimate['beta']*xs
plt.plot(xs,ys);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with y_model:
    trace = pm.sample(tune=1000, draws=8000)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with y_model:
    az.plot_trace(trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.summary(trace).round(2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
trace.stack(sample=['chain', 'draw'], inplace=True)
post=trace.posterior
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
print(post)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots()
ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
ys = map_estimate['alpha']+map_estimate['beta']*xs
plt.plot(xs,ys,'orange',label='MAP');
ys = np.mean(post['alpha'].data)+np.mean(post['beta'].data)*xs
plt.plot(xs,ys,'red', label='mean');
ys = post['alpha'].median().data+ post['beta'].median().data*xs
plt.plot(xs,ys,'green', label='median');
plt.legend();
plt.close()
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
trace.posterior.sizes['sample']
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
xs = np.linspace(50,250,100)
for i in range(128):
    k  =np.random.randint(0, trace.posterior.sizes['sample'])
    ys = post['alpha'][k].data+post['beta'][k].data*xs
    plt.plot(xs,ys,'grey', alpha=0.25);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs = np.linspace(50,250,100)
ys = np.mean(post['alpha'].data)+np.mean(post['beta'].data)*xs
yss = post['alpha'].data+np.outer(xs,post['beta'].data)
err = yss.std(axis=1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
_ = ax.errorbar(clean_data[:,0], clean_data[:,1],  yerr=clean_data[:,2], fmt='o')
plt.plot(xs,ys,color='red', linewidth=1);
plt.fill_between(xs,ys-err,ys+err,color='orange', alpha=0.5);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hexbin(post['alpha'].data, post['beta'].data);
plt.xlabel('$\\alpha$', fontsize=20);
plt.ylabel('$\\beta$', fontsize=20);
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Uncertainties on both axes

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots() 
ax.set_xlim(0,300)
ax.set_ylim(0,700)
ax.scatter(clean_data[:,0], clean_data[0:,1],marker='.', color='black')
ax.errorbar(clean_data[:,0], clean_data[0:,1], 
            xerr=clean_data[:,3], yerr=clean_data[:,2],fmt='none',
           elinewidth=1, capsize=2)
plt.close()    
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
def o_d(phi,s):
    cos  = np.cos(phi)
    sin  = np.sin(phi)
    return  (np.array([s*cos, s*sin]), 
                      np.array([-sin, cos])
                     )
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from matplotlib.patches import Arc, FancyArrowPatch
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
phi=0.7
s = 0.47
fig, ax = plt.subplots(figsize=(7,7))
ax.axvline(0, linewidth=1, color='gray');ax.axhline(0, linewidth=1, color='gray')
ax.set_aspect(1.0);ax.set_xlim(-1,1); ax.set_ylim(-1,1);
plt.scatter([0],[0],color='black');
o,d = o_d(phi,s)
ax.scatter(o[0],o[1]);ax.plot([0,o[0]],[0,o[1]],linestyle='--');
ts = np.linspace(-1.,1.0,10); ls = d*ts.reshape(-1,1)+o;
ax.plot(ls[:,0], ls[:,1],color='black');
arc=Arc((0,0), width=.5, height=0.5, theta1=0, theta2=np.rad2deg(phi));ax.add_patch(arc);
ax.text(0.25*np.cos(phi/2), 0.25*np.sin(phi/2),"$\\phi$")
ax.text(0.3*np.cos(phi+.1), 0.3*np.sin(phi+.1),"$s$")
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
  slide_type: slide
---
n = len(clean_data)
with pm.Model() as model:
    s   = pm.HalfFlat('s')
    phi = pm.Uniform('phi',lower = -np.pi, upper = np.pi)  
    t = pm.Flat('t',shape=n)
    o = pm.math.stack([s*np.cos(phi), s*np.sin(phi)])
    d = pm.math.stack([-np.sin(phi), np.cos(phi)])
   
    p = pm.Deterministic('p', d*t.reshape((-1,1))+o)
    
    obs_y = pm.Normal('obs_y',mu = p[:,1], sigma=clean_data[:,2], 
                      observed=clean_data[:,1])
    obs_x = pm.Normal('obs_x', mu = p[:,0], sigma=clean_data[:,3], 
                      observed=clean_data[:,0])
  
   
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with model:
    trace = pm.sample(draws=10000, tune=8000, chains=4, target_accept = 0.99)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with model:
    az.plot_trace(trace, var_names=["phi","s"]);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
az.summary(trace, var_names=['phi', 's'])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
trace.stack(sample=['chain', 'draw'], inplace=True)
post2 = trace.posterior
p_m = post2['p'].data.mean(2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
phi_m = post2["phi"].data.mean()
s_m = post2["s"].data.mean()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots() 
ax.set_xlim(0,250)
ax.set_ylim(0,600)
ax.scatter(clean_data[:,0], clean_data[0:,1],marker='.')
ax.errorbar(clean_data[:,0], clean_data[0:,1], 
            xerr=clean_data[:,3], yerr=clean_data[:,2],fmt='none',
           elinewidth=1, capsize=2)
o,d  = o_d(phi_m,s_m)  
times = np.linspace(-600,-100,100)
ps = o+d*times.reshape(-1,1)
ax.plot(ps[:,0], ps[:,1]);
ax.scatter(p_m[:,0], p_m[:,1],color='green',s=10);
plt.close()
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
plt.hexbin(post2['phi'].data,post2['s'].data);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(-100,250)
ax.set_ylim(-100,600)
ax.scatter(clean_data[:,0], clean_data[0:,1],marker='.')
ax.errorbar(clean_data[:,0], clean_data[0:,1], 
            xerr=clean_data[:,3], yerr=clean_data[:,2],fmt='none',
           elinewidth=1, capsize=2)
o,d  = o_d(phi_m,s_m)  
times = np.linspace(-600,100,100)
ps = o+d*times.reshape(-1,1)
ax.plot(ps[:,0], ps[:,1]);
ax.scatter(p_m[:,0], p_m[:,1],color='green',s=10);
ax.scatter([0],[0],color='red',s=20);
plt.close()
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
  slide_type: ''
---

```

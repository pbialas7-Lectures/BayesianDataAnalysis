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
import numpy as np
import scipy
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
figsize=(8,6)
plt.rcParams["figure.figsize"] = figsize
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.special import  expit, log_expit
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Logistic regression
# Analysis of bioassay experiment 
## (from "Bayesian Data Analysis" sec. 3.7)

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
dose = np.array([-0.86,-0.3, -0.05, 0.73]) # log(g/ml)
n_animals = np.array([5,5,5,5]) 
n_deaths = np.array([0,1,3,5])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.scatter(dose, n_deaths/n_animals,s=50, edgecolors='blue', facecolors='none')
plt.xlabel('dose [log g/ml]', fontsize=16);
plt.ylabel('$\\frac{\\# deaths}{\\# animals}$', rotation='horizontal', fontsize=16, labelpad=10);
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Model

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We assume that a given dose $x_i$ results in probability of death $\theta_i$ that is dependent on this dose

+++ {"slideshow": {"slide_type": "slide"}}

$$\theta_i=\theta(x_i)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Given that the number of deaths in $n_i$ subjects will follow the  binomial distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$y_i|\theta_i \sim \operatorname{Bin}(n_i,\theta_i)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

And the probability mass function is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y_i|\theta_i,n_i)=\binom{n_i}{y_i}\theta_i^{y_i}(1-\theta_i)^{n_i-y_i}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Logistic regression

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Next  step is to assume some functional dependence of $\theta$ on $x$. The simplest possible would be the linear relation

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$\newcommand{\logit}{\operatorname{logit}}$
$$\theta_i = \alpha+\beta x_i $$

+++ {"slideshow": {"slide_type": "skip"}}

however this does not fulfill the constrain $0\le \theta_i \le 1$. To make it so we use the logistic sigmoid function

+++ {"slideshow": {"slide_type": "fragment"}}

$$z_i=\alpha+\beta x_i$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\theta_i=s(z_i)\equiv\frac{e^z_i}{1+e^z_i}$$

+++ {"slideshow": {"slide_type": "skip"}}

$z$ in this formula is often called _logit_. The sigmoid function is provided in `scipy` as `scipy.special.expit`

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs = -np.linspace(-10,10,200)
plt.xlabel("$z$", fontsize=16); plt.ylabel("$s(z)$", rotation='horizontal')
plt.plot(xs,expit(xs)); 
```

+++ {"slideshow": {"slide_type": "skip"}}

Combining this together we obtain so called  _logistic regression_

+++ {"slideshow": {"slide_type": "slide"}}

$$\theta(x)=s(\alpha+\beta x)=\frac{e^{\alpha+\beta x}}{1+e^{\alpha+\beta x}}$$

+++ {"slideshow": {"slide_type": "skip"}}

The posterior on $\alpha$ and $\beta$ is given by the usual formula

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\alpha,\beta|y,n,x)=P(y|\alpha,\beta,n,x)P(\alpha,\beta)=P(\alpha,\beta)\prod_k P(y_i|\alpha,\beta,n_i,x_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

The last equality follows from the fact that we have assumed all trials to be independent. The likelihood is given by

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y_i|\alpha,\beta,n_i,x_i)\propto\theta_i^{y_i}(1-\theta_i)^{n_i-y_i}= s(\alpha+\beta x_i)^{y_i}(1-s(\alpha+\beta x_i))^{n_i-y_i}$$

+++ {"slideshow": {"slide_type": "skip"}}

but it will be more convenient to use its logarithm

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log P(y_i|\alpha,\beta,n_i,x_i)=y_i\log s(\alpha+\beta x_i)^{y_i}+(n_i-y_i)\log (1-s(\alpha+\beta x_i))+ const$$

+++ {"slideshow": {"slide_type": "skip"}}

In the following we will assume the uninformative prior

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\alpha,\beta)\propto 1$$

+++ {"slideshow": {"slide_type": "skip"}}

so finally

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\log P(\alpha,\beta|y,n,x) =\\ \sum_i \left(y_i\log s(\alpha+\beta x_i)^{y_i}+(n_i-y_i)\log (1-s(\alpha+\beta x_i))\right)+ const
$$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Numerical calculations

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We will make use of the relation

+++ {"slideshow": {"slide_type": "fragment"}}

$$1-\theta_i=1-\frac{e^{z_i}}{1+e^{z_i}}=\frac{1}{1+e^{z_i}}=\frac{e^{-z_i}}{1+e^{-z_i}}=s(-z_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

The implementation of the  logarithm of posterior is straightforward

```{code-cell}
---
slideshow:
  slide_type: slide
---
def log_P_alpha_beta(a,b,x,y,n):
    z = a+b * x
    log_theta = log_expit(z)
    log_theta_conj = log_expit(-z)
    return np.sum(y*log_theta+(n-y)*log_theta_conj)
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and we can use the` scipy.otimize.mininimize` function the find the MAP estimate of $\alpha$ and $\beta$

```{code-cell}
---
slideshow:
  slide_type: slide
---
min_results = scipy.optimize.minimize(lambda arg: -log_P_alpha_beta(*arg,dose, n_deaths, n_animals),[10,10])
print(min_results)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
ab_map=min_results.x
print(ab_map)
```

+++ {"slideshow": {"slide_type": "skip"}}

We can check how  those estimates compare to data

```{code-cell}
---
slideshow:
  slide_type: skip
---
def lr(x,a,b):
    z = a+b*x
    return expit(z)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
xs = np.linspace(-1,1,500)
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(dose, n_deaths/n_animals,s=50, edgecolors='blue', facecolors='none')
ax.plot(xs,lr(xs,*ab_map),linewidth=1)
ax.set_xlabel('dose [log g/ml]', fontsize=16);
ax.set_ylabel('$\\frac{\\# deaths}{\\# animals}$', rotation='horizontal', fontsize=16, labelpad=10);
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

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

It look as the choice of logistic regression as our model was justified.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def log_P_alpha_beta_tensor(a,b,x,y,n):
    z = a + np.multiply.outer(x,b)
    log_theta = log_expit(z)
    log_theta_conj = log_expit(-z)
    sh = np.ones(len(log_theta.shape),dtype='int')
    sh[0] = -1
    return np.sum((log_theta*y.reshape(sh))+
                   ((log_theta_conj)*(n-y).reshape(sh)), axis=0)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
alphas = np.linspace(-5,10, 500)
betas  = np.linspace(-10,40,500)
a_mesh, b_mesh = np.meshgrid(alphas, betas)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
lr_zs = log_P_alpha_beta_tensor(a_mesh, b_mesh, dose, n_deaths, n_animals)
lr_zs = lr_zs - lr_zs.max()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig,ax = plt.subplots()
def fmt(x):
    return f"{np.exp(x):.2f}"
levels = np.log([0.01,0.1, 0.25, 0.5,0.75,0.9,1.0])
ax.contourf(alphas, betas, lr_zs, levels=levels );
CS = ax.contour(alphas, betas, lr_zs, levels=levels, colors='red',negative_linestyles='-');
ax.clabel(CS, fmt=fmt)
ax.scatter([ab_map[0]], [ab_map[1]], color='red');
ax.set_xlabel("$\\alpha$",fontsize=16);ax.set_ylabel("$\\beta$",fontsize=16, rotation='horizontal');
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
fig,ax = plt.subplots()
def fmt(x):
    return f"{x:.2f}"
levels = [0.01,0.1, 0.25, 0.5,0.75,0.9,1.0]
ax.contourf(alphas, betas, np.exp(lr_zs), levels=levels );
CS = ax.contour(alphas, betas, np.exp(lr_zs), levels=levels, colors='red',negative_linestyles='-');
ax.clabel(CS, fmt=fmt)
ax.scatter([ab_map[0]], [ab_map[1]], color='red');
ax.set_xlabel("$\\alpha$",fontsize=16);ax.set_ylabel("$\\beta$",fontsize=16, rotation='horizontal');
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

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Marginal distribution for $\alpha$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
log_p_alphas  = scipy.special.logsumexp(lr_zs,0)
log_p_alphas -= scipy.special.logsumexp(log_p_alphas)
map_alpha = alphas[log_p_alphas.argmax()]
print(map_alpha)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def log_trapz(log_y,x):
    dx = x[1:]-x[:-1]
    log_f = np.logaddexp(log_y[1:],log_y[:-1])
    return scipy.special.logsumexp(a=log_f, b= 0.5*dx)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
log_Z = log_trapz(log_p_alphas,alphas)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.plot(alphas, np.exp(log_p_alphas-log_Z) );
plt.axvline(map_alpha, color='orange');
plt.xlabel("$\\alpha$", fontsize=16);
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Marginal distribution for $\beta$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
log_p_betas  = scipy.special.logsumexp(lr_zs,1)
log_p_betas -= scipy.special.logsumexp(log_p_betas)
map_beta = betas[log_p_betas.argmax()]
print(map_beta)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
log_Z_beta = log_trapz(log_p_betas,betas)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.plot(betas, np.exp(log_p_betas-log_Z_beta) );
plt.axvline(map_beta, color='orange');
plt.xlabel("$\\beta$", fontsize=16);
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Monte-Carlo

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
import pymc as pm
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
lr_model = pm.Model()

with lr_model:
    alpha = pm.Flat("alpha")
    beta = pm.Flat("beta")

    z = beta*dose+alpha
    theta = pm.math.sigmoid(z)

    obs = pm.Binomial("obs", n=n_animals, p=theta, observed=n_deaths)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with lr_model:
    lr_MAP = pm.find_MAP()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
lr_MAP
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
with lr_model:
    lr_trace = pm.sample(tune=2000, draws=16000)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import arviz as az
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
az.plot_trace(lr_trace);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
ab_posterior = lr_trace.posterior.stack({'z':['chain','draw']})
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
ab_posterior['alpha'].shape
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots(figsize=figsize)
levels = np.log([0.01,0.1, 0.25, 0.5,0.75,0.9,1.0])
ax.contour(alphas, betas, lr_zs, levels= levels);
ax.hexbin(ab_posterior['alpha'], ab_posterior['beta']);
ax.scatter([ab_map[0]], [ab_map[1]], color='orange');
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

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Correlations

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
a_posterior=ab_posterior['alpha'].values
b_posterior=ab_posterior['beta'].values
ab_array = np.stack((a_posterior, b_posterior),1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
ab_array.shape
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
np.corrcoef(ab_array,rowvar=False)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## LD50 Dose

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\theta = \frac{1}{2}$$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$s(z)=\frac{1}{2}\qquad z=0$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\alpha +\beta x = 0,\quad x = -\frac{\alpha}{\beta}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig,ax = plt.subplots()
ax.hist(-ab_posterior['alpha']/ab_posterior['beta'],bins=100, histtype='step', density=True, color='red', linewidth=2)
ax.set_xlabel("LD50 [log g/ml]")
for x in dose:
    ax.axvline(x,linewidth=0.5)
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

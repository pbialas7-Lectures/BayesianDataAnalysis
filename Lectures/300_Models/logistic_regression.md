---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
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
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Analysis of bioassay experiment 
## (from "Bayesian Data Analysis" sec. 3.7)

```{code-cell}
---
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
plt.plot(dose, n_deaths/n_animals,'o')
plt.xlabel('dose [log g/ml]', fontsize=16);
plt.ylabel('$\\frac{\\# deaths}{\\# animals}$', rotation='horizontal', fontsize=16, labelpad=10);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Model

+++ {"slideshow": {"slide_type": "skip"}}

We assume that a given dose $x_i$ results in probability of death $\theta_i$ that is dependent on this dose

+++ {"slideshow": {"slide_type": "slide"}}

$$\theta_i=\theta(x_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

Given that the number of deaths in $n_i$ subjects will follow the  binomial distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$y_i|\theta_i \sim \operatorname{Bin}(n_i,\theta_i)$$

+++ {"slideshow": {"slide_type": "skip"}}

And the probability mass function is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(y_i|\theta_i,n_i)=\binom{n_i}{y_i}\theta_i^{y_i}(1-\theta_i)^{n_i-y_i}$$

+++ {"slideshow": {"slide_type": "slide"}}

## Logistic regression

+++ {"slideshow": {"slide_type": "skip"}}

Next  step is to assume some functional dependence of $\theta$ on $x$. The simplest possible would be the linear relation

+++ {"slideshow": {"slide_type": "fragment"}}

$\newcommand{\logit}{\operatorname{logit}}$
$$\theta_i = \alpha+\beta x_i $$

+++ {"slideshow": {"slide_type": "skip"}}

however this does not fulfill the constrain $0\le \theta_i \le 1$. To make it so we use the logistic sigmoid function

+++ {"slideshow": {"slide_type": "fragment"}}

$$z_i=\alpha+\beta x_i$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\theta_i=s(z_i)\equiv\frac{e^z_i}{1+e^z_i}$$

+++ {"slideshow": {"slide_type": "skip"}}

$z$ in this formula is often called _logit_. The sigmoid function is provided in `scipy` as `scipy.special.expit`

```{code-cell}
---
slideshow:
  slide_type: slide
---
from scipy.special import  expit
```

```{code-cell}
---
slideshow:
  slide_type: fragment
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

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\log P(\alpha,\beta|y,n,x) =\\ \sum_i \left(y_i\log s(\alpha+\beta x_i)^{y_i}+(n_i-y_i)\log (1-s(\alpha+\beta x_i))\right)+ const
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Numerical calculations

+++ {"slideshow": {"slide_type": "skip"}}

We start by importing the logarithm of logistic sigmoid function from `scipy.special`

```{code-cell}
---
slideshow:
  slide_type: fragment
---
from scipy.special import log_expit
```

+++ {"slideshow": {"slide_type": "skip"}}

We will also make use of the relation

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

+++ {"slideshow": {"slide_type": "skip"}}

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
slideshow:
  slide_type: slide
---
xs = np.linspace(-1,1,500)
plt.scatter(dose, n_deaths/n_animals)
plt.plot(xs,lr(xs,*ab_map))
plt.xlabel("$x\\; [g/ml]$");plt.ylabel("$\\theta$",rotation='horizontal');
```

+++ {"slideshow": {"slide_type": "skip"}}

It look as the choice of logistic regression as our model was justified.

```{code-cell}
---
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
slideshow:
  slide_type: skip
---
alphas = np.linspace(-5,10, 500)
betas  = np.linspace(-10,40,500)
a_mesh, b_mesh = np.meshgrid(alphas, betas)
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
lr_zs = log_P_alpha_beta_tensor(a_mesh, b_mesh, dose, n_deaths, n_animals)
lr_zs = lr_zs - lr_zs.max()
```

```{code-cell}
---
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
slideshow:
  slide_type: slide
---
fig
```

```{code-cell}
---
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
slideshow:
  slide_type: slide
---
fig
```

```{code-cell}
---
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
slideshow:
  slide_type: fragment
---
log_Z = log_trapz(log_p_alphas,alphas)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
plt.plot(alphas, np.exp(log_p_alphas-log_Z) );
plt.axvline(map_alpha, color='orange');
plt.xlabel("$\\alpha$", fontsize=16);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Sampling from the posterior distribution

+++ {"slideshow": {"slide_type": "slide"}}

### Ancestor sampling

+++ {"slideshow": {"slide_type": "-"}}

$$P(\alpha,\beta)=P(\beta|\alpha)P(\alpha)$$

```{code-cell}
---
slideshow:
  slide_type: slide
---
alphas_dist = scipy.stats.rv_discrete(0,len(log_p_alphas)-1,
               values=( range(len(log_p_alphas)), np.exp(log_p_alphas) ))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
p_alpha_beta=np.exp(lr_zs)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
b_dist=np.asarray(
    [scipy.stats.rv_discrete(0,len(betas)-1,
                values=(
                    range(len(betas)),p_alpha_beta[:,i]/p_alpha_beta[:,i].sum()
                             ))
                        for i in range(len(betas))  ] )
    
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def gen(x):
    return x.rvs(size=1)

gen = np.vectorize(gen)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
def gen_alpha_beta(n):
    da = (alphas[-1]-alphas[0])/(len(alphas-1))
    db =  (betas[-1]-betas[0])/(len(betas-1))
    ia = alphas_dist.rvs(size=n)
    als = alphas[ia]+scipy.stats.uniform(loc=da/2,scale=da).rvs(size=n)
    bes = betas[gen(b_dist[ia])]+scipy.stats.uniform(loc=db/2,scale=db).rvs(size=n)
    return np.stack((als,bes), axis=1)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
%%time
ab = gen_alpha_beta(250000)
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
ab
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
levels = np.log([0.01,0.1, 0.25, 0.5,0.75,0.9,1.0])
plt.contour(alphas, betas, lr_zs, levels= levels);
plt.hexbin(ab[:,0], ab[:,1]);
plt.scatter([ab_map[0]], [ab_map[1]], color='orange');
```

+++ {"slideshow": {"slide_type": "slide"}}

### LD50 Dose

+++ {"slideshow": {"slide_type": "fragment"}}

$$\theta = \frac{1}{2}$$

+++

$$s(z)=\frac{1}{2}\qquad z=0$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\alpha +\beta x = 0,\quad x = -\frac{\alpha}{\beta}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig,ax = plt.subplots()
ax.hist(-ab[:,0]/ab[:,1],bins=500, histtype='step', density=True, color='red')
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

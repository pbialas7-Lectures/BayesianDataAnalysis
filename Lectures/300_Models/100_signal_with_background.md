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
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Signal with background

> "Data Analysis, A Bayesian Tutorial" D.S.Sivia, J. Skilling

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In this notebook we will analyze a simple model of radioactive decay. We assume that we have a radioactive source that emits particles in a 1D space. The source is located at $x_0$ and emits particles with a rate $D(x)$. The rate is given by

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$D_k = n_0A\, e^{\displaystyle -\frac{1}{2w^2}(x_k - x_0)^2 }$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Additionally, we have a background source that emits particles at a constant rate $n_0 B$, so the total rate is

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$D_k = n_0 \left(A\, e^{\displaystyle -\frac{1}{2w^2}(x_k - x_0)^2 } +B \right)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Poisson distribution

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Given the rate $D_k$ the number of particles emitted in some time interval is given by a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution)

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(n|D) = e^{-D}\frac{D^n}{n!} $$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$X\sim Poisson(D)\qquad E[X]=D,\qquad var[X]=D$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
from scipy.stats import poisson
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
D = 5.77
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
ks = np.arange(0,20)
plt.stem(ks, poisson(D).pmf(ks), basefmt='C0');
plt.axvline(D, color='orange', label='mean')
plt.xlabel('n');
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Posterior distribution for Poisson distribution

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let $n_i$, $i=1,2,\ldots,M$ denote $M$ samples from the Poisson distribution with parameter $D$. Then assuming the uniform prior for parameter $D$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(D)=1$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

the posterior distribution is given by

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(D|\{n\}_i)=\prod_{i=1}^M e^{-D}\frac{D^{\displaystyle n_i}}{n_i!}\propto
e^{\displaystyle-M D}
\,  D^{\displaystyle \sum_{i=1}^M n_i}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

As a function of $D$ the posterior distribution is a [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$p(x|k,\theta) = \frac{1}{\Gamma(k)\theta^k}x^{k-1} e^{-\frac{x}{\theta}},
\qquad 
p(x|\alpha,\beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1} e^{-\beta x}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$D|n\sim \operatorname{Gamma}\left(k = 1+\sum_k n_k,\theta = \frac{1}{M}\right)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The Gamma distribution is a conjugate prior for Poisson distribution

+++

$$e^{-M D}
\,  D^{\sum_i n_i} 
\frac{1}{\Gamma(k)\theta^k}D^{k-1} e^{-\frac{D}{\theta}} 
\propto 
D^{\sum_i n_i+k-1} e^{-D(\frac{1}{\theta}+M)} 
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This is Gamma distribution with new parameters

+++

$$k'=\sum_{i=1}^M n_i+k,\qquad \theta'=\frac{1}{\theta^{-1}+M}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We regain the uniform prior with $k=1$ and $\theta=\infty$.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
np.random.seed(23487576)
M=7
nk_pois = poisson(D).rvs(size=M) 
print(nk_pois)
nk_sum = nk_pois.sum()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
from scipy.stats import gamma
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs = np.linspace(0,20,200)
ys = gamma(nk_sum,scale=1./M).pdf(xs)
plt.plot(xs,ys)
plt.axvline(D, color='red');
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Signal and background    posterior distribution for $A$ and $B$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

#### Sampling distribution/Likelihood

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$P(\{n_k\}|A,B)=\prod_{k=1}^M P(n_k|A,B)=\prod_{k=1}^M P(n_k|D_k(A,B)) $$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(A,B|\{n_k\})=P(A,B)\prod_{k=1}^M P(n_k|A,B)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Again we assume an uniform prior on $A$ and $B$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(A,B) = \begin{cases}1 & 
A\ge0, B\ge0\\
0 & \text{otherwise}
\end{cases}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\log P(A,B|\{n_k\})=\sum_k \log P(n_k|A,B)$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\log P(n_k|A,B)=n_k \log D_k(A,B)-D_k(A,B) -\log( n_k!)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

So let's simulate some measurements

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
w  = 2.12
x0 = 0
A_true = 1
B_true = 2
n0 = 32
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
def D_rate_scalar(x, A,B):
    b = 1.0/(2*w**w)
    return n0*(A*np.exp(-b*(x-x0)*(x-x0))+B)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

First we compute the decay rates in different positions $\{x\}_{k=1}^{13}=\{-6,-5,\ldots,6\}$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
xk = np.arange(-6,7)
dk = D_rate_scalar(xk,A_true, B_true) 
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.scatter(xk,dk, label='signal')
plt.axhline(n0*B_true,label='background')
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

From those rates we generate the actual counts from Poisson distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
nk = poisson(dk).rvs(size=len(dk))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
bottom = 60
plt.bar(xk,nk-bottom,bottom=bottom,label='measured counts')
plt.scatter(xk,dk,label='true activity')
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Given $A$, $B$ and counts $n_k$ the log posterior can be calculated as

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def log_pk(nk,A,B):
    lpk=0.0
    for k,x in enumerate(xk):
        dk = D_rate_scalar(x,A,B)
        lpk+=  np.log(dk)*nk[k] - dk
    return lpk            
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We will calculate the posterion numerically. To this end we choose discretised intervals on for $A$ and $B$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
As = np.linspace(0.0,2,500)
Bs = np.linspace(1,3,400)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We will need the $D_k(A,B)$ for all possible combinations of $k$, $A$ and $B$. This can be done using the `for` loops. It is not an efficient implementation

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
%%time
posterior = np.zeros((len(Bs), len(As)))
for i,a in enumerate(As):
    for j,b in enumerate(Bs):
        posterior[j,i] = log_pk(nk,a,b) 
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
posterior_n = posterior - posterior.max()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
idx = np.unravel_index(np.argmax(posterior), posterior.shape)
AB_map = As[idx[1]], Bs[idx[0]]
AB_map
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.contourf(As,Bs,posterior_n, levels=np.log(np.array([0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])))
ax.contour(As,Bs,posterior_n, levels=np.log(np.array([0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])), colors='black', 
           linestyles='-', linewidths=0.5)
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.scatter([A_true],[B_true], color='orange');ax.scatter(*AB_map, color='red')
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

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

A more efficient python calculation will have to the tensor capabilities of the `numpy` library. The loops can be substitudef for the `outer` functions.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

`numpy` provides a set of [such functions](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html#numpy.ufunc.outer): `op.outer(x,y)`. Those functions loops over all possible combination of values in $x$ and $y$ and calculate the `op` operation. 
For one dimensional arrays such function is equivalent to

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

```
# op.outer

out = np.empty(len(x),len(y))
for i,x_i in enumerate(x):
    for j,y_i in enumerate(y):
        out[i,j]=op(x_i,y_i)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Using those functions we can rewrite the rates functions as

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def D_rate_vector(x,A,B):
    b = 1.0/(2*w**w)
    A_xk = np.multiply.outer(A, np.exp(-b*(x-x0)*(x-x0))) 
    return n0*np.add.outer(B, A_xk)            
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and log posterior as

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
def log_pk_vector(A,B):
    dk = D_rate_vector(xk,A,B)
    lk =  np.log(dk)*nk.reshape(1,1,-1) - dk
    return lk.sum(-1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
%%time
zs = log_pk_vector(As, Bs)
nzs=zs-np.max(zs)
```

This function is about 30 times faster than the previous implementation.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
idx = np.unravel_index(np.argmax(zs), zs.shape)
AB_map=(As[idx[1]], Bs[idx[0]])
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
print(A_true, B_true)
print(AB_map)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.contourf(As,Bs,nzs, levels=np.log(np.array([0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])))
ax.contour(As,Bs,posterior_n, levels=np.log(np.array([0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])), colors='black', 
           linestyles='-', linewidths=0.5)
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.scatter([A_true],[B_true], color='orange');ax.scatter(*AB_map, color='red')
ax.grid()
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

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Marginal distributions

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
from scipy.special import logsumexp
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### $A$ distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
ys = np.exp(logsumexp(nzs,axis=0))
A_map = As[ys.argmax()]
print(A_map)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
AB_map
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.plot(As,ys/np.trapezoid(ys, As))
ax.grid()
ax.axvline(A_true, color='orange');ax.axvline(AB_map[0],color='blue');ax.axvline(A_map,color='red')
ax.set_xlabel("A");
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### $B$ distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
ys = np.exp(logsumexp(nzs,axis=1))
B_map = Bs[ys.argmax()]
print(B_map)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
AB_map
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.plot(Bs,ys/np.trapezoid(ys, Bs))
ax.grid()
ax.axvline(B_true, color='orange');ax.axvline(AB_map[1],color='red');ax.axvline(B_map,color='red')
ax.set_xlabel("B");
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

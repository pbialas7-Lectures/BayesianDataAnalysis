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
import numpy as np
import scipy as sp
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "slide"}}

### Signal with background

> "Data Analysis, A Bayesian Tutorial" D.S.Sivia, J. Skilling

+++ {"slideshow": {"slide_type": "slide"}}

$$D_k = n_0 \left(A\, e^{\displaystyle -\frac{1}{2w^2}(x_k - x_0)^2 } +B \right)$$

+++ {"slideshow": {"slide_type": "slide"}}

### Poisson distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n|D) = e^{-D}\frac{D^n}{n!} $$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(D|n)=\prod_{k=1}^M e^{-D}\frac{D^{\displaystyle n_k}}{n_k!}\propto e^{\displaystyle-M D+\log D\sum_{k=1}^M n_k}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(D|n)\propto e^{\displaystyle-M D+\log D\sum_{k=1}^M n_k}=
e^{\displaystyle-M D} D^{\displaystyle \sum_k n_k}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$D|n\sim \operatorname{Gamma}\left(1+\sum_k n_k,\frac{1}{M}\right)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
np.random.seed(23487576)
lb = 5.77
M=7
nk_pois = st.poisson(lb).rvs(size=M) 
print(nk_pois)
nk_sum = nk_pois.sum()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs = np.linspace(0,20,200)
ys = st.gamma(nk_sum,scale=1./M).pdf(xs)
plt.plot(xs,ys)
plt.axvline(lb, color='red');
```

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\{n_k\}|A,B)=\prod_{k=1}^M P(n_k|A,B)=\prod_{k=1}^M P(n_k|D_k(A,B)) $$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A,B|\{n_k\})=P(A,B)\prod_{k=1}^M P(n_k|A,B)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$P(A,B) = \begin{cases}1 & 
A\ge0, B\ge0\\
0 & \text{otherwise}
\end{cases}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log P(A,B|\{n_k\})= \log P(A,B) +\sum_k \log P(n_k|A,B)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$\log P(n_k|A,B)=n_k \log D_k-D_k -\log( n_k!)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
w  = 2.12
x0 = 0
A_true = 1
B_true = 2
n0=32
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
def D_rate(x,A,B):
    b = 1.0/(2*w**w)
    return n0*(np.multiply.outer(np.exp(-b*(x-x0)*(x-x0)),A)+B)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
D_rate(np.array([1,2,3]),np.array([1,2]),np.array([1,2]) )
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xk = np.arange(-6,7)
dk = D_rate(xk,A_true, B_true) 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
nk = st.poisson(dk).rvs(size=len(dk))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.step(xk,dk, where='mid')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.plot(xk,nk, drawstyle='steps-mid')
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
def log_pk(A,B):
    dk = D_rate(xk,A,B)
    dim = len(dk.shape)
    sh=np.ones(dim).astype('int')
    sh[0]=-1
    lk =  np.log(dk)*nk.reshape(sh) - dk
    return lk

def log_p(A,B):
    lk = log_pk(A,B)
    return lk.sum(axis=0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
As = np.linspace(0.0,2,200)
Bs = np.linspace(1,3,200)
xs, ys = np.meshgrid(As,Bs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
zs = log_p(xs,ys)
nzs=zs-np.max(zs)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
idx = np.unravel_index(np.argmax(zs), zs.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(xs[idx], ys[idx])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.contourf(xs,ys,nzs, levels=np.log(np.array([0.001,0.01,0.1, 0.3, 0.5, 0.7, 0.9,1])))
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.scatter([A_true],[B_true], color='red')
plt.show()
```

```{code-cell} ipython3
from scipy.special import logsumexp
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ys = np.exp(logsumexp(nzs,axis=0))
ax.plot(As,ys/np.trapz(ys, As))
ax.grid()
ax.axvline(A_true, color='red')
ax.set_xlabel("A")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
ys = np.exp(logsumexp(nzs,axis=1))
ax.plot(Bs,ys/np.trapz(ys, Bs))
ax.grid()
ax.axvline(B_true, color='red')
ax.set_xlabel("B")
```

```{code-cell} ipython3

```

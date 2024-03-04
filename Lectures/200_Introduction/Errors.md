---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import beta
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import sys
sys.path.append('../../src/')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
dc='#1f77b4' #default color
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import billiard as bl
import bda.stats as bst
```

+++ {"slideshow": {"slide_type": "slide"}}

# Errors

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
p =  np.pi/10.0
y =  0.786
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
np.random.seed(87)
x = st.uniform(loc=0, scale=1).rvs(size=(100,2))
left=(x[:,0]<=p) + 0
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
prior = np.vectorize(lambda x: 1.0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
beta_posteriors=[]
for i in range(1,101):
    n_l = left[:i].sum()
    n_r = i-n_l
    beta_posteriors.append(beta(n_l+1, n_r+1))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Standard deviation

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

So far we did not consider any errors on our estimators. As we have the whole posterior distribution at our disposal, there are many possible ways to define errors. One obvious way would be us use the standard deviation $\sigma$  of the posterior distribution.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\begin{split}
\mu &=\int_0^1\text{d}p\, p\, P_{post}(p)\\
\sigma^2 &= \int_0^1\text{d}p (p-\mu)^2 P_{post}(p)
\end{split}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This of course assumes that we are using the mean $\mu$  as the estimate.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

For $Beta(\alpha,\beta)$ distribution the variance is

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
def var(a, b):
    return a*b/((a+b)**2*(a+b+1))

def stats(left):
    n = len(left)
    n_l = left.sum()
    n_r = n-n_l
    a = n_l+1;
    b = n_r +1
    mu = a/(a+b)
    s = np.sqrt(var(a,b))
    return n_l, n_r, mu,s
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
for i in [1,2,3,100]:
    n_l, n_r, mu, s = stats(left[:i])
    print(f"{n_l+n_r:3d} {n_l:3d} {n_r:3d} {mu:6.3f} {s:6.3f}")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
nb=3;n=3
xs=np.linspace(0,1,500)
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
alpha=1
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);
n_l, n_r, mu, s = stats(left[:n])
post = beta_posteriors[n-1]
p_mean = post.mean()
pax.axvline(p_mean)

pax.fill_between(xs,beta_posteriors[2].pdf(xs),0, where = (xs>p_mean-s) & (xs<p_mean+s), alpha=0.5);
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Standard deviation around mode

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
n_l, n_r, mu, s = stats(left[:n])
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
s = np.sqrt(bst.var_central_f(post.pdf,p_map,0,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
nb=3;n=3
xs=np.linspace(0,1,500)
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
alpha=1
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');

pax.fill_between(xs,beta_posteriors[2].pdf(xs),0, where = (xs>p_map-s) & (xs<p_map+s), alpha=0.5);
plt.close();
```

```{code-cell} ipython3
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Standard deviation around median

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
n_l, n_r, mu, s = stats(left[:n])
post = beta_posteriors[n-1]
p_median = bst.median_f(post.pdf,0,1)
s = np.sqrt(bst.var_central_f(post.pdf,p_median,0,1))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
nb=3;n=3
xs=np.linspace(0,1,500)
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
alpha=1
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);

pax.axvline(p_median,color='orange')
pax.fill_between(xs,beta_posteriors[2].pdf(xs),0, where = (xs>p_median-s) & (xs<p_median+s), alpha=0.5);
plt.close();
```

```{code-cell} ipython3
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

## Confidence interval

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from bda.stats import  cdi_left_f, cdi_central_f
```

+++ {"slideshow": {"slide_type": "skip"}}

Another way to express uncertainty is to provide a _confidence interval_. We will call $\beta$-confidence interval any interval $[a,b]$ such that the probability that $p$ lies in this interval is equal to $\beta$.

+++ {"slideshow": {"slide_type": "slide"}}

$$P(p\in [a,b])=\beta$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
nb=3;n=3
xs=np.linspace(0,1,500)
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
#plot_balls(ax, nb, x,left, cs)
alpha=1
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
_,l,r=cdi_left_f(post.pdf, 0.0,0.75)
pax.fill_between(xs,beta_posteriors[2].pdf(xs),0, where = (xs>l) & (xs<r), alpha=0.5);
_,l,r=cdi_left_f(post.pdf, 0.2,0.75)
pax.fill_between(xs,beta_posteriors[2].pdf(xs),0, where = (xs>l) & (xs<r), alpha=0.5);
ax.set_title(f"Confidence intervals $\\beta=0.75$")
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
  slide_type: skip
tags: [aux_code]
---
from scipy.optimize import brentq
nb=3;n=3
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
#plot_balls(ax, nb, x,left, cs)
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);

n_l = left[:n].sum(); n_r = n- n_l;
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
_,l,r=cdi_central_f(post.pdf, p_map,0.75,0,1)
pax.axvline(l, color='lightblue')
pax.axvline(r, color='lightblue')
pax.fill_between(xs,post.pdf(xs),0, where = (xs>l) & (xs<=r),alpha=0.5, color='lightblue');
yl=post.pdf(l)
pax.axhline(yl)
rr=r+0.05
pax.fill_between(xs,post.pdf(xs),0, where = (xs>=r) & (xs<rr),alpha=0.5, color='red');
yrr=post.pdf(rr)
pax.axhline(yrr, color='red')
xm = brentq(lambda x: post.pdf(x)-yrr,l,r)
pax.fill_between(xs,post.pdf(xs),0, where = (xs>=l) & (xs<xm),alpha=0.5, color='orange');
ax.set_title("Confidence interval symmetric around $p_{MAP}$")
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

## Highest density region (HDR)

+++ {"slideshow": {"slide_type": "skip"}}

$\beta$ HDR is a region where at least $\beta$ of probability is concentrated and has smallest possible volume in the parameter space, hence highest density. More formal definition given below.

+++ {"slideshow": {"slide_type": "skip"}}

Let $P_X(x)$ be de density function of  some random variable $X$ with values in $R$. Let' $R_X(r)$ be the subset of $R_X$ such  that

+++ {"slideshow": {"slide_type": "skip"}}

$$ R_X(r) = \{x\in R: P_X(x)\ge r\}$$

+++ {"slideshow": {"slide_type": "skip"}}

The $\beta$ HDR is equal to $R(r_\beta)$ where $r_\beta$ is the largest constant such that

+++ {"slideshow": {"slide_type": "skip"}}

$$P\left(x\in R(r_\beta)\right)\ge \beta$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from bda.hdr_plot import plot_All
from ipywidgets import interact
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
interactive_plot = interact(lambda r: plot_All(beta_posteriors[2].pdf,r),r=(0.0,1.9,0.001))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from bda.stats import hdr_f
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
nb=3;n=3
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
pax.axvline(p, color='black')
alpha=1    
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);
n_l = left[:n].sum(); n_r = n- n_l;
n_l = left[:n].sum(); n_r = n- n_l;
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
lr,_,_=hdr_f(post.pdf, 0.95,a=0,b=1)
pax.fill_between(xs,post.pdf(xs),0, where = (xs>lr[0]) & (xs<lr[1]), alpha=0.5);
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

and here is the HDR(0.95)  after 100 balls have been thrown

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
nb=100;n=100
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,10))
pax.axvline(p, color='black')
alpha=1    
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);
n_l = left[:n].sum(); n_r = n- n_l;
n_l = left[:n].sum(); n_r = n- n_l;
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
lr,_,_=hdr_f(post.pdf, 0.95,a=0,b=1)
pax.fill_between(xs,post.pdf(xs),0, where = (xs>lr[0]) & (xs<lr[1]), alpha=0.5);
plt.close()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Female births (France)

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
f_births = 241945  
m_births = 251527
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
f_post = beta(f_births+1, m_births+1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
f_births/(f_births+m_births)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
f_post.mean()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
f_post.std()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
(0.5-f_post.mean())/f_post.std()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fig,ax = plt.subplots()
ps = np.linspace(0.48,0.51,1000)
ax.plot(ps, f_post.pdf(ps));
b=1-1e-10
hdr,_,_=hdr_f(f_post.pdf,.1,a=0.48,b=0.51)
ax.fill_between(ps,f_post.pdf(ps),0,where = (ps>hdr[0]) & (ps<hdr[1]),  alpha=0.5);
ax.text(.4930,500,f"$\\beta={b:.12g}$\n $[{hdr[0]:.4f} {hdr[1]:.4f}]$")
ax.axvline(0.5,c='orange')
plt.close();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

$$ P\left(p_f>=\frac{1}{2}\right) = \frac{\int_{\frac{1}{2}}^1 p_f^{241945}(1-p_f)^{251527}\text{d}p_f}
{\int_{0}^1  p_f^{241945}(1-p_f)^{251527}\text{d}p_f} \approx 1.15 \times 10^{-42}$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

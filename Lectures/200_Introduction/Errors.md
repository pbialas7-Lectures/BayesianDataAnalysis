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
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.stats import beta
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
import sys
sys.path.append('../../src/')
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
dc='#1f77b4' #default color
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
import billiard as bl
import bda.stats as bst
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Errors

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This is the position of the initial ball

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
p =  np.pi/10.0
y =  0.786
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We throw 100 balls and check wheter they land on the left of the initial ball

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
np.random.seed(87)
x = st.uniform(loc=0, scale=1).rvs(size=(100,2))
left=(x[:,0]<=p) + 0
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We assume the initial uniform prior ($\alpha=\beta=1$)

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
prior = np.vectorize(lambda x: 1.0)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
beta_posteriors=[]
for i in range(1,101):
    n_l = left[:i].sum()
    n_r = i-n_l
    beta_posteriors.append(beta(n_l+1, n_r+1))
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Standard deviation

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

So far we did not consider any errors on our estimators. As we have the whole posterior distribution at our disposal, there are many possible ways to define errors. One obvious way would be us use the standard deviation $\sigma$  of the posterior distribution.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\newcommand{\avg}[1]{\left\langle#1\right\rangle}$$
$$\begin{split}
\avg{p} &=\int_0^1\text{d}p\, p\, P_{post}(p)\\
\sigma^2 &= \int_0^1\text{d}p (p-\avg{p})^2 P_{post}(p)
\end{split}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This of course assumes that we are using the mean $\avg{p}$  as the estimate.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

For $Beta(\alpha,\beta)$ distribution the mean is

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$\avg{p} = \frac{\alpha}{\alpha+\beta}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which gives for the posterior mean

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\avg{p} = \frac{n_l+\alpha}{n_l+n_r+\alpha+\beta}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

For $Beta(\alpha,\beta)$ distribution the variance is

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

giving the posterior variance

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
\sigma^2 &= \frac{(n_l+\alpha)(n_r+\beta)}{(\alpha+\beta+n_l+n_r)^2(\alpha+\beta+n_l+n_r+1)}\\&
=\frac{
\avg{p}(1-\avg{p})}{
\alpha+\beta+n_l+n_r+1}
\end{split}
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For uniform prior we obtain

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$\avg{p} = \frac{n_l+1}{n_l+n_r+2}$$

+++ {"slideshow": {"slide_type": ""}, "editable": true}

$$\sigma^2 = \frac{
\avg{p}(1-\avg{p})}{n_l+n_r+3}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
def var(a, b):
    """Calculates the variance of Beta(a,b) distribution
    """
    return a*b/((a+b)**2*(a+b+1))

def stats(left):
    """Given an array containing one when ball landed to the left, and zero if on the right
calculates the posterior mean and variance    
    """
    n = len(left)
    n_l = left.sum()
    n_r = n-n_l
    a = n_l+1;
    b = n_r +1
    mu = a/(a+b)
    s = np.sqrt(var(a,b))
    return n_l, n_r, mu,s
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
print(f"True position = {p:.5f}")
for i in [1,2,3,50,100]:
    n_l, n_r, mu, s = stats(left[:i])
    print(f"{n_l+n_r:3d} {n_l:3d} {n_r:3d} {mu:6.3f} {s:5.3f} ({mu-p:.4f})")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Below we plot the $\pm\sigma$ interval around mean

```{code-cell}
---
editable: true
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Standard deviation around mode

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In the same way we can define standard deviation around mode i.e. the MAP estimator

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$
\sigma^2 = \int_0^1\text{d}p (p-p_{MAP})^2 P_{post}(p)
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Please recall that for Beta distribution

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$p_{MAP}=\frac{\alpha-1}{\alpha+\beta-2}$$

+++ {"editable": true, "slideshow": {"slide_type": "notes"}}

translating into posterior mode

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_{MAP}=\frac{n_l}{n_l+n_r}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
n_l, n_r, mu, s = stats(left[:n])
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
s = np.sqrt(bst.var_central_f(post.pdf,p_map,0,1))
```

```{code-cell}
---
editable: true
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Standard deviation around median

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

And finally we can use median which for Beta distribution is approximatelly

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$p_{med.}\approx\frac{\alpha-\frac{1}{3}}{\alpha+\beta-\frac{2}{3}}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
n_l, n_r, mu, s = stats(left[:n])
post = beta_posteriors[n-1]
p_median = bst.median_f(post.pdf,0,1)
s = np.sqrt(bst.var_central_f(post.pdf,p_median,0,1))
```

```{code-cell}
---
editable: true
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Confidence interval

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.stats import  cdi_left_f, cdi_central_f
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Another way to express uncertainty is to provide a _confidence interval_. We will call $\beta$-confidence interval any interval $[a,b]$ such that the probability that $p$ lies in this interval is equal to $\beta$.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(p\in [a,b])=\beta$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Below we construct two such possible intervals

```{code-cell}
---
editable: true
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

One way of choosing a confidence interval is to make it symmetric around $p_{MAP}$

```{code-cell}
---
editable: true
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The problem with such an interval is that the probabilities on the edges are not equal. If we look at the picture above we can see that the yellow are which is included in the interval corresponds to smaller probability then the excluded red area to the right.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Highest density region (HDR)

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

$\beta$ HDR is a region where at least $\beta$ of probability is concentrated and has smallest possible volume in the parameter space, hence highest density. More formal definition given below.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Let $P_X(x)$ be de density function of  some random variable $X$ with values in $R$. Let' $R_X(r)$ be the subset of $R_X$ such  that

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$ R_X(r) = \{x\in R: P_X(x)\ge r\}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The $\beta$ HDR is equal to $R(r_\beta)$ where $r_\beta$ is the largest constant such that

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P\left(x\in R(r_\beta)\right)\ge \beta$$

```{code-cell}
---
slideshow:
  slide_type: skip
---
from bda.hdr_plot import plot_All
from ipywidgets import interact
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
interactive_plot = interact(lambda r: plot_All(beta_posteriors[2].pdf,r),r=(0.0,1.9,0.001))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.stats import hdr_f
```

```{code-cell}
---
editable: true
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and here is the HDR(0.95)  after 100 balls have been thrown

```{code-cell}
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

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Female births (France)

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's now take the problem considered by Laplace in 18th century. The problem was to check wheter there is less female births then male.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The data he had was the following

```{code-cell}
---
slideshow:
  slide_type: fragment
---
f_births = 241945  
m_births = 251527
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

based on that and assuming flat prior the posterior distribution is $Beta(n_f+1,n_m+1)$

```{code-cell}
---
slideshow:
  slide_type: fragment
---
f_post = beta(f_births+1, m_births+1)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

with mode

```{code-cell}
---
slideshow:
  slide_type: slide
---
f_births/(f_births+m_births)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and mean

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
f_post.mean()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The standard deviation is

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
f_post.std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Value $\frac{1}{2}$ is well outside the errors around the mean,it is over 13 standard deviations away

```{code-cell}
---
slideshow:
  slide_type: fragment
---
(0.5-f_post.mean())/f_post.std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Next we will calculate the HDR with $\beta=1-10^{-9}$, that is an interval such that the probabilty that percentage of female births lies within this interval is  $1-10^{-9}$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig,ax = plt.subplots()
ps = np.linspace(0.485,0.505,10000)
ax.plot(ps, f_post.pdf(ps));
b=1-1e-9
hdr,_,_=hdr_f(f_post.pdf,b,a=0.48,b=0.51)
ax.fill_between(ps,f_post.pdf(ps),0,where = (ps>hdr[0]) & (ps<hdr[1]),  alpha=0.5);
ax.text(.4930,500,f"$\\beta={b:.12g}$\n $[{hdr[0]:.4f} {hdr[1]:.4f}]$")
ax.axvline(0.5,c='orange')
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

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Again we see that the value one half lies far beyond this interval.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

And finally the probability that there are more females births then males is

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
P\left(p_f>=\frac{1}{2}\right) &= \frac{\int_{\frac{1}{2}}^1 p_f^{241945}(1-p_f)^{251527}\text{d}p_f}
{\int_{0}^1  p_f^{241945}(1-p_f)^{251527}\text{d}p_f}\\
& \approx 1.15 \times 10^{-42}
\end{split}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which is astronomicaly small.

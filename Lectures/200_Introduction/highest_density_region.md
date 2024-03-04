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
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
```

```{code-cell} ipython3

```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
import sys
sys.path.append('../../src/')
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Highest density region (HDR)

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs = np.linspace(-10,20,1000)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: '-'
---
from scipy.stats import norm
dist = norm(4,2).pdf(xs)
dist += norm(-2,1).pdf(xs)
dist/=  np.trapz(dist,xs)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax =plt.subplots()
ax.plot(xs, dist);
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$\beta$-HDR is a region where at least $\beta$ of probability is concentrated and has smallest possible volume in the sample space, hence highest density. 

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

More formal definition given below.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

Let $P_X(p)$ be de density function of  some random variable $X$ with values in $R_X$. Let $R_X(p)$ be the subsets of $R_X$ such  that

+++ {"slideshow": {"slide_type": "-"}, "editable": true}

$$ R(p) = \{x\in R_X: P_X(x)\ge p\}$$

+++ {"slideshow": {"slide_type": "-"}, "editable": true}

The $\beta$ HDR is equal to $R(p_\beta)$ where $p_\beta$ is the largest constant such that

+++ {"slideshow": {"slide_type": "-"}}

$$P\left(x\in R(p_\beta)\right)\ge \beta$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax =plt.subplots()
ax.plot(xs, dist);
```

```{code-cell} ipython3
from bda.stats import R_d, PofR_d, hdr_d
from bda.hdr_plot import plot_rp_d
```

```{code-cell} ipython3
hdr_d(xs,dist,0.9)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
from ipywidgets import interactive, FloatSlider
f=lambda p: plot_rp_d(xs,dist,p)
interactive_plot = interactive(f, p=FloatSlider(min=0.0, max=0.25,step=1e-4, value=0.22, readout_format=".3f"))
output = interactive_plot.children[-1]
#output.layout.height = '650px'
interactive_plot
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots()
ax.plot(xs, dist);
Rx,p,mass=hdr_d(xs,dist,.9)
for itr in Rx.reshape(-1,2):
    ax.fill_between(xs,dist,0, where = ( (xs>itr[0]) & (xs<itr[1])),color='lightgray');
ax.axhline(p, linewidth=1)    
ax.text(.7,0.8,f"$p={p:.3f}$ ${mass:.2f}$", fontsize=14, transform=ax.transAxes);
plt.close()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
from scipy.stats import beta
```

```{code-cell} ipython3
ps=np.linspace(0,1,1000)
fig,ax=plt.subplots()
ax.plot(ps, beta(20,5).pdf(ps));
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: slide
---
from bda.stats import hdr_f
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
hdr,_,_ = hdr_f(beta(a=20,b=5).pdf,beta=0.95,a=0,b=1)
ps=np.linspace(0,1,1000)
fig,ax=plt.subplots()
ax.plot(ps, beta(20,5).pdf(ps));
ax.fill_between(ps, beta(20,5).pdf(ps),0, where=((ps>hdr[0]) & (ps<hdr[1])), color='lightgray');
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

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
  slide_type: ''
---
%load_ext autoreload
%autoreload 2
```

```{code-cell}
import numpy as np
import scipy as sp
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
from matplotlib.patches import Arc, FancyArrowPatch
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
import lighthouse as lh
```

# Lighthouse

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This problem is taken from "Data Analysis, a Bayesian Tutorial" by D.S. Silva with J. Skiling. A lighthouse distance $h=1$ from the shore is rotating with constant angular frequency and emitting thin beams of light at random. The probability of emission is uniform in time. The signals are picked up on the shore by an array of detectors and their location is saved in the file `lighthouse.txt`.  The horizontal location of the lighthouse $x_{lh}$ is unknown. The task is to estimate this position.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
h=1
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
flash_x = np.loadtxt('lighthouse.txt')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The figure below  presents the geometry of the problem. The orange dot indicates the lighthouse. Blue dots are the points were the flashes were recorded. The directions of the first 10 flashes are shown as lightblue lines.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x0=10;
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
fig,ax=plt.subplots();
ax.set_ylim(0,1.1)
ax.set_xlim(-350,200)
aspect = lh.get_aspect(ax)

ax.scatter(flash_x,np.zeros_like(flash_x), zorder=10, c='blue');
ax.axvline(x0, color='black')

for i in range(10):
    ax.plot([flash_x[i],x0],[0,1], c='lightblue', zorder=-10)    
ax.scatter([x0],[1],s=200, zorder=10, c='orange');
thetas=np.arctan((flash_x[:10]-x0)*aspect)
arc3=Arc((x0,h),width=0.75/aspect,height=0.75, angle=0, theta1=np.rad2deg(-np.pi), theta2=0,
         edgecolor='black', linewidth=0.5);
ax.add_patch(arc3);
ax.scatter(*lh.polar2xy((x0,h),-np.pi/2+thetas,0.75/(2*aspect), aspect=aspect));
xy = lh.polar2xy((x0,h),-np.pi/2+np.pi/30,0.85/(2*aspect), aspect=aspect)
ax.annotate(r"$\phi$",xy, fontsize=24);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Sampling distribution

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Under those conditions the sampling distribution is the Cauchy distribution with probability density function

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p(x|x_{lh},h)=\frac{h}{\pi}\frac{1}{\left(x-x_{lh}\right)^2+h^2}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In the following we will drop the subscript $X$ on $p$.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Assuming an uniform prior on interval $[-x_{lim}, x_{lim})$ with $x_{lim}=100$ for $x_{lh}$  plot the posterior distribution after performing one measurement.  Represent posterior density function by an array $p_{post}(xs_i)$ where $xs_i$ are uniformly distributed on the interval  $[-x_{lim}, x_{lim})$:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x_lim=100.0
xs=np.linspace(-x_lim, x_lim,5000)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Normalize the $p(xs_i)$ such that

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\sum_i p(xs_i)=1$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Find and printout the maximal a posteriori (MAP) estimate for $x_{lh}$.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Add the plot of the posterior after two measurments. 

**Hint** perform same calculations as in problem 1, but use the calculated posterior as the new prior. Calculate the MAP estimate.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}, "raw_mimetype": ""}

Write an iterative procedure that calculates all the posteriors ans all MAP estimates corresponding to $1,\ldots,100$ measurements. Plot first 10 and last 10 posterior distributions. What is the final MAP? Find 95% probability symmetric interval around this value.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 4

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Write a procedure that takes as the input $xs$ array, the prior array,  array of measurments and returns the posterior "in one go". 

**Hint** it is easier to work with logarithms of probabilities. For normalizing you can use `logsumexp` function from `scipy.special`. If you need to use a loop in your code, please loop over the measurments.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
from scipy.special import logsumexp
```

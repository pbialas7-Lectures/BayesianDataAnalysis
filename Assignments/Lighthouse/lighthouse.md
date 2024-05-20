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

We will start by  calculating the probability density function for distribution of  flashes. But first I will recall some definitions.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Continuous random variables

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Univariate  (one dimensional)

+++ {"editable": true, "slideshow": {"slide_type": ""}}

By continous random variables we will understand variables with have a connected subset  $S$ of $\mathbb{R}$ e.g. an interval as the outcome set.

+++

When the set of the outcomes is not countable _i.e._ we cannot enumerate them, we cannot  specify probability of the event by adding probabilities of elementary events it contains.  Actually for most of the interesting continous random variables the probability of a single outcome is zero

+++

$$P(X=x) = 0$$

+++

### Cummulative distribution function

+++

However we can ask for the probability that the outcome is smaller then some number:

+++

$$F_X(x) = P(X\le x)$$

+++

This is called a cummulative distribution function (cdf) or _cummulant_.

+++

### Probability density function

+++

We can also ask for the probability that the outcome lies in a small interval $\Delta x$

+++

$$P(x<X\le x+\Delta x)$$

+++

For small intervals and "well behaved" random variables we expect that this probability will be proportional to $\Delta x$, so let's take the ratio and go to the limit $\Delta x\rightarrow 0$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\frac{P(x<X<x+\Delta x)}{\Delta x}\underset{\Delta x\rightarrow 0}{\longrightarrow} p_X(x)$$

+++

If this limit exists it's called probability density function (pdf).

+++

There is a relation between cdf and pdf

+++

$$ p_X(x) =\frac{\text{d}}{\text{d}x}F_X(x)\qquad F_X(x) = \int\limits_{-\infty}^x p_X(x')\text{d}x'$$

+++

## Sampling distribution -- cont

+++

Let $X$ be the random variable corresponding to flash location and we are looking for $p_X(x)$. It is easier to start with the cummulative distribution function

+++

$$F_X(x)=P(X<x)=P(\phi(X)<\phi(x))$$

+++

where $\phi(x)$ is the angle measured from the vertical line to the beam direction at $x$

+++

$$\phi(x)=\arctan\left(\frac{x-x_{lh}}{h}\right)$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Using the fact that the distribution of angles is uniform we obtain

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$P(\phi(X)<\phi(x))=\frac{1}{\pi}\left(\phi(x)+\frac{\pi}{2}\right)$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

so finally

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$F_X(x)=\frac{1}{\pi}\left(\arctan\left(\frac{x-x_{lh}}{h}\right)+\frac{\pi}{2}\right)$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Differentating with respect to $x$ we obtain the probability density function

+++

$$p_X(x)=\frac{h}{\pi}\frac{1}{\left(x-x_{lh}\right)^2+h^2}$$

+++

In the following we will drop the subscript $X$ on $p$.

+++

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Assuming an uniform prior on interval $[-x_{lim}, x_{lim})$ with $x_{lim}=100$ for $x_{lh}$  plot the posterior distribution after performing one measurment.  Represent posterior density function by an array $p_{post}(xs_i)$ where $xs_i$ are uniformly distributed on the interval  $[-x_{lim}, x_{lim})$:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x_lim=100.0
xs=np.linspace(-x_lim, x_lim,50000)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Normalize the $p(xs_i)$ such that

+++

$$\sum_i p(xs_i)=1$$

+++

Find the maximal a posteriori (MAP) estimate for $x_{lh}$.

+++

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Add the plot of the posterior after two measurments. *Hint* perform same calculations as in problem 1, but use the calculated posterior as the new prior. Caclculate the MAP estimate.

+++

## Problem 3

```{raw-cell}
Write an iterative procedure that calculates all the posteriors ans all MAP estimates corresponding to $1,\ldots,100$ measurements. Plot first 10 and last 10 posterior distributions. What is the final MAP? Find 95% probability interval around this value. 
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 4

+++

Write a procedure that takes as the input $xs$ array, the prior array,  array of measurments and returns the posterior "in one go". *Hint* it is easier to work with logarithms of probabilities. For normalizing you can use `logsumexp` function from `scipy.special`. If you need to use a loop in your code, please loop over the measurments.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
from scipy.special import logsumexp
```

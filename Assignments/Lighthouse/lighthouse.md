---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
import numpy as np
import scipy as sp
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
from matplotlib.patches import Arc, FancyArrowPatch
```

```{code-cell} ipython3
import lighthouse as lh
```

# Lighthouse

+++

This problem is taken from "Data Analysis, a Bayesian Tutorial" by D.S. Silva with J. Skiling. A lighthouse distance $h=1$ from the shore is rotating with constant angular frequency and emitting thin beams of light at random. The probability of emission is uniform in time. The signals are picked up on the shore by an array of detectors and their location is saved in the file `lighthouse.txt`.  The horizontal location of the lighthouse $x_{lh}$ is unknown. The task is to estimate this position.

```{code-cell} ipython3
h=1
```

```{code-cell} ipython3
flash_x = np.loadtxt('lighthouse.txt')
```

The figure below  presents the geometry of the problem. The orange dot indicates the lighthouse. Blue dots point were the flashes were recorded. The directions of the first 10 flashes are shown in lighblue.

```{code-cell} ipython3
x0=10;
```

```{code-cell} ipython3
fig,ax=plt.subplots();
ax.set_ylim(0,1.1)
ax.set_xlim(-350,200)
aspect = lh.get_aspect(ax)

ax.scatter(flash_x+x0,np.zeros_like(flash_x), zorder=10, c='blue');
ax.axvline(x0, color='black')

for i in range(10):
    ax.plot([flash_x[i],x0],[0,1], c='lightblue')    
ax.scatter([x0],[1],s=200, zorder=10, c='orange');
thetas=np.arctan((flash_x[:10]-x0)*aspect)
arc3=Arc((x0,h),0.75/aspect,0.75, 0, np.rad2deg(-np.pi), 0,edgecolor='black', linewidth=0.5);ax.add_patch(arc3);
ax.scatter(*lh.polar2xy((x0,h),-np.pi/2+thetas,0.75/(2*aspect), aspect=aspect));
xy = lh.polar2xy((x0,h),-np.pi/2+np.pi/30,0.85/(2*aspect), aspect=aspect)
ax.annotate("$\phi$",xy, fontsize=24);
```

## Sampling distribution

+++

We will start by  caclulating the probability density function for distribution of  flashes. But first I will recall some definitions

+++

## Continuous random variables

+++

### Univariate  (one dimensional)

+++

By continous random variables we will understand variables with have a connected subset of $S\in \mathbb{R}$ e.g. an interval as the outcome set.

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

+++

from scipy.special import logsumexp$$\frac{P(x<X<x+\Delta x)}{\Delta x}\underset{\Delta x\rightarrow 0}{\longrightarrow} P_X(x)$$

+++

If this limit exists it's called probability density function (pdf).

+++

There is a relation between cdf and pdf

+++

$$ P_X(x) =\frac{\text{d}}{\text{d}x}F_X(x)\qquad F_X(x) = \int\limits_{-\infty}^x P_X(x')\text{d}x'$$

+++

## Sampling distribution -- cont

+++

Let $X$ be the random variable corresponding to flash location and we are looking for $P_X(x)$. It is easier to start with the cummulative distribution function

+++

$$F_X(x)=P(X<x)=P(\phi(X)<\phi(x))$$

+++

where $\phi(x)$ is the angle measured from the vertical line to the beam direction at $x$

+++

$$\phi(x)=\arctan\left(\frac{x-x_{lh}}{h}\right)$$

+++

Using the fact that the distribution of angles is uniform we obtain

+++

$$P(\phi(X)<\phi(x))=\frac{1}{\pi}\left(\phi(x)-\frac{\pi}{2}\right)$$

+++

so finally

+++

$$F_X(x)=\frac{1}{\pi}\left(\arctan\left(\frac{x-x_{lh}}{h}\right)-\frac{\pi}{2}\right)$$

+++

Differentating with respect to $x$ we obtain the probability density function

+++

$$P_X(x)=\frac{h}{\pi}\frac{1}{\left(x-x_{lh}\right)^2+h^2}$$

+++

In the following we will drop the subscript $X$ on $P$.

+++

## Problem 1 

+++

Assuming an uniform prior on interval $[-x_{lim}, {x_lim})$ with $x_{lim=100}$ for $x_{lh}$  plot the posterior distribution after performing one measurment.  Represent posterior by an array $P_{post}(xs_i)$ where $xs_i$ are uniformly distributed on the interval  $[-x_{lim}, {x_lim})$:

```{code-cell} ipython3
x_lim=100
xs=np.linspace(-x_lim, x_lim,5000)
```

Normalize the $P(xs_i)$ such that

+++

$$\sum_i P(xs_i)=1$$

+++

Find the MAP estimate for $x_{lh}$. 

+++

## Problem 2

+++

Add the plot of the posterior after two measurments. *Hint* perform same calculations as in problem 1, but use the caclulated posterior as the new prior. Caclculate the MAP estimate. 

+++

## Problem 3

```{raw-cell}
Write an iterative procedure that calculates all the posteriors ans all MAP estimates corresponding to $1,\ldots,100$ measurments. Plot first 10 and last 10 posterior distributions. What is the final MAP? Find 95% probability interval around this value. 
```

## Problem 4

+++

Write a procedure that takes as the input $xs$ array, the prior array,  array of measurments and returns the prior "in one go". *Hint* it is easier to work with logarithms of probabilities. For normalizing you can use `logsumexp` function from `scipy.special`. If you need to use a loop in your code, please loop over the measurments. 

```{code-cell} ipython3
from scipy.special import logsumexp
```

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
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
import numpy as np
import scipy as sp
```

```{code-cell} ipython3
import pymc3 as pm
print(f"Running on PyMC3 v{pm.__version__}")
import arviz as az
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

The figure below  presents the geometry of the problem. The orange dot indicates the lighthouse. Blue dots are the points were the flashes were recorded. The directions of the first 10 flashes are shown as lightblue lines.

```{code-cell} ipython3
x0=10;
```

```{code-cell} ipython3
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
arc3=Arc((x0,h),0.75/aspect,0.75, angle=0, theta1=np.rad2deg(-np.pi), theta2=0,edgecolor='black', linewidth=0.5);ax.add_patch(arc3);
ax.scatter(*lh.polar2xy((x0,h),-np.pi/2+thetas,0.75/(2*aspect), aspect=aspect));
xy = lh.polar2xy((x0,h),-np.pi/2+np.pi/30,0.85/(2*aspect), aspect=aspect)
ax.annotate("$\phi$",xy, fontsize=24);
```

$$x = h*\tan(\phi)+x_0\qquad \phi=\frac{1}{h}\arctan(x-x_0)$$

+++

## Problem 1

+++

Estimate the position of the lighthouse using PyMC3.

1. Formulate the model. Hint: The sampling distribution is called [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution).
2. Find the MAP estimate.
3. Simulate the posterior and find the mean and highest density interval.

+++

## Problem 2

+++

Estimate the position of the lighthouse and its distance from the shore using PyMC3. Use the new dataset

```{code-cell} ipython3
flash_x_2d = np.loadtxt('lighthouse_2d.txt')
```

```{code-cell} ipython3

```

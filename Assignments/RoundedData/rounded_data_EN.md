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

+++ {"tags": ["EN"]}

# Rounded data  (from "Bayesian Data Analysis")

+++ {"tags": ["EN"]}

It is a common problem for measurements to be observed in rounded form. For a simple example, suppose we weigh an object five times and measure weights, rounded to the nearest pound, of 10, 10, 12, 11, 9. Assume the unrounded measurements are normally distributed with some mean $\mu$ and variance $\sigma^2$.

+++

## Problem 1

+++ {"tags": ["EN"]}

Give the posterior distribution for $(\mu, \sigma^2)$ obtained by pretending that the observations are exact unrounded measurements. Assume a noninformative prior on $\mu$ and $\sigma$

+++

$$\mu\propto 1\qquad \sigma^2\propto \frac{1}{\sigma^2}$$

+++

## Problem 2

+++ {"tags": ["EN"]}

Give the correct posterior distribution for $(\mu, \sigma^2)$ treating the measurements as rounded.

+++

Proszę podać rozkład a posteriori dla zaokrąglonych danych

+++

## Problem 3

+++ {"tags": ["EN"]}

How do the incorrect and correct posterior distributions differ? Compare means, variances, and contour plots.

+++ {"tags": ["EN"]}

#### Marginal distribution

+++ {"tags": ["EN"]}

To calculate mean and variance of $\mu$ and $\sigma^2$ we need marginal distributions. We can approximate them  numerically by symming over one axis of the grid.
Do not forget to exponentiate the log of probability before summing!

+++

Do obliczenia średniej i wariancji

+++ {"tags": ["EN"]}

### Contour plots

+++ {"tags": ["EN"]}

To make a contour plot of a function of two variables we need its values dsitributed over a grid. This function takes a function and x and y ranges (arrays of values) and returns a grid with x values corresponding to columns and y values to rows

```{code-cell}
def make_grid(f,xs,ys):
    grid = np.zeros((len(ys), len(xs)))
    for iy in range(len(ys)):
         for ix in range(len(xs)):
                grid[iy,ix]=f(xs[ix],ys[iy])
            
    return grid    
```

```{code-cell}
def quad (x,y):
    return -x*x-0.25*y*y 
```

```{code-cell}
xs = np.linspace(-3,3,100)
ys = np.linspace(-3,3,100)
grid = make_grid(quad, xs,ys)
```

```{code-cell}
import matplotlib.pyplot as plt
```

```{code-cell}
plt.contour(xs,ys,grid)
plt.show()
```

+++ {"tags": ["EN"]}

We can specify contour levels

```{code-cell}
plt.contour(xs,ys,grid, levels=[-4,-3,-2,-1,0])
plt.show()
```

+++ {"tags": ["EN"]}

and fill areas between them

```{code-cell}
plt.contourf(xs,ys,grid, levels=[-4,-3,-2,-1,0])
plt.show()
```

+++ {"tags": ["EN"]}

When solving the problem is better to calculate not the posterior probability but the log of posterior probability.

```{code-cell}
isig, imu = np.unravel_index(ps.argmax(), ps.shape)
print(mus[imu], sigmas[isig])
```

```{code-cell}
isig, imu = np.unravel_index(ps_correct.argmax(), ps_correct.shape)
print(mus[imu], sigmas[isig])
```

```{code-cell}
(ps - ps_correct).max()
```

```{code-cell}
np.unravel_index(ps.argmax(),ps.shape)
```

```{code-cell}
sigmas[62]
```

```{code-cell}
mus[259]
```

```{code-cell}
np.unravel_index(ps_correct.argmax(),ps.shape)
```

```{code-cell}
sigmas[56]
```

```{code-cell}
mus[260]
```

```{code-cell}
plt.contourf(mus,sigmas,np.exp(ps)-np.exp(ps_correct));
plt.colorbar();
```

```{code-cell}
%time st.norm(0,1).cdf(0.0)
```

```{code-cell}
%time erf(0)
```

```{code-cell}

```

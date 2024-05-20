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

+++ {"tags": ["EN"], "editable": true, "slideshow": {"slide_type": ""}}

# Rounded data  (from "Bayesian Data Analysis")

+++ {"tags": ["PL"]}

## Zaokrąglone dane (z "Bayesian Data Analysis")

+++ {"tags": ["EN"]}

It is a common problem for measurements to be observed in rounded form. For a simple example, suppose we weigh an object five times and measure weights, rounded to the nearest pound, of 10, 10, 12, 11, 9. Assume the unrounded measurements are normally distributed with some mean $\mu$ and variance $\sigma^2$.

+++ {"tags": ["PL"]}

Często dane z którymi mamy do czynienia są zokrąglone.  Jako prosty przykład weźmy pięć pomiarów wagi zaokraglonych do jednego funta (funt = 0.45359237 kg) (10,10,12,11,9). Proszę założyć, że niezaokrąglone pomiary pochodzą z rozkładu normalnego ze średnia $\mu$ i wariancją $\sigma^2$.

+++

## Problem 1

+++ {"tags": ["EN"]}

Give the posterior distribution for $(\mu, v=\sigma^2)$ obtained by pretending that the observations are exact unrounded measurements. Assume a uninformative prior on $\mu$ and $v=\sigma^2$

+++ {"tags": ["PL"]}

Proszę podać rozkład a posteriori dla parametrów $\mu$ i $v = \sigma^2$ zakładając, że dane nie są zaokraglone. Proszę założyć mało informacyjny prior dla  tych parametrów

+++

$$\mu\propto 1\qquad v\propto \frac{1}{v}$$

```{code-cell}
import numpy as np
```

## Problem 2

+++ {"tags": ["EN"]}

Give the correct posterior distribution for $(\mu, v)$ treating the measurements as rounded.

+++

Proszę podać rozkład a posteriori dla zaokrąglonych danych

+++

## Problem 3

+++ {"tags": ["EN"]}

How do the incorrect and correct posterior distributions differ? Represent the posterior as a two  dimensional array. Construct the array in such way that $\mu$ corresponds to columns (x direction) and $v$ to rows (y direction).   Compare the mean and variance for each marginal distribution separately. Compare the contour plots.

+++ {"tags": ["PL"]}

Jak te dwa posteriory się różnią? Proszę porównać średnie i wariancje dla każdego rozkładu osobno oraz wykresy poziomicowe(?) (contour plots).

+++ {"tags": ["EN"]}

#### Marginal distribution

+++ {"tags": ["PL"]}

#### Rozkłady brzegowe

+++ {"tags": ["EN"]}

We can approximate marginal distributions numerically by summing over one axis of the grid.
Do not forget to exponentiate the log of probability before summing! You can and should use the `logsumexp` function from `scipy.special`.

+++

Rozkłady brzegowe możemy przybliżyć numerycznie sumując po jednej osi rozkładu dwuwymiarowego.

+++ {"tags": ["EN", "PL"]}

### Contour plots

+++ {"tags": ["EN"]}

To make a contour plot of a function of two variables we need its values distributed over a grid. This function takes a function and x and y ranges (arrays of values) and returns a grid with x values corresponding to columns and y values to rows

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
xs = np.linspace(-4,4,200)
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
plt.contour(xs,ys,grid, levels=[-5,-4,-3,-2,-1,0])
plt.show()
```

+++ {"tags": ["EN"]}

and fill areas between them

```{code-cell}
plt.contourf(xs,ys,grid, levels=[-5, -4,-3,-2,-1,0])
plt.show()
```

+++ {"tags": ["EN"]}

When solving the problem is better to calculate not the posterior probability but the log of posterior probability.

```{raw-cell}

```

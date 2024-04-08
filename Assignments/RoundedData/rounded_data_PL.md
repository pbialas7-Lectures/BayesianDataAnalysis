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

+++ {"tags": ["PL"]}

## Zaokrąglone dane (z "Bayesian Data Analysis")

+++ {"tags": ["PL"]}

Często dane z którymi mamy do czynienia są zokrąglone.  Jako prosty przykład weźmy pięć pomiarów wagi zaokraglonych do jednego funta (funt = 0.45359237 kg) (10,10,12,11,9). Proszę założyć, że niezaokrąglone pomiary pochodzą z rozkładu normalnego ze średnia $\mu$ i wariancją $\sigma^2$.

+++

## Problem 1

+++ {"tags": ["PL"]}

Proszę podać rozkład a posteriori dla parametrów $\mu$ i $\sigma^2$ zakładając, że dane nie są zaokraglone. Proszę założyć mało informacyjny prior dla  tych parametrów

+++

$$\mu\propto 1\qquad \sigma^2\propto \frac{1}{\sigma^2}$$

```{code-cell}
import numpy as np
```

## Problem 2

+++

Proszę podać rozkład a posteriori dla zaokrąglonych danych

+++

## Problem 3

+++ {"tags": ["PL"]}

Jak te dwa posteriory się różnią? Proszę porównać średnie i wariancje dla każdego rozkładu osobno oraz wykresy poziomicowe(?) (contour plots).

+++ {"tags": ["PL"]}

#### Rozkłady brzegowe

+++

Do obliczenia średniej i wariancji dla $\mu$ i $\sigma$ będziemy potrzebowali rozkładów brzegowych. Możemy przybliżyć je analitycznie sumując po jednej osi rozkładu dwuwymiarowego.

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

```{code-cell}
plt.contour(xs,ys,grid, levels=[-4,-3,-2,-1,0])
plt.show()
```

```{code-cell}
plt.contourf(xs,ys,grid, levels=[-4,-3,-2,-1,0])
plt.show()
```

```{code-cell}

```

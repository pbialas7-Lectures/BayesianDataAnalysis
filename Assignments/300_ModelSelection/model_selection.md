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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Model selection

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
import scipy
import pymc as pm
import arviz as az
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"Running on PyMC version {pm.__version__} and ArviZ version {az.__version__}")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,6)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Your aim in this assignment is to find the model that best fits the data

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
data= np.loadtxt('data.txt')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The data is from an discrete distribution. You can use the function below to plot the histogram.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def plot_discrete_hist(data, ax = None, **kwargs):
    if ax is None:
        ax =plt.gca()
    min, max = data.min(), data.max()
    n_bins=int(max-min+1)
    return ax.hist(data, bins=n_bins, range=(min-0.5, max+0.5), **kwargs)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fig, ax = plt.subplots()
plot_discrete_hist(data, ax=ax);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Check if the data can be explained by the binomial distribution with $n=35$. Make a posterior predictive check.

```{code-cell} ipython3
tune = 1000
draws = 8000
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for binomial distribution with $n=45$.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for poisson distribution.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p(k)=e^{-\lambda}\frac{\lambda^k}{k!}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

 Use $\Gamma$ distribution for prior on the $\lambda$ parameter of the distribution.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 4

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Compare the three models using the "leave one out" cross validation.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If you can, explain the results :)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

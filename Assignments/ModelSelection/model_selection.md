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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Model selection

+++ {"editable": true, "slideshow": {"slide_type": ""}}

For this assignment  you should install PyMC version 5.x. It should probably be enough to run
```shell
mamba install "pymc>=5.0"
```
but maybe you would have to remove `pymc3` first
```
mamba deinstall pymc3
```
Your python version should be 3.10. 

```{code-cell} ipython3
import numpy as np
import scipy
import pymc as pm
import arviz as az
```

```{code-cell} ipython3
print(f"Running on PyMC version {pm.__version__} and ArviZ version {az.__version__}")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,6)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
data= np.loadtxt('data.txt')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Check if the data can be explained by the binomial distribution with $n=35$. Make a posterior predictive check.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for binomial distribution with $n=45$.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for poisson distribution. Use $\Gamma(\alpha, \beta)$ distribution for prior. 

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 4

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Compare the three models using the "leave one out" cross validation. 

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If you can explain the results :)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

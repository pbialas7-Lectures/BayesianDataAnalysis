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
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
import scipy as sp
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import pymc as pm
print(f"Running on PyMC v{pm.__version__}")
import arviz as az
```

```{code-cell}
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
from matplotlib.patches import Arc, FancyArrowPatch
```

```{code-cell}
import lighthouse as lh
```

# Lighthouse

+++

In this assignment you will continue to work on the estimatio of the position of the lighthouse, but this time using the Monte-Carlo method.

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

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Estimate the position of the lighthouse using PyMC.

1. Formulate the model. When formulating the model please take care that all the parameters of the distributions are specified as floating point numbers (use the decimal point). Specifying them as integers may lead to errors :(  
2. Find the MAP estimate.
3. Simulate the posterior and find the mean and 95% highest density interval. The needed flash distribution is called Cauchy. Report the mean and HDI to two decimal places.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Estimate the position of the lighthouse and its distance from the shore using PyMC. Use the lighthouse_`2d.txt dataset`. As in previous problem:
1. Formulate the model. When formulating the model please take care that all the parameters of the distributions are specified as floating point numbers (use the decimal point). Specifying them as integers may lead to errors :(  
2. Find the MAP estimate.
3. Simulate the posterior and find the mean and 95% highest density interval for each parameter separately.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
flash_x_2d = np.loadtxt('lighthouse_2d.txt')
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

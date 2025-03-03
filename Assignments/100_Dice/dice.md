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
import numpy as np
import scipy
import matplotlib.pyplot as plt
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Dice

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Throw the provided dice more than 100 times. Count how many times each face apeared in the results and write it down in array

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
data =np.asarray([0,0,0,0,0,0])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

First number in the array should be the number of ones in the results and last one should be the number of sixes.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Plot the results using the [`matplotlib.pyplot.bar`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html) function.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Assuming an uniform prior, what is the posterior for the probability of getting a six? Plot this function.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Provide the MAP, mean and median of the distribution. Please print them out and mark them on the plot.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["hint"]}

__Hint__ For median use the `isf` function of the distribution from `scipy.stats`.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Estimate the HDR containing the 90% probability. Use the `bda.stats.hdr_f` function.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
import bda 
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
help(bda.stats.hdr_f)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

 In case of numerical problems you may need to constrain the (a,b) interval.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the following but this time for the prior use Beta distribution with mean equal to 1/6 and standard deviation equal to 1/100.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Let's suppose that the dice is crooked and

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p_1=\frac{1}{6}-\epsilon,\, p_2=p_3=p_4=p_5=
\frac{1}{6},\, p_6 =\frac{1}{6}+\epsilon$$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

with $\epsilon=0.01$. Assuming uniform prior estimate  how many throws are needed to ascertain that a dice is crooked with 90% accuracy.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["hint"]}

__Hint__: This is a binomial distribution with $p=\frac{1}{6}+\epsilon$. Assume that for large $n$ the number of sixes is equal to $p\cdot n$. Calculate the posterior and 90% HDR. By trial and error estimate smallest $n$ such that $p=\frac{1}{6}$ lies outside of the HDR.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Perform same calculations for the prior from Problem 2.

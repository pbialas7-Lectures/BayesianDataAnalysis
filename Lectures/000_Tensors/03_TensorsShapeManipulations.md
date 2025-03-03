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
```

```{code-cell}
from matplotlib.image import imread
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

```{code-cell}
grumpy =  imread('GrumpyCat.jpg').copy() # we are taking the copy to make our grumpy writable
grumpy.shape
```

## Concatenating, stacking and spliting

+++

Let's start by combining one or more arrays together. 

Concatenating two or more arrays does not change they dimensionality, but only increases the size of the concatenated dimension.
Below we copy grumpy picture by adding more rows:

```{code-cell}
two_grumpies_portrait = np.concatenate((grumpy, grumpy), axis=0)
print(two_grumpies_portrait.shape)
plt.imshow(two_grumpies_portrait);
```

and here we add columns

```{code-cell}
two_grumpies_landscape = np.concatenate((grumpy, grumpy), axis=1)
print(two_grumpies_landscape.shape)
plt.imshow(two_grumpies_landscape);
print(two_grumpies_landscape.base)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Same effect can be obtained by appending one tensor to another

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(np.append(grumpy, grumpy, axis=1));
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Stacking in adds one dimension more. Below we reasemble our grumpy from separate color chanels:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
red = grumpy[:,:,0]
green = grumpy[:,:,1]
blue  = grumpy[:,:,2]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

All this chanel arrays are two-dimensional:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
blue.shape
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(blue, cmap='gray');
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

and we will stack them creating a third dimension, which we chose to be the last:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
stacked_grumpy = np.stack((red, green, blue), axis=2)
print(stacked_grumpy.shape)
plt.imshow(stacked_grumpy);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We could sack along some other axis:

```{code-cell}
stacked_grumpy_wrong = np.stack((red, green, blue), axis=0)
stacked_grumpy_wrong.shape
```

But now we cannot intepret it as an image. It is a stack of chanels.

```{code-cell}
plt.imshow(stacked_grumpy_wrong[0], cmap='Reds');
```

The inverse operation to concatenating would be spliting. 

We can split an array along an axis. Below we split the picture into three color channels.  The rusult is a list of arrays. Please note however that the parts resulting from split are still three dimensional, the last dimension being "degenerate" and having size one. I will comment on this more later. We can get rid of those degenerate dimensions by using function `squeeze`.

```{code-cell}
chanels = np.split(grumpy,3,axis=2)
print(chanels[0].base is grumpy)
print(type(chanels))
print(chanels[0].shape)
plt.imshow(np.squeeze(chanels[0]), cmap='gray');
```

Please try to reasamble chanels back into three color picture. Which function will you use?

+++ {"editable": true, "slideshow": {"slide_type": ""}}

But be aware that split creates the views of the array not the copies! So changing one of the resulting arrays alters the original:

```{code-cell}
chanels[0][:]=0
plt.imshow(grumpy);
```

## Reshaping

+++

Arrays are stored as contiguous pieces of memory. In this sense all arrays are vectors and the dimensions are just artefacts of how we view them. An array of twelve elements

```{code-cell}
ar = np.arange(0,12)
```

can be viewed as 3x4 matrix by reashaping it:

```{code-cell}
print(ar.shape)
ar3x4 = np.reshape(ar,(3,4))
print(ar3x4.shape)
print(ar3x4)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Reshaping does not create a new array just another view into array similarly to slice. Assigning to reshaped array we are still assigning to original array:

```{code-cell}
ar3x4[0]=0
ar3x4[1]=1
ar3x4[2]=2
ar
```

Similarly we can view it as a 4x3 matrix:

```{code-cell}
ar = np.arange(0,12)
```

```{code-cell}
print(ar.shape)
ar2=np.reshape(ar,(4,3))
print(ar2.shape)
print(ar2)
```

a 2x6 matrix:

```{code-cell}
print(ar.shape)
ar2=np.reshape(ar,(2,6))
print(ar2.shape)
print(ar2)
```

or higher dimensional tensor:

```{code-cell}
print(ar.shape)
ar2=np.reshape(ar,(2,3,2))
print(ar2.shape)
print(ar2)
```

We can let the numpy to deduce the size of one of the dimensions by marking it with -1:

```{code-cell}
ar2.reshape(-1)
```

```{code-cell}
ar2.reshape(-1,3)
```

```{code-cell}
ar2.reshape(2,-1,3)
```

## Transpositions

+++

Sometimes we need to change the order of dimensions. Let's for example take our `stacked_grumpy_wrong`

```{code-cell}
stacked_grumpy_wrong.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As we have already explained this cannot be interpreted as an image beacuse we expect the chanels to be the last dimension. We can fix this by using `moveaxis` or `transpose

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
stacked_grumpy_right = np.moveaxis(stacked_grumpy_wrong, 0 ,2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(stacked_grumpy_right);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

But again what we obtain is a view not a copy and if we modify the untransposed array

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
stacked_grumpy_wrong[0,:]=0
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

we will also modify the transposed one:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(stacked_grumpy_right);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(grumpy)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

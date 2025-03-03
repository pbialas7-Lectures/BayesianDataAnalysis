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

# Tensors Basics

```{code-cell}

```

## Why tensors

Tensors can be  understood as a multidimensional  extension of more familiar object like  vectors (1D tensors) or matrices (2D tensors). They  provide  basic building blocks for all known machine learning, deep or otherwise, frameworks. 
That is because the data is mostly presented in form of multidimensional tables i.e. tensors.  Also many models have their internal structure expressed in terms of tensors.
So whatever you do, you will be using tensors to manipulate your data. 

Apart for being a very useful mathematical abstraction, the use of tensors is very efficient.  Tensors libraries provide optimized  implementations of tensor operations. This is crucial for interpreted languages like Python. As you will see the difference between "naive" python implementation and  tensor operations can be staggering. This is because those operations are written in C/C++ and compiled to native code. Usually their are optimized for given hardware and use multithreading and/or vector instructions. Without using tensor functions  Python would be to slow for any practical purposes of machine learning. 

That's why being confortable with tensors is absolutelly crucial for any serious machine learning practitioner. I strongly encourage you to experiment with the ideas presented here to be sure that you understand them.

+++

### numpy

The tensor package for python is `numpy`. Most of other libraries build on top of it. However you must be aware that many machine learning frameworks (e.g. PyTorch) define their own tensors. Usually they provide  functions to convert to and from `numpy` tensors. Also they implement a similar set of functions and operators. That's why in this notebook I will make a quick introduction to `numpy` library. All the main concepts introduced here will translate to any tensor library, but the notation may differ. 

Please note that I will not explain in detail each introduced function or method. You are expected to familiarize yourself with common functions by consulting the documentation.

+++

Let's start by importing the `numpy` module:

```{code-cell}
import numpy as np
```

As already mentioned tensor is a multidimensional table i.e. something that requires several indices to access its elements. If we need $D$ indices we say that the tensor is $D$-dimensional. For example tensor below is 3-dimenional.

+++

$$x_{ijk}$$

+++

Another important feature is dimension size. Each index corresponds to one dimension and the range the index can take is the size of this dimension. In the example below we create a 3-dimensional random tensor, where each tensor element is drawn idependently from gaussian distribution.  The last argument specifies the tensor shape i.e. the number of dimensions and the size of each dimension. Actually the name for tensors in `numpy` is `array` and we will be using this from now on.

```{code-cell}
x = np.random.normal(0,1,(3,5,2))
print(x)
```

The array `x` has three dimensions with sizes equal to three, five and two. We can verify it by using attribute `shape`:

```{code-cell}
x.shape
```

Arrays of 0 to 2 dimensions are also commonly refered by other names:

+++

### Scalar

Scalar or "just a number" can be interpreted as zero-dimensional array that does not require any indices.

```{code-cell}
scal = np.pi
```

### Vector

Vector is a one-dimenional array:

```{code-cell}
vec = np.zeros(7)
print(vec)
```

### Matrix

Matrix is a two-dimensional array:

```{code-cell}
mat = np.ones((3,4))
print(mat)
```

### Higher dimensions

+++

A RGB image is a three dimensional array of shape (height, width,3).

```{code-cell}
from matplotlib.image import imread
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

the last dimension containing three color channels.

```{code-cell}
grumpy =  imread('GrumpyCat.jpg')
grumpy.shape
```

Please note that the dimensions are reversed compared to usuall notation 'width x height'. That is because the first dimension denotes the rows and the second columns.

+++

And here is our, unfortunately late,  king of the internet!

```{code-cell}
plt.imshow(grumpy); #The semicolon at the end prevents the results of the function to be written to output.
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If we load more then one image of the same size, we can store them in four-dimensional array of shape (N_images, height, width,3). In the example below we load 1024 32x32x3 color images.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
cifar = np.load("cifar_10_1024.npy")
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
cifar.shape
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
fig, ax = plt.subplots(1,8,figsize=(8,1))
for i,a in enumerate(ax.ravel()):
  a.axis('off')
  a.imshow(cifar[i])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

And finally  we usually divide the data in batches. That would give us a five-dimensional array (N_batches, N_images_in_batch, height, width, 3). You will encounter more examples as we progress.

```{code-cell}
cifar_batched = cifar.reshape(-1,8,32,32,3) # The reshape function will be explained later. 
cifar_batched.shape
```

Our data now consist of 128 batches of eights 32x32x3 images each.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
fig, ax = plt.subplots(4,8,figsize=(8,4))
for batch in range(4): #loop over batches
  for j in range(8): #loop over images in batch
    ax[batch][j].axis('off')
    ax[batch][j].imshow(cifar_batched[batch][j])
```

## Creating tensors

+++

Above we have seen some examples of functions creating tensors.  All of them  initialize the elements of the array. Either with random numbers or a constant like zero or one. When we want to imediatelly assign other value to an tensor this is wasteful. For this we have yet another function that creates an empty, that is uninitialized array:

```{code-cell}
np.empty((5,4))
```

All of those tensor creation functions also take a shape argument. Some of them have a version which takes another array as an "blueprint"

```{code-cell}
x = np.ones((2,4))
y = np.zeros_like(x)
y.shape
```

Yet another option is to create numpy arrays out of Python data structures: list, tuples using `array` or `asarray`.

```{code-cell}
np.asarray([0,0])
```

```{code-cell}
np.array((0,0))
```

```{code-cell}
np.array([[0,1],[1,0]])
```

For more information see [Array creation](https://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation) in `numpy` documentation.

+++

## Array data type

Arrays do not only have shape but also data type for its elements which can be queried using the attribute `dtype'

```{code-cell}
d = np.ones(3)
d.dtype
```

As you can see the default type for floating point arrays is double precision (64bit)

```{code-cell}
r = np.arange(10)
r.dtype
```

and for integer typye it is 64bit int. Some creation function accept the `dtype`   argument:

```{code-cell}
f = np.ones(3, dtype='float32')
f.dtype
```

```{code-cell}
i = np.ones(3, dtype='int32')
i.dtype
```

```{code-cell}
rf = np.arange(10, dtype='float32')
rf.dtype
```

For functions that do not provide `dtype` argument we have to use `astype` array method

```{code-cell}
rnd_f = np.random.uniform(size=10).astype('float32')
rnd_f.dtype
```

`astype` returns a copy of the array with appriopriate data type:

```{code-cell}
ri = np.arange(10)
rf = ri.astype('float32')
```

`ri` and `rf` are two distinct arrays.

+++

The types are important because some functionality  requires a specific data type. For example we can only use integer arrays to index other arrays. 

More importantly GPU  uses `float32` arithmetics and so most of the deep learning packages expect `float32` dtype. Even if they hadle the conversion themself we waste lot of memory by using the `float64`.

+++

## Indexing and slicing

+++

We index the individual elements of the tensor by providing the values for all of its indices. As in C/C++ the indices start from zero. To access red chanel of the upper left pixel we use

```{code-cell}
grumpy[0,0,0]
```

The greatest value an index can have its size of its dimension minus one. The code below access red chanel of the lower right pixel.

```{code-cell}
grumpy[599,460,0]
```

This requires us to remember the size of all dimensions. However we can index counting from the back:

```{code-cell}
grumpy[-1,-1,0]
```

One can understand this as substracting from the dimension size.

```{code-cell}
gs= grumpy.shape
grumpy[gs[0]-1,gs[1]-1,0]
```

If we do not specify all indices a subtensor is returned. In that way we can acces a whole pixel:

```{code-cell}
pixel = grumpy[0,0]
print(pixel.shape)
pixel
```

and in that way a row:

```{code-cell}
row = grumpy[0]
print(row.shape)
print(row)
```

What if we want to skip indices which are not at the end? For example how we access a single column? This can be achieved using slice notation. We mark the missing index by a colon ':'. Here we access first column:

```{code-cell}
column = grumpy[:,0]
print(column.shape)
column
```

and here the blue chanel:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
chanel = grumpy[:,:,2]
print(chanel.shape)
plt.imshow(chanel, cmap='gray');
```

The ':' is in fact a special case of the more general notation. The general  slice has the format

+++

```python
start:end:step

```

+++

which denotes values of the index starting at `start` then increasing by `step` until it is equal or greater then `end`. This means that the `end` is not included in this range. When omited, `start` defaults to zero, `end` to size and `step` to one.
Single colon ':' is equivalent to `0:size:1`.

+++

Here we take a portion of the photo:

```{code-cell}
sub = grumpy[100:200, 100:200]
print(sub.shape)
plt.imshow(sub);
```

And here we take every eight pixel:

```{code-cell}
small = grumpy[::8, ::8]
print(small.shape)
plt.imshow(small);
```

A frequent idiom is reversing the array along one dimension by using negative steps:

```{code-cell}
plt.imshow(grumpy[::-1]);
```

```{code-cell}
plt.imshow(grumpy[:,::-1]);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(grumpy[::-1,::-1]);
```

Please play with slicing until you are confident that you have mastered this notation.

+++

## Copies, views, references and asignment

+++

It's very important to understand when we are dealing with copies of arrays and when with references to them or views.

+++

### References

+++

The line below formaly creates a new ten-element array and stores reference to it in variable x:

```{code-cell}
x = np.arange(10)
x
```

Assignment in Python, like in Java, only copies the references creating an alias for x,

```{code-cell}
y = x
```

After this assignement both variables x and y point to the same object, which we can check by using the operator `is`:

```{code-cell}
y is x
```

So changing array using one of the references will change same array as pointed by others:

```{code-cell}
y[0]=10
x
```

However assigning a new array to y will replace the reference and the original x array will be unaffected:

```{code-cell}
y = np.zeros_like(x)
x
```

### Views

+++

Assigning references is not the only way we can create an alias. Many, if not most, of the tensor operations creates co called **views** of the array. A view is a array that shares memory with another array. When taking slices of the array we actually creating a view.

```{code-cell}
x_view = x[:]
x_view
```

now the x and x_view point to distinct objects

```{code-cell}
x_view is x
```

but share the underlaying arrays. Assigning to `x_view` will change `x`

```{code-cell}
x_view[6] = 42
x
```

A view holds a reference to original array which we can retrieve using `base` property:

```{code-cell}
x_view.base
```

while the original array has base None

```{code-cell}
print(x.base)
```

Of course we can use any slice:

```{code-cell}
x_half = x[::2]
x_half[:] = 7
x
```

Please note an technical but important point. Why we could n ot just write `x_half=7` but used `x_half[:]=7` instead? Let's check:

```{code-cell}
x_half=8
x
```

```{code-cell}
x_half
```

`x_half` is a variable holding reference to a python object which happens to be a view (ndarray). Assignment `x_half=8` just stores number 8 in this variable, replacing the array reference. The slice notation forces the elementwise assignment.

+++

Of course we can use slices directly without using the intermediate references:

```{code-cell}
x[1::3]=3
x
```

### Copy

+++

A physical (deep) copy is provided by the copy method of the array:

```{code-cell}
grumpy_copy  =  grumpy.copy()
```

Now changing the copy does not change the original:

```{code-cell}
fig, ax = plt.subplots(1,3, figsize=(21,7))
ax[0].imshow(grumpy_copy);
ax[0].set_title("Copy")
grumpy_copy[:,:,:] = np.ones_like(grumpy_copy)*255
ax[1].set_title("Copy after assignment")
ax[1].imshow(grumpy_copy);
ax[2].set_title("Original")
ax[2].imshow(grumpy);
```

We can use slices to selectively alter a part of the picture:

```{code-cell}
grumpy_copy = grumpy.copy()
fig, ax = plt.subplots(1,2, figsize=(16,8))
ax[0].imshow(grumpy_copy);
grumpy_copy[100:200, 100:200]=np.array([255,255,255])
ax[1].imshow(grumpy_copy);
```

But again this has changed only the copy:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(grumpy);
```

## Saving and loading

+++

 Array can be saved to disk using `save` function

```{code-cell}
np.save('grumpy.npy', grumpy)
```

and loaded back

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
another_grumpy = np.load('grumpy.npy')
plt.imshow(another_grumpy);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

#### Problem

Display the blue channel of the grumpy cat picture in more familiar way using the hues of blue.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

To this end we must create a  three dimensional tensor of same shape as the `grumpy` and assign zero to red and green channels and leave the blue chanel intact. Actually we will create an array filled with zeros and only assign the blue channel

```{code-cell}
---
editable: true
jupyter:
  source_hidden: true
slideshow:
  slide_type: ''
tags: [answer]
---
blue_channel = np.zeros_like(grumpy)
blue_channel[:,:,2]=grumpy[:,:,2]
```

```{code-cell}
---
editable: true
jupyter:
  source_hidden: true
slideshow:
  slide_type: ''
tags: [answer]
---
plt.imshow(blue_channel);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This concludes a very brief introduction to `numpy` arrays. In the next section we will show how to manipulate  and change the shape of the arrays.

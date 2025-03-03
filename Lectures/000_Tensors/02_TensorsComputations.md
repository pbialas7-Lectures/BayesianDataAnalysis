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

import matplotlib.image as img
import matplotlib.pyplot as plt
%matplotlib inline
```

## Arithmetical operations

+++

`numpy` supports all arithmetic operations and many functions in form of elementwise operations. For example  for multiplication

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$ z_{ijk}=x_{ijkl} \cdot y_{ijk}\quad\text{for all}\; i,j,k$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x = np.random.normal(0,1,(3,5,2))
y = np.random.normal(0,1,(3,5,2))
%time z = x * y 
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Please note that arithmetic operations create a new array.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(z.base)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This is equivalent to the following loop but faster

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
%%time
z = np.empty((3,5,2))
s = z.shape
for i in range(s[0]):
    for j in range(s[1]):
        for k in range(s[2]):
            z[i,j,k]=x[i,j,k] * y[i,j,k]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Time difference in this case is not  very big, but for bigger arrays it can becomes very large:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
import timeit
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
xb = np.random.normal(0,1,(30,50,20))
yb = np.random.normal(0,1,xb.shape)
start_time = timeit.default_timer()
zb = xb * yb 
end_time = timeit.default_timer()
elapsed_implicit = end_time-start_time
print("Took %s " % (elapsed_implicit,))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
s = xb.shape
start_time = timeit.default_timer()
zbloop = np.empty_like(xb)
for i in range(s[0]):
    for j in range(s[1]):
        for k in range(s[2]):
            zbloop[i,j,k]=xb[i,j,k] * yb[i,j,k]
end_time = timeit.default_timer()            
elapsed_explicit = end_time-start_time
print("Took %fs which is %f times longer!" %(elapsed_explicit, elapsed_explicit/elapsed_implicit))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As you can see this pure python implementation is almost 200 times slower! That is the main reason you should become fluent in tensor operations.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Similarly we can apply a numpy function to every element of the tensor just by calling it with tensor argument:

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$s_{ijk} = \sin(x_{ijk})\quad\text{for all}\; i,j,k$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
%time s = np.sin(x)
```

Please compare yourself the time of the execution of this operation to an explicit loop.

+++

You can also use a scalar argument in tensor operations with the common sense interpretation:

```{code-cell}
grumpy = img.imread("GrumpyCat.jpg")
```

```{code-cell}
normalized_grumpy = grumpy/255
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Reduction

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Another common operations are  reductions. Those are the functions that can be applied to a subset of dimensions "reducing" them  to a single number.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
n = 1000;m =4
data = np.random.normal(0,1,(n,m))
data.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

A common reduction operation is sum. Without any additional parameters sum sums all the element of the array

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.sum(data)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

But we can specify the dimension(s) along which the reduction operation will be applied.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
row_sum = np.sum(data, axis=1)
row_sum.shape
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
row_sum[:8]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As we can see the secod dimension  was "reduced" and we are left with a one dimensional array.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In the same way we can calculate the mean of every column:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.mean(data, axis=0)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

or standard deviation

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.std(data, axis=0)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can reduce more then one dimension at the time. Below we calculate the mean value of each chanel in grumpy

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.mean(grumpy, axis=(0,1))
```

or max and min  values

```{code-cell}
np.min(grumpy, axis=(0,1))
```

```{code-cell}
np.max(grumpy, axis=(0,1))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Contractions -- inner product

+++

Another class of operations are contraction. In contraction we sum over two dimensions of a product of two arrays. The examples include the dot (scalar) product

+++

$$ x\cdot y =\sum_{i} x_{i} \cdot y_{i}$$

+++

matrix vector multiplication:

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$ v_j =\sum_{i} A_{ji} \cdot w_{i}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

and matrix multiplication

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$   z_{ij}=\sum_{k} x_{ik} \cdot y_{kj}
$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

`numpy` has special operators for both operations but we can use more general `inner` and `tensordot`. 

`inner` takes two arrays and contracts last dimensions in each of them. That means that the sizes of those dimensions must match. 

When both arrays are vectors this is normal scalar product:

```{code-cell}
x = np.random.normal(0,1,10)
y = np.ones_like(x)
np.inner(x,y)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

When first is  a matrix and other is a vector this is matrix vector multiplication:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
m = np.asarray([[1,-1],[-1,1]])
v = np.array([0.5, -0.5])
np.inner(m,v)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Can you tell what the operation below is doing?

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
w =np.asarray([0.3, 0.59, 0.11])
G = np.inner(grumpy,w)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.imshow(G, cmap='gray');
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Similar to `inner` is `dot. Please check out its documentation [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Matrix multiplication requires contraction of last and first dimension. That's why it's more convenient to use `tensordot(A,B,n)` which contracts last `n` dimensions of array `A` with first `n` dimensions of array `B`.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
A = np.random.normal(0,1,(2,3))
B = np.random.normal(0,2,(3,4))
C = np.tensordot(A,B,1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(C.shape)
C
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If we want to do matrix multiplication it's better to use 
`matmul` function which is described [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul). This function can be invoked using operator `@`

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
A@B
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Here we contract all dimensions resulting in scalar

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
A2 = np.random.normal(0,1,(4,3))
B2 = np.random.normal(0,2,(4,3))
C2 = np.tensordot(A2,B2,2)
print(C2.shape)
C2
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In the above expression `C2` is calculated as: 
$$ C = \sum_{ij}A_{ij} B_{ij}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can also specify which dimensions will be contracted, by providing lists of dimensions in each array:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
A3 = np.random.normal(0,1,(4,3))
B3 = np.random.normal(0,2,(3,4))
C3 = np.tensordot(A3,B3,((0,1), (1,0)))
print(C3.shape)
C3
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Which corresponds to

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$ C = \sum_{ij}A_{ij} B_{ji}=\operatorname{Tr}A\cdot B$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

#### Problem

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

You have a 3x4 matrix W and a set of N 4-vectors in a form of array X of shape (N,4). How to produce an array Y of shape (N,3) where each row is the product of matrix W and corresponding row of X ?

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
N = 1000
W = np.random.normal(1,1,(3,4))
X = np.random.normal(-1,0.5,(N,4))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

It is always good to start with writing the problem in the index notation. What we need is

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$Y_{ij} = \sum_{k} W_{jk} X_{i,k}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

which means that we have the contract the last dimensions of $W$ and $X$. However the order of the arguments is important

```{code-cell}
---
editable: true
jupyter:
  source_hidden: true
slideshow:
  slide_type: ''
tags: [answer]
---
Y = np.inner(X,W)
Y.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

What would happen if we revers the arguments?

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Outer product

+++ {"editable": true, "slideshow": {"slide_type": ""}}

What happens when we request zero dimension contraction in `tensordot`? For two vectors this should correspond to

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$ z_{ij} = x_i \cdot y_j$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Let's check this.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x = np.arange(4)
y = np.arange(5)
z  = np.tensordot(x,y,0)
print(z.shape)
z
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This operation is called outer or tensor product. We can achieve same result with function `outer`

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x = np.arange(4)
y = np.arange(5)
z  = np.outer(x,y)
print(z.shape)
z
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

However those two functions behave the same same only for 1-dimensional arrays.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

For more dimensional arrays

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
x = np.random.normal(0,1,(3,4))
y = np.random.normal(0,1,(2,2))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

`tensordot` creates an outer product  with dimensions concatenated

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
out_tdot =  np.tensordot(x,y,0)
out_tdot.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

and `outer` firts flatten both input arrays making them one dimensional and then calculates outer product

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
out_out=np.outer(x,y)
out_out.shape
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Actually the numbers are the same, only indexed differently.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## "Degenerate" dimensions

This a technical but a quite important point. It concerns dimensions with size one. While it may seem that such dimensions are spurious or "degenerate" they nevertheless change the dimensionality of the array and can impact the result of the operations.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Let's start by creating a vector

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
vector = np.random.normal(0,1,(4,))
print(vector.shape)
vector
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

a one row matrix

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
vector_row = np.random.normal(0,1,(1,4))
print(vector_row.shape)
vector_row
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

and one column matrix:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
vector_column = np.random.normal(0,1,(4,1))
print(vector_column.shape)
vector_column
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Now make some experiments:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.inner(vector, vector)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.inner(vector_row, vector_row)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.inner(vector_column, vector_column)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This actually the outer product:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.outer(vector_column.squeeze(), vector_column.squeeze())
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The `squeeze` method eliminates those degenerate dimensions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The only two other combinations that will match are:

```{code-cell}
np.inner(vector, vector_row)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
np.inner(vector_row, vector)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

#### Problem

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

Please explain the results of all the above operations. Write down using indices what each operation actually does.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

In the first case we have a onde dimensional vector $x_i$ and the inner product is

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$y = \sum_i x_i\cdot x_i$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

resulting in a single scalar.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

In the second case we have a two dimensional vector $x_{ij}$ but $i$ can take only one value: zero. The  inner product is formally a two dimensional tensor

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$y_{ik}=\sum_{j}x_{ij}x_{kj}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

which reduces to

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$y_{00}=\sum_{j}x_{0j}x_{0j}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

In the third, most interestng case, $x$ is a column vector $x_{ij}$ with $j$ being restricted to zero. The inner product is again

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$y_{ik}=\sum_{j}x_{ij}x_{kj}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

which reduces to

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$y_{ik}=x_{i0}x_{k0}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

resulting in outer product.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

And finally the last two examples can be writte as

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$y_0 = \sum_{i} x_i\cdot x_{0i}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

and

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$y_0 = \sum_{i} x_{0i}\cdot x_{i}$$.

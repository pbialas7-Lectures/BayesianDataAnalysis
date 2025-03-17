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
  slide_type: skip
---
%load_ext autoreload
%autoreload 2
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import scipy
from scipy.stats import beta
import matplotlib.pyplot as plt
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
%matplotlib inline
plt.rcParams["figure.figsize"] = [9,6]
dc='#1f77b4' #default color
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import defaults
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import billiard as bl
import bda.stats as bst
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Estimates and  errors

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We return to the billiard example that we have studied in the previous notebook "100_Introduction.md". In this notebook we will explain how to attach errors to the estimates of $p$ that we have calculated previously. But first let's again establish the starting point.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
p =  np.pi/10.0
y =  0.786
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We throw 100 balls and check wheter they land on the left of the initial ball

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
np.random.seed(87)
xy = scipy.stats.uniform(loc=0, scale=1).rvs(size=(100,2))
left=(xy[:,0]<=p) + 0
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We have assumed the initial uniform prior

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$P(p)=1$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
prior = np.vectorize(lambda x: 1.0)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which is an special case of a more general

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$PDF[Beta(\alpha,\beta)]=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Beta distribution with  $\alpha=\beta=1$. The $Beta(\alpha,\beta)$ distribution is a _conjugate_ prior for binomial distribution so the posterior is again a Beta distribution

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$P(p|n_L,n_R)=PDF[Beta(n_L+\alpha,n_R+\beta)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

where $n_L$ and $n_R$ is the number of balls that have landed on the left and right of the original ball.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
beta_posteriors=[]
for i in range(1,101):
    n_l = left[:i].sum()
    n_r = i-n_l
    beta_posteriors.append(beta(n_l+1, n_r+1))
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

After five throws we have

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
n_b=5;n_l = left[:n_b].sum();n_r=n_b-n_l
print(f"After throwing {n_b} balls, {n_l} ball(s) landed on the left and {n_r} balls on the right")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and the posterior distribution is $Beta(2,5)$ illustrated below. The true value of $p$ is denoted by the black line.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
post = beta_posteriors[n_b-1]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
xs=np.linspace(0,1,500)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
tags: [aux_code]
---
fig, ax = plt.subplots();ax.set_xlim(0,1);ax.axhline(0,lw=0.5,c='gray')
ax.axvline(p, linewidth=0.5,c ='black')
ax.plot(xs,post.pdf(xs), zorder=1, c='blue');
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This distribution summarises _all_ our knowledge about the balls position however often we would like to obtain a single _estimate_ of the parameters. This can be achieved in several ways. One of them is the _mode_ of the distribution, that we have already discussed. This leads to the

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Maximal a posteriori (MAP)

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{MAP}=\operatorname{argmax}_p P(p|D)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The mode of the $Beta(\alpha,\beta)$ distribution is

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\operatorname{Mode}[Beta(\alpha,\beta)]=\frac{\alpha-1}{\alpha+\beta-2}\qquad \alpha>1,\;\beta>1$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which translates into

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_{MAP}=\frac{n_L+\alpha-1}{n_L+n_R+\alpha+\beta-2}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In this particular example

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_{MAP} =  \frac{1}{5}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
_a,_b = beta_posteriors[n_b-1].args
p_map = (_a-1)/(_a+_b-2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
fig, ax = bl.simple_plot(p=p)
ax.axvline(p_map, linewidth=0.5, c=defaults.map_color, label='MAP')
ax.plot(xs,beta_posteriors[n_b-1].pdf(xs), zorder=1, c='blue');
plt.legend();
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Mean

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Another possible estimate is the mean

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\newcommand{\avg}[1]{\left\langle#1\right\rangle}$$
$$\avg{p}=\int_{0}^1\!\text{d}p\,P_{post}(p|D) p$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\operatorname{Mean}[Beta(\alpha,\beta)]=\frac{\alpha}{\alpha+\beta}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\newcommand{\avg}[1]{\left\langle#1\right\rangle}$$
$$\avg{p}=\frac{n_L+\alpha}{n_L+n_R+\alpha+\beta}$$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\newcommand{\avg}[1]{\left\langle#1\right\rangle}$$
$$\avg{p}=\frac{2}{7}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
p_mean=beta_posteriors[n_b-1].mean()
print(p_mean)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
fig,ax = bl.simple_plot(p=p)
ax.axvline(p, linewidth=0.5,c ='black',label='$p$')
ax.axvline(p_map, linewidth=0.5, c=defaults.map_color, label='MAP')
ax.axvline(p_mean, linewidth=0.5, c=defaults.mean_color, label='mean')
ax.plot(xs,beta_posteriors[n_b-1].pdf(xs), zorder=1, c='blue');
ax.legend()
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Median

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

And finally we have the median. Median has a very clear interpretation: it is a value such that probability that $p$ is less than this value is same as the probability that $p$ is greater than this value, or more exactly

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P\left(p\leq p_{med}\right)\geq\frac{1}{2}\quad\text{and}\quad P\left(p\geq p_{med}\right)\geq\frac{1}{2}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

There is no close formula for the median of the Beta distribution. It can be calculated by solving the equation

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$
CDF[Beta(\alpha,\beta)](x)=\frac{1}{2}
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

where $CDF$ stands for _cumulative distribution function.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots()
ax.set_xlim(0,1);ax.set_ylim(0,1);
ax.plot(xs, beta_posteriors[n_b-1].cdf(xs))
ax.axvline(beta_posteriors[n_b-1].isf(0.5),lw=0.5)
ax.axhline(0.5, lw=0.5); 
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
from scipy.optimize import fsolve
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
p_med = fsolve(lambda x: beta_posteriors[n_b-1].cdf(x)-0.5, x0=[0.2]).item()
p_med
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
beta_posteriors[n_b-1].isf(0.5)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
fig,ax = bl.simple_plot(p=p)
ax.axvline(p, linewidth=0.5,c ='black',label='true')
ax.axvline(1/5, linewidth=0.5, c=defaults.map_color, label='MAP')
ax.axvline(2/7, linewidth=0.5, c=defaults.mean_color, label='mean')
ax.axvline(p_med, linewidth=0.5, c=defaults.median_color, label='median')
ax.plot(xs,beta_posteriors[n_b-1].pdf(xs), zorder=1, c='blue');
ax.legend()
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The median of the Beta distribution can be approximated as

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$median[Beta(\alpha,\beta)]
\approx
\frac{\alpha-\frac{1}{3}}{\alpha+\beta-\frac{2}{3}}$$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_{med}=\frac{n_L+\alpha-\frac{1}{3}}{n_L+n_R+\alpha+\beta-\frac{2}{3}}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
(_a-1/3)/(_a+_b-2/3)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Which is not very far from the true value

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
p_med
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Errors

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Standard deviation

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

So far we did not consider any errors on our estimators. As we have the whole posterior distribution at our disposal, there are many possible ways to define errors. One obvious way would be us use the standard deviation $\sigma$  of the posterior distribution.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\newcommand{\avg}[1]{\left\langle#1\right\rangle}$$
$$\begin{split}
\avg{p} &=\int_0^1\text{d}p\, p\, P_{post}(p)\\
\sigma^2 &= \int_0^1\text{d}p (p-\avg{p})^2 P_{post}(p)
\end{split}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This of course assumes that we are using the mean $\avg{p}$  as the estimate.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

For $Beta(\alpha,\beta)$ distribution the variance is

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}=\frac{\mu(1-\mu)}{\alpha+\beta+1}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

giving the posterior variance

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
\sigma^2 &= \frac{
\avg{p}(1-\avg{p})}{
\alpha+\beta+n_L+n_R+1}
\end{split}
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For uniform prior we obtain

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\sigma^2 = \frac{
\avg{p}(1-\avg{p})}{n_L+n_R+3}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
def var(a, b):
    """Calculates the variance of Beta(a,b) distribution
    """
    return a*b/((a+b)**2*(a+b+1))

def stats(n_l,n_r,*,a=1,b=1):
    """Given an array containing one when ball landed to the left, and zero if on the right
calculates the posterior mean and variance    
    """
    n=n_l+n_r
    a_post = n_l+1;
    b_post = n_r +1
    mu = a_post/(a_post+b_post)
    s = np.sqrt(var(a_post,b_post))
    return n_l, n_r, mu,s
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Below we plot the $\pm\sigma$ interval around mean

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
fig,ax = bl.simple_plot(p=p)
post = beta_posteriors[n_b-1]
ax.plot(xs,post.pdf(xs), zorder=1, c='blue');
n_l, n_r, mu, s = stats(n_l, n_r)
p_mean = post.mean()
ax.axvline(p_mean, lw=0.5, c=defaults.mean_color)
ax.fill_between(xs,post.pdf(xs),0, where = (xs>p_mean-s) & (xs<p_mean+s), alpha=0.5);
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Please note that contrary to the Gaussian distribution the probability contained in within the interval of one standard deviation around mean is not constant

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
d1 = beta(n_l+1,n_r+1)
p1 = d1.mean()
std1 = d1.std()
d1.cdf(p1+std1)-d1.cdf(p1-std1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
d2 = beta(2,1)
p2 = d2.mean()
std2 = d2.std()
d2.cdf(p2+std2)-d2.cdf(p2-std2)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

so the interpretation of this quantity is not so clear.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
print(f"True position = {p:.5f}")
for n in [1,2,3,50,100]:
    n_l = left[:n].sum()
    n_r = n-n_l
    n_l, n_r, mu, s = stats(n_l, n_r)
    print(f"{n_l+n_r:3d} {n_l:3d} {n_r:3d} {mu:6.3f} {s:5.3f} ({mu-p:.4f})")
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Standard deviation around central point

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In the same way we can define standard deviation around any value $c$ e.g. mode or median. We will call this value a _central point_.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The definition is given by

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$
\int_0^1\text{d}p P_{post}(p) (p-c)^2  = \int_0^1\text{d}p  P_{post}(p) (p^2-2 c p +c^2)
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which is equal to

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\avg{p^2} -2 c\avg{p}+ c^2$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

So if we know the first $\avg{p}$ and second moment $\avg{p^2}$ of the distribution, we can calculate the variance around any central point $c$.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The second moment of the Beta distribution is

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\frac{\alpha(1+\alpha)}{(\alpha+\beta)(1+\alpha+\beta)}=\mu\frac{\alpha+1}{\alpha+\beta+1}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
def c_std_beta(c,a,b):
  mean = a/(a+b)
  x2 = mean*(1+a)/(a+b+1)
  return np.sqrt(x2-2*c*mean+c*c)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can also use the `moment` method of the `scipy.stats` distribution.

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
post.moment(2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
s = c_std_beta(p_map, n_l+1, n_r+1)
xs=np.linspace(0,1,500)
fig,ax = bl.simple_plot(p=p)
ax.plot(xs,beta_posteriors[n_b-1].pdf(xs), zorder=1, c='blue');
ax.axvline(p_map,lw=0.5, c=defaults.map_color)
ax.fill_between(xs,beta_posteriors[n_b-1].pdf(xs),0, where = (xs>p_map-s) & (xs<p_map+s), alpha=0.5);
ax.set_title("Standard deviation around MAP")
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Mean absolute deviation (MAD)

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\int d p P(p) |p-c| $$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
import bda
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
bda.stats.mad_c_f(beta(2,6).pdf ,p_map, left=0, right=1)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Confidence interval

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.stats import  cdi_left_f, cdi_central_f
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Another way to express uncertainty is to provide a _confidence interval_. We will call $\beta$-confidence interval any interval $[a,b]$ such that the probability that $p$ lies in this interval is equal to $\beta$.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p\in [a,b])=\beta$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Below we construct two such possible intervals

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
fig,ax = bl.simple_plot(p=p)
ax.plot(xs,post.pdf(xs), zorder=1, c='blue');
_,l,r=cdi_left_f(post.pdf, 0.0,0.75)
ax.fill_between(xs,beta_posteriors[n_b-1].pdf(xs),0, where = (xs>l) & (xs<r), alpha=0.5, color='blue');
_,l,r=cdi_left_f(post.pdf, 0.1,0.75)
ax.fill_between(xs,beta_posteriors[n_b-1].pdf(xs),0, where = (xs>l) & (xs<r), alpha=0.5, color='orange');
ax.set_title(f"Confidence intervals $\\beta=0.75$")
plt.close();
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

One way of choosing a confidence interval is to make it symmetric around $p_{MAP}$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax=bda.plotting.plot_c_vs_hdr_example(post,p_map, beta=0.7,delta=0.05, show=False);
ax.axvline(p_map, c='blue');
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
bda.plotting.plot_c_vs_hdr_example(post,p_map, beta=0.7,delta=0.05);
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The problem with such an interval is that the probabilities on the edges are not equal. If we look at the picture above we can see that the dark blue region is included but  corresponds to smaller probability then the excluded red area to the right.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Highest density region (HDR)

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

$\beta$ HDR is a region where at least $\beta$ of probability is concentrated and has smallest possible volume in the parameter space, hence highest density. More formal definition given below.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Let $P_X(x)$ be de density function of  some random variable $X$ with values in $R$. Let' $R_X(r)$ be the subset of $R_X$ such  that

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$ R_X(r) = \{x\in R: P_X(x)\ge r\}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.hdr_plot import plot_All
from ipywidgets import interact
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plot_All(beta_posteriors[4].pdf,2.0);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plot_All(beta_posteriors[4].pdf,0.8);
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
def plot(r=2.5):
  plot_All(beta_posteriors[4].pdf,r)
  plt.show()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
interactive_plot = interact(plot,r=(0, 2.5,0.01));
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The $\beta$ HDR is equal to $R(r_\beta)$ where $r_\beta$ is the largest constant such that

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P\left(x\in R(r_\beta)\right)\ge \beta$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.stats import hdr_f
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
beta=0.7
lr,th,mass=hdr_f(post.pdf, beta,a=0,b=1)
_,ax=plot_All(beta_posteriors[n_b-1].pdf,th)
ax.axvline(p_map);
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and here is the HDR(0.95)  after all

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
len(left)
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

balls have been thrown

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
beta=0.7
lr,th,mass=hdr_f(beta_posteriors[-1].pdf, beta,a=0,b=1)
fig,ax=plot_All(beta_posteriors[-1].pdf,th)
n_l = left.sum(); n_r = len(left)- n_l;
n_l = left.sum(); n_r = len(left)- n_l;
p_map = n_l/(n_l+n_r);
ax.axvline(p_map);
plt.close()
```

```{code-cell}
fig
```

```{code-cell}

```

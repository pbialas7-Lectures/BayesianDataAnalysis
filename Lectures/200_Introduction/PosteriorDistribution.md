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

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
---
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats as st
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
dc='#1f77b4' #default color
```

+++ {"slideshow": {"slide_type": "slide"}}

# Posterior distribution

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
np.random.seed(1149753)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: skip
tags: [function]
---
def beta_mu_var(mu, s2):
    """Returns Beta distribution object (from scipy.stats) with specified mean and variance"""
    
    nu = mu*(1-mu)/s2 -1
    if nu>0:
        alpha = mu*nu
        beta = (1-mu)*nu
        return st.beta(a=alpha,b=beta)
    else:
        print("s2 must be less then {:6.4f}".format(mu*(1-mu)))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Coin toss

+++ {"slideshow": {"slide_type": "fragment"}}

Let's assume that the $p$ values of coins produced by sloppy blacksmith have Beta distribution  with mean $\mu=0.45$ and standard deviation $\sigma=0.1$.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
prior = beta_mu_var(0.45, 0.1*0.1)
pars = prior.kwds
alpha =pars['a']
beta = pars['b']
print("alpha = {:.2f}, beta={:.2f}".format(alpha,beta))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
uni_prior = st.beta(1,1)
```

+++ {"slideshow": {"slide_type": "notes"}}

We will compare this to uniform prior with $\alpha=\beta=1$. This gives a constant probability density function $P(p)=1$

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,2000)
plt.plot(xs, prior.pdf(xs),    label="$\\alpha = {:5.2f}$ $\\beta = {:5.2f}$".format(alpha, beta));
plt.plot(xs,uni_prior.pdf(xs), label="$\\alpha = {:5.2f}$ $\\beta = {:5.2f}$".format(1, 1));
plt.xlabel('p');
plt.ylabel('P(p)');
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

### Posterior

+++ {"slideshow": {"slide_type": "skip"}}

If we chose $Beta(\alpha, \beta)$ as the prior $P(p)$ the posterior is again Beta distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$Beta(\alpha+n_h,\beta+n_t)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
p_coin = 0.35
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
coin = st.bernoulli(p=p_coin)
```

+++ {"slideshow": {"slide_type": "skip"}}

We will "toss" it 10000 times

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
n_tosses = 10000
tosses = coin.rvs(n_tosses)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [function]
---
def ht(tosses):
    """Takes a list of toss results and returns number of successes and failures"""
    h = tosses.sum()
    t = len(tosses)-h
    return (h,t)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ht(tosses)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [function]
---
def ab_string(a,b):
    return f"$\\alpha = {a:.2f}$ $\\beta = {b:.2f}$"

def draw_beta_prior(a,b,**kargs):
    xs=np.linspace(0,1,1000)
    plt.plot(xs, st.beta(a,b).pdf(
        xs), **kargs)

def draw_beta_posterior(a,b,tosses, **kargs):
    """Draw posterior distribution after  seing tosses assuming Beta(a,b) prior"""
    (h,t)=ht(tosses)
    xs=np.linspace(0,1,1000)
    plt.plot(xs, st.beta(a+h,b+t).pdf(xs), **kargs)
```

```{code-cell} ipython3
ht(tosses[:10])
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's draw the posterior after 10 tosses

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
draw_beta_posterior(alpha,beta,tosses[:10], label=ab_string(alpha, beta))
draw_beta_posterior(1,1,tosses[:10], label=ab_string(1, 1))
plt.legend();
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's discuss again what does this probability distribution mean?

+++ {"slideshow": {"slide_type": "slide"}}

You can thing about it as an outcome of following experiment:
 1. You draw a value for $p$  from the prior distribution
 1. You draw 10 ten times from the Bernoulli distribution with $p$ selected above.
 1. You  repeat the two points above noting each time  $p$ and number of successes.
 1. From the results you select only those where number of successes was equal to `tosses[:10].sum()`
 1. The distributiion of $p$ in this selected results should match our posterior!

+++ {"slideshow": {"slide_type": "skip"}}

 Let's check this.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def experiment(prior,n,size):
    p = prior.rvs(size=size)
    return np.stack((p,st.binom(n=n, p=p).rvs()), axis=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
res = experiment(st.beta(alpha, beta),10,1000000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(res[res[:,1]==tosses[:10].sum()][:,0], bins=50, density=True, label=ab_string(alpha, beta));
draw_beta_posterior(alpha,beta,tosses[:10], label=ab_string(alpha, beta))
draw_beta_posterior(3,3,tosses[:10], label=ab_string(3, 3))
draw_beta_posterior(1,1,tosses[:10], label=ab_string(1, 1))
plt.legend(title = 'Prior');
```

+++ {"slideshow": {"slide_type": "notes"}}

As we can see we indeed get the predicted posterior distribution. Unfortunatelly this requires us to get our prior right.

+++ {"slideshow": {"slide_type": "notes"}}

With more data the dependence of the posterior on the prior diminishes

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
res_100 = experiment(st.beta(alpha, beta),100,1000000)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(res_100[res_100[:,1]==tosses[:100].sum()][:,0], bins=50, density=True);
draw_beta_posterior(alpha,beta,tosses[:100], label=ab_string(alpha, beta))
draw_beta_posterior(3,3,tosses[:100], label=ab_string(1,1), c='red')
draw_beta_posterior(1,1,tosses[:100], label=ab_string(1,1))
plt.legend(title='Prior');
```

+++ {"slideshow": {"slide_type": "notes"}}

So let's see how the posterior distribution evolves with increasing number of tosses. Below we draw posterior distribution after different number of tosses

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
for n in [0, 1,2,3,4,5,10,20,50,100]:
    draw_beta_posterior(alpha,beta,tosses[:n], label="{:d}".format(n))
plt.legend();
plt.axvline(p_coin);
```

+++ {"slideshow": {"slide_type": "skip"}}

And below after some more tosses

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.xlim(0.2,0.5)
for n in [100,200,500,1000,5000]:
    draw_beta_posterior(alpha,beta,tosses[:n], label="{:d}".format(n))
plt.legend();
plt.axvline(p_coin);
```

+++ {"slideshow": {"slide_type": "notes"}}

Let's compare  how the estimated value converges to the real one for different priors. We will use the maximal a posteriori estimate of $p$

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

#### MAP (Maximal a posteriori)

+++ {"slideshow": {"slide_type": "skip"}}

Because mode of the Beta distribution is

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$\frac{\alpha-1}{\alpha+\beta-2},\qquad \alpha, \beta>1$$

+++ {"slideshow": {"slide_type": "skip"}}

the mode of posterior is:

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$ p_{MAP} = \frac{\alpha-1+n_l}{\alpha-1+n_l+\beta-1+n_r}$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
cums  = np.cumsum(tosses)
ns  = np.arange(1.0, len(cums)+1)
avs = cums/ns
post_avs = (cums + alpha-1)/(ns+alpha+beta -2 )
```

+++ {"slideshow": {"slide_type": "notes"}}

So adding a Beta prior amounts to adding $\alpha-1$  and $\beta-1$  repectively to  $n_l$ and $n_r$.

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
l = 500
plt.plot(ns[:l],avs[:l],'.', label='uniform prior');
plt.plot(ns[:l],post_avs[:l],'.', label='prior');
plt.axhline(p_coin, linewidth=1, c='grey')
plt.legend();
```

+++ {"slideshow": {"slide_type": "notes"}}

We can see that after few tens/houndreds of tosses both estimate behave in the same way, but with informative prior  we get better results for small number of tosses.

+++ {"slideshow": {"slide_type": "slide"}}

#### Problem

+++ {"slideshow": {"slide_type": "-"}}

Assume that the coin is fair _.i.e._ the prior has mean equal 1/2 and standard deviation 1 /50. We after $n$ tosses we get $n$ heads. 

How big must $n$ be so  that the posterior probability $P(p>0.75|n,0)$ is greater then 10% ?

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fair_prior = beta_mu_var(0.5, (1/50)**2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
ps = np.linspace(0,1,500)
plt.plot(ps,fair_prior.pdf(ps));
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fair_a = fair_prior.kwds['a']
fair_b = fair_prior.kwds['b']
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
def fair_posterior_p(n):
        d = st.beta(a=fair_a+n, b=fair_b)
        return 1-d.cdf(0.75)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
for n in range(1,1000):
    P= fair_posterior_p(n)
    if P >0.1:
        break
print(n,P)        
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
d = st.beta(a=fair_a+n, b=fair_b)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
plt.plot(ps,d.pdf(ps));
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
d.cdf(0.75)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
plt.axvline(0.75,c='grey', linewidth=1)
plt.axhline(0.1, c='grey', linewidth=1)
plt.plot(ps,1-d.cdf(ps));
```

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

### Posterior predictive distribution

+++ {"slideshow": {"slide_type": "notes"}}

We can ask what is the probability of the coin comming head up after seing it come head up $n_l$ times in $n$ trials? The answer is the integral

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$
P(X=1|n_l,n_r)=\int\limits_0^1\text{d}p \,P(X=1|p) P(p|n_l,n_r) = \int\limits_0^1\text{d}p\, p\, P(p|n_l,n_r)
$$

+++ {"slideshow": {"slide_type": "skip"}}

which is an expectation value (mean) of the posterior  distribution leading to:

+++ {"slideshow": {"slide_type": "fragment", "slideshow": {"slide_type": "fragment"}}}

$$P(X=1|n_l,n_r)=\frac{\alpha+n_l}{\alpha+n_l+\beta+n_r}$$

+++ {"slideshow": {"slide_type": "notes"}}

With uniform prior we obtain the so called

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}}

#### Laplace Rule of succession

+++ {"slideshow": {"slide_type": "-"}}

The probability of succes after seing $n_l$ successes and $n_r$ failures  is

+++ {"slideshow": {"slide_type": "-", "slideshow": {"slide_type": "fragment"}}}

$$P(succes) =  \frac{n_l+1}{n_l+n_r +2}$$

+++ {"slideshow": {"slide_type": "notes"}}

This is also known as  _Laplace smoothing_.

+++ {"tags": ["problem"], "slideshow": {"slide_type": "slide"}}

__Problem__  Amazon reviews

+++ {"slideshow": {"slide_type": "-", "slideshow": {"slide_type": "-"}}, "tags": ["problem"]}

You can buy same item from two  sellers one with 90 positive  and 10 negative reviews and another with 6 positive  and no negative reviews.
From which you should buy ? What assumption you had to make?

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

Let's assume that for each seller sale is  an independent Bernoulli trial with success denoting no problems for the buyer. The other assuption that we are going to make is that all buyers write the reviews.  If so then by the rule of succession probability of success for the first buyer on the next deal is

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
(90+1)/(100+2)
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

and for the second

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
(6+1)/(6+2)
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

We should buy from the first.

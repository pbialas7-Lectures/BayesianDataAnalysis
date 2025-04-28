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

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Monte Carlo methods

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
%matplotlib inline
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [6, 4]
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Posterior distribution

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The goal of the Bayesian inference is to find the posterior distribution of the parameters $\theta$ given the data $y$. The $\theta$ can be a vector of parameters $\theta = (\theta_1,\theta_2,\ldots,\theta_k)$ This distribution can be obtained from sampling distribution $P(y|\theta)$
and the prior distribution of the parameters $P(\theta)$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$P(\theta|y) = \frac{ P(y|\theta) P(\theta)}{P(y)}=\frac{ P(y|\theta) P(\theta)}{\int\text{d}\theta P(y|\theta) P(\theta)}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The problem is the integral in the denominator. In general it is not possible to calculate it analytically. Also standard numerical methods will fail when the dimension of the parameter space is higher than three or four. It's true that not we do not need this integral at all times, for example for MAP estimation. But even then we may need to calculate the marginal distrubutions for only a couple of parameters:

+++ {"slideshow": {"slide_type": ""}, "editable": true}

$$P(\theta_1|y)=\int\text{d}\theta_2\text{d}\theta_2\ldots\text{d}\theta_k P(\theta_1, \theta_2, \ldots,\theta_k |y)$$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Monte Carlo methods

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

A possible solution is to use Monte Carlo methods. An expectation value of random variable and/or its functions is defined by the integral over the probability $P(\theta)$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$E_{P(\theta)}[f(\theta)] \equiv\int\text{d}{\theta} f(\theta) P(\theta)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

If we can sample from the posterior distribution we can calculate the approximate expectation values

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$ E_{P(\theta)}[f(\theta)] \approx \frac{1}{N}\sum_{i=1}^N f(\theta^i), \quad \theta^{i}\sim P(\theta)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Here the $\theta\sim P(\theta)$ means that $\theta$ is sampled from distribution $P(\theta)$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For example to  estimate the mean of the parameter $\theta_1$ we can use

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$E_{P(\theta_1)}[\theta_1] \approx \frac{1}{N}\sum_{i=1}^N \theta_1^i, \qquad \theta^{i}\sim P(\theta^{i})$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The MAP estimate would be more difficult to estimate, but we can still do it by binning the obtained values and finding the bin with highest numbers of counts. This is only possible for a small number of dimensions.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Generating  random numbers

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For this to work we a need a way to generate random number from an arbitrary, often complex distribution.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Discrete distributions

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

A discrete distribution is defined by a set of probabilities $p_i$ for $i=0,1,\ldots,N-1$ such that

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_i,\quad \sum_ip_i=1,\quad i=0,\ldots,N-1$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
p = np.array([1 / 48, 3 / 48, 1 / 3, 1 / 2, 1 / 12])
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can view this as a set of intervals  or bins

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$[0,p_0),[p_0,p_0+p_1),[p_0+p_1,p_0+p_1+p_2),\ldots,[\sum_{i=0}^{N-2}p_i,1)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

or

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$[0,c_0), [c_0,c_1),\cdots\qquad  c_i = \sum_{j=0}^i p_j$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Coefficients $c_i$ can be calculated by `cumsum` function

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
cum = np.cumsum(p)
cum
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can now draw a random number $u$ from uniform distribution $u\in [0,1)$ and check in which interval it falls.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$u\in [0,1)$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
u = np.random.uniform(0, 1)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The most effient way to do this is to use `searchsorted` function that uses the binary search algorithm.

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
np.searchsorted(cum, u)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
n_samples = 1000
u = np.random.uniform(0, 1, n_samples)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
sample = np.searchsorted(cum, u)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
plt.hist(sample, bins=len(p), range=(-0.5, len(p) - 0.5), histtype='bar', rwidth=0.75, label='true')
plt.scatter(np.arange(len(p)), p * n_samples, color='red', s=100, label='generated')
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Markov Chain Monte-Carlo

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For continuous distribution and/or high dimensional parameter space we can use Markov Chain Monte-Carlo methods. The idea is to generate a sequence of random numbers $\theta^0,\theta^1,\theta^2,\ldots$ such that the distribution of $\theta^n$ converges to the desired distribution $P(\theta|y)$. The step from $\theta^n$ to $\theta^{n+1}$ usually entails only a small change in $\theta$ and is determined by a transition probability

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(\theta^{n+1}|\theta^n)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We require that

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$ \int \text{d}\theta^{n+1} P(\theta^{n+1}|\theta^{n}) = 1$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and that every possible $\theta$ can be reached from any other $\theta$ in a finite number of steps.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The  distribution after $n+1$ steps is given by the recursive relation

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P_{n+1}(\theta^{n+1})=\int\text{d}{\theta^{n}} P_n(\theta^{n})P(\theta^{n+1}|\theta^{n})$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and the stationary distribution is given by

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(\theta)=\int\text{d}{\theta'} P(\theta')P(\theta|\theta')$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Detailed balance

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

If the transition probability satisfies so called detailed balance condition

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(\theta')P(\theta|\theta') = P(\theta) P(\theta'|\theta)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

then the stationary distribution is the desired distribution $P(\theta)$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\int\text{d}{\theta'} P(\theta')P(\theta|\theta')= \int\text{d}{\theta'} P(\theta)P(\theta'|\theta)$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\int\text{d}{\theta'} P(\theta)P(\theta'|\theta) = P(\theta)\underbrace{\int\text{d}{\theta'} P(\theta'|\theta)}_1 = P(\theta)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Metropolis-Hastings algorithm

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

One way of satysfy the detailed balance is to use Metropolis-Hastings algorithm. We start with proposing a new trial value $\theta$ from a trial distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P_{trial}(\theta|\theta')$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We then accept the new value with probability

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{acp}=\min\left\{1,
\frac{P(\theta)P_{trial}(\theta'|\theta)}{P(\theta')P_{trial}(\theta|\theta')}
\right\}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

If the trial probability is symmetric

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P_{trial}(\theta|\theta') =P_{trial}(\theta'|\theta) $$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$p_{acp}=\min\left\{1,
\frac{P(\theta)}{P(\theta')}
\right\}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

That means that if the new value is more probable than the old value we always accept it. If the new value is less probable, we accept it with probability which is the ratio of the probability of the new value to previous value. Rejecting the new value means that we keep the old value.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In practice we will usually work with the logarithm of the probabilities

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(x) = e^{\log P(x)}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The algorithm is then

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

```
if log P(theta_proposed) > log P(theta):
    theta = theta_proposed
else:
   r = np.random.uniform()
   if r< exp(log P(theta_proposed) -log P(theta) ):
   theta = theta_proposed
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Example  - Normal distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(x|\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{\displaystyle -\frac{(x-\mu)^2}{2\sigma^2}}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\log P(x|\mu,\sigma) =  -\frac{(x-\mu)^2}{2\sigma^2} -\frac{1}{2}\log(2\pi)-\log \sigma $$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The two last terms are constant and can be omitted. Code below implements the algorithm described above

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def log_norm(x, mu=0, s=1):
    return -0.5 * (x - mu) * (x - mu) / (s * s)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def mcmc(log_p, x, size, eps=0.1):
    accepted = 0
    prev_x = x;
    chain = [prev_x] 
    prev_log_p = log_p(prev_x)
    for i in range(size):
        trial_x = prev_x + np.random.uniform(-eps, eps) # trial value
        trial_log_p = log_p(trial_x)
        accept = True
        if (trial_log_p < prev_log_p):
            r = np.random.uniform(0, 1)
            if (r > np.exp(trial_log_p - prev_log_p)):
                accept = False
                
        if accept:
            prev_x = trial_x
            prev_log_p = trial_log_p
            accepted +=1
            
        chain.append(prev_x)
    return np.asarray(chain), accepted/size
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
%%time
sigma = 0.5
mu = 1
chain, acceptance = mcmc(lambda x: log_norm(x, mu, sigma), -10 , 1000000, 1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
acceptance
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's look at the first samples

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.plot(chain[:100])
plt.axhline(mu, color='red');
plt.axhline(mu+sigma, color='red', linestyle='--')
plt.axhline(mu-sigma, color='red', linestyle='--')
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This picture show a very important point: it can take some time before our process  converges to the desired stationary distribution. That's why we have to discard some number of initial samples. How much depends unfortunately on the process.  From the plot it look that one hundred may be enough.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
clean_chain = chain[100:]
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's check if the distribution is correct

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain, bins=50, density=True, align='mid')
xs = np.linspace(-4 * sigma, 4 * sigma, 100)
plt.plot(xs, st.norm(mu, sigma).pdf(xs))
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This looks good, although more formal test would be needed to really ascertain the correctness of the algorithm.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Statistical errors

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Once we have a mean to produce samples from a distributions we need  a way to estimate the error of the estimated quantities. Let's take the mean

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\bar x = \frac{1}{N}\sum_{i=1}^N x_i.$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Because each $x_i$ a random variable from distribution $P(X)$ the mean itself is also a random variable so we may ask what is its the expectation value. Using tha fact that expectation value is linear we obtain

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$E[\bar x] = \frac{1}{N}\sum_{i=1}^N E[x_i] = E[x]$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which prooves that $\bar x$ is a _unbiased_ estimator. To estimate the error we need the variance of this estimator

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$var[\bar x]\equiv E\left[(\bar x-E[\bar x])^2)\right]=E\left[(\bar x-E[x])^2)\right]$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Inserting the definition of $\bar x$ we obtain

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$E\left[(\bar x-E[\bar x])^2)\right] = E\left[\frac{1}{N^2}\sum_{i,j=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right]$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The sum has $N^2$ term and may be split in two part: $N$ term such that $i=j$  and the remaining $N(N-1)$ terms where $i\neq j$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
E\left[\frac{1}{N^2}\sum_{i,j=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right] &=
E\left[\frac{1}{N^2}\sum_{i=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_i-E[\bar x])\bigr)\right]\\
&\phantom{=}+
E\left[\frac{1}{N^2}\sum_{i\ne j} \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right].
\end{split}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The first term is just the variance of $x$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$E\left[\frac{1}{N^2}\sum_{i=1}^N \bigl(x_i-E[\bar x]\bigr)\bigl(x_i-E[\bar x])\bigr)\right]=\frac{1}{N}Var[x] $$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The second term is the sum correlation coefficients between $x_i$ and $x_j$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
E\left[\frac{1}{N^2}\sum_{i\ne j} \bigl(x_i-E[\bar x]\bigr)\bigl(x_j-E[\bar x])\bigr)\right] &=\frac{1}{N^2}\sum_{i\ne j}cov(x_i,x_j)
\end{split}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

If the $x_i$ are _uncorrelated_  which is usual the case with independent measurements then  $cov(x_i,x_j)=0$ and we are left with  the usual formula

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$var[\bar x] = \frac{1}{N}var[x]$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which means that the error falls down like $N^{-\frac{1}{2}}$.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

However the samples produced from the Markov Chain are *not* idependent. That is because at each step we make only small changes to the previous configuration.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Autocorrelation

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

It is often better to use the correlation coefficient instead of covariance

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$corr(x_i,x_j)=\frac{cov(x_i,x_j)}{var[x]}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Below is a plot of the correlations between $x_i$ and $x_j$  as a function of $|j-i|$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import sys
sys.path.append('../../src')
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.autocorr import ac_and_tau_int
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
tau_int, ac = ac_and_tau_int(clean_chain)
plt.plot(ac,'.');
plt.grid();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Clearly the successive samples are not independent. Please contrast this with the samples from build in random number generator that produces (almost) independent samples

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
_, ac = ac_and_tau_int(np.random.uniform(-1, 1, 100000))
plt.plot(ac,'.');
plt.grid();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

After some manipulations we can write the final formula for the variance of $\bar x$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$Var[\bar x] =  \frac{Var[x]}{N}\left(1+2\sum_{j=1}^{\frac{N}{2}}  corr(x_0,x_{j})\right)= 2\frac{Var[x]}{N}\tau_{int}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

It still falls off like $N^{-1}$ but there is an additional factor called _integrated autocorrelation time_

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\tau_{int}=\frac{1}{2}+\sum_{j=1}^{\infty}  corr(x_0,x_{j})$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In practice the integrated autocorrelation time is tricky to estimate from the finite sample. You can use the `ac_and_tau_int` function from the `bda.autocorr` module.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
tau_int, ac = ac_and_tau_int(clean_chain)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

It returns both $\tau_{int}$ and the autocorrelation function. In this case

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
tau_int
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

So finallly our error on the mean

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
clean_chain.mean()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

as measured by standard deviation is

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
np.std(clean_chain)*np.sqrt(2*tau_int/len(clean_chain))
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We can write the formula for variance also as

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$var[\bar x]=\frac{var[x]}{N_{eff}}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

where $N_{eff}$ is the _effective sample size

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$N_{eff}=\frac{N}{2\tau_{int}}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

which in this case equals

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
len(clean_chain)/(2*tau_int)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
import arviz as az
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
az.ess(clean_chain)
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Acceptance

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

If the autocorrelation is caused by the small changes between the samples then maybe increasing the magnitude of the changes may result in smaller autocorrelation time? Unfortunately  the bigger the change the smaller probability of accepting such a change. If the acceptance rate becomes small then many proposed smaples are rejected and we are left with streaks of same samples leading again to large autocorrelations. Let's check this out

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
chain, acceptance = mcmc(lambda x: log_norm(x, mu, sigma), 1.0 , 100000, 2)
tau_int, ac = ac_and_tau_int(chain)
print(acceptance, tau_int)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Increasing the `eps` to two reduced the acceptance as expected but the autocorrelation time droped. Let's increase the `eps` even further

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
chain, acceptance = mcmc(lambda x: log_norm(x, mu, sigma), 1.0 , 100000, 4)
tau_int, ac = ac_and_tau_int(chain)
print(acceptance, tau_int)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Aceeptance droped and  autocorrelation time increased. On the plot below we can clearly see the streaks of rejections

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.plot(chain[:100],'.-', linewidth=1);
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Further increase of `eps` leads to further increase of autocorrelation time.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
chain, acceptance = mcmc(lambda x: log_norm(x, mu, sigma), 1.0 , 100000, 10)
tau_int, ac = ac_and_tau_int(chain)
print(acceptance, tau_int)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Reducing `eps` leads to increased acceptance but also increased autocorrelation time

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
chain, acceptance = mcmc(lambda x: log_norm(x, mu, sigma), 1.0 , 100000, 0.5)
tau_int, ac = ac_and_tau_int(chain)
print(acceptance, tau_int)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

So in practice the `eps` or equivalent parameter should be carefully tuned to minimize the autocorrelation time.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Example - Normal data with uninformative priors

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Our next example will be the normal model with unknown mean $\mu$ and variance $\sigma^2$ with uninformative prior

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$P(\mu,\sigma^2)=\frac{1}{\sigma^2}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The posterior probability distribution that we need to sample is

+++ {"slideshow": {"slide_type": ""}, "editable": true}

$$
P(\mu,\sigma^2|y) \propto  (\sigma^2)^{-\frac{n+2}{2}} 
e^{\displaystyle -\frac{n}{2\sigma^2}\left(\bar y -\mu\right)^2 -\frac{n-1}{2\sigma^2}s^2},\quad s^2=\frac{n}{n-1}\left(\overline{y^2} -{\bar y }^2\right)
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The function below calculates the log probability of this distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
def log_mu_sig2(mu, sig2, n, y_bar, s2):
    log_p = -0.5 * (n + 2) * np.log(sig2)
    log_p -= 0.5 * n * (mu - y_bar) * (mu - y_bar) / (sig2)
    log_p -= 0.5 * (n - 1) * s2 / sig2
    return log_p
```

```{raw-cell}
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: skip
---
This time our distribution is two dimensional and one of the components, $\sigma^2$ is constrained to be positive. This is reflected in the generating function below
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def gen(x, eps):
        trial = x + np.random.uniform(-1, 1, 2) * eps
        trial[1] = np.abs(trial[1])
        return trial
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The `eps` variable can be a tuple of numbers $(\epsilon_\mu, \epsilon_{\sigma^2})$  i.e. we ca have different values for $\mu$ and $\sigma^2$.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["problem"]}

__Problem__

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["problem"]}

Please show that this trial distribution is symmetric.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The function that generates the chain is almost identical to the previous one, but we also append the $\log P(\mu^i,(\sigma^2)^i)$ for each $i$.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
def mcmc2(log_p,x0, size, eps):
    prev_x = x0;
    prev_log_p = log_p(*prev_x)
    chain = []
    chain.append(np.append(x0, prev_log_p))
    for i in range(size):
        trial_x = gen(prev_x, eps)
        trial_log_p = log_p(*trial_x)
        accept = True
        if (trial_log_p < prev_log_p):
            r = np.random.uniform(0, 1)
            if (r > np.exp(trial_log_p - prev_log_p)):
                accept = False
        if accept:
            prev_x = trial_x
            prev_log_p = trial_log_p
        save = np.append(prev_x, prev_log_p)
        chain.append(save)
    return np.asarray(chain)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Let's first generate some artificial data by drawing 20 samples from a normal distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
mu = 1
sigma = 2
var = sigma**2
y = np.random.normal(mu, sigma, 20)
y_bar = y.mean()
s2 = y.var(ddof=1)
print(y_bar, s2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
log_p = lambda m, s: log_mu_sig2(m, s, 20, y_bar, s2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
%%time
chain2 = mcmc2(log_p, x0=[0,1], size=500000, eps=(1,1))
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Plotting the first 500 values of $\mu$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.plot(chain2[:500, 0])
plt.ylabel("$\\mu$")
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

we may notice that there are some   correlation. The effect is more pronounced for $\sigma^2$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.plot(chain2[:500, 1])
plt.ylabel("$\\sigma$")
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We will drop the first 5000 samples

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
clean_chain2=chain2[5000:]
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

First we check the autocorellation time for $\mu$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
tau_int_mu, ac_mu = ac_and_tau_int(clean_chain2[:,0])
print(f"tau_int_mu = {tau_int_mu:.2f}")
plt.plot(ac_mu,'.');
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and then for $\sigma^2$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
tau_int_var, ac_var = ac_and_tau_int(clean_chain2[:,1], maxlen=500)
print(f"tau_int_var = {tau_int_var:.2f}")
plt.plot(ac_var,'.');
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The correlations for $\sigma^2$ are quite big.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Marginal distribution $P(\mu|y)$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Using the simulated samples we can plot the resulting marginal $\mu$  distribution  $P(\mu|y)$ and compare it to the true value of $\mu$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain2[:, 0], bins=60, histtype='step', density=True,label=r"$P(\mu|y)$");
plt.axvline(mu, color='red',label=r"$\mu_{true}$");
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The mean of this posterior distribution is

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(f"{clean_chain2[:,0].mean():.6f}")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

In this particular case we know this distribution analytically (see the normal model lectures). It is a Student's $t-$distribution

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

$$\frac{\mu-\bar y}{\sqrt{\frac{s^2}{n}}} \sim t(n-1)$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
t_dist = st.t(df=len(y) - 1, loc=y_bar, scale=np.sqrt(s2/len(y)))
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The mean of this distribution is $\bar y$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
y_bar
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

So as we can see the Monte-Carlo estimate is very  good

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The standard deviation of this distribution is

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
t_dist.std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

again in good agreement with the Monte-Carlo estimate

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
clean_chain2[:,0].std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The highest density  interval can be calculated using the `hdi` function from `arviz`

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
hdi_mu = az.hdi(clean_chain2[:,0], hdi_prob=.95)
print(hdi_mu)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain2[:, 0], bins=60, histtype='step', density=True,label=r"$P(\mu|y)$");
plt.axvline(mu, color='red',label=r"$\mu_{true}$");
plt.axvspan(*hdi_mu, color='grey',alpha=0.5, label='hdi')
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Finally let's compare the whole distribution with the Monte-Carlo histogram

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain2[:, 0], bins=50, histtype='step', density=True)
mus = np.linspace(-1, 5, 100)
plt.plot(mus, t_dist.pdf(mus));
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

### Marginal distribution $P(\sigma^2|y)$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Now let's repeat this with the marginal $\sigma^2$ distribution $P(\sigma^2|y)$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain2[:, 1], bins=60, histtype='step', density=True, label=r"$P(\sigma^2|y)$");
plt.axvline(var, color='red',label=r"$\sigma^2_{true}$");
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

This distribution is also known analytically and is the [scaled inverse $\chi^2$ distribution](https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution)

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\sigma^2|y \sim \operatorname{Scaled-Inv-}\chi^2(n-1,s^2)=  \operatorname{Inv-\Gamma}(\frac{n-1}{2},\frac{(n-1)s^2}{2})$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
sc_inv_gamma_dist = st.invgamma((len(y)-1) / 2., scale=(len(y)-1) * s2 / 2.)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The mean of this distribution is

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
sc_inv_gamma_dist.mean()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

to be compared with

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
clean_chain2[:,1].mean()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For the standard deviation we obtain

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
sc_inv_gamma_dist.std()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and

```{code-cell}
clean_chain2[:,1].std()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
hdi_sigma2 = az.hdi(clean_chain2[:,1],hdi_prob=0.95)
print(hdi_sigma2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain2[:, 1], bins=60, histtype='step', density=True, label=r"$P(\sigma^2|y)$");
plt.axvline(var, color='red',label=r"$\sigma^2_{true}$");
plt.axvspan(*hdi_sigma2,color='grey',alpha=0.5, label='hdi');
plt.legend();
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

And finally the full distribution comparison

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
plt.hist(clean_chain2[:, 1], bins=50, histtype='step', density=True)
xs = np.linspace(0, 15, 100)
plt.plot(xs, sc_inv_gamma_dist.pdf(xs))
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

## Summary

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---

```

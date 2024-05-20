---
jupytext:
  cell_metadata_json: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
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
slideshow:
  slide_type: skip
---
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats as st
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
import sys
sys.path.append("../../src")
import bda.stats as bst
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
dc='#1f77b4' #default color
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import billiard as bl
```

+++ {"slideshow": {"slide_type": "slide"}}

# Bayesian Data Analysis

+++

> ### Statistical inference is concerned with drawing conclusions, from numerical data, about quantities that are not observed.
>  "Bayesian Data Analysis" A. Gelman, J. B. Carlin, H. S. Stern,  D. B. Dunson, A. Vehtari, D. B. Rubin

+++ {"slideshow": {"slide_type": "slide"}}

### Literature

+++ {"slideshow": {"slide_type": "-"}}

- "Bayesian Data Analysis" A. Gelman, J. B. Carlin, H. S. Stern,  D. B. Dunson, A. Vehtari, D. B. Rubin.[[pdf](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)]
- "Data Analysis, a Bayesian Tutorial" D.S. Silva with J. Skiling.
- "Bayesian methods for Hackers, Probabilistic Programming adn Bayesian Inference" C. Davidson-Pilon.  [[online](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)]
- "From Laplace to Supernova SN 1987A: Bayesian Inference in Astrophysics" T. J. Loredo. [[pdf](https://bayes.wustl.edu/gregory/articles.pdf)]

+++ {"slideshow": {"slide_type": "slide"}}

## Reverend Thomas Bayes' original example -- pool table

+++ {"slideshow": {"slide_type": "skip"}}

We will start with the original example considered by the Reverend Thomas Bayes in his paper [["An essay towards solving a problem in the doctrine of chances"](ThomasBayespaper.pdf)] which was published posthumously by Richard Price in Philosophical Transactions __53__  (1763).

+++ {"slideshow": {"slide_type": "skip"}}

Imagine that somebody places a  billiard ball at random on the billiard table. The location of the ball is unknown. Then we start to throw the balls at random at the table and somebody is reporting to us if the balls have landed on the left or on the right of the original ball. The problem is really one dimensional as we are only interested in ball position along one side of the table and we can describe it by one number $p\in[0,1]$. 

The problem considered by T. Bayes was how our belief about the ball position changes with each new ball thrown on the table.

+++ {"slideshow": {"slide_type": "skip"}}

So let's place our original ball (black) on the table

```{code-cell}
---
slideshow:
  slide_type: skip
---
p =  np.pi/10.0
y =  0.786
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig,ax,pax = bl.make_pool_figure(p)
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True);
```

+++ {"slideshow": {"slide_type": "slide"}}

## Probability

+++ {"slideshow": {"slide_type": "skip"}}

Our "belief" as to the position of the ball will be expressed by a _probability distribution_ characterized by the _probability distribution function_ (pdf) (see `100_Probability/continuous_distributions.md` notebook)

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p)$$

+++ {"slideshow": {"slide_type": "skip"}}

This assigns a non-negative number to each possible position $p$ on the table. The higher the number the stronger our belief that the ball is near this location. The distribution function has of course to be normalized

+++ {"slideshow": {"slide_type": "fragment"}}

$$\int_0^1\!\!\text{d}p\, P(p)=1$$

+++ {"slideshow": {"slide_type": "skip"}}

Given that, the probability that the ball lies in the interval $[a,b]$ is given by the integral

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p\in [a,b])=\int_a^b\!\!\text{d}p\, P(p)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For each $a<=b$ this number is a _probability_ and it fulfills all the probability axioms presented in the `100_Probability/probability.md` notebook.

+++ {"slideshow": {"slide_type": "slide"}}

### What is probability ?

+++ {"slideshow": {"slide_type": "slide"}}

>#### "One sees, from this Essay, that the theory of probabilities is basically just common sense reduced to calculus; it makes one appreciate with exactness that which accurate minds feel with a sort of instinct, often without being able to account for it."
> "Théorie Analytique des Probabilités" Pierre-Simon Laplace

+++ {"slideshow": {"slide_type": "skip"}}

The meaning of this number (probability) is not always clear, I will stick with the Bayesian interpretation as the measure of our certainty of that this event will happen. In particular zero probability means that we are sure this event will not happen, and probability one means that we are certain that it will happen.

+++ {"slideshow": {"slide_type": "slide"}}

## Prior

+++ {"slideshow": {"slide_type": "skip"}}

We start with our belief as to the position of the ball on the table before we get any additional information. That is called a _prior_ from latin _a priori_. 

We will assume that the placement was uniformly random

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p)=1$$

+++ {"slideshow": {"slide_type": "skip"}}

which is indicated in the plot below by the blue line.

```{code-cell}
---
slideshow:
  slide_type: skip
---
xs = np.linspace(0,1,1000)
prior = np.vectorize(lambda x: 1.0)
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
pax.plot(xs,prior(xs), zorder=1, c='blue', label="$P(p)$");
pax.legend();
```

+++ {"slideshow": {"slide_type": "slide"}}

#### First ball

+++ {"slideshow": {"slide_type": "skip"}}

Now we throw another ball marked in red on the plot below, which lands on the right side of the original ball. Actually I will throw 100 balls at once, but use them one by one.

```{code-cell}
---
slideshow:
  slide_type: skip
---
np.random.seed(87) ## Sets the seed so that each time the notebook is started it will  give same results.
x = st.uniform(loc=0, scale=1).rvs(size=(100,2)) #location of the balls
left=(x[:,0]<=p) + 0
```

```{code-cell}
---
slideshow:
  slide_type: fragment
---
fig,ax,pax = bl.make_pool_figure(p,ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, x,left, bl.cs)
```

+++ {"slideshow": {"slide_type": "skip"}}

Intuitively it changes our belief as to where the first ball is located notably it becomes less likely that the ball is close to the right edge. Such change of probability distribution is described by the _conditional_ probability.

+++ {"slideshow": {"slide_type": "slide"}}

## Conditional probability

+++ {"slideshow": {"slide_type": "skip"}}

Conditional probability $P(A|B)$ is the probability of event $A$ provided that event $B$ happened and is given by the formula

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A|B) = \frac{P(A\cap B)}{P(B)}$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": ["example", "heading"]}

#### Example

We are rolling two dices, what is the probability that a three appeared in the results?

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [hide]
---
from itertools import product
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
S = list(product(range(1,7), range(1,7)) )
```

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [example, hide]
---
print(S)
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
A = list(filter(lambda x: x[0]==3 or x[1]==3, S))
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
print(A)
```

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [example, hide]
---
from fractions import Fraction
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
print(Fraction(len(A),len(S)) )
```

+++ {"slideshow": {"slide_type": "-"}}

How does that probability changes when we know that the sum of the to results is odd? What if the result is even?

```{code-cell}
---
slideshow:
  slide_type: skip
---
rem = 1
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
B = list(filter(lambda x: (x[0]+x[1])%2 == rem, S))
len(B)
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
AcapB = list(filter(lambda x: (x[0]+x[1])%2 == rem, A))
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
print(Fraction(len(AcapB),len(B)))
```

+++ {"slideshow": {"slide_type": "slide"}}

### Product rule

+++ {"slideshow": {"slide_type": "skip"}}

The definition of the conditional probability can be rewritten as the _product rule_

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A\cap B)= P(A|B)P(B)$$

+++ {"slideshow": {"slide_type": "skip"}}

which can be generalized to multiple sets

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$B = \bigcup_{i}  B_i,\; \bigvee_{i\neq j} B_i\cap B_j=\emptyset \implies P(B)=\sum_i P(B_i)$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A\cap B)= \sum_i P(A|B_i)P(B_i)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

In special case when $B=S$ this gives

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\bigcup_i B_i = S \implies P(A)= \sum_i P(A|B_i)P(B_i)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This is very useful for defining more complex probability distributions.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Height distribution

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

 For example the height distribution in human population can be decomposed in this way into height distribution for women and men, each separately being a normal distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(h)=P(h|f)P(f)+P(h|m)P(m)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Back to biliard

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Going back to our billiard example what we are looking for it the conditional probability of ball being at $p$ on condition that the second ball landed on the right of it

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

$$P(p|r)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Sampling distribution/ likelihood

+++ {"slideshow": {"slide_type": "skip"}}

We will start by calculating the _reverse_ conditional probability

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(r|p)$$

+++ {"slideshow": {"slide_type": "skip"}}

that the second ball lands on the right provided that first ball is at $p$. This called the _sampling distribution_ and  in this case this is the Bernoulli distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax,pax = bl.make_pool_figure(p,ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, x,left, bl.cs)
```

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(r|p)=1-p\quad P(l|p)=p$$

+++ {"slideshow": {"slide_type": "skip"}}

If we consider $P(r|p)$ as a function of $p$ we will call it the _likelihood_

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$L(p|r)\equiv P(r|p)$$

+++ {"slideshow": {"slide_type": "skip"}}

Please note that this __is not__ a probability density function.

+++ {"slideshow": {"slide_type": "slide"}}

## Bayes' theorem

+++ {"slideshow": {"slide_type": "skip"}}

To go from $P(r|p)$ to $P(p|r)$ we will use the Bayes theorem which can be simply derived from the definition of the conditional probability. Substituting the product rule into the definition of $P(B|A)$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(B|A)= \frac{P(A\cap B)}{P(A)}$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$\boxed{P(B|A)=\frac{P(A|B)P(B)}{P(A)}}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Posterior

+++ {"slideshow": {"slide_type": "skip"}}

With setting  $B\equiv p$ and $A\equiv r$ we obtain the _posterior_ distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p|r)=\frac{P(r|p)P(p)}{P(r)}$$

+++ {"slideshow": {"slide_type": "skip"}}

The denominator $P(r)$ is the normalization factor which in this case can be simply calculated

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(r)=\int_{0}^1\text{d}p P(r|p)P(p)= \int_0^1\text{d}p (1-p)= 1-\frac{1}{2}=\frac{1}{2}$$

+++ {"slideshow": {"slide_type": "skip"}}

leading finally to the distribution

+++ {"slideshow": {"slide_type": "slide"}}

$$P(p|r)=2-2p$$

+++ {"slideshow": {"slide_type": "skip"}}

denoted by the dark blue line on the plot below.

```{code-cell}
---
slideshow:
  slide_type: skip
---
posteriors = [lambda x: 2-2*x]
```

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
n=1
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, x,left, bl.cs)
alpha=bl.plot_posteriors(posteriors[:2])    
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   

plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Maximal a posteriori

+++ {"slideshow": {"slide_type": "skip"}}

The probability distribution $P(p|r)$ contains  most complete information about the location of the first ball, however what we  often want is an _estimate_ for $p$. Similarly to [[Maximal likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)] we can define a maximal a posteriori (MAP) estimator

+++ {"slideshow": {"slide_type": "fragment"}}

$$p_{map}=\operatorname{argmax}_p P(p|r)$$

+++ {"slideshow": {"slide_type": "skip"}}

which in this case is equal to zero.

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
def find_pmap(posterior):
    xs = np.linspace(0,1,1000)
    post = posterior(xs)
    i_max = np.argmax(post)
    return xs[i_max], post[i_max]
```

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
n=1
fig,ax,pax = bl.make_pool_figure(p,ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, x,left, bl.cs)
alpha=bl.plot_posteriors(posteriors[:n], ax=pax)
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map, y_map = find_pmap(posteriors[n-1])
pax.annotate(f'MAP $p={p_map:.1f}$',(p_map, y_map),(p_map+0.2, y_map), 
             fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

#### Second ball

+++ {"slideshow": {"slide_type": "skip"}}

We can continue by throwing another ball which again lands on the right of the original ball

```{code-cell}
---
slideshow:
  slide_type: skip
tags: [aux_code]
---
nb=2;n=1
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, x,left, bl.cs)
plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

We can repeat the process but this time we will use the posterior as our new prior, as this represents our present belief as to where the original ball is located

+++ {"slideshow": {"slide_type": "slide"}}

$$P(p|r,r) = \frac{P(r|p) P(p|r)}{\int_0^1\text{d}p\,P(r|p) P(p|r)}$$

+++ {"slideshow": {"slide_type": "skip"}}

The normalizing factor is again easy to calculate

+++ {"slideshow": {"slide_type": "fragment"}}

 $$P(r|p) P(p|r)=2(1-p)^2,\quad 2\int_0^1\text{d}p (1-p)^2=\frac{2}{3}$$

+++ {"slideshow": {"slide_type": "skip"}}

leading to the new posterior distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p|r,r)=3(1-p)^2$$

+++ {"slideshow": {"slide_type": "skip"}}

which does not change the MAP estimate which remains zero.

```{code-cell}
---
slideshow:
  slide_type: skip
---
posteriors.append(lambda x: 3*(1-x)**2)
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
nb=2;n=2
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, x,left, bl.cs)
alpha=bl.plot_posteriors(posteriors[:n], ax=pax)
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
    
p_map, y_map = find_pmap(posteriors[n-1])
pax.annotate(f'MAP $p={p_map:.1f}$',
             (p_map, y_map),(p_map+0.2, y_map-0.25), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
plt.close()
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}, "tags": ["problem"]}

#### Problem

How did the probability that $p<0.05$ changed?

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

This probability is given by the integral

+++ {"slideshow": {"slide_type": "skip"}}

$$\int_0^{0.05}\text{d}p \tilde{P}(p)$$

+++ {"slideshow": {"slide_type": "skip"}}

Substituting prior $P$ for $\tilde{P}$ we obtain

+++ {"slideshow": {"slide_type": "skip"}}

$$\int_0^{0.05}\text{d}p =0.05$$

+++ {"slideshow": {"slide_type": "skip"}}

For two priors we get

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$
\int_0^{0.05}\text{d}p P(p|r) = 
\int_0^{0.05}\text{d}p 2(1-p) = 2\cdot(0.05 -\frac{1}{2}0.05^2)=0.0975
$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$
\int_0^{0.05}\text{d}p P(p|r) = 
\int_0^{0.05}\text{d}p 3(1-p)^2\approx 0.142
$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

We see that this probability successively  increases.

+++ {"slideshow": {"slide_type": "slide"}}

#### Third ball

+++ {"slideshow": {"slide_type": "skip"}}

Throwing a third ball we notice that it lands on the left of the original ball.

```{code-cell}
---
slideshow:
  slide_type: skip
---
nb=3;n=2
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, x,left, bl.cs)
plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "slide"}}

### Binomial distribution

+++ {"slideshow": {"slide_type": "skip"}}

We could continue in the similar fashion as before that but for larger amount of balls it would quickly become tedious. Instead will calculate the posterior in "one go". Assuming that w $n_l$ balls had landed on the left and $n_r$ on the right, the posterior is given by

+++ {"slideshow": {"slide_type": "slide"}}

$$P(p|n_l,n_r)=\frac{P(n_l,n_r|p)P(p)}{P(n_l,n_r)}$$

+++ {"slideshow": {"slide_type": "skip"}}

The sampling distribution is given by the binomial distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n_l,n_r|p) = \binom{n_l+n_r}{n_l}p^{n_l}(1-p)^{n_r}$$

+++ {"slideshow": {"slide_type": "skip"}}

The normalizing factor is now more complicated but can be calculated using e.g. _Mathematica_.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

\begin{split}
\int_0^1\text{d}p'& P(n_l,n_r|p')P(p') \\&= \binom{n_l+n_r}{n_l}
\frac{n_l!n_r!}{(n_l+n_r+1)!}=\frac{1}{n_l+n_r+1}
\end{split}

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

leading to the posterior

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(p|n_l,n_r)=\binom{n_l+n_r}{n_l}(n_l+n_r+1)p^{n_l}(1-p)^{n_r}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Beta distribution

+++ {"slideshow": {"slide_type": "skip"}}

After some manipulations

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

\begin{split}
&\binom{n_l+n_r}{n_l}(n_l+n_r+1)\\
&\kern5mm=\frac{(n_l+n_r)!(n_l+n_r+1)}{n_l!n_r!}
=\frac{(n_l+n_r+1)!}{n_l!n_r!}
\end{split}

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\frac{(n_l+n_r+1)!}{n_l!n_r!}=\frac{\Gamma(n_l+n_r+2)}{\Gamma(n_l+1)\Gamma(n_r+2)}$$

+++ {"slideshow": {"slide_type": "skip"}}

and substituting

+++ {"slideshow": {"slide_type": "slide"}}

$$\alpha = n_l+1,\qquad \beta= n_r+1$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain that the posterior is given by the [_Beta_ distribution](https://en.wikipedia.org/wiki/Beta_distribution)

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p|n_l,n_r)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}$$

+++ {"slideshow": {"slide_type": "skip"}}

Actually we could get away without calculating the normalizing factor. By noting that the posterior is a probability distribution and it is proportional to the pdf of the Beta distribution with $\alpha=n_l+1$ and $\beta=n_r+1$

+++ {"slideshow": {"slide_type": "fragment"}}

$$p^{n_l}(1-p)^{n_r} \propto \operatorname{PDF}[\operatorname{Beta}(n_l+1,n_r+1),p] $$

+++ {"slideshow": {"slide_type": "skip"}}

we automatically obtain that it has to be a Beta distribution.

+++ {"slideshow": {"slide_type": "skip"}}

Then we can use know properties of the beta distribution like e.g. the position of the mode.

+++ {"slideshow": {"slide_type": "slide"}}

$$\operatorname{Mode}[Beta(\alpha,\beta)]=\frac{\alpha-1}{\alpha+\beta-2}$$

+++ {"slideshow": {"slide_type": "skip"}}

which translates to

+++ {"slideshow": {"slide_type": "fragment"}}

$$p_{MAP}=\frac{n_l}{n_l+n_r}$$

+++ {"slideshow": {"slide_type": "skip"}}

which is the usual estimate for the parameter of binomial distribution.  This is not surprising as we have assumed that our initial prior was uniform, so our MAP estimator is essentially the maximal likelihood estimator.

+++ {"slideshow": {"slide_type": "skip"}}

Before we proceed with the third ball we will check if we can recreate the results for the two balls. Setting $n_l=0$ and $n_r=2$ we obtain identical answer

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
p(p|r,r)&=PDF[Beta(1,3)]\\&=\frac{\Gamma(4)}{\Gamma(1)\Gamma(3)}(1-p)^{2}=\frac{3!}{2!}(1-p)^{2}
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

It show that one can either update our knowledge incrementally or use all data at once.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This is a general feature. Let's say we have collected two independent sets of data $D_1$ and $D_2$. Using all the data we obtain for the posterior

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$
\begin{split}
P(\theta|D_2, D_1)&=\frac{P(D_2,D_1|\theta)P(\theta)}{P(D_2, D_1)}
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

However using the independence of the two sets we can factorize this expression to obtain

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

\begin{split}
\frac{P(D_2|\theta)P(D_1|\theta)P(\theta)}{P(D_2)P(D_1)}&=\frac{P(D_2|\theta)}{P(D_2)}\frac{P(D_1|\theta)P(\theta)}{P(D_1)}\\
&= \frac{P(D_2|\theta)}{P(D_2)}P(\theta|D_1)
\end{split}

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The last expression shows that we can first calculate the posterior conditioned on set $D_1$ and use the likelihood of the set $D_2$ to get final posterior.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Going back to our billiard table with our third ball on the left we obtain for posterior

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
p(p|r,r,l)&=PDF[Beta(2,3)]=\frac{\Gamma(5)}{\Gamma(2)\Gamma(3)}p(1-p)^{2}\\
&=\frac{4!}{2!}(1-p)^{2}
=12 p (1-p)^2
\end{split}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and MAP estimate

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{map}=\frac{1}{3}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.stats import beta
beta_posteriors=[]
for i in range(1,101):
    n_l = left[:i].sum()
    n_r = i-n_l
    beta_posteriors.append(beta(n_l+1, n_r+1))
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
nb=3;n=3
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, x,left, bl.cs)
alpha=bl.plot_posteriors(beta_posteriors[:n], ax=pax)
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_l = left[:n].sum(); n_r = n- n_l;
p_map = n_l/(n_l+n_r); y_map=beta_posteriors[n-1].pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), 
             fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
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

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior mean

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Instead of the mode we can also use the mean of the distribution as the estimator

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\operatorname{Mean}[Beta(\alpha,\beta)]=\frac{\alpha}{\alpha+\beta}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\frac{n_l+1}{n_l+n_r+2}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

denoted below by the light blue vertical line below

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior median

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Very often it is better to use median instead of the mean. Median has a very clear interpretation: it is a value such that probability that $p$ is less than this value is same as the probability that $p$ is greater than this value, or more exactly

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P\left(p\leq p_{med}\right)\geq\frac{1}{2}\quad\text{and}\quad P\left(p\geq p_{med}\right)\geq\frac{1}{2}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The median value is marked with a orange line on the plot below.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
nb=3;n=3
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
#plot_balls(ax, nb, x,left, bl.cs)
alpha=1
pax.plot(xs,beta_posteriors[n-1].pdf(xs), zorder=1, c='blue', alpha=alpha);

n_l = left[:n].sum(); n_r = n- n_l;
p_map = n_l/(n_l+n_r); y_map=beta_posteriors[n-1].pdf(p_map)
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
p_mean = beta_posteriors[n-1].mean(); 
pax.axvline(p_mean);
p_median = bst.median_f(beta_posteriors[2].pdf,0,1)
pax.axvline(p_median, color='orange')
plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

And finally here is the result after 100 throws

```{code-cell}
---
slideshow:
  slide_type: skip
---
nb=100;n=100
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,10))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, x,left, bl.cs, draw_line=False, alpha=0.7)
alpha=1    
pax.plot(xs,beta_posteriors[99].pdf(xs), zorder=1, c='blue', alpha=alpha);
n_l = left[:n].sum(); n_r = n- n_l;
post = beta_posteriors[n-1]
p_map = n_l/(n_l+n_r); y_map=post.pdf(p_map)
p_mean= post.mean()
pax.annotate(f'MAP',(p_map, y_map),(p_map, y_map+0.5), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05), va='center');
ax.axvline(p_mean)
p_median = bst.median_f(post.pdf,0,1)
pax.axvline(p_median, color='orange')
plt.close()
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}}

Because of relatively large number of throws there is very little difference between MAP, posterior mean and median.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Conjugate  priors

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

So far we have assumed that the prior is uniform over the whole interval $[0,1]$. What happens if our original ball has the tendency to land near the middle? Or near the right edge of the table? We can model that using again the Beta distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p)\sim Beta(\alpha,\beta)$$

```{code-cell}
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,2,5,10]:
    ys = st.beta(a,a).pdf(xs)
    plt.plot(xs,ys, label='%4.2f' %(a,))
plt.legend(loc=1);
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,2,5,10]:
    ys = st.beta(a,3).pdf(xs)
    plt.plot(xs,ys, label='%4.2f' %(a,))
plt.legend(loc=1);
```

+++ {"slideshow": {"slide_type": "skip"}}

If we combine this with Bernoulli likelihood

+++ {"slideshow": {"slide_type": "slide"}}

$$ p^{n_l}(1-p)^{n_r} p^{\alpha-1}(1-p)^{\beta-1}=
p^{n_l+\alpha-1}(1-p)^{n_+\beta-1}$$

+++ {"slideshow": {"slide_type": "skip"}}

we see that the functional dependence of the posterior on $p$ is same as for $Beta(n_l+\alpha, n_r+\beta)$. Which means that we do not have to calculate any normalizing factors and just assume

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\begin{split}
P(p| n_r, n_l )  &= \operatorname{PDF}[  Beta(n_l+\alpha, n_r+\beta),p]
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

In this context $\alpha$ and $\beta$ are called pseudo throws, as we are effectively adding $\alpha -1$ and $\beta-1$  respectively  to the number of balls on the left and balls on the right.

+++ {"slideshow": {"slide_type": "skip"}}

A prior with a property that the posterior distribution has same form as the prior is called _conjugate_ prior to the sampling distribution. So the Beta distribution is a conjugate prior to Bernoulli distribution.

+++ {"slideshow": {"slide_type": "skip"}}

It can be more convenient to parameterize  Beta distribution by its mean and variance. The mean and variance of Beta distribution are

+++ {"slideshow": {"slide_type": "slide"}}

$$\mu = \frac{\alpha}{\alpha+\beta}\quad\text{and}\quad \sigma^2=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

+++ {"slideshow": {"slide_type": "skip"}}

and so

+++ {"slideshow": {"slide_type": "fragment"}}

$$\nu = \frac{\mu(1-\mu)}{\sigma^2}-1,\quad \alpha = \mu\nu, \quad
\beta = (1-\mu) \nu,\quad \sigma^2<\mu(1-\mu)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Categorical variables

+++ {"slideshow": {"slide_type": "skip"}}

A natural generali<ation of the Bernoulli distribution is the multinouilli or categorical distribution and the generalization of the binomial distribution is the _multinomial_ distribution.

+++ {"slideshow": {"slide_type": "skip"}}

Let's say we have $m$ categories with probability $p_k$ for each category. Then after $n$ trials the probability that we $n_k$ results in category $k$ is:

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n_1,\ldots, n_{m}|p_1,\ldots, p_{m}) = \frac{m!}{n_1!\cdots n_{m}!}p_1^{n_1}\cdots p_{m}^{n_{m}}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Example: Dice

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$m=6,\quad p_i=\frac{1}{6}$$ $$ P(n_1,\ldots, n_{6}) = \frac{N!}{n_1!\cdots n_{m}!}\frac{1}{6^m}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true, "jp-MarkdownHeadingCollapsed": true}

#### Problem

We are rolling four dices. What is the probability that we roll four ones?  What is the probability that we roll numbers from one to four?

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Dirichlet distribution

+++ {"slideshow": {"slide_type": "skip"}}

Conjugate prior  to this distribution is the Dirichlet distribution which is a generalization of the Beta distribution. It has $m$ parameters $\alpha_k$ and its probability mass function is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P_{Dir}(p_1,\ldots,p_{m}|\alpha_1,\ldots,\alpha_{m}) = \frac{\Gamma\left(\sum\limits_{i=1}^{m} \alpha_i\right)}{\prod\limits_{i=1}^{m}\Gamma(\alpha_i)}
\prod\limits_{i=1}^{m}p_i^{\alpha_i-1}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior

+++ {"slideshow": {"slide_type": "skip"}}

It is easy to check that the posterior probability density for $P(p_1,\ldots, p_{m}|n_1,\ldots, n_{m})$ with prior given by Dirichlet distribution with parameters $\alpha_k$  is again given with by the  Dirichlet distribution with parameters $\alpha_1+n_1,\ldots, \alpha_{m}+n_{m}$.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p_1,\ldots, p_{m}|n_1,\ldots, n_{m})=P_{Dir}(p_1,\ldots,p_{m}|\alpha_1+n_1,\ldots,\alpha_{m}+n_m)$$

+++ {"slideshow": {"slide_type": "slide"}}

### MAP

+++ {"slideshow": {"slide_type": "skip"}}

The maximal a posteriori estimate is:

+++ {"slideshow": {"slide_type": "fragment"}}

$$p_{MAP\,k} = \frac{n_k+\alpha_k-1}{n + \sum_i \alpha_k-m}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Mean

+++ {"slideshow": {"slide_type": "skip"}}

and the mean

+++ {"slideshow": {"slide_type": "fragment"}}

$$\bar{p_k} = \frac{n_k+\alpha_k}{\sum_{k=1}^m n_k  + \sum_{k=1}^m \alpha_k}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Poisson distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(k|\lambda) = e^{-\lambda}\frac{\lambda^k}{k!}$$

+++ {"slideshow": {"slide_type": "skip"}}

Poisson distribution appears when  some events occur with uniform probability in time. For example there are $\lambda$ customers per our on average walking into a store and probability of this happening is same every second then number of customers that visit the store each hour is a discrete random variable with probability mass function

+++ {"slideshow": {"slide_type": "skip"}}

Assuming uniform prior on the parameter $\lambda$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\lambda)=1$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

(this is an _improper_ prior, as this is not a normalizable probability distribution) we obtain for posterior after  observing a series of counts $\{k_i\}$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(\{k_i\}|\lambda) = \prod_{i=1}^n e^{-\lambda}\frac{\lambda^{k_i}}{k_i!}\propto \lambda^{\displaystyle n\bar{k}}e^{\displaystyle -n\lambda}$$ $$ k=\frac{1}{n}\sum_{i=1}^n k_i$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

From this formula we can infer that distribution with a pdf of the form

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\lambda)\propto e^{-\beta\lambda}\lambda^{\alpha-1}$$

+++ {"slideshow": {"slide_type": "skip"}}

would be a conjugate prior to Poisson distribution, as

+++ {"slideshow": {"slide_type": "fragment"}}

$$\lambda^{n\bar k}e^{-n\lambda}\cdot e^{-\beta\lambda}\lambda^{\alpha-1} = e^{-(\beta+n)\lambda}\lambda^{n\bar k+\alpha-1}$$

+++ {"slideshow": {"slide_type": "skip"}}

The distribution of this form is called [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)

+++ {"slideshow": {"slide_type": "slide"}}

$$\operatorname{PDF}[Gamma(\alpha,\beta),x]=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$$

+++ {"slideshow": {"slide_type": "skip"}}

and  posterior distribution for Poisson distribution after observing $\{k_i\}$ counts, with prior $\Gamma(\alpha,\beta)$ is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\lambda|\{k_i\}) =PDF[Gamma(\alpha+n \bar k,\beta+n)]$$

```{code-cell}
---
slideshow:
  slide_type: skip
---
from scipy.stats import gamma
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
xs = np.linspace(1,20,500)
fig, ax = plt.subplots()
b=0.75
ax.set_title(f"Gamma distribution $\\beta={b:.2f}$")
for a in [0.5,0.75,1,2,5,7]:
    pdf = gamma(a=a,scale=1/b).pdf
    ax.plot(xs, pdf(xs), label=f"$\\alpha={a:.2f}$");
ax.legend();    
plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell}
---
slideshow:
  slide_type: skip
---
xs = np.linspace(1,20,500)
fig, ax = plt.subplots()
b=2
ax.set_title(f"Gamma distribution $\\beta={b:.2f}$")
for a in [0.5,0.75,1,2,5,7]:
    pdf = gamma(a=a,scale=1/b).pdf
    ax.plot(xs, pdf(xs), label=f"$\\alpha={a:.2f}$");
ax.legend();    
plt.close();
```

```{code-cell}
---
slideshow:
  slide_type: slide
---
fig
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---

```

---
jupytext:
  cell_metadata_json: true
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
from fractions import Fraction
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import beta
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
import bda
import bda.stats as bst
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
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Bayesian Data Analysis

+++ {"editable": true, "slideshow": {"slide_type": ""}}

> ### Statistical inference is concerned with drawing conclusions, from numerical data, about quantities that are not observed.
>  "Bayesian Data Analysis" A. Gelman, J. B. Carlin, H. S. Stern,  D. B. Dunson, A. Vehtari, D. B. Rubin

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Literature

+++ {"slideshow": {"slide_type": "-"}, "editable": true}

- "Bayesian Data Analysis" A. Gelman, J. B. Carlin, H. S. Stern,  D. B. Dunson, A. Vehtari, D. B. Rubin.[[pdf](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)]
- "Data Analysis, a Bayesian Tutorial" D.S. Silva with J. Skiling.
- "Bayesian methods for Hackers, Probabilistic Programming adn Bayesian Inference" C. Davidson-Pilon.  [[online](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)]
- "From Laplace to Supernova SN 1987A: Bayesian Inference in Astrophysics" T. J. Loredo. [[pdf](https://bayes.wustl.edu/gregory/articles.pdf)]

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Reverend Thomas Bayes' original example -- pool table

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We will start with the original example considered by the Reverend Thomas Bayes in his paper [["An essay towards solving a problem in the doctrine of chances"](ThomasBayespaper.pdf)] which was published posthumously by Richard Price in Philosophical Transactions __53__  (1763).

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Imagine that somebody places a  billiard ball at random on the billiard table. The location of the ball is unknown. Then we start to throw the balls at random at the table and somebody is reporting to us if the balls have landed on the left or on the right of the original ball. 
The problem considered by T. Bayes was how our belief about the ball position changes with each new ball thrown on the table.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

So let's place our original ball (black) on the table, the problem is really one dimensional as we are only interested in ball position along one side of the table and we can describe it by one number $p\in[0,1]$.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
p =  np.pi/10.0
y =  0.786
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax,pax = bl.make_pool_figure(p)
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True);
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Probability

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Our "belief" as to the position of the ball will be expressed by a _probability distribution_ characterized by the _probability density function_ (pdf) (see "100_Probability/continuous_distributions.md" notebook)

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This assigns a non-negative number to each possible position $p$ on the table. The higher the number the stronger our belief that the ball is near this location. The distribution function has of course to be normalized

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\int_0^1\!\!\text{d}p\, P(p)=1.$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Given that, the probability that the ball lies in the interval $[a,b]$ is given by the integral

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p\in [a,b])=\int_a^b\!\!\text{d}p\, P(p)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

For each $a\leq b$ this number is a _probability_ and it fulfills all the probability axioms presented in the "100_Probability/probability.md" notebook.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### What is probability ?

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

>#### "One sees, from this Essay, that the theory of probabilities is basically just common sense reduced to calculus; it makes one appreciate with exactness that which accurate minds feel with a sort of instinct, often without being able to account for it."
> "Théorie Analytique des Probabilités" Pierre-Simon Laplace

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The meaning of this number (probability) is not always clear, I will stick with the Bayesian interpretation as the measure of our certainty of that this event will happen. In particular zero probability means that we are sure this event will not happen, and probability one means that we are certain that it will happen.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Prior

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We start with our belief as to the position of the ball on the table before we get any additional information. That is called a _prior_ from latin _a priori_.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We will assume that that we have no information as the where the ball has landed, so we choose an uniformly random prior

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p)=1$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

which is indicated in the plot below by the blue line.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
xs = np.linspace(0,1,1000)
prior = np.vectorize(lambda x: 1.0)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
pax.plot(xs,prior(xs), zorder=1, c='blue', label="$P(p)$");
pax.legend();
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### First ball

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Now we throw another ball marked in red on the plot below, which lands on the right side of the original ball. Actually I will throw 100 balls at once, but use them one by one.
For each ball we generate two numbers specyfying its $(x,y)$ position on the table. The second $y$ coordinate is only used for rendering.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
N_balls = 100
np.random.seed(87) ## Sets the seed so that each time the notebook is started it will  give same results.
xy = scipy.stats.uniform(loc=0, scale=1).rvs(size=(N_balls,2)) #location of the balls
left= (xy[:,0]<=p) + 0
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The last line rquires an explanation. By comparing a array to scalar we obtain an array filled with boolean (true/false) values.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
(xy[:,0]<p)[:5]
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Adding zero to it converts true to one and false to zero.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
(xy[:,0]<p)[:5] + 0
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
fig,ax,pax = bl.make_pool_figure(p,ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, xy,left, bl.cs)
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Intuitively it changes our belief as to where the first ball is located, notably it becomes less likely that the ball is close to the right edge. Such change of probability distribution is described by the _conditional_ probability.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Conditional probability

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Conditional probability $P(A|B)$ is the probability of event $A$ provided that event $B$ happened and is given by the formula

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(A|B) = \frac{P(A\cap B)}{P(B)}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Intuitively this means that we are restricting our sample space to $B$. Let's look at an example.

+++ {"slideshow": {"slide_type": "slide"}, "tags": ["example", "heading"], "editable": true}

### Example

We are rolling two dices, what is the probability that we role a sum of eleven or twelve?

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from bda.plotting import plot_cartesian
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
f_A = lambda i,j: i+j >10
fig, ax = plot_cartesian(6,6, radius=0.3, cfunc=lambda i,j: 'red' if f_A(i,j) else 'lightgrey' )
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

We have 36 possible outcomes and only three of them have a sum larger than ten, so the probability is

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
p_A = Fraction(3,36)
print(f"{p_A} \u2248 {float(p_A):.4f}")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Now let's supose that somebody has told us that there is five on one of the dices. How would this change the probability of rolling the sum greater than ten? Now we have to look only in the combinations that fulfill this condition

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
f_B = lambda i,j: i==5 or j==5
fig, ax = plot_cartesian(6,6, radius=0.3, cfunc= lambda i,j: 'green' if f_B(i,j) else None )
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

There is only eleven such combination so $P(B)$ is equal to

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
p_B=Fraction(11,36)
print(f"{p_B} \u2248 {float(p_B):.4f}")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Of those only two fullfill the condition so

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
def f_cond_A_B(i,j):
  pa = f_A(i,j)
  pb = f_B(i,j)
  if pa and pb:
    return (1.0, 1.0, 0.0)
  if pb:
    return 'green'
  return None
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig, ax = plot_cartesian(6,6, radius=0.3, cfunc= f_cond_A_B )
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The result is

```{code-cell}
---
editable: true
slideshow:
  slide_type: fragment
---
p_A_cond_B =Fraction(2,11)
print(f"{p_A_cond_B} \u2248 {float(p_A_cond_B):.4f}")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

and as we can see it is  substantially higher then $P(A)$.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

On the other hand if we new that there is a number smaller then five on one of the dices, then our chances drop to zero, as it is not possible to attain eleven or twelve in this situation.

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
less_than_five = lambda i,j: i<5  or j<5
fig, ax = plot_cartesian(6,6, radius=0.3, cfunc= lambda i,j: 'green' if less_than_five(i,j) else None )
plot_cartesian(6,6, radius=0.3, ax=ax, cfunc=lambda i,j: 'red' if f_A(i,j) else None );
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Product rule

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The definition of the conditional probability can be rewritten as the _product rule_

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(A\cap B)= P(A|B)P(B)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

which can be generalized to multiple disjoint sets

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$B = \bigcup_{i}  B_i,\;\; \bigvee_{i\neq j} B_i\cap B_j=\emptyset \implies P(B)=\sum_i P(B_i)$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(A\cap B)= \sum_i P(A|B_i)P(B_i)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

In special case when contains the whole sample space $B=S$ this gives

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\bigcup_i B_i = S \implies P(A)= \sum_i P(A|B_i)P(B_i)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

This is very useful for defining more complex probability distributions.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Example

__Height distribution__

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

 For example the height distribution in human population can be decomposed in this way into height distribution for women and men, each separately being a normal distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(h)=P(h|f)P(f)+P(h|m)P(m)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Back to biliard

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Going back to our billiard example what we are looking for it the conditional probability of ball being at $p$ on condition that the second ball landed on the right of it

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p|R)$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Sampling distribution/ likelihood

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We will start by calculating the _reverse_ conditional probability

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(R|p)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

that the first  ball lands on the right provided that original ball is at $p$. This called the _sampling distribution_ and  in this case this is the Bernoulli distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig,ax,pax = bl.make_pool_figure(p,ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, xy,left, bl.cs)
bda.plotting.darrow((0,0.5), (p,0.5), ax=ax, c='green', label='$p$')
bda.plotting.darrow((p, 0.5), (1,0.5), ax=ax, c='red', label ='$1-p$')
```

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(R|p)=1-p\quad P(L|p)=p$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

If we consider $P(R|p)$ as a function of $p$ we will call it the _likelihood_

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$L(p|R)\equiv P(R|p)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Please note that in general this __is not__ a probability density function.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Bayes' theorem

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

To go from $P(r|p)$ to $P(p|r)$ we will use the Bayes theorem which can be simply derived from the definition of the conditional probability. Substituting the product rule into the definition of $P(B|A)$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(B|A)= \frac{P(A\cap B)}{P(A)}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

we obtain

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\boxed{P(B|A)=\frac{P(A|B)P(B)}{P(A)}}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

With setting  $B\equiv p$ and $A\equiv r$ we obtain the _posterior_ distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p|R)=\frac{P(R|p)P(p)}{P(R)}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The denominator $P(R)$ is the normalization factor which in this case can be simply calculated

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(R)=\int_{0}^1\text{d}p P(R|p)P(p)= \int_0^1\text{d}p (1-p)= 1-\frac{1}{2}=\frac{1}{2}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

leading finally to the distribution

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(p|R)=2-2p$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

denoted by the dark blue line on the plot below.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
posteriors = [lambda x: 2-2*x]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
n=1
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, xy,left, bl.cs)
alpha=bl.plot_posteriors(posteriors[:1])    
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
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

### Maximal a posteriori (MAP)

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The probability distribution $P(p|r)$ contains  most complete information about the location of the first ball, however what we  often want is an _estimate_ for $p$. Similarly to [[Maximal likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)] we can define a maximal a posteriori (MAP) estimator

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{MAP}=\operatorname{argmax}_p P(p|R)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

which in this case is equal to zero.

```{code-cell}
---
editable: true
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
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
n=1
fig,ax,pax = bl.make_pool_figure(p,ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, 1, xy,left, bl.cs)
alpha=bl.plot_posteriors(posteriors[:n], ax=pax)
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);       
p_map, y_map = find_pmap(posteriors[n-1])
# ax.annotate(f'$p_{{MAP}}={p_map:.1f}$',(p_map, 0),(p_map, -0.15), 
#              fontsize=16, arrowprops=dict(color='blue',  arrowstyle='->,head_width=0.15', relpos=(0,0)), va='center',ha='left');
ax.axvline(p_map+.001, c=defaults.map_color);
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

### Second ball

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We can continue by throwing another ball which again lands on the right of the original ball

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
tags: [aux_code]
---
nb=2;n=1
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, xy,left, bl.cs)
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

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We will repeat the process but this time we will use the posterior as our new prior, as this represents our present belief as to where the original ball is located

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(p|R,R) = \frac{P(R|p) P(p|R)}{\int_0^1\text{d}p\,P(R|p) P(p|R)}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The normalizing factor is again easy to calculate

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

 $$P(R|p) P(p|R)=2(1-p)^2,\quad 2\int_0^1\text{d}p (1-p)^2=\frac{2}{3}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

leading to the new posterior distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p|R,R)=3(1-p)^2$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

which does not change the MAP estimate which remains zero.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
posteriors.append(lambda x: 3*(1-x)**2)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
nb=2;n=2
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, xy,left, bl.cs)
alpha=bl.plot_posteriors(posteriors[:n], ax=pax)
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);     
p_map, y_map = find_pmap(posteriors[n-1])
# ax.annotate(f'$p_{{MAP}}={p_map:.1f}$',(p_map, 0),(p_map, -0.1), 
#              fontsize=16, arrowprops=dict(color='blue',  arrowstyle='->,head_width=0.15', relpos=(0,0)), va='center',ha='left');
pax.axvline(p_map+0.001,lw=1, c=defaults.map_color, label='MAP')
plt.legend();
plt.close()
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

Althought the MAP estimate did not change, the distribution did (see the problem below).

+++ {"slideshow": {"slide_type": "slide"}, "tags": ["problem"], "editable": true}

#### Problem

How did the probability that $p<\frac{1}{2}$ changed?

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "editable": true, "jupyter": {"source_hidden": true}}

This probability is given by the integral

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true, "tags": ["problem"]}

$$P\left(p<\frac{1}{2}\right)=\int_0^{b=\frac{1}{2}}\text{d}p P_{post}(p|D)$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true, "tags": ["answer"], "jupyter": {"source_hidden": true}}

At the begining posterior is equal to prior

+++ {"slideshow": {"slide_type": "skip"}, "editable": true, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$P(p<b)=\int_0^b\!\!\text{d}p = b$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$P\left(p<\frac{1}{2}\right)=\frac{1}{2}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true, "tags": ["answer"], "jupyter": {"source_hidden": true}}

After throwing the first ball we obtain

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "editable": true, "jupyter": {"source_hidden": true}}

$$
\int_0^b\text{d}p P(p|r) = 
2\int_0^{b}\!\text{d}p\; (1-p) = \left.2\left(p-\frac{1}{2}p^2\right)\right|^b_0=
2\left(b-\frac{1}{2}b^2\right)
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$P\left(p<\frac{1}{2}\right)=\frac{3}{4}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

and after the second

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "editable": true, "jupyter": {"source_hidden": true}}

$$
\int_0^{b}\text{d}p P(p|rr) = 
3\int_0^{b}\text{d}p\,(1-p)^2=3\int_0^{b}\text{d}p\,(1-2p+p^2)=3\left(b - b^2 +\frac{1}{3}b^3\right)
$$

```{code-cell}
---
editable: true
jupyter:
  source_hidden: true
slideshow:
  slide_type: skip
tags: [answer]
---
b = Fraction(1,2)
3*(b-b*b+b*b*b/3)
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$P\left(p<\frac{1}{2}\right)=\frac{7}{8}$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"], "editable": true, "jupyter": {"source_hidden": true}}

We see that this probability successively  increases.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Third ball

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Throwing a third ball we notice that it lands on the left of the original ball.

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
nb=3;n=2
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,3))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, xy,left, bl.cs)
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

That should shift our distribution away from $p=0$.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Binomial distribution

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We could continue in the similar fashion as before, but for larger amount of balls it would quickly become tedious. Instead will calculate the posterior in "one go". Assuming that w $n_L$ balls had landed on the left and $n_R$ on the right, the posterior is given by

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(p|n_L,n_R)=\frac{P(n_L,n_R|p)P(p)}{P(n_L,n_R)}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The sampling distribution is given by the binomial distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(n_L,n_R|p) = \binom{n_L+n_R}{n_L}p^{n_L}(1-p)^{n_R}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The normalizing factor is now more complicated but can be calculated using e.g. _Mathematica_.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$
\begin{split}
\int_0^1\text{d}p'& P(n_L,n_R|p')P(p') \\
&=\binom{n_L+n_R}{n_R}\int_0^1\text{d}p^{'n_L}p'(1-p')^{n_R}\\[5mm]
&= \binom{n_L+n_R}{n_R}
\frac{n_L!n_R!}{(n_L+n_R+1)!}=\frac{1}{n_L+n_R+1}
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

leading to the posterior

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$P(p|n_L,n_R)=\binom{n_L+n_R}{n_L}(n_L+n_R+1)p^{n_L}(1-p)^{n_R}$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Beta distribution

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

After some manipulations

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
&\binom{n_L+n_R}{n_L}(n_L+n_R+1)\\
&\kern5mm=\frac{(n_L+n_R)!(n_L+n_R+1)}{n_L!n_R!}
=\frac{(n_L+n_R+1)!}{n_L!n_R!}
\end{split}
$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\frac{(n_L+n_R+1)!}{n_L!n_R!}=\frac{\Gamma(n_L+n_R+2)}{\Gamma(n_L+1)\Gamma(n_R+2)}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

where $\Gamma(n)$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function) which for integer arguments is given by the factorial

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\Gamma(n)=(n-1)!$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Substituting

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\alpha = n_L+1,\qquad \beta= n_R+1$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

we obtain that the posterior is given by the [_Beta_ distribution](https://en.wikipedia.org/wiki/Beta_distribution)

+++

$$P(p|n_L,n_R)=PDF[Beta(n_L+1,n_R+1)$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$PDF[Beta(\alpha,\beta)]=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha-1}(1-p)^{\beta-1}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Actually we could get away without calculating the normalizing factor. By noting that the posterior is, by definition, a probability distribution and it is proportional to the pdf of the Beta distribution with $\alpha=n_L+1$ and $\beta=n_R+1$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$p^{n_L}(1-p)^{n_R} \propto \operatorname{PDF}[\operatorname{Beta}(n_L+1,n_R+1),p] $$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

we automatically obtain that it has to be a Beta distribution.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

We then can use know properties of the beta distribution like e.g. the position of the mode.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\operatorname{Mode}[Beta(\alpha,\beta)]=\frac{\alpha-1}{\alpha+\beta-2}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

which translates to

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{MAP}=\frac{n_L}{n_L+n_R}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

which is the usual estimate for the parameter of binomial distribution.  This is not surprising as we have assumed that our initial prior was uniform, so our MAP estimator is essentially the maximal likelihood estimator.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Before we proceed with the third ball we will check if we can recreate the results for the two balls. Setting $n_L=0$ and $n_R=2$ we obtain identical answer

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
p(p|R,R)&=PDF[Beta(1,3)]\\&=\frac{\Gamma(4)}{\Gamma(1)\Gamma(3)}(1-p)^{2}=\frac{3!}{2!}(1-p)^{2}
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

However using the independence of the two sets, we can factorize this expression to obtain

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\begin{split}
\frac{P(D_2|\theta)P(D_1|\theta)P(\theta)}{P(D_2)P(D_1)}&=\frac{P(D_2|\theta)}{P(D_2)}\frac{P(D_1|\theta)P(\theta)}{P(D_1)}\\
&= \frac{P(D_2|\theta)}{P(D_2)}P(\theta|D_1)
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

The last expression shows that we can first calculate the posterior conditioned on set $D_1$ and use the likelihood of the set $D_2$ to get final posterior.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Going back to our billiard table with our third ball on the left we obtain for posterior

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\begin{split}
p(p|R,R,L)&=PDF[Beta(2,3)]=\frac{\Gamma(5)}{\Gamma(2)\Gamma(3)}p(1-p)^{2}\\
&=\frac{4!}{2!}(1-p)^{2}
=12 p (1-p)^2
\end{split}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and MAP estimate

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$p_{MAP}=\frac{1}{3}$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Using the `scipy.stats` module we will generate all the posteriors

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
from scipy.stats import beta
beta_posteriors=[]
for n in range(1,101): # n is the number of balls
    n_L = left[:n].sum() # the number of balls to the left
    n_R = n-n_L # and to the right
    beta_posteriors.append(beta(n_L+1, n_R+1))
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
bl.plot_balls(ax, nb, xy,left, bl.cs)
alpha=bl.plot_posteriors(beta_posteriors[:n], ax=pax)
pax.plot(xs,prior(xs), zorder=1, c='blue', alpha=alpha);   
n_L = left[:n].sum(); n_R = n- n_L;
p_map = n_L/(n_L+n_R); y_map=beta_posteriors[n-1].pdf(p_map)
# ax.annotate(f'$p_{{MAP}}={p_map:.1f}$',(p_map, 0),(p_map, -0.1), 
#              fontsize=16, arrowprops=dict(color='blue',  arrowstyle='->,head_width=0.15', relpos=(0,0)), va='center',ha='left');
pax.axvline(p_map+0.001, c=defaults.map_color,label='MAP')
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

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": ["problem"]}

#### Problem

Calculate $P\left(p<\frac{1}{2}\right)$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

In general this is given by the cummulative distribution function

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

$$P\left(p<b\right)=CDF[Beta(2,3)](b)$$

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["problem"]}

Use Wikipedia and `scipy.special` to obtain the value.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

From [Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution) we obtain that

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

$$CDF[Beta(\alpha,\beta)](x)=I_x(\alpha,\beta)$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}, "tags": ["answer"], "jupyter": {"source_hidden": true}}

where $I_x(\alpha,\beta)$ [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function) and its value can be calculated using `scipy.special.betainc` function

```{code-cell}
---
editable: true
jupyter:
  source_hidden: true
slideshow:
  slide_type: skip
tags: [answer]
---
print(f"P(p<1/2) \u2248 {scipy.special.betainc(2,3,0.5):.2f}")
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Posterior mean

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

Instead of the mode we can also use the mean of the distribution as the estimator

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\operatorname{Mean}[Beta(\alpha,\beta)]=\frac{\alpha}{\alpha+\beta}$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\frac{n_L+1}{n_L+n_R+2}$$

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

n_L = left[:n].sum(); n_R = n- n_L;
p_map = n_L/(n_L+n_R); y_map=beta_posteriors[n-1].pdf(p_map)
# ax.annotate(f'$p_{{MAP}}={p_map:.1f}$',(p_map, 0),(p_map, -0.1), 
#              fontsize=16, arrowprops=dict(color='blue',  arrowstyle='->,head_width=0.15', relpos=(0,0)), va='center',ha='left');
p_mean = beta_posteriors[n-1].mean(); 
pax.axvline(p_mean, label='mean', color=defaults.mean_color);
p_median = bst.median_f(beta_posteriors[2].pdf,0,1)
pax.axvline(p_median, color=defaults.median_color, label='median')
pax.axvline(p_map,lw=1, c=defaults.map_color, label='MAP')
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

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
nb=100;n=100
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,10))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
alpha=1.0
for i in reversed([0,1,2,5,10,25,50,75,90,95,99]):
  plt.plot(xs,beta_posteriors[i].pdf(xs), zorder=1, c='blue', alpha=alpha, linewidth=1);
  alpha*=0.9
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

And finally here is the result after 100 throws

```{code-cell}
---
editable: true
slideshow:
  slide_type: skip
---
nb=100;n=100
fig,ax,pax = bl.make_pool_figure(p, ylim=(0,10))
bl.plot_ball(ax, p, y, bc='k', lc='darkgrey', bs=200, draw_line=True)
bl.plot_balls(ax, nb, xy,left, bl.cs, draw_line=False, alpha=0.7)
alpha=1    
pax.plot(xs,beta_posteriors[99].pdf(xs), zorder=1, c='blue', alpha=alpha);
n_L = left[:n].sum(); n_R = n- n_L;
post = beta_posteriors[n-1]
p_map = n_L/(n_L+n_R); y_map=post.pdf(p_map)
p_mean= post.mean()
ax.annotate(f'$p_{{MAP}}={p_map:.1f}$',(p_map, 0),(p_map, -0.1), 
             fontsize=16, arrowprops=dict(color='blue',  arrowstyle='->,head_width=0.15', relpos=(0,0)), va='center',ha='left');
ax.axvline(p_mean)
p_median = bst.median_f(post.pdf,0,1)
pax.axvline(p_median, color='orange')
plt.close()
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
fig
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

Because of relatively large number of throws there is very little difference between MAP, posterior mean and median.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
print(f"p_MAP = {p_map:.4f} p_true = {p:.4f}")
```

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

As we can see we got close to the true value of $p$ but not quite. In the next notebook we will investigate how to calculate the errors on our estimates.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Conjugate  priors

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

So far we have assumed that the prior is uniform over the whole interval $[0,1]$. What happens if our original ball has the tendency to land near the middle? Or near the right edge of the table? We can model that again using the Beta distribution

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$P(p) = PDF[Beta(\alpha,\beta),p]$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,2,5,10]:
    ys = beta(a,a).pdf(xs)
    plt.plot(xs,ys, label=f'{a:.1f}')
plt.title("Symmetric Beta distribution");
plt.legend(loc=1, title="$\\alpha=\\beta$");
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: slide
---
xs =np.linspace(0,1,250)
for a in [0.25,0.5,1,2,5,10]:
    ys = beta(a,3).pdf(xs)
    plt.plot(xs,ys, label='%4.2f' %(a,))
plt.title("Assymetric Beta distribution $\\beta=3.0$")  
plt.legend(loc=1, title="$\\alpha$");
```

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

If we combine this with Bernoulli likelihood

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$ p^{n_L}(1-p)^{n_R} p^{\alpha-1}(1-p)^{\beta-1}=
p^{n_L+\alpha-1}(1-p)^{n_R+\beta-1}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

we see that the functional dependence of the posterior on $p$ is same as for $Beta(n_L+\alpha, n_R+\beta)$. Which means that we do not have to calculate any normalizing factors

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\begin{split}
P(p| n_R, n_L )  &= \operatorname{PDF}[  Beta(n_L+\alpha, n_R+\beta),p]
\end{split}
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

In this context $\alpha$ and $\beta$ are called pseudo throws, as we are effectively adding $\alpha -1$ and $\beta-1$  respectively  to the number of balls on the left and balls on the right.

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

#### Conjugate prior

A prior with a property that the posterior distribution has same form as the prior is called _conjugate_ prior to the sampling distribution. So the Beta distribution is a conjugate prior to Bernoulli distribution.

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

It can be more convenient to parameterize  Beta distribution by its mean and variance. The mean and variance of Beta distribution are

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

$$\mu = \frac{\alpha}{\alpha+\beta}\quad\text{and}\quad \sigma^2=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

and so

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\nu = \frac{\mu(1-\mu)}{\sigma^2}-1,\quad \alpha = \mu\nu, \quad
\beta = (1-\mu) \nu$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

on condition that.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$\sigma^2<\mu(1-\mu)$$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Summary

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

The posterior probability distribution for the parameters $\theta$ given the data $D$ is given by the Bayes thorem

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$\Large
\boxed{\overbrace{P(\theta|D)}^{posterior} = \frac{\overbrace{P(D|\theta)}^{likelihood}\overbrace{P(\theta)}^{prior}}{P(D)}}
$$

+++ {"editable": true, "slideshow": {"slide_type": "skip"}}

where $P(D|\theta)$ is the sampling distribution or likelihood and $P(\theta)$ is a prior. $P(D)$ is a normalization constant

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

$$P(D) = \int\text{d}\theta P(D|\theta)P(\theta)$$

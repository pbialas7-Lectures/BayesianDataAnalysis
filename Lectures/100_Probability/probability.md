---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from IPython.display import Markdown as md
```

# Probability

+++ {"slideshow": {"slide_type": "slide"}}

>#### "One sees, from this Essay, that the theory of probabilities is basically just common sense reduced to calculus; it makes one appreciate with exactness that which accurate minds feel with a sort of instinct, often without being able to account for it."
> "Théorie Analytique des Probabilités" Pierre-Simon Laplace

+++ {"tags": ["description"], "slideshow": {"slide_type": "skip"}}

Because they deal with uncertain events, most of the machine learning methods can be framed in the language of probability. 
In this notebook I will very briefly recall the basics concepts of the probability calculus and introduce the notation I will be using, hopefully consistently, throughout the lecture.

+++ {"tags": ["description"], "slideshow": {"slide_type": "skip"}}

But keep in mind that this is not a supposed to be a textbook  on probability! Please treat this as a list of concepts and definitions that you have to refresh. It will also serve as a brief introduction to various Python packages. But again this is not a tutorial on Python. The code is provided as a guidance for you and it's up to you to lookup  explanantion in documentation if  needed. I  am of course also happy to help. You can consult me on the chat on Teams.

+++ {"tags": ["description"], "slideshow": {"slide_type": "skip"}}

The lecture includes some simple problems to help you check your understanding of the subject. Some problems have answers right in the notebook. I will try to hide the content of this cells, please try to solve the problem before looking at the answer.

+++ {"slideshow": {"slide_type": "slide"}}

## Random events

+++ {"slideshow": {"slide_type": "fragment"}}

### Sample space

+++ {"slideshow": {"slide_type": "skip"}}

Imagine any process that can have an upredictable outcome. This could be the results of a coin toss,  number of passengers on the bus etc. Let's  assume that we know the set of all possible outcomes of this process and call this set $S$. This set is often called _sample space_.

+++ {"slideshow": {"slide_type": "fragment"}}

$$S$$

+++

For example for coin and dice toss the sampling space would be respectively

+++ {"slideshow": {"slide_type": "fragment"}}

$$\{H,T\},\qquad \{1,2,3,4,5,6\}$$

+++ {"slideshow": {"slide_type": "slide"}}

### Event

+++ {"slideshow": {"slide_type": "skip"}}

Any subset $A$ of $S$ denoted

+++ {"slideshow": {"slide_type": "fragment"}}

$$A\subseteq S$$

+++ {"slideshow": {"slide_type": "skip"}}

 will be called an _event_. If process has an outcome $s\in S$ then we say that the event $A$ happened if $s\in A$.

+++ {"slideshow": {"slide_type": "skip"}}

An event that contain only one  element

+++ {"slideshow": {"slide_type": "fragment"}}

$$A=\{s\}$$

+++ {"slideshow": {"slide_type": "skip"}}

 will be called an _elementary_ event, _atomic_ event or _sample point_.

+++ {"slideshow": {"slide_type": "skip"}}

For coin toss there are two elementary events $\{H\}$ nad $\{T\}$ and four possible events (including the empty set).

+++ {"slideshow": {"slide_type": "fragment"}}

$$\emptyset, \{H\}, \{T\}, \{H,T\}$$

+++

For rolling of a dice the event "rolling a even number" would be

+++ {"slideshow": {"slide_type": "fragment"}}

$$\{2,4,6\}$$

+++ {"slideshow": {"slide_type": "slide"}}

#### Example: Dice roll

+++ {"slideshow": {"slide_type": "fragment"}}

What is the sets of all possible outcomes of a rolling  two dice? How many elements it contains? Write down the event $A$ - "the sum of the points is 9".

+++ {"slideshow": {"slide_type": "fragment"}, "tags": ["answer"]}

$$S=\{(i,j): i,j\in\{1,2,3,4,5,6\}\},\quad \#S=36,\quad A=\{(3,6), (4,5), (5,4), (6,3)\}\quad \#A = 4$$

+++ {"slideshow": {"slide_type": "skip"}}

Where $\#A$ denotes the number of elements in set $A$.

+++ {"slideshow": {"slide_type": "skip"}}

For larger examples this would be impractical, but just for fun let's code this in Python

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from itertools import product
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
S_dice =  {(i,j) for i,j in product(range(1,7), range(1,7))}
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print(len(S_dice))
print(S_dice)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
A = set( filter(lambda s: sum(s)==9, S_dice) )
print(A)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Probability of an event

+++ {"slideshow": {"slide_type": "skip"}}

Because the outcome of a process is unpredictable, so are the events.    However some events are more likely to happen then the others and we can quantify this by assigning  a number to each event that we call _probability_ of that event:

+++ {"slideshow": {"slide_type": "fragment"}}

$$0\leq P(A) \leq 1$$

+++ {"slideshow": {"slide_type": "skip"}}

What this number really means is still subject to discussion and interpretation and I will not address this issue. Personaly I suport the Bayesian interpretation where probability is a measure of "degree of certainty" with zero probability denoting _impossible_ event and one denoting a _certain_ event.  What is important is that those numbers cannot be totaly arbitrary. To be considered a valid measure, probabilities must satisfy several  axioms consistent with our common sense:

+++ {"slideshow": {"slide_type": "fragment"}}

1. Probability is non-negative 

$$P(A)\ge 0$$

+++ {"slideshow": {"slide_type": "fragment"}}

2. Probability of event $S$ is one as one of the possible outcomes _must_ happen

$$P(S)=1$$

+++ {"slideshow": {"slide_type": "slide"}}

3. Probability of a sum of disjoint events is the sum of the probabilities of each event.</br></br>
    For any integer $k>1$ including $k=\infty$ if events $A_i$ are mutually disjoint, that is for each $i\neq j$ $A_i \cap A_j =\varnothing$ then 

    $$P(A_1\cup A_2\cup\cdots \cup A_k) = P(A_1)+P(A_2) + \cdots + P(A_k)$$

    An important colorary is that when the set of outcomes is countable the probability of an event $A$ is the sum of the probabilities for each elementary event contained in $A$:

    $$P(A) = \sum_{s\in A}P(\{s\})$$

+++ {"slideshow": {"slide_type": "skip"}}

A set is countable when we can assign an unique natural number to each of its elements, in other word we can count its elements. All finite sets are of course countable. An example of not countable set is provided e.g. by the real numbers or any interval $[a,b)$ with $b>a$.

+++ {"slideshow": {"slide_type": "skip"}}

It follows from 3. that in case of countable outcomes it is enough to specify the probability of each elementary event, as they are clearly disjoint and each event can be  represented as a sum of elementary events. 

In the following  I will ommit the set parenthesis for the elementary events i.e. assume

+++ {"slideshow": {"slide_type": "slide"}}

$$P(s)\equiv P(\{s\}).$$

+++ {"slideshow": {"slide_type": "skip"}}

From axiom 2.  we have

+++ {"slideshow": {"slide_type": "fragment"}}

$$\sum_{s\in S} P(s) = 1$$

+++ {"slideshow": {"slide_type": "skip"}}

Because

+++ {"slideshow": {"slide_type": "slide"}}

$$P(\emptyset\cup S)=P(\emptyset)+P(S)=P(S)$$

+++ {"slideshow": {"slide_type": "skip"}}

it follows that

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(\emptyset)=0$$

+++ {"slideshow": {"slide_type": "slide"}, "tags": ["problem"]}

#### Problem: Complementary event

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["problem"]}

Prove that

+++ {"slideshow": {"slide_type": "fragment"}, "tags": ["problem"]}

$$P(S\setminus A)= 1-P(A)\quad\text{ where }\quad S\setminus A = \{s\in S: s\notin A\}$$

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"]}

__Answer__

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

It follows directly from the second and third axiom after noting that

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

$$(S\setminus A) \cup A = S \text{ and } (S\setminus A) \cap A = \varnothing$$

+++ {"slideshow": {"slide_type": "slide"}}

## Calculating probability

+++ {"slideshow": {"slide_type": "fragment"}}

### Model

+++ {"slideshow": {"slide_type": "skip"}}

The concept of the probability can be somewhat hazy and verges upon philosophy. My take on this is that to calculate the probability we need a _model_ of the process. E.g. for the dice example the model is that all elementary events are equally probable, leading to assignement of probability $1/36$ to every possible two dice roll outcome.

+++ {"slideshow": {"slide_type": "slide"}}

### Law of large numbers

+++ {"slideshow": {"slide_type": "skip"}}

The connection with experiment (reality) is given by the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers). It states that if you repeat an experiment independently a large number of times and average the result, what you obtain should be close to the expected value of the random variable.

+++ {"slideshow": {"slide_type": "fragment"}}

$$M_n(X_i) = \frac{X_1+X_a+\cdots+X_n}{n},\qquad X_i\text{ i.i.d.},\; E[X_i]=\mu$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$M_n(X_i)\longrightarrow\mu$$

+++ {"slideshow": {"slide_type": "skip"}}

From that it follows that fraction of times an event happens will converge to the probability of this event.

+++ {"slideshow": {"slide_type": "fragment"}}

$$\chi_{i,A}=\begin{cases}1 & X_i\in A\\
0& \text{otherwise}\end{cases}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$E[\chi_{i,A}]= 1\cdot P(A) + 0 \cdot P(S\setminus A)=P(A)$$

+++ {"slideshow": {"slide_type": "skip"}}

This is a fundation of _frequentist_ interpretation of probability.

+++ {"slideshow": {"slide_type": "skip"}}

It is harder to interpret the probability of one-off events _e.g._ "there is a 30% chance that it will rain tomorrow", or "there is 80% chance that Real Madrid will win La Liga this year" in view of the frequentist interpretation. However we can still use the Bayesian "degree of certainty(belief)" interpretation in this case.

+++ {"slideshow": {"slide_type": "slide"}}

#### Problem

How would you interpret the phrase: "There is 75% chance that I will pass this exam?".

+++ {"slideshow": {"slide_type": "slide"}}

## Conditional probability

+++ {"slideshow": {"slide_type": "skip"}}

How does a probability of an event change when we know that some other event happed? That is a central question in machine learning and is  answered by _conditional_ probability

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A|B)$$

+++ {"slideshow": {"slide_type": "skip"}}

This denotes the probability that event $A$ happened on  condition that the event also $B$ happend. The formal definition is

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A|B) = \frac{P(A\cap B)}{P(B)}$$

+++ {"slideshow": {"slide_type": "skip"}}

From this defintion it follows  that

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A\cap B)=P(A|B) P(B)$$

+++ {"slideshow": {"slide_type": "skip"}}

This is called [_product or chain rule_](https://en.wikipedia.org/wiki/Chain_rule_(probability)) and is very useful for specyfying the probability.

+++ {"slideshow": {"slide_type": "slide"}}

#### Example

+++ {"slideshow": {"slide_type": "-"}}

Let's take as an example roll of two dice. What is the probability that  the sum is six ?

+++

There are only five possible combinations that add up to six

+++

$$\{(1,5),(2,4),(3,3),(4,2),(5,1)\}$$

+++

which we can verify by some python code

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
A = set( filter(lambda s: sum(s)==6, S_dice) )
print(len(A))
print(A)      
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# Just to have nice fractions instead of floats
from fractions import Fraction 
P_A =  Fraction(len(A),len(S_dice))
print(P_A, float(P_A))
```

+++ {"slideshow": {"slide_type": "skip"}}

And now suppose that someone tells us that we have rolled three on one die. Did the the probability change?

+++

The event $B$ contains 11 elementary events:

+++

$$\{
(3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (1,3), (2,3), (4,3), (5,3), (6,3)
\}$$

+++ {"slideshow": {"slide_type": "skip"}}

Again I will use some Python code althought it is probably faster to   calculate this "by hand".

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
B = set( filter(lambda s: s[0]==3 or s[1]==3 , S_dice) )
print(len(B))
print(B)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
P_B = Fraction(len(B), len(S_dice))
print(P_B)
```

The event $A\cap B$ contains only one event $(3,3)$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
A_cap_B = A.intersection(B)
print(A_cap_B)
P_A_cap_B = Fraction(len(A_cap_B), len(S_dice))
```

so

+++

$$P(A\cap B)=\frac{1}{36}$$

+++ {"slideshow": {"slide_type": "skip"}}

And finally

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
P_A_cond_B = P_A_cap_B/P_B
print(P_A_cond_B, float(P_A_cond_B))
```

$$P(A|B) = \frac{P(A\cap B)}{P(B)}=\frac{1}{36}\cdot\frac{36}{11}=\frac{1}{11}<P(A)=\frac{5}{36}$$

+++ {"slideshow": {"slide_type": "skip"}}

So this is smaller  then $P(A)=5/36$.

+++ {"slideshow": {"slide_type": "skip"}}

#### Problem

+++ {"slideshow": {"slide_type": "skip"}}

__1.__ What if we are told that we have rolled one on one die? Has the conditional probability of rolling six changed?

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [remove, skip, answer]
---
B1 = set( filter(lambda s: s[0]==1 or s[1]==1 , S_dice) )
A_cap_B1 = A.intersection(B1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
Fraction(len(A_cap_B1), len(B1))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Bayes theorem

+++ {"slideshow": {"slide_type": "skip"}}

It is very important to keep in mind that conditional probability $P(A|B)$ is not symetric! _E.g._ when it rains the probability that sidewalk will be wet is one. On the other hand when the sidewalk is wet it does not imply  with certainty that it has rained, it may have  been _e.g._ washed by our neighbour. But as we will see many times in course of this lecture the ability to "invert" conditional probability comes in very handy.

+++ {"slideshow": {"slide_type": "skip"}}

By definition

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(B|A) = \frac{P(A \cap B)}{P(A)}\quad\text{and}\quad P(A|B) = \frac{P(A \cap B)}{P(B)}$$

+++ {"slideshow": {"slide_type": "skip"}}

we can use second expression to calculate $P(A\cap B)$ and subsitute it into first to obtain

+++ {"slideshow": {"slide_type": "fragment"}}

$$\large\boxed{P(B|A) = \frac{P(A|B)P(B)}{P(A)}}$$

+++ {"slideshow": {"slide_type": "skip"}}

This formula is know as Bayes theorem.

+++ {"slideshow": {"slide_type": "slide"}}

#### Problem: Wet sidewalk

+++ {"slideshow": {"slide_type": "-"}}

Let's apply it to the "wet sidewalk problem". We look in the morning through our window and see wet sidewalk. What is the probability that it has rained at night?

+++ {"slideshow": {"slide_type": "skip"}}

#### Answer

+++ {"slideshow": {"slide_type": "skip"}}

If $wet$ is the event "sidewalk is wet" and $rain$ is the event "it has rained" then $P(wet|rain)=1$ and according to Bayes theorem

+++ {"slideshow": {"slide_type": "skip"}}

$$P(rain|wet)=\frac{P(rain)}{P(wet)}$$

+++ {"slideshow": {"slide_type": "skip"}}

We will make a reasonable assumption that our neighbour does not wash the sidewalk when it has rained: $P(wash|rain)=0$. Also obviously $P(wet|wash)=1$ so

+++ {"slideshow": {"slide_type": "skip"}}

$$P(wet) = P(rain) + P(wash|\neg rain)P(\neg rain) = P(rain) + P(wash|\neg rain)(1-P(rain))$$

+++ {"slideshow": {"slide_type": "skip"}}

and

+++ {"slideshow": {"slide_type": "skip"}}

$$P(rain|wet) = \frac{P(rain)}{P(rain)+P(wash|\neg rain)(1-P(rain))}$$

+++ {"slideshow": {"slide_type": "skip"}}

Let's consider some "corner cases". If our neigbour always washes the sidewalk when it does not rain then the results is $P(rain)$ - sidewalk is always wet, we do not have any additional information.

+++ {"slideshow": {"slide_type": "skip"}}

If our neigbour never washes the sidewalk then results is one - the only reason for wet sidewalk is rain so when it is wet it must have rained.

+++ {"slideshow": {"slide_type": "skip"}}

If our neighbour washed the sidewalk only half of the times when it does not rain we obtain

+++ {"slideshow": {"slide_type": "skip"}}

$$P(rain|wet) = \frac{P(rain)}{P(rain)+\frac{1}{2}(1-P(rain))} = \frac{ 2 P(rain)}{1+P(rain)}$$

+++ {"slideshow": {"slide_type": "skip"}}

So if _e.g._ $P(rain)=1/7$  seeing wet sidewalk increses that chance to

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print(2 * Fraction(1,7)/(1+Fraction(1,7)))
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's plot this using `matplotlib`  and `numpy` libraries.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [12,8]
```

+++ {"slideshow": {"slide_type": "skip"}}

We can plot the whole family of plots corresponding to different values of $P(wash|\neg rain)$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [hide_src]
---
ps = np.linspace(0,1,100)
plt.xlabel("P(rain)")
plt.ylabel("P(rain|wet)");
plt.plot(ps, ps, c='grey', linewidth = 1);
for pw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]:
    plt.plot(ps, ps/(ps+pw*(1-ps)),label = "P(w|not r) = {:.2f}".format(pw)); 
plt.grid()
plt.legend();
```

+++ {"slideshow": {"slide_type": "slide", "slideshow": {"slide_type": "slide"}}, "tags": ["problem"]}

#### Problem: Base rate fallacy

+++ {"tags": ["problem"], "slideshow": {"slide_type": "-"}}

You are tested for a rare disease (1 person in 250). Test has 80%  true positive rate and  10% false positive rate. i.e. test gives positive (you are ill) result for 80% of ill patients and for 10% of healthy patients.   

Your are tested positive, what are the chances you have the disease?

+++ {"slideshow": {"slide_type": "skip"}}

#### Answer

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

What we need is the  probability that we are ill on condition that we have been tested positive:

+++ {"slideshow": {"slide_type": "skip"}, "tags": ["answer"]}

$$P(ill|P)= \frac{P(ill, P)}{P(P)}$$

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

The probability of being ill and tested positive is

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
p_ill_p = 0.004 * 0.8  
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

The probability of being tested positive is

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

$$P(P)=P(ill,P)+P(\neg ill, P)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
p_p = .004*0.8 + 0.996*0.1
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

and finally

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: [answer]
---
p_ill_cond_p = p_ill_p/p_p
print("{:4.1f}%".format(100*p_ill_cond_p))
```

+++ {"tags": ["answer"], "slideshow": {"slide_type": "skip"}}

So there is no cause to despair yet :)

+++ {"slideshow": {"slide_type": "skip", "slideshow": {"slide_type": "slide"}}}

### Increase of information (learning)

+++ {"slideshow": {"slide_type": "skip"}}

One could say that this test is useless if positive  result gives only $3\%$ chance of being ill. And  this particular test was actually discarde but it is not totaly useless. Before taking the test our chance of being ill was $0.4\%$. After seing the positive result it "jumped" more then ten times to $3.1\%$.

+++ {"slideshow": {"slide_type": "skip"}}

$$0.004 \longrightarrow 0.031$$

+++ {"slideshow": {"slide_type": "skip"}}

After seing a negative result our chances of being ill dropped four times:

+++ {"slideshow": {"slide_type": "skip"}}

$$0.004 \longrightarrow 0.001 $$

+++ {"slideshow": {"slide_type": "slide"}}

## Independent events

+++ {"slideshow": {"slide_type": "skip"}}

It may happen that  knowledge that $B$ happened does not change  the probability of $A$

+++ {"slideshow": {"slide_type": "-"}}

$$P(A|B) = P(A)$$

+++ {"slideshow": {"slide_type": "skip"}}

We say then that  events $A$ and $B$ are _independent_.

+++ {"slideshow": {"slide_type": "skip"}}

For example when tossing the coin the outcome of toss does not depend in any way on the outcome of previous tosses or in case of dice the  face they land on are independent etc.

+++ {"slideshow": {"slide_type": "skip"}}

Substituting the definition of conditional independence

+++ {"slideshow": {"slide_type": "fragment"}}

$$\frac{P(A\cap B)}{P(B)}  = P(A)$$

+++ {"slideshow": {"slide_type": "skip"}}

we obtain  a more familiar factorisation criterion for joint probability of independent events

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(A\cap B) = P(A)\cdot P(B)$$

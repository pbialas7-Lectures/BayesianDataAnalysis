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

```{code-cell}
%load_ext autoreload
%autoreload 2
```

### Problem 2

+++ {"tags": ["EN"]}

#### Estimation of differences between two independent experiments  (from "Bayesian Data Analysis")

+++ {"tags": ["EN"]}

An experiment was performed on the effects of magnetic fields on the flow of calcium out of chicken brains. Two groups of chickens were involved: a control group of 32 chickens and an exposed group of 36 chickens. One measurement was taken on each chicken, and the purpose of the experiment was to measure the average flow $\mu_c$ in untreated (control) chickens and the average flow $\mu_t$ in treated chickens. The 32 measurements on the control group had a sample mean of 1.013 and a sample standard deviation of 0.24. The 36 measurements on the treatment group had a sample mean of 1.173 and a sample standard deviation of 0.20.

+++

#### Problem 2.1

+++ {"tags": ["EN"]}

Assuming the control measurements were taken at random from a normal distribution with mean $\mu_c$ and variance $\sigma_c^2$, what is the posterior distribution of $\mu_c$? Similarly, use the treatment group measurements to determine the marginal posterior distribution of $\mu_t$. Assume an uninformative  prior distribution on $\mu_c$, $\mu_t, \sigma^2_c, \sigma^2_t \sim 1,1,\sigma_c^{-2}, \sigma_t^{-2}$.

+++

#### Problem 2.2

+++ {"tags": ["EN"]}

What is the posterior distribution for the difference, $\mu_t-\mu_c$? To get this, you may sample from the independent $t$ distributions you obtained in part (1) above. Plot a histogram of your samples and give an approximate 95% posterior interval (highest density region) for $\mu_t-\mu_c$.

+++ {"tags": ["EN"]}

### Random numbers

+++ {"tags": ["EN"]}

To generate random numbers with given distribution we can use functions in the module <code>scipy.stats</code> ([docs](https://docs.scipy.org/doc/scipy/reference/stats.html)) or <code>numpy.random</code>.

```{code-cell}
import numpy as np
import scipy.stats as st
```

+++ {"tags": ["EN"]}

Here we generate 10000 random numbers from  normal distribution with mean 1 and standard deviation 2

```{code-cell}
randoms = st.norm(1,2).rvs(size=10000)
```

+++ {"tags": ["EN"]}

We will use <code>matplotlib</code> library for plotting

```{code-cell}
import matplotlib.pyplot as plt
```

+++ {"tags": ["EN"]}

for example

```{code-cell}
plt.plot(randoms[:500],'.');
```

+++ {"tags": ["EN"]}

displays the "time series" of first 500 numbers.

+++ {"tags": ["EN"]}

### Histograms

+++ {"tags": ["EN"]}

More useful way of visualizing random number is the [histogram](https://en.wikipedia.org/wiki/Histogram) which shows how many numbers falls in each interval (bin). There are many ways to specify bins, the simplest is just to pass the number of bins to the plotting function. The range which is divided into the bins is then calculated automatically as to contain all the enumber. See the [documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) for more information on how to use this function.

```{code-cell}
c,b,p = plt.hist(randoms,bins=50)
plt.show()
```

+++ {"tags": ["EN"]}

Apart from  plotting the histogram this function returns the number of counts in each bin (variable `c` above), and list of bin edges (variable `b` above). Please note that `len(b)=len(c)+1` ([docs](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)).

+++ {"tags": ["EN"]}

Often we will need count values normalized so that the integral (area) under the histogram is one, so it can be compared with probability distribution. This can be achived using the density argument:

+++

Często będziemy potrzebowali znormalizowej wersji histogramu  tak aby pole pod wykresem wynosiło jeden. Umożliwia to porównywanie histogramu z gęstością prawdopodobieństwa zmiennej losowej. Znormlizowany histogram otrzymamy ustawiając  argument `density` na `True`

+++ {"tags": ["EN"]}

### Highest density region

+++ {"tags": ["EN"]}

Highest density region was already defined  in the notebooks  `Lectures/200_introduction/Errors/` and `Lectures/200_Introduction/highest_density_region`. The associated functions can be found in the file `src/bda/stats.py`. You can import it, but first you have add its path to python path:

```{code-cell}
import sys
sys.path.append('../../src/')
from bda.stats import hdr_d
```

+++ {"tags": ["EN"]}

Function `hdr_d` calculates the highest density interval for a distribution defined by two arrays, one containing the values of the random variable, second their probability.
Given the `c` and `b` obtained from the `plt.hist` function we can transform it to the representation used in `hdr_d`:

```{code-cell}
dist=c
xs = (b[1:]+b[:-1])/2 # centers of bins
```

```{code-cell}
hdr95 = hdr_d(xs, dist, 0.95) # return intervals, mass  and level p
```

```{code-cell}
hdr95
```

```{code-cell}
plt.hist(randoms,bins=50, density=True, histtype='step');
plt.fill_between(xs,dist,0,where = ( (xs>hdr95[0][0]) & (xs<=hdr95[0][1])), color='lightgray' );
```

```{code-cell}

```

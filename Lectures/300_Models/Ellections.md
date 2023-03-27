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
import numpy as np
import scipy
import matplotlib.pyplot  as plt
import json
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import sys
sys.path.append('../../src')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
n_samples = 1013
poll = {
    'ZP':0.327,
    'KO':0.28,
    'Lewica': 0.093,
    'Polska 2050': 0.076,
    'PSL': 0.055,
    'Konfederacja': 0.052,
    'Porozumienie': 0.02,
    'Kukiz15':0.009,
    'Inna': 0.021,
    'Nie wiem': 0.068
}
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
prob = np.asarray(list(poll.values()))
responses = np.array(np.round(n_samples*prob), dtype='int')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
poll_dict = {p: np.round(poll[p]*n_samples) for p in poll.keys()}
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
with open("poll.json","w") as f:
    json.dump(poll_dict, f)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Ellection polls

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
with open("poll.json") as f:
    poll = json.load(f)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
poll
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
responses= np.array(list(poll.values()), dtype='int')
names=list(poll.keys())
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
n=np.sum(responses)
probs = responses/n
```

+++ {"slideshow": {"slide_type": "slide"}}

### Multinomial distribution

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(n_1,\ldots,n_k|p_1,\ldots,p_k)=\frac{(\sum_{i=1}^k n_i )!}{n_1!\cdots n_k!}p_1^{n_1}\cdots p_k^{n_k}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p_1,\ldots,p_k|\alpha_1,\dots,\alpha_k)=\frac{\Gamma(\sum_{i=1}^k \alpha_i)}{\Gamma(\alpha_1)
\cdots \Gamma(\alpha_k)}p_1^{\alpha_1-1}\cdots p_k^{\alpha_k-1}$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$P(p_1,\ldots,p_k|n_1,\ldots,n_k,\alpha_1,\ldots,\alpha_k)\propto P(n_1,\ldots,n_k|p_1,\ldots,p_k)P(p_1,\ldots,p_k|\alpha_1,\dots,\alpha_k)$$

+++ {"slideshow": {"slide_type": "slide"}}

$$p_1,\ldots,p_k|n_1,\ldots,n_k,\alpha_1,\ldots,\alpha_k\sim \operatorname{Dirichlet}(\alpha_1+n_1,\ldots,\alpha_k+n_k)$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
sample =  scipy.stats.dirichlet(alpha=responses+1).rvs(size=(100000))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample[:,0], bins=100, histtype='step', density=True, label=names[0])
plt.hist(sample[:,1], bins=100, histtype='step', density=True, label=names[1])
plt.axvline(probs[0]);plt.axvline(probs[1],c="C1")
plt.legend();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
h = ax.hist2d(sample[:,0], sample[:,1], bins=100)
plt.colorbar(h[3],ax=ax)
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
h = ax.hexbin(sample[:,0], sample[:,1], gridsize=100)
plt.colorbar(h,ax=ax)
plt.grid()
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample[:,0]-sample[:,1], bins=100, histtype='step', density=True, label=f"{names[0]} - {names[1]}")
plt.axvline(0.0)
plt.legend();
```

```{code-cell} ipython3
diff = sample[:,0]-sample[:,1]
(diff>0).sum()/len(diff)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
plt.hist(sample[:,4], bins=100, histtype='step', density=True, label=names[4])
plt.axvline(0.05)
plt.legend();
```

```{code-cell} ipython3
np.sum(sample[:,4]>0.05)/len(sample)
```

```{code-cell} ipython3

```

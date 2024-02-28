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
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
import numpy as np

import matplotlib.pyplot as plt
```

```{code-cell} ipython3
import hdr
```

```{code-cell} ipython3
from scipy.stats import beta
```

```{code-cell} ipython3
pdf = beta(2,4).pdf
```

```{code-cell} ipython3
p = 1.5
r = hdr.Rf(pdf,p,0,1)
ps=np.linspace(0,1,500)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(ps, pdf(ps));
hdr.plot_R(pdf,r,0,1, ax=ax, alpha=0.3)
ax.axhline(p,c='darkblue', linewidth=1)
ax.text(0.75,0.8,f"p={p:.3f} {hdr.PofR(pdf,r):.3f}",transform=plt.gca().transAxes, fontsize=16);
```

```{code-cell} ipython3
from scipy.optimize import minimize_scalar
def plot_All(f, p,*,a=0,b=1):
    max_p = -minimize_scalar(lambda x:-f(x),bounds=(a,b), method="Bounded").fun
    r = hdr.Rf(f,p,a,b)
    fig, ax = plt.subplots(figsize=(12,8))
    ps=np.linspace(a,b,500)
    ax.set_xlim(a,b)
    ax.set_ylim(0,1.1*max_p)
    ax.plot(ps, f(ps));
    hdr.plot_R(f,r,a,b, ax=ax, alpha=0.5, color='lightblue')
    ax.axhline(p,c='darkblue', linewidth=1)
    ax.text(0.75,0.8,f"p={p:.3f} {hdr.PofR(f,r):.3f}",transform=plt.gca().transAxes, fontsize=16);
```

```{code-cell} ipython3
plot_All(pdf,1.5)
```

```{code-cell} ipython3
from ipywidgets import interact
```

```{code-cell} ipython3
interactive_plot = interact(lambda x: plot_All(pdf,x),x=(0.0,2.5,0.001))
```

```{code-cell} ipython3
p, r=hdr.hdrf(pdf,0.95,a=0,b=1)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
pdf_mmod = lambda x: beta(6,2).pdf(x)/3 + 2/3*beta(2,9).pdf(x)
```

```{code-cell} ipython3
interactive_plot = interact(lambda x: plot_All(pdf_mmod,x),x=(0.0,2.5,0.001))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
p, r=hdr.hdrf(pdf_mmod,0.95,a=0,b=1)
```

```{code-cell} ipython3
p
```

```{code-cell} ipython3
r
```

```{code-cell} ipython3

```

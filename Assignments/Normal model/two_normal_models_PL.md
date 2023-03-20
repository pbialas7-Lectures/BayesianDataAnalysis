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

+++ {"tags": ["PL"]}

#### Ocena różnic pomiędzy dwoma niezależnymi eksperymentami

+++ {"tags": ["PL"]}

Przeprowadzono eksperyment mający na celu zbadanie wpływu pola magnetycznego na wypływ wapna z mózgów kurczaków. Do eksperymentu użyto dwie grupy kurczaków: w grupie kontrolnej były 32 kurczaki a w grupie badanej wystawionej na działanie pola magnetycznego było 36 kurczaków. Pomiar przepływu wapna dokonany został dla każdego kurczaka w każdej z obu grup. W grupie kontrolnej średni przepływ wyniósł $1.013$ a standardowe odchylenie  wyniosło $0.24$. W grupie badanej było to odpowiednio $1.173$ i $0.20$.

+++

#### Problem 2.1

+++ {"tags": ["PL"]}

Zakładając, że pomiary w grupie kontrolnej pochodziły z rozkładu normalnego o średniej $\mu_c$ i wariancji $\sigma_c^2$ proszę podać rozkład _a posteriori_ dla $\mu_c$. Proszę założyć  prior $\mu_c\sim 1$ i $\sigma_c^2\sim \sigma_c^{-2}$. Podobnie proszę podać rozkład _a posteriori_ dla średniej w grupie kontrolnej $\mu_t$.

+++

#### Problem 2.2

+++ {"tags": ["PL"]}

Jaki rozkład _a posteriori_ ma różnica średnich $\mu_t-\mu_c$? Aby to obliczyć proszę wylosować po $1e6$ (1000000) liczb dla każdej grupy z rozkładów które otrzymali Państwo w pierwszym punkcie. Proszę narysować histogram rokładu różnic $\mu_t-\mu_c$ pomiędzy dwoma grupami i oszacować 95% region największej gestości (HDR).

+++ {"tags": ["PL"]}

#### Liczby losowe

+++ {"tags": ["PL"]}

Do wygenerowania liczb losowych posłużymy się fukcjami znajdującymi sie w module <code>scipy.stats</code> ([docs](https://docs.scipy.org/doc/scipy/reference/stats.html)) albo <code>numpy.random</code>.

```{code-cell}
import numpy as np
import scipy.stats as st
```

+++ {"tags": ["PL"]}

Poniżej generujemy 10000 liczb losowych z rozkładu normalnego ze średnią $1$ i standardowym odchyleniem $2$.

```{code-cell}
randoms = st.norm(1,2).rvs(size=10000)
```

+++ {"tags": ["PL"]}

Do rysowania wykresów posłużymy się biblioteką `matplotlib`

```{code-cell}
import matplotlib.pyplot as plt
```

+++ {"tags": ["PL"]}

na przykład polecenie

```{code-cell}
plt.plot(randoms[:500],'.');
```

+++ {"tags": ["PL"]}

wyświetla "szereg czasowy" dla pierwszych 500 liczb.

+++ {"tags": ["PL"]}

Bardziej przydatnym sposobem wizualizacji liczb losowych są [histogramy](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html), które pokazują ile liczb znajduje się każdym przedziale (ang. bin). Jest wiele sposobów określenia tych przedziałów, ale najprostszym jest podanie liczby przedziałów w funkcji rysującej. Zakres który zostanie podzielony na te przedziały jest wtedy określony automatycznie, tak aby zawarły sie w nim wszystkie liczby.  Więcej informacji znajda państwo w [dokumentacji](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html).

```{code-cell}
c,b,p = plt.hist(randoms,bins=50)
plt.show()
```

+++ {"tags": ["PL"]}

Fukcja `plt.hist` poza rysowaniem histogramu zwraca  liczbe elementów w każdym przedziale (zmienna `c`) oraz granice przedziałów (zmienna `b`). Proszę zwrócić uwagę, że `len(b)=len(c)+1` ([docs](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)).

+++

Często będziemy potrzebowali znormalizowej wersji histogramu  tak aby pole pod wykresem wynosiło jeden. Umożliwia to porównywanie histogramu z gęstością prawdopodobieństwa zmiennej losowej. Znormlizowany histogram otrzymamy ustawiając  argument `density` na `True`

```{code-cell}
:tags: [PL]

c,b,p = plt.hist(randoms,bins=50, density=True, histtype='step')
xs = np.linspace(-7,7, 200)
plt.plot(xs, st.norm(1,2).pdf(xs))
plt.show()
```

+++ {"tags": ["PL"]}

### Przedział największej gęstości

+++ {"tags": ["PL"]}

Przedział największej gęstości został zdefiniowany w notebookach `Lectures/200_introduction/Errors/` i `Lectures/200_Introduction/highest_density_region`. Skojarzone z  nimi funkcje można znajdują się w pliku `src/bda/stats.py`. Można je zaimportować, ale najpierw trzeba dodać ścieżkę tego pliku do ściezki systemowej pythona

```{code-cell}
import sys
sys.path.append('../../src/')
from bda.stats import hdr_d
```

+++ {"tags": ["PL"]}

Funcja `hdr_d` oblicza przedział największej gęstości dla dystrubycji zdefiniowanej przez dwie tablice. Jedną zawierającą wartości zmiennej losowej i drugą zawierająca ich prawdopodobieństwa. Odpowiada to zmiennym `b` i` c` zwróconym przez `plt.hist`, z tym, że zmienna `b` wymaga przekształcenia

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

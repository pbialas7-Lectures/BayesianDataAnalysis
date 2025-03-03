import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import sys

sys.path.append('')

from .stats import R_f, PofR_f, R_d, PofR_d


def plot_R_f(f, r, a, b, *, alpha=0.5, n=500, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    xs = np.linspace(a, b, n)
    ra = np.array(r).reshape(-1, 2)
    for i in range(len(ra)):
        ax.fill_between(xs, f(xs), 0, where=(xs > ra[i, 0]) & (xs < ra[i][1]), alpha=alpha, **kwargs)


def plot_All(f, p, *, a=0, b=1,figsize=None):
    max_p = -minimize_scalar(lambda x: -f(x), bounds=(a, b), method="Bounded").fun
    r = R_f(f, p, a, b)
    fig, ax = plt.subplots(figsize=figsize)
    ps = np.linspace(a, b, 500)
    ax.set_xlim(a, b)
    ax.set_ylim(0, 1.1 * max_p)
    ax.plot(ps, f(ps));
    plot_R_f(f, r, a, b, ax=ax, alpha=0.5, color='lightblue')
    ax.axhline(p, c='darkblue', linewidth=1)
    ax.text(0.4, 0.8, f"r={p:.3f}  $P\\left(x\\in R(r)\\right) = {PofR_f(f, r):.3f}$",
            transform=plt.gca().transAxes,
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'));
    return fig, ax


def plot_rp_d(xs, dist, p):
    fig, ax = plt.subplots()
    ax.plot(xs, dist);
    ax.set_ylim(0, 1.25 * dist.max())
    Rp = R_d(dist, p)
    Rx = xs[Rp]
    mass = PofR_d(dist, Rp)
    ax.axhline(p)
    for itr in Rx.reshape(-1, 2):
        ax.fill_between(xs, dist, 0, where=((xs > itr[0]) & (xs < itr[1])), color='lightgray');
    ax.text(.8, 0.8, f"${mass:.2f}$", fontsize=14, transform=ax.transAxes);
    ax.text(10, 0.005, f"${mass:.2f}$", fontsize=14);

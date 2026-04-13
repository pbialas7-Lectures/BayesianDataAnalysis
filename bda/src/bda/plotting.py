import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import brentq
from scipy.special import logsumexp


def darrow(start, end, *, ax, c, label):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='<->', color=c))
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    ax.text(midpoint[0], midpoint[1], label, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

def plot_cartesian(N_row, N_col, *, figsize=None, radius=0.4, cfunc=lambda i, j: "red", ax=None):
    fig = None
    if ax is None:
        if figsize is None:
            figsize = (N_col, N_row)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(N_col / N_row)
        ax.set_xticks([])  # Remove x-axis tick marks
        ax.set_yticks([])  # Remove y-axis tick marks
        ax.set_xticklabels([])  # Remove x-axis tick labels
        ax.set_yticklabels([])
        ax.set_xlim(0.5, N_col + 0.5)
        ax.set_ylim(0.5, N_row + 0.5)

    for r in range(1, N_row + 1):
        for c in range(1, N_col + 1):
            if cfunc(c, r) is not None:
                ax.add_patch(Circle((c, r), radius, color=cfunc(c, r), alpha=1.0))

    return fig, ax

def plot_cartesian(N_row, N_col, *, figsize=None, radius=0.4, cfunc=lambda i, j: "red", ax=None):
    fig = None
    if ax is None:
        if figsize is None:
            figsize = (N_col, N_row)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(N_col / N_row)
        ax.set_xticks([])  # Remove x-axis tick marks
        ax.set_yticks([])  # Remove y-axis tick marks
        ax.set_xticklabels([])  # Remove x-axis tick labels
        ax.set_yticklabels([])
        ax.set_xlim(0.5, N_col + 0.5)
        ax.set_ylim(0.5, N_row + 0.5)

    for r in range(1, N_row + 1):
        for c in range(1, N_col + 1):
            if cfunc(c, r) is not None:
                ax.add_patch(Circle((c, r), radius, color=cfunc(c, r), alpha=1.0))

    return fig, ax

def simple_plot(p=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, lw=0.5,color='gray')
    ax.axvline(1, lw=0.5,color='gray')
    ax.axhline(0.0, lw=0.5,color='gray')
    if p is not None:
        ax.axvline(p, color='black', lw=0.5)
    ax.set_xlim(0, 1)
    return fig, ax

def plot_c_vs_hdr_example(post, c,*,beta,delta, figsize=None,show=True):
    from bda.stats import cdi_central_f
    fig, ax = simple_plot(figsize=figsize, p=None)
    xs = np.linspace(0,1,1000)
    ax.plot(xs, post.pdf(xs))
    from bda.stats import cdi_central_f
    _,l,r=cdi_central_f(post.pdf, c,beta,0,1)
    ax.axvline(l, color='blue', lw=0.75);
    ax.axvline(r, color='blue', lw=0.55)
    ax.fill_between(xs, post.pdf(xs), 0, where=(xs > l) & (xs <= r), alpha=0.5, color='lightblue');
    if show:
        if post.pdf(r)>post.pdf(l):
            rr = r + delta
            ax.fill_between(xs, post.pdf(xs), 0, where=(xs <= rr) & (xs > r), alpha=0.5, color='red');
            yrr = post.pdf(rr)
            ax.axhline(yrr, color='red', lw=0.5)
            xm = brentq(lambda x: post.pdf(x) - yrr, l, r)
            ax.fill_between(xs, post.pdf(xs), 0, where=(xs > l) & (xs < xm), alpha=1, color='blue');
        else:
            ll = l - delta
            ax.fill_between(xs, post.pdf(xs), 0, where=(xs >= ll) & (xs < l), alpha=0.5, color='red');
            yll = post.pdf(ll)
            ax.axhline(yll, color='red', lw=0.5)
            xm = brentq(lambda x: post.pdf(x) - yll, l, r)
            ax.fill_between(xs, post.pdf(xs), 0, where=(xs > xm) & (xs < r), alpha=1, color='blue');

    return fig, ax
  

def plot_marginals(xs, ys, density,*, log=False, figsize=(7,7), levels=10):
  density_= density/density.sum()  if not log else density - logsumexp(density)
  density_ = np.exp(density_) if log else density_
  fig = plt.figure(figsize=figsize);
  gs = gridspec.GridSpec(2,3,
      width_ratios=[1,4,0.10],height_ratios=[1, 4],   # top marginal narrow, main plot tall
      hspace=0.05,
      wspace=0.05,
  )

  Z = np.trapezoid(np.trapezoid(density_, x=ys, axis=0), x=xs)
  density_/=Z
  
  ax_main  = fig.add_subplot(gs[1, 1])                        # bottom-right
  ax_top   = fig.add_subplot(gs[0, 1], sharex=ax_main)        # top-right
  ax_left  = fig.add_subplot(gs[1, 0], sharey=ax_main) 
  ax_right = fig.add_subplot(gs[1,2])
  
  cf = ax_main.contourf(xs,ys, density_, levels=levels, cmap="viridis")
  ax_main.contour(xs,ys, density_, levels=levels,
                  colors="white", linewidths=0.5, alpha=0.4)
  ax_main.tick_params(axis='y', left=False, labelleft=False);
  fig.colorbar(cf, cax=ax_right)
  ax_right.yaxis.set_tick_params(labelsize=7)

  marginal_x = np.exp(logsumexp(density,0)) if log else np.sum(density,0)
  marginal_x/=np.trapezoid(marginal_x, xs)
  ax_top.plot(xs, marginal_x, color="steelblue",
              linewidth=0.4)
  ax_top.set_xlim(ax_main.get_xlim())
  ax_top.tick_params(axis='x', labelbottom=False)
  ax_top.tick_params(axis='x', top='True', labeltop=True,direction='in')

  ax_top.yaxis.set_tick_params(labelsize=7)
  ax_top.grid(True);

  marginal_y = np.exp(logsumexp(density,1)) if log else np.sum(density,1)
  marginal_y/=np.trapezoid(marginal_y,ys)
  ax_left.plot(marginal_y ,ys, color="steelblue", 
                linewidth=0.4)          # rotated!
  ax_left.set_ylim(ax_main.get_ylim())
  ax_left.yaxis.set_visible(True)
  ax_left.xaxis.set_tick_params(labelsize=7, direction='in')
  ax_left.tick_params(axis='y', direction='in')
  ax_left.invert_xaxis();
  ax_left.grid(True);
  return fig, ax_main, ax_top, ax_left
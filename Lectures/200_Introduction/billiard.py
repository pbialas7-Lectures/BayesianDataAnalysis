import numpy as np
import matplotlib.pyplot as plt


def simple_plot(p=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, lw=0.5,color='gray')
    ax.axvline(1, lw=0.5,color='gray')
    ax.axhline(0.0, lw=0.5,color='gray')
    if p is not None:
        ax.axvline(p, color='black', lw=0.5)
    ax.set_xlim(0, 1)
    return fig, ax



def make_pool_figure(p, *, ylim=(0, 1), width=9, aspect=3 / 2, fontsize=20):
    fig, ax = plt.subplots(figsize=(width, width / aspect))
    plt.subplots_adjust(bottom=0.2)
    ax.xaxis.set_visible(True);
    ax.yaxis.set_visible(False);
    # ax.annotate('$p$', (p, -.1 ), fontsize=fontsize, annotation_clip=False,ha='center');
    ax.set_xlim(0, 1);
    ax.set_ylim(0, 1);
    pax = ax.twinx()
    pax.get_yaxis().set_visible(False)
    pax.set_ylim(*ylim)
    return fig, ax, pax


def plot_ball(ax, x, y, *, bc, bs=100, lc=None, draw_line=False, empty=False, **kwargs):
    fill = bc
    if empty:
        fill = 'white'
    ax.scatter(x, y, marker='o', s=bs, edgecolor=bc, color=fill, zorder=1, **kwargs);
    if draw_line:
        if lc:
            linecolor = lc
        else:
            linecolor = bc
        ax.axvline(x, color=linecolor, zorder=-1)
    return ax


cs = np.asarray(['red', 'darkgreen'])


def plot_balls(ax, n, x, left, cs, *, draw_line=True, **kwargs):
    for i in range(n):
        plot_ball(ax, x[i, 0], x[i, 1], bc=cs[left[i]], bs=200, draw_line=draw_line, empty=True, **kwargs);


def plot_posteriors(posteriors, *, a=0, b=1, n_points=500, gamma=0.5, ax=None):
    xs = np.linspace(a, b, n_points)
    if ax is None:
        ax = plt.gca()
    alpha = 1
    for post in reversed(posteriors):
        if hasattr(post, 'pdf'):
            ax.plot(xs, post.pdf(xs), zorder=1, c='blue', alpha=alpha);
        else:
            ax.plot(xs, post(xs), zorder=1, c='blue', alpha=alpha);
        alpha *= 0.5
    return alpha

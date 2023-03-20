import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad


# Function

# Mean
def mass_f(f, a, b):
    """Integral of f between a and b"""
    return quad(f, a, b)[0]


def mean_f(f, a, b):
    """Mean of the distribution f on interval [a,b]"""
    norm = quad(f, a, b)[0]
    mean = quad(lambda x: x * f(x), a, b)[0]
    return mean / norm


def median_f(f, a, b):
    """Median of distribution f on interval [a,b]"""
    norm = quad(f, a, b)[0]

    def cdf_med(x):
        return quad(f, a, x)[0] / norm - 0.5

    return brentq(cdf_med, a, b)


def var_central_f(f, c, a, b):
    """Variance around central point of distribution f on interval [a,b]"""
    norm = quad(f, a, b)[0]

    def v(x):
        return (x - c) ** 2 * f(x) / norm

    return quad(v, a, b)[0]


# Confidence interval
def cdi_central_f(f, center, beta, a, b):
    """Symmetric confidence interval around central point"""

    x_max = min(center - a, b - center)

    def mass(w):
        return quad(f, center - w, center + w)[0]

    if mass(x_max) < beta:
        return False, center - x_max, center + x_max
    else:
        width = brentq(lambda x: mass(x) - beta, 0, x_max)
        return True, center - width, center + width


def cdi_left_f(f, left, beta, *, a=0, b=1):
    """Confidence interval starting from left edge"""

    def mass(r):
        return quad(f, left, r)[0]

    if mass(b) < beta:
        return False, a, b
    else:
        right = brentq(lambda x: mass(x) - beta, left, b)
    return True, left, right


# Highest density region


def fzeros(f, a, b, n=100):
    """Find all zeros of function on an interval"""
    xs = np.linspace(a, b, n)
    delta = xs[1] - xs[0]
    fxs = f(xs)
    cross = fxs[:-1] * fxs[1:] <= 0
    fz = xs[:-1][cross]
    return [brentq(f, x, x + delta) for x in fz]


def R_f(f, p, a, b):
    """Region R(p) for distribution f"""
    fzs = fzeros(lambda x: f(x) - p, a, b)
    if f(a) > p:
        fzs = [a] + fzs
    if f(b) > p:
        fzs += [b]
    return fzs


def PofR_f(f, r):
    """Probability of region with respect to distribution"""
    p = 0.0
    ra = np.array(r).reshape(-1, 2)
    for i in range(len(ra)):
        p += quad(f, *ra[i])[0]
    return p


def hdr_f(f, beta, *, a, b):
    """Highest density region for distribution f on interval [a,b]"""
    max_p = -minimize_scalar(lambda x: -f(x), bounds=(a, b), method="Bounded").fun

    def g(p):
        r = R_f(f, p, a, b)
        return PofR_f(f, r) - beta

    p_zero = brentq(g, 0, max_p)
    rp = R_f(f, p_zero, a, b)
    return rp, p_zero, PofR_f(f, rp)


# Discrete
def R_d(dist, p):
    gt = dist > p
    return np.where(np.logical_xor(gt[1:], gt[:-1]))[0]


def PofR_d(dist, r):
    assert (r.size % 2) == 0
    cdf = np.cumsum(dist)
    mass = cdf[r].reshape(-1, 2)
    return (np.sum(mass[:, 1] - mass[:, 0])) / cdf[-1]


def hdr_d(xs, dist, beta):
    z = np.trapz(dist, xs)
    dist_n = dist / z
    p_max = dist_n.max()
    p_min = 0.0
    for i in range(24):
        p = (p_max + p_min) / 2
        Rp = R_d(dist_n, p)
        mass = PofR_d(dist_n, Rp)
        if mass > beta:
            p_min = p
        else:
            p_max = p
    r = R_d(dist, (p_min + p_max) / 2)
    return xs[r], p_max, PofR_d(dist, r)

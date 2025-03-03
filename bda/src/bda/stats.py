"""
This is a collection of functions for computing statistics and confidence intervals for distributions.

The functions are divided into two groups: those that work with a distribution defined by its density function
distinguished by the suffix _f and those that work with a distribution defined by its histogram distinguished by the
suffix _d.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

arviz_imported = False
try:
    import arviz as az
    import xarray
    arviz_imported = True
except ImportError:
    pass

#  Distributions defined by their density function distinguished by suffix _f
# f is the density function of the distribution, but it does not need to be normalized.

def mass_f(f, a, b):
    """Integral (mass) of f between a and b

    Parameters
    ----------
    f : function
        Density function of the distribution
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval
    """
    return quad(f, a, b)[0]


def mean_f(f, a, b):
    """Mean of the distribution f on an interval [a,b]

    Parameters
    ----------
    f : function
        Density function of the distribution
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval

    Returns
    -------
    float
        Mean of the distribution
    """
    norm = mass_f(f, a, b)
    mean = quad(lambda x: x * f(x), a, b)[0]
    return mean / norm


def median_f(f, a, b):
    """Median of distribution f on an interval [a,b]

    Parameters
    ----------
    f : function
        Density function of the distribution
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval

    Returns
    -------
    float
        Median of the distribution
    """
    norm = quad(f, a, b)[0]

    def cdf_med(x):
        return quad(f, a, x)[0] / norm - 0.5

    return brentq(cdf_med, a, b)


def var_central_f(f, c, a, b):
    """Variance around a central point c of distribution f on an interval [a,b]

    Parameters
    __________
    f : function
        Density function of the distribution
    c : float
        Central point
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval

    Returns
    -------
    float
        Variance of the distribution
    """
    norm = quad(f, a, b)[0]

    def v(x):
        return (x - c) ** 2 * f(x) / norm

    return quad(v, a, b)[0]


# Confidence interval
def cdi_central_f(f, c, beta, a, b):
    """Symmetric beta-confidence interval around central point c of distribution f on an interval [a,b]

    Parameters
    ----------
    f : function
        Density function of the distribution
    c : float
        Central point
    beta : float
        Confidence level
    a : float
        Lower bound of the interval
    b : float

    Returns
    -------
    status : bool
        True if the interval is valid, False otherwise
    left : float
        Left end of the interval
    right : float
        Right end of the interval
    """

    x_max = min(c - a, b - c)

    def mass(w):
        return quad(f, c - w, c + w)[0]

    if mass(x_max) < beta:
        return False, c - x_max, c + x_max
    else:
        width = brentq(lambda x: mass(x) - beta, 0, x_max)
        return True, c - width, c + width


def cdi_left_f(f, left, beta, *, a=0, b=1):
    """beta-confidence interval starting from left of the distribution f on an interval [a,b]

    Parameters
    ----------
    f : function
        Density function of the distribution
    left : float
        Left end of the interval
    beta : float

    Returns
    -------
    status : bool
        True if the interval is valid, False otherwise
    left : float
        Left end of the interval
    right : float
        Right end of the interval
    """

    def mass(r):
        return quad(f, left, r)[0]

    if mass(b) < beta:
        return False, a, b
    else:
        right = brentq(lambda x: mass(x) - beta, left, b)
    return True, left, right


# Highest density region


def zeros_f(f, a, b, n=100):
    """Find all zeros of function f on an interval (a,b)

    It divides the interval into n-1 subintervals, finds the intervals that contain zeros by checking the sign of f
    at each end of the interval and then uses brentq to find the zero.


    Parameters
    ----------
    f : function
        Function to find zeros of
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval
    n : int
        Number of subintervals  (default is 100)

    Returns
    -------
    list
        List of zeros of f
    """
    xs = np.linspace(a, b, n)
    delta = xs[1] - xs[0]
    fxs = f(xs)
    cross = fxs[:-1] * fxs[1:] <= 0
    fz = xs[:-1][cross]
    return [brentq(f, x, x + delta) for x in fz]


def R_f(f, p, a, b):
    """Region R(p) for distribution f on an interval [a,b]

    Parameters
    ----------
    f : function
        Density function of the distribution
    p : float
        Threshold
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval

    Returns
    -------
    list
        List of intervals where f(x) > p.
    """

    fzs = zeros_f(lambda x: f(x) - p, a, b)
    if f(a) > p:
        fzs = [a] + fzs
    if f(b) > p:
        fzs += [b]
    return fzs


def PofR_f(f, r):
    """Integral of f over the region r. The region is a list of intervals.

    Parameters
    ----------
    f : function
        Density function of the distribution
    r : list
        List of intervals

    Returns
    -------
    float
        Mass of the region r
    """
    p = 0.0
    ra = np.array(r).reshape(-1, 2)
    for i in range(len(ra)):
        p += quad(f, *ra[i])[0]
    return p


def hdr_f(f, beta, *, a, b):
    """Highest density region for distribution f on an interval [a,b]

    Parameters:
    ----------
    f : function
        Density function of the distribution
    beta : float
        Confidence level
    a : float
        Lower bound of the interval
    b : float

    Returns:
    --------
    tuple
        Tuple with the region, the threshold and the mass of the region
    """
    max_p = -minimize_scalar(lambda x: -f(x), bounds=(a, b), method="Bounded").fun

    def g(p):
        r = R_f(f, p, a, b)
        return PofR_f(f, r) - beta

    p_zero = brentq(g, 0, max_p)
    rp = R_f(f, p_zero, a, b)
    return rp, p_zero, PofR_f(f, rp)

def mad_c_f(f, c, *, left,right):
    """Mean absolute deviation around c for distribution f on an interval [left,right]

    Parameters
    ----------
    f : function
        Density function of the distribution
    c : float
        Central point
    left : float
        Left end of the interval
    right : float
        Right end of the interval

    Returns
    -------
    float
        Mean absolute deviation of the distribution
    """
    norm = quad(f, left, right)[0]

    def v(x):
        return abs(x - c) * f(x) / norm

    return quad(v, left, right)[0]


# Distribution defined by its histogram distinguished by suffix _d
# dist is the histogram of the distribution, but it does not need to be normalized.

def R_d(dist, p):
    """
    Find the regions where dist > p

    Parameters
    ----------
    dist: np.array
        Histogram of the distribution
    p: float
        Threshold

    Returns
    -------
    np.array
        Array of indices where dist > p
    """
    gt = dist > p
    return np.where(np.logical_xor(gt[1:], gt[:-1]))[0]


def PofR_d(dist, r):
    """
    Compute the mass of the region r

    Parameters
    ----------
    dist: np.array
        Histogram of the distribution
    r: np.array
        Array of indices where dist > p

    Returns
    -------
    float
        Mass of the region r
    """
    assert (r.size % 2) == 0
    cdf = np.cumsum(dist)
    mass = cdf[r].reshape(-1, 2)
    return (np.sum(mass[:, 1] - mass[:, 0])) / cdf[-1]


def hdr_d(xs, dist, beta):
    """
    Compute the highest density region for a distribution defined by its histogram

    Parameters
    ----------
    xs: np.array
        Array of x values
    dist: np.array
        Histogram of the distribution
    beta: float
        Confidence level

    Returns
    -------
    tuple
        Tuple with the region, the threshold and the mass of the region
    """
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



# Random variables

if arviz_imported:
    def mode_rvs(x, **kwargs):
        """

        Parameters
        ----------
        x
        kwargs

        Returns
        -------

        """
        if isinstance(x, xarray.core.dataarray.DataArray):
            x = x.stack(__z__=x.dims)
            x = x.values
        grid, pdf = az.kde(x)
        return grid[pdf.argmax()]
else:
    print("Arviz not imported, mode_rvs not available")



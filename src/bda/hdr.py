import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar, fsolve
from scipy.integrate import quad


def confidence_interval(dist, center, alpha):
    cdf = dist.cdf
    f = lambda x: cdf(center) - cdf(x) - alpha / 2
    left = fsolve(f, center)
    f = lambda x: cdf(x) - cdf(center) - alpha / 2
    right = fsolve(f, center)
    return left.item(), right.item()


def cdif_left(f, left, beta, *, a=0, b=1):
    def mass(right):
        return quad(f, left, right)[0]

    if mass(b) < beta:
        return False, a, a
    else:
        right = brentq(lambda x: mass(x) - beta, left, b)
    return True, left, right


def R(p, dist):
    gt = dist > p
    return np.where(np.logical_xor(gt[1:], gt[:-1]))[0]


def R_mass(Rp, dist):
    assert (Rp.size % 2) == 0
    cdf = np.cumsum(dist)
    mass = cdf[Rp].reshape(-1, 2)
    return np.sum(mass[:, 1] - mass[:, 0])




def hdr(xs, dist, beta, p_max=1, eps=1e-6):
    p_min = 0.0
    for i in range(24):
        p = (p_max - p_min) / 2
        Rp = R(p, dist)
        mass = R_mass(Rp, dist)
        if mass > beta:
            p_min = p;
        else:
            p_max = p
    return xs[R(p_max, dist)], R_mass(Rp, dist)


def fzeros(f, a, b, n=100):
    xs = np.linspace(a, b, n)
    delta = xs[1] - xs[0]
    fxs = f(xs)
    cross = fxs[:-1] * fxs[1:] <= 0
    fz = xs[:-1][cross]
    return [brentq(f, x, x + delta) for x in fz]


def Rf(f, p, a, b):
    fzs = fzeros(lambda x: f(x) - p, a, b)
    if f(a) > p:
        fzs = [a] + fzs
    if f(b) > p:
        fzs += [b]
    return fzs


def PofR(f, r):
    p = 0.0
    ra = np.array(r).reshape(-1, 2)
    for i in range(len(ra)):
        p += quad(f, *ra[i])[0]
    return p


def hdrf(f, beta, *, a, b):
    max_p = -minimize_scalar(lambda x: -f(x), bounds=(a, b), method="Bounded").fun

    def g(p):
        r = Rf(f, p, a, b)
        return PofR(f, r) - beta

    p_zero = brentq(g, 0, max_p)

    return p_zero, Rf(f, p_zero, a, b)

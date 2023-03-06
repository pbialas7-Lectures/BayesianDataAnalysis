import numpy as np
import scipy as sp

def _swapaxes(a,axis1, axis2, ndim=1):
    if isinstance(a,np.ndarray):
        if a.ndim>ndim:
            return np.swapaxes(a,axis1, axis2)
    return a

def rate(x,A,B,mu, sigma):
    res = np.subtract.outer(mu,x)
    res2 = -0.5*res*res
    exponent = _swapaxes(np.multiply.outer(1.0/(sigma*sigma), res2),0,1)
    signal = np.multiply.outer(A,np.exp(exponent))
    output = np.add.outer(B,signal)
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.swapaxes(output,0,1)
    else:
        return output

def log_prob(counts, rates):
    return np.sum(counts*np.log(rates) - rates-sp.special.loggamma(counts), axis = -1)

class Signalbackground():
    def __init__(self, A, B, mu, sigma):
        self.A = A
        self.B = B
        self.mu = mu
        self.sigma = sigma
        self._x = None
        self._counts = None

    def rate(self, x, n0=1):
        return n0*rate(x, self.A, self.B, self.mu, self.sigma)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self,x):
        self._x = x
    @property
    def rates(self):
        return self._rates

    @property
    def counts(self):
        return self._counts

    def gen_rates(self, n0):
        self._rates = self.rate(self.x,n0)
        self.n0 = n0
        return self._rates

    def gen_counts(self,n0=1, seed=122132143):
        self.gen_rates(n0)
        np.random.seed(seed)
        self._counts = np.random.poisson(self._rates,len(self.x))
        return self._counts

    def set_counts(self, counts, n0):
        self.n0 = n0
        self._counts = counts

    def log_prob(self, A, B ,mu, sigma):
        rates = self.n0 * rate(self.x, A, B ,mu, sigma)
        return  log_prob(self.counts, rates)

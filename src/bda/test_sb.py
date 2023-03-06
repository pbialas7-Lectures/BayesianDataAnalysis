import unittest

import sys
sys.path.append('.')

import numpy as np

import bda.signal_background as sb

class TestRates(unittest.TestCase):

    def test_x(self):
        x = np.array([-1,0,1])
        A = 1.0
        B = 2.0
        mu = 0.0
        sigma = 2.0
        rates =  sb.rate(x,A, B, mu, sigma)
        rate0 = A+B
        rate1 = A*np.exp(-0.5/4)+B
        np.testing.assert_array_almost_equal(rates,np.asarray([rate1,rate0, rate1]))

    def test_x_A_B(self):
        x = np.array([-1.0, 0.0, 0.5])
        A = np.array([0.0, 1.0])
        B = np.array([1.0, 2.0, 3.0])
        mu = 0.0
        sigma = 2.0
        e0 = 1
        e1 = np.exp(-0.5/(sigma*sigma))
        e2 = np.exp(-0.5*(0.5*0.5)/(sigma*sigma))
        rates =  sb.rate(x,A, B, mu, sigma)
        expected =  np.asarray([
            [#A=0.0
                [1.0, 1.0, 1.0], #B=1.0
                [2.0, 2.0, 2.0], #B=2.0
                [3.0, 3.0, 3.0]  #B=3.0
            ],
            [#A=1.0
                [e1+1, e0+1, e2+1], #B=1.0
                [e1+2, e0+2, e2+2], #B=2.0
                [e1+3, e0+3, e2+3]  #B=3.0
            ]
        ])
        np.testing.assert_array_almost_equal(rates, expected)


if __name__ == '__main__':
    unittest.main()


x = np.array([-1.0, 0.0, 1.0])
A = np.array([0.0, 1.0])
B = np.array([1.0, 2.0, 3.0])
mu = 0.0
sigma = 2.0

rates =  sb.rate(x,A, B, mu, sigma)
rates.shape

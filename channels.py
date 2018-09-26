import numpy as np
from scipy.stats import norm


class BSC(object):
    """ Simulates the effect of passing data through a Binary Symmetric
    Channel. """

    def __init__(self, eb, no):
        """ Initializes the channel with the provided
        signal-to-noise-ratio (SNR). """
        self.p = None
        self.set_p(eb, no)
        self.past = True

    def set_p(self, eb, no):
        self.p = norm.sf(np.sqrt(2 * eb / no))

    def send(self, data):
        """ Flips any bit with probability p. """
        return np.mod(data + (np.random.rand(20000) < self.p), 2)


class GaussianChannel(object):
    """ Simulates the effect of passing data through a Gaussian channel.

     Assumes real-valued constellation, and 20000 uses per call of `send()`. """

    def __init__(self, eb, no):
        """ Initializes the channel with the provided
        signal-to-noise-ratio (SNR). """
        self.var = None
        self.set_snr(eb, no)

    def set_snr(self, eb, no):
        self.var = no / 2

    def gaussian_noise(self, n):
        """ Generates bits of Gaussian noise on the real line, of shape `n`. """
        return np.random.normal(scale=np.sqrt(self.var),
                                size=n)

    def send(self, data):
        """ Encodes data using BPSK, adds Gaussian noise to the data, and returns the result. Simulates
        passing data through an actual Gaussian channel. """
        return (-1) ** np.array(data) + self.gaussian_noise(20000)

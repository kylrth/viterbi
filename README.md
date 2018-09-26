# Viterbi decoding

When data is sent over a channel, there will always be noise. If care is not taken to protect data, errors could be produced and data could be corrupted. One of the most effective ways of encoding data so that errors can be corrected is by using [convolutional codes](https://en.wikipedia.org/wiki/Convolutional_code).

The convolutional code used here can be represented by the following matrix:

```text
G(x)=[ 1 + x^2   1 + x + x^2 ]
```

Since the simulation time increases exponentially with SNR, a graph of the probability of decoding error for various SNRs is included in this repository. In this graph we see that with Viterbi decoding we can achieve the same probability of error with a lower SNR. This is referred to as [coding gain](https://en.wikipedia.org/wiki/Coding_gain).

## Hard- and Soft-Decision decoding

In this simulation, both hard- and soft-decision Viterbi decoding are tested. Often, noise on a channel follows a Gaussian distribution. In the hard-decision version, simulating with a [binary symmetric channel](https://en.wikipedia.org/wiki/Binary_symmetric_channel) (BSC) is equivalent to simulating with Gaussian noise because errors are counted as the number of mismatched entries in the decoded tuple. Here is the distance method used by the hard-decision class in this repository:

```python
def dist(self, one, two):
    """ Calculates the "distance" from `one` to `two` using the 1-norm.

     Assumes len(one) == len(two) == 2, and binary entries. """
    return sum((one[0] != two[0], one[1] != two[1]))
```

Essentially, the hard-decision form measures error using the L1-norm. The soft-decision form uses the L2-norm, which requires us to use the Gaussian channel model in simulation:
```python
def dist(self, one, two):
    """ Calculates the "distance" from `one` to `two` using the 2-norm.
     That's the difference between hard and soft decoding!

     Assumes len(one) == len(two) == 2. """
    return np.sqrt((one[0] - two[0]) ** 2 + (one[1] - two[1]) ** 2)
```

The included graph shows that soft-decision decoding outperforms hard-decision decoding significantly.
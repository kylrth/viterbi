import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from channels import BSC, GaussianChannel


def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i + n]
        if len(val) == n:
            yield tuple(val)


def generate_bits(n, p):
    """ Generates `n` bits of data to transmit. Zeros occur with
    probability `p`. """
    try:
        return (np.random.rand(*n) > p).astype(int)
    except TypeError:
        return (np.random.rand(n) > p).astype(int)


class ViterbiHard(object):
    """ Hard-decision Viterbi encoding and decoding. """

    def __init__(self, eb, no):
        self.message = []
        self.channel = BSC(eb, no)
        # states
        self.zero_zero_w = 0
        self.zero_one_w = 0
        self.one_zero_w = 0
        self.one_one_w = 0
        self.zero_zero_path = []
        self.zero_one_path = []
        self.one_zero_path = []
        self.one_one_path = []
        self.flop1 = 0
        self.flop2 = 0
        # channel-coded data for comparing errors
        self.zero_zero = (0, 0)
        self.zero_one = (0, 1)
        self.one_zero = (1, 0)
        self.one_one = (1, 1)

    def conv_encode(self, message=None):
        """ Returns 10000 bits of random (p=0.5) binary data (or `message` if given) using the convolutional
        encoder below:

         G(x)=[ 1 + x^2   1 + x + x^2 ].

         Stores the generated bits in `self.message` so the decoder can compare them.
         """
        output = []
        if message:
            for bit in message + [0, 0]:
                output += [np.mod(bit + self.flop2, 2), np.mod(bit + self.flop1 + self.flop2, 2)]
                self.flop2 = self.flop1
                self.flop1 = bit
        else:
            bucket = list(generate_bits(10000, 0.5))  # generate lots at once
            self.message += bucket
            # decrease the weights equally every so often
            current_min = min(self.zero_zero_w, self.zero_one_w, self.one_zero_w, self.one_one_w)
            self.zero_zero_w -= current_min
            self.zero_one_w -= current_min
            self.one_zero_w -= current_min
            self.one_one_w -= current_min
            for bit in bucket:
                output += [np.mod(bit + self.flop2, 2), np.mod(bit + self.flop1 + self.flop2, 2)]
                self.flop2 = self.flop1
                self.flop1 = bit
        return output

    def dist(self, one, two):
        """ Calculates the "distance" from `one` to `two` using the 1-norm.

         Assumes len(one) == len(two) == 2, and binary entries. """
        return sum((one[0] != two[0], one[1] != two[1]))

    def run(self):
        """ Runs the decoder on data as it's produced by `conv_encode`. Yields the original bit from `self.message`
         along with the received bit. """
        # initialize the states so beginning at 00 will be the cheapest path
        self.zero_zero_w = 0
        self.zero_one_w = 10
        self.one_zero_w = 10
        self.one_one_w = 10
        self.zero_zero_path = []
        self.zero_one_path = []
        self.one_zero_path = []
        self.one_one_path = []

        so_far = 0
        while True:
            # pass bits through BSC
            received = self.channel.send(self.conv_encode())

            for bit_pair in group(received, 2):
                # 00->00 case
                weight1 = self.zero_zero_w + self.dist(bit_pair, self.zero_zero)
                # 01->00 case
                weight2 = self.zero_one_w + self.dist(bit_pair, self.one_one)
                if weight1 < weight2:
                    zero_zero_path_new = self.zero_zero_path + [0]
                    zero_zero_w_new = weight1
                else:
                    zero_zero_path_new = self.zero_one_path + [0]
                    zero_zero_w_new = weight2

                # 10->01 case
                weight1 = self.one_zero_w + self.dist(bit_pair, self.zero_one)
                # 11->01 case
                weight2 = self.one_one_w + self.dist(bit_pair, self.one_zero)
                if weight1 < weight2:
                    zero_one_path_new = self.one_zero_path + [0]
                    zero_one_w_new = weight1
                else:
                    zero_one_path_new = self.one_one_path + [0]
                    zero_one_w_new = weight2

                # 00->10 case
                weight1 = self.zero_zero_w + self.dist(bit_pair, self.one_one)
                # 01->10 case
                weight2 = self.zero_one_w + self.dist(bit_pair, self.zero_zero)
                if weight1 < weight2:
                    one_zero_path_new = self.zero_zero_path + [1]
                    one_zero_w_new = weight1
                else:
                    one_zero_path_new = self.zero_one_path + [1]
                    one_zero_w_new = weight2

                # 10->11 case
                weight1 = self.one_zero_w + self.dist(bit_pair, self.one_zero)
                # 11->11 case
                weight2 = self.one_one_w + self.dist(bit_pair, self.zero_one)
                if weight1 < weight2:
                    one_one_path_new = self.one_zero_path + [1]
                    one_one_w_new = weight1
                else:
                    one_one_path_new = self.one_one_path + [1]
                    one_one_w_new = weight2

                # save the new best paths
                self.zero_zero_path = zero_zero_path_new
                self.zero_one_path = zero_one_path_new
                self.one_zero_path = one_zero_path_new
                self.one_one_path = one_one_path_new
                self.zero_zero_w = zero_zero_w_new
                self.zero_one_w = zero_one_w_new
                self.one_zero_w = one_zero_w_new
                self.one_one_w = one_one_w_new
                so_far += 1

                # give output from 20 received bits (10 message bits) back
                if so_far > 10:
                    best_weight = np.argmin([self.zero_zero_w, self.zero_one_w, self.one_zero_w, self.one_one_w])
                    if best_weight == 0:
                        yield (self.message.pop(0), self.zero_zero_path.pop(0))
                        del self.zero_one_path[0]
                        del self.one_zero_path[0]
                        del self.one_one_path[0]
                    elif best_weight == 1:
                        del self.zero_zero_path[0]
                        yield (self.message.pop(0), self.zero_one_path.pop(0))
                        del self.one_zero_path[0]
                        del self.one_one_path[0]
                    elif best_weight == 2:
                        del self.zero_zero_path[0]
                        del self.zero_one_path[0]
                        yield (self.message.pop(0), self.one_zero_path.pop(0))
                        del self.one_one_path[0]
                    else:
                        del self.zero_zero_path[0]
                        del self.zero_one_path[0]
                        del self.one_zero_path[0]
                        yield (self.message.pop(0), self.one_one_path.pop(0))


class ViterbiSoft(ViterbiHard):
    """ Soft-decision Viterbi encoding and decoding. """

    def __init__(self, eb, no):
        super(ViterbiSoft, self).__init__(eb, no)
        self.channel = GaussianChannel(eb, no)
        self.zero_zero = (1, 1)
        self.zero_one = (1, -1)
        self.one_zero = (-1, 1)
        self.one_one = (-1, -1)

    def dist(self, one, two):
        """ Calculates the "distance" from `one` to `two` using the 2-norm.
         That's the difference between hard and soft decoding!

         Assumes len(one) == len(two) == 2. """
        return np.sqrt((one[0] - two[0]) ** 2 + (one[1] - two[1]) ** 2)


def compare():
    """ Plot the three curves (two experimental, one theoretical) together for comparison. """
    eb = 1
    p = 0.5
    R = 0.5
    N = 150  # set the error count to reach

    domain = np.linspace(0, 10, 11)
    hard_domain = np.linspace(0, 8, 9)
    soft_domain = np.linspace(0, 6, 7)
    hard_decibels = 10 ** (hard_domain / 10)
    soft_decibels = 10 ** (soft_domain / 10)

    # get results for hard decoding
    nos = eb / hard_decibels
    hard_results = []
    for no in nos:
        print(np.where(nos == no)[0][0])
        experiment = ViterbiHard(eb, no / R)
        bit_count = 0
        error_count = 0
        for output in experiment.run():
            bit_count += 1
            if output[0] != output[1]:
                error_count += 1
                if error_count == N:
                    hard_results.append(error_count / bit_count)
                    break

    # get results for soft decoding
    nos = eb / soft_decibels
    soft_results = []
    for no in nos:
        print(np.where(nos == no)[0][0])
        experiment = ViterbiSoft(eb, no / R)
        bit_count = 0
        error_count = 0
        for output in experiment.run():
            bit_count += 1
            if output[0] != output[1]:
                error_count += 1
                if error_count == N:
                    soft_results.append(error_count / bit_count)
                    break

    decibels = 10 ** (domain / 10)
    nos = eb / decibels
    taus = nos / 4 / np.sqrt(eb) * np.log((1 - p) / p)
    plt.semilogy(hard_domain, np.array(hard_results), label='Hard-decision')
    plt.semilogy(soft_domain, np.array(soft_results), label='Soft-decision')
    plt.semilogy(domain, norm.sf((taus + np.sqrt(eb)) / np.sqrt(nos / 2)) * (1 - p) +
                 norm.sf((np.sqrt(eb) - taus) / np.sqrt(nos / 2)) * p, label='Expectation ($p=0.5$)')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Probability of bit error')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    compare()
    # the coding gain appears to be about 1.3 dB for hard-decision decoding,
    # and 3.4 dB for soft-decision decoding.

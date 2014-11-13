from random import randint
import numpy as np
import math

"""
These methods are util replacements for numpy methods so we can use pypy to JIT
optimize this section. This also includes other utils methods.
"""


def random_normalized_vector(n):
    frequency_vector = [randint(0, 100) for _ in range(0, n)]
    sum_of_frequencies = sum(frequency_vector)
    normalized_vector = [x / float(sum_of_frequencies) for x in frequency_vector]
    return normalized_vector


def nm_add_scalar(scalar, arr):
    return [scalar + arr[i] for i in xrange(len(arr))]


def nm_subtract(arr1, arr2):
    return [arr1[i] - arr2[i] for i in xrange(len(arr1))]


def nm_exp(arr):
    return [x ** 2 for x in arr]


def nm_log(matrix):
    new_matrix = []
    for row in matrix:
        new_row = [math.log(x) for x in row]
        new_matrix.append(new_row)
    return new_matrix


def nm_log_vector(vector):
    return nm_log([vector])


def nm_factorial(matrix):
    new_matrix = []
    for row in matrix:
        new_row = [math.factorial(x) for x in row]
        new_matrix.append(new_row)
    return new_matrix


def nm_factorial_vector(vector):
    return nm_factorial([vector])


def nm_sum_transpose(matrix):
    return [sum(vector) for vector in matrix]


def nm_logsumexp(lnv):
    """
    Sum exp(item) for item in lnv (log-normal vector) without overflow.
    SOURCE: https://github.com/ekg/freebayes/blob/master/python/logsumexp.py
    """
    n = lnv[0]
    maxAbs = n
    minN = n
    maxN = n
    c = n
    for item in lnv[1:]:
        n = item
        if n > maxN:
            maxN = n
        if abs(n) > maxAbs:
            maxAbs = abs(n)
        if n < minN:
            minN = n
    if maxAbs > maxN:
        c = minN
    else:
        c = maxN
    return c + math.log(sum([math.exp(i - c) for i in lnv]))


"""
The Multinomial Mixture model section implements the EM algorithm generically
"""


class MultinomialMixture:
    def __init__(self, n_clusters, count_vectors, n_iterations=None, verbose=False):
        self.count_vectors = count_vectors
        self.vocabulary_size = len(count_vectors[0])
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.verbose = verbose

        # Huh?
        self.n = sum(self.count_vectors[0])
        self.log_factorial_n = math.log(math.factorial(self.n))

        self.lambda_value = self.generate_lambda()
        self.beta_matrix = self.generate_beta_matrix()

    def generate_lambda(self):
        """
        Generate a random lambda value given the size of data
        """
        return random_normalized_vector(self.n_clusters)

    def generate_beta_matrix(self):
        """
        Generate a beta matrix given the number of clusters and data size
        """
        return [random_normalized_vector(self.vocabulary_size) for _ in range(0, self.n_clusters)]

    def log_joint_probabilities(self):
        # TODO: Remove NumPy
        a = np.log(np.array(self.lambda_value)).reshape((self.n_clusters, 1))
        b = np.log(np.array(self.beta_matrix)).dot(np.array(self.count_vectors).T)
        return a + b

    def expected_count_totals(self, probabilities):
        # TODO: Remove NumPy
        return probabilities.dot(self.count_vectors), probabilities.sum(axis=1)

    def learn_parameters(self):
        for i in xrange(self.n_iterations):
            # calculate un-normalized log posteriors, and likelihood
            log_probabilities = self.log_joint_probabilities()
            log_likelihood = self.log_likelihood(log_probabilities)

            print log_probabilities
            return

            # normalize the log posteriors TODO: Fix this
            log_probabilities -= nm_logsumexp(log_probabilities)
            probabilities = nm_exp(log_probabilities)

            # get expected count totals
            counts1, counts2 = self.expected_count_totals(probabilities)

            # estimate new parameters
            self.lambda_value = counts2 / self.a
            self.beta_matrix = counts1 / counts1.sum(axis=1).reshape((self.n_clusters, 1))

            # if self.verbose:
            # print np.array(self.intermediate_data)

    def log_likelihood(self, log_probabilities):
        log_factorial_matrix = nm_log(nm_factorial(self.count_vectors))
        a = nm_add_scalar(self.log_factorial_n, log_probabilities)
        b = nm_subtract(a, nm_sum_transpose(log_factorial_matrix))

        print b

        return sum(nm_logsumexp(b))


if __name__ == '__main__':
    MultinomialMixture()
from random import randint
import numpy as np
import math

np.set_printoptions(suppress=True, precision=4)
np.set_printoptions(threshold=np.nan, linewidth=10000)

"""
These methods are util replacements for numpy methods or numpy methods themselves (from source).
This also includes other utils methods.
"""


def np_logsumexp(a, axis=None, b=None):
    """
    Taken from the numpy source
    """
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
    else:
        a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    if b is not None:
        b = np.asarray(b)
        if axis is None:
            b = b.ravel()
        else:
            b = np.rollaxis(b, axis)
        out = np.log(np.sum(b * np.exp(a - a_max), axis=0))
    else:
        out = np.log(np.sum(np.exp(a - a_max), axis=0))
    out += a_max
    return out


def nm_gammaln(n):
    return math.log(abs(math.factorial(n - 1)))


def random_normalized_vector(n):
    frequency_vector = [randint(1, 10) for _ in range(0, n)]
    sum_of_frequencies = sum(frequency_vector)
    normalized_vector = [x / float(sum_of_frequencies) for x in frequency_vector]
    return normalized_vector


def nm_add_scalar(scalar, arr):
    return [scalar + arr[i] for i in xrange(len(arr))]


def nm_subtract(arr1, arr2):
    return [arr1[i] - arr2[i] for i in xrange(len(arr1))]


def nm_exp(arr):
    return np.array([x ** 2 for x in arr])


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


"""
The Multinomial Mixture model section implements the EM algorithm generically
"""


class MultinomialMixture:
    def __init__(self, n_clusters, count_vectors, n_iterations=None, verbose=False, lambda_values=None,
                 beta_matrix=None, smoothing=False, confusion_matrix=None, document_types=None):

        self.count_vectors = np.array(count_vectors)
        self.vocabulary_size = len(count_vectors[0])
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.smoothing = smoothing
        self.degree, self.b = self.count_vectors.shape
        self.document_types = document_types

        # Create the confusion matrix
        self.confusion_matrix = confusion_matrix

        self.n = sum(self.count_vectors[0])
        self.log_factorial_n = nm_gammaln(self.n + 1)

        if lambda_values:
            self.lambda_value = np.array(lambda_values)
        else:
            self.lambda_value = np.array(self.generate_lambda())

        if beta_matrix:
            self.beta_matrix = np.array(beta_matrix)
        else:
            self.beta_matrix = np.array(self.generate_beta_matrix())

        if self.verbose:
            self.intermediate_data = []

    def generate_lambda(self):
        """
        Generate a random lambda value given the size of data
        """
        return random_normalized_vector(self.n_clusters)

    def generate_beta_matrix(self):
        """
        Generate a beta matrix given the number of clusters and data size
        """
        return [random_normalized_vector(self.vocabulary_size) for _ in xrange(self.n_clusters)]

    def log_joint_probabilities(self):
        a = np.log(self.lambda_value).reshape((self.n_clusters, 1))
        b = np.log(self.beta_matrix).dot(self.count_vectors.T)
        return a + b

    def expected_count_totals(self, probabilities):
        return probabilities.dot(self.count_vectors), probabilities.sum(axis=1)

    def learn_parameters(self):
        for i in xrange(self.n_iterations):
            # calculate un-normalized log posteriors, and likelihood
            log_probabilities = self.log_joint_probabilities()
            log_likelihood = self.log_likelihood(log_probabilities)

            # normalize the log posteriors
            log_probabilities -= np_logsumexp(log_probabilities, axis=0)
            probabilities = np.exp(log_probabilities)

            # Verbose statements from the TAs
            if self.verbose:
                self.intermediate_data.append(
                    np.hstack((i + 1, self.lambda_value[0], self.beta_matrix[:, 0], probabilities[0], log_likelihood)))

            # get expected count totals
            beta_counts, lambda_counts = self.expected_count_totals(probabilities)

            # estimate new parameters
            self.lambda_value = self.estimate_lambda(lambda_counts)
            self.beta_matrix = self.estimate_beta_matrix(beta_counts)

        # Verbose statements from the TAs
        if self.verbose:
            for array in np.array(self.intermediate_data):
                # I think this is where we can say cm.add_observation(...) (see method for a description)
                # print: it# lambda b1 ... bn
                # print array[:2 + self.n_clusters]
                # print: it# log_likelihood
                print [array[0], array[-1]]

        # Run the Cluster Assignment
        self.assign_to_clusters()

    def estimate_lambda(self, counts):
        return counts / self.degree

    def estimate_beta_matrix(self, counts):
        denominator = counts.sum(axis=1).reshape((self.n_clusters, 1))
        if self.smoothing:
            counts += 1
            denominator += self.vocabulary_size

        return counts / denominator

    def log_likelihood(self, log_probabilities):
        a = nm_add_scalar(self.log_factorial_n, log_probabilities)
        log_factorial_vector = np.log(nm_factorial(self.count_vectors)).sum(axis=1)
        b = a - log_factorial_vector
        return np_logsumexp(b, axis=0).sum()

    def assign_to_clusters(self):
        for document_index in xrange(len(self.count_vectors)):
            document = self.count_vectors[document_index]
            cluster_probabilities = []
            for cluster_index in xrange(self.n_clusters):
                # Find the probability of each cluster
                probability = 1.
                for index in xrange(len(document)):
                    probability *= self.beta_matrix[cluster_index][index] or 1
                cluster_probabilities.append(probability)

            # Get the max cluster is its 'index + 1'
            max_cluster = cluster_probabilities.index(max(cluster_probabilities)) + 1
            # print max_cluster
            self.confusion_matrix.add_observation(self.document_types[document_index], max_cluster)

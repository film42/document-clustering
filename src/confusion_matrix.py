import sys


class ConfusionMatrix:
    def __init__(self, labels):

        self.doc_labels = {}
        self.labels = labels
        self.cluster_labels = range(1, len(labels) + 1)
        self.n_clusters = len(labels)

        # Initializing the matrix with 0's
        self.matrix = []
        for i in xrange(self.n_clusters):
            self.matrix.append([0 for _ in xrange(self.n_clusters)])

    def add_observation(self, label, cluster_label):
        """
        Increments the count of an observation at [label][cluster]
        """
        label_index = self.labels.index(label)
        cluster_index = cluster_label - 1
        self.matrix[label_index][cluster_index] += 1

    def accuracy(self):
        numerator = sum(self.matrix[i][i] for i in range(len(self.matrix)))
        denominator = sum(sum(row) for row in self.matrix)
        return numerator / float(denominator)

    def print_matrix(self):
        for label_index in xrange(len(self.matrix)):
            # Print the label
            label = self.labels[label_index]
            sys.stdout.write("%s%s" % (label, " " * (28 - len(label))))
            # Print the row
            for count_index in xrange(len(self.matrix[label_index])):
                sys.stdout.write("%d  " % self.matrix[label_index][count_index])
            sys.stdout.write("\n")

        print "Accuracy %f" % self.accuracy()



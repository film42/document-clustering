
class ConfusionMatrix:


    def __init__(self, clusters):

        self.clusters = clusters
        self.matrix = [[0 for x in xrange(len(clusters))] for x in xrange(len(clusters))]
        pass


    def add_observations(self, document, cluster):
        self.matrix
        pass


    def calc_accuracy(self):
        t = sum(sum(l) for l in self.matrix)
        return sum(self.matrix[i][i] for i in range(len(self.matrix))) / t

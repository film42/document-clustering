
class ConfusionMatrix:


    def __init__(self, n_clusters):

        self.doc_labels = {}
        self.n_clusters = n_clusters

        #initializing the matrix with 0's
        self.matrix = [[0 for x in xrange(len(self.n_clusters))] for x in xrange(len(self.n_clusters))]

    def add_data(self, count_vector, label):
        """
        stores the label and the count vector which gives us the correct labeling
        """
        self.doc_labels[count_vector] = label


    def add_observation(self, count_vector, cluster):
        """
        increments the cell in the matrix of the corresponding [count_vector, cluster] where cluster is the index of the
        mode of the betas for that row
        """
        


    def calc_accuracy(self):
        t = sum(sum(l) for l in self.matrix)
        return sum(self.matrix[i][i] for i in range(len(self.matrix))) / t

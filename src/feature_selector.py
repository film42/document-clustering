import os
from tf_idf import TFIDF


class BuildModel:
    def __init__(self, path):
        self.path = path
        self.corpus = {}
        self.documents = {}

        self.import_features()

    def load_feature(self, name, words):
        # Save document
        self.documents[name] = words
        # Save the words for counting later
        for word in words:
            self.corpus[word] = self.corpus.get(word, 0) + 1

    def import_features(self):
        # First import all features
        for dir_name in os.listdir(self.path):
            if dir_name == '.DS_Store':
                continue

            for file_name in os.listdir("%s/%s" % (self.path, dir_name)):
                if file_name == '.DS_Store':
                    continue

                document_name = "%s/%s" % (dir_name, file_name)
                with open("%s/%s" % (self.path, document_name), 'r') as f:
                    words = f.read().split("\n")[:-1]
                    self.load_feature(document_name, words)

    def count_vectors(self):
        """
        The point of build is to load all documents into memory and generate a count vector for each document
        """
        count_vectors = []
        for document, words in self.documents.iteritems():
            count_vector = {}
            print "Doc %s" % document
            for corpus_word in self.corpus.keys():
                # Count the number of times the word in the corpus occurs in the document
                count_vector[corpus_word] = words.count(corpus_word)
            # Save the corpus vector
            count_vectors.append(count_vector.values())

        return count_vectors


class FeatureSelector:
    def __init__(self):
        self.table = TFIDF()

    def run(self):
        """
        Generate the features using Top N algorithm
        """
        for dir_name in os.listdir("../data/groups/"):
            if dir_name == '.DS_Store':
                continue

            for file_name in os.listdir("../data/groups/%s" % dir_name):
                if file_name == '.DS_Store':
                    continue

                document_name = "%s/%s" % (dir_name, file_name)
                with open("../data/groups/%s" % document_name, 'r') as f:
                    self.table.add_document(document_name, f.read().lower())

        new_data_set = self.table.top_n_words(3)
        for document_name, words in new_data_set.iteritems():

            directory_name, file_name = document_name.split('/')
            path_name = "../data/features/%s" % directory_name

            if not os.path.exists(path_name):
                os.makedirs(path_name)

            with open("%s/%s" % (path_name, file_name), 'w') as f:
                for word in words:
                    f.write(word)
                    f.write("\n")


if __name__ == '__main__':
    FeatureSelector().run()
from feature_selector import BuildModel
from multinomial_mixture import MultinomialMixture


class Main:
    def __init__(self):
        pass

    def run(self):
        model = BuildModel("../sample/features")
        count_vectors = model.count_vectors()
        mm = MultinomialMixture(20, count_vectors, n_iterations=5, verbose=True)
        mm.learn_parameters()
        pass


if __name__ == "__main__":
    app = Main()
    app.run()
from feature_selector import BuildModel
from multinomial_mixture import MultinomialMixture


class Main:
    def __init__(self):
        pass

    def assignment_5_875_test(self):
        l_value = 0.3
        b1_value = 0.3
        b2_value = 0.6
        lambda_values = [l_value, 1 - l_value]
        beta_matrix = [[b1_value, 1 - b1_value], [b2_value, 1 - b2_value]]
        count_vectors = [[3., 0.],
                         [0., 3.],
                         [3., 0.],
                         [0., 3.],
                         [3., 0.]]
        mm = MultinomialMixture(2, count_vectors, n_iterations=4, verbose=True, beta_matrix=beta_matrix,
                                lambda_values=lambda_values)
        mm.learn_parameters()

    def run(self):
        model = BuildModel("../sample/features")
        count_vectors = model.count_vectors()
        mm = MultinomialMixture(20, count_vectors, n_iterations=100, verbose=True)
        mm.learn_parameters()


if __name__ == "__main__":
    app = Main()
    app.assignment_5_875_test()
    # app.run()
from feature_selector import BuildModel
from multinomial_mixture import MultinomialMixture


class Main:
    def __init__(self):
        pass

    def coin_and_spinner(self):
        """
        This is the last example using a coin and then spinner of 4 words from 5.875
        """
        l_value = 0.3
        lambda_values = [l_value, 1 - l_value]
        count_vectors = [[1, 0, 1, 1],
                         [0, 1, 1, 1],
                         [2, 1, 0, 0],
                         [0, 0, 2, 1],
                         [0, 0, 1, 2]]
        beta_matrix = [[.41, .39, .11, .09],
                       [.11, .09, .35, .45]]

        mm = MultinomialMixture(2, count_vectors, n_iterations=7, verbose=True, beta_matrix=beta_matrix,
                                lambda_values=lambda_values)
        mm.learn_parameters()

    def assignment_5_875_test(self):
        """
        This is the first example given in 5.875
        """
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
        mm = MultinomialMixture(2, count_vectors, n_iterations=10, verbose=True, beta_matrix=beta_matrix,
                                lambda_values=lambda_values)
        mm.learn_parameters()

    def run(self):
        """
        This is for assignment 6, the data is from 20 news groups
        """
        model = BuildModel("../data/features")
        count_vectors = model.count_vectors()
        mm = MultinomialMixture(20, count_vectors, n_iterations=25, verbose=True, smoothing=True)
        mm.learn_parameters()


if __name__ == "__main__":
    app = Main()
    # app.assignment_5_875_test()
    # app.coin_and_spinner()
    app.run()
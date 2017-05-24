import numpy as np


class Regression:
    def __init__(self):
        self.coef_arr = None
        self.degree = 0

    def train(self, X, Y, degree=2):
        print("======")
        # print(Y)

        X = np.squeeze(np.asarray(X))
        Y = np.squeeze(np.asarray(Y))

        print(X)
        print(Y)



        self.coef_arr = np.polyfit(X, Y, degree)
        print(self.coef_arr)

        exit()

    def predict(self, X):
        print("-------")
        result_arr = []

        for x in np.squeeze(np.asarray(X)):
            result = 0

            for coef in reversed(self.coef_arr):
                result = coef + x * result

            result_arr.append(result)

        return result_arr


def learn_method(X, Y):
    pass


def find_index_list(in_list, func):
    rtn_index_list = []
    for i in range(len(in_list)):
        if func(in_list[i]):
            rtn_index_list.append(i)

    return rtn_index_list


def hw3(S, X, T, v, r, n, k):
    """ Monte Carlo program to price American puts
    
    :param S: Stock price at time 0
    :param X: Strike price
    :param T: Maturity in years
    :param v: Annual volatility
    :param r: Continuously compounded annual interest rate
    :param n: Number of periods
    :param k: Number of simulation paths
    :return: 
    """
    delta_t = T / n
    path_arr = np.random.randn(k, n)

    sample_prices_matrix = []

    for i in range(len(path_arr)):
        path = path_arr[i]
        stock_price = S
        stock_price_arr = [stock_price]

        for rv in path:
            stock_price = stock_price * np.exp((r - v ** 2 / 2) * delta_t + v * np.sqrt(delta_t) * rv)
            stock_price_arr.append(stock_price)
        sample_prices_matrix.append(stock_price_arr)

    sample_prices_matrix = np.matrix(sample_prices_matrix)

    for i in range(len(sample_prices_matrix[:, -1])):
        last_price = sample_prices_matrix[:, -1][i]
        sample_prices_matrix[:, -1][i] = max([0, X - last_price])

    print(sample_prices_matrix)

    for i in reversed(range(n)):
        legal_index_list = find_index_list(sample_prices_matrix[:, i], lambda _: _ > X)
        print(legal_index_list)

        train_X = sample_prices_matrix[:, i][legal_index_list]
        train_Y = sample_prices_matrix[:, i + 1][legal_index_list]

        model = Regression()
        model.train(train_X, train_Y)

        print(train_X)
        print(train_Y)
        print(model.predict(train_X))
        exit()
        pass


hw3(101, 105, 1, 0.15, 0.02, 4, 10)

# hw3(101, 105, 1, 0.15, 0.02, 50, 100000)
# desired result: 7.2977
# model = Regression()
# model.train(np.array([0, 1]), np.array([1, 3]))
# print(model.predict(np.array([2, 3])))

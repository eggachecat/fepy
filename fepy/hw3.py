import numpy as np


class Regression:
    def __init__(self):
        self.coef_arr = None
        self.degree = 0

    def train(self, X, Y, degree=2):
        X = np.squeeze(np.asarray(X))
        Y = np.squeeze(np.asarray(Y))

        self.coef_arr = np.polyfit(X, Y, degree)

    def predict(self, X):
        result_arr = []

        predicter = np.poly1d(self.coef_arr)

        for x in np.squeeze(np.asarray(X)):
            # result = 0
            #
            # for coef in self.coef_arr:
            #     result = coef + x * result

            result_arr.append([predicter(x)])

        return np.asmatrix(result_arr)


def learn_method(X, Y):
    pass


def find_index_list(in_list, func):
    rtn_index_list = []
    for i in range(len(in_list)):
        if func(in_list[i]):
            rtn_index_list.append(i)

    return rtn_index_list


def hw3(initial_stock_price, strike_price, T, v, r, n, k):
    """ Monte Carlo program to price American puts
    
    :param initial_stock_price: Stock price at time 0
    :param strike_price: Strike price
    :param T: Maturity in years
    :param v: Annual volatility
    :param r: Continuously compounded annual interest rate
    :param n: Number of periods
    :param k: Number of simulation paths
    :return: 
    """
    delta_t = T / n
    path_arr = np.random.randn(k, n)
    discount_rate = 1 / np.exp(r * delta_t)

    sample_prices_matrix = []

    print("Start constructing sample price matrix...")
    for i in range(len(path_arr)):
        if i % int((k / 10)) == 0 and i > 0:
            print("\t{i} paths ({p}%)  Done".format(i=i, p=100*i/k))
        path = path_arr[i]
        stock_price = initial_stock_price
        stock_price_arr = [stock_price]

        for rv in path:
            stock_price = stock_price * np.exp((r - v ** 2 / 2) * delta_t + v * np.sqrt(delta_t) * rv)
            stock_price_arr.append(stock_price)
        sample_prices_matrix.append(stock_price_arr)
    print("All Done!")

    sample_prices_matrix = np.matrix(sample_prices_matrix)
    print("Start backreducing...")
    for i in range(len(sample_prices_matrix[:, -1])):
        last_price = sample_prices_matrix[:, -1][i]
        sample_prices_matrix[:, -1][i] = max([0, strike_price - last_price])

    for i in reversed(range(1, n)):
        if i % int((n / 10)) == 0 and i > 0:
            print("\tNow in period-{i} ({p}%)".format(i=i, p=100-100*i/n))
        legal_index_list = find_index_list(sample_prices_matrix[:, i], lambda _: _ < strike_price)

        train_X = sample_prices_matrix[:, i][legal_index_list]
        train_Y = discount_rate * sample_prices_matrix[:, i + 1][legal_index_list]

        if len(train_X) > 1:
            model = Regression()
            model.train(train_X, train_Y)

            continuation_value_arr = model.predict(train_X)
            exercise_value_all = strike_price - train_X

            exercise_indices = exercise_value_all > continuation_value_arr

            sample_prices_matrix[:, i] = discount_rate * sample_prices_matrix[:, i + 1]
            for idx in range(len(legal_index_list)):
                if exercise_indices[idx]:
                    sample_prices_matrix[:, i][legal_index_list[idx]] = exercise_value_all[idx]
        else:
            sample_prices_matrix[:, i] = discount_rate * sample_prices_matrix[:, i + 1]
    print("Done!")
    print("The price is about {p} and the standard error is about {se}.".format(p=np.mean(sample_prices_matrix[:, 1]),
                                                                                se=np.std(sample_prices_matrix[:,
                                                                                          1]) / np.sqrt(k)))


# hw3(101, 105, 3, 0.05, 0.05, 3, 1000)

hw3(101, 105, 1, 0.15, 0.02, 50, 100000)
# desired result: 7.2977

#
#
# model = Regression()
# model.train(np.array([1,2]),np.array([2,3]))
# model.train(np.array([92.5815, 103.6010, 98.7120, 101.0564, 93.727, 102.4177]),
#             np.array([0, 0, 0, 0.44565, 5.347, 3.8786]))
# result = model.predict(np.array([92.5815, 103.6010, 98.7120, 101.0564, 93.727, 102.4177]))
# print("res",result)

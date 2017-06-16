import numpy as np


class Entry:
    def __init__(self, h, v):
        self.h = h
        self.v = v

    def __str__(self):
        return "h={h} C={C}".format(h=self.h, C=self.v)


class Table:
    def __init__(self, K):
        # h in ascending
        self.K = K
        self.entry_arr = []

    def add_entry(self, en):
        self.entry_arr.append(en)
        self.entry_arr.sort(key=lambda x: x.h)

    def get_interpolation(self, h):
        _i = self.find_interval(h)

        if _i == 0 or _i == self.K - 1:
            return self.entry_arr[_i].v

        if self.entry_arr[_i - 1].h == h:
            return self.entry_arr[_i].v

        ratio = (self.entry_arr[_i].h - h) / (self.entry_arr[_i].h - self.entry_arr[_i - 1].h)
        return ratio * self.entry_arr[_i - 1].v + (1 - ratio) * self.entry_arr[_i].v

    def find_interval(self, h):
        _i = 0
        while h < self.entry_arr[_i].h and h < self.K:
            _i += 1

        return max[_i, self.K - 1]


class Node:
    def __init__(self, log_price, conditional_var, risk_free_rate, gamma, n_split, d=0, loc=0, prev_loc=0, h=None):

        self.loc = loc
        self.prev_loc = prev_loc

        self.value = 0
        self.d = d
        self.y = log_price
        self.S = np.exp(self.y)
        self.r = risk_free_rate
        self.v = conditional_var
        if h is None:
            h = np.sqrt(self.v)
        self.h = h
        self.gamma = gamma
        self.n = n_split

        self.in_mean = self.y + self.r - 0.5 * self.v
        self.in_var = self.v
        self.grid = float(gamma / n_split)
        self.eta, self.prob_list = self.find_eta()

        self.in_node = []
        self.out_node = [self.loc + d * self.eta for d in [-1, 0, 1]]

    def get_principle_parameter(self):
        return self.y, self.v, self.r, self.gamma, self.n

    def find_eta(self, eta=None):

        eta = int(np.ceil(self.h / self.gamma))

        prob_list = self.compute_associated_probability(eta)

        while np.any(prob_list < 0):
            eta += 1
            prob_list = self.compute_associated_probability(eta)

        # self.eta = eta
        return eta, prob_list

    def evaluate(self, value_func, *args):
        self.value = value_func(self, *args)

    def get_log_price(self, j):
        return self.y + j * self.eta * self.grid

    def get_conditional_var(self, j, b_0, b_1, b_2, c):
        e = (j * self.eta * self.grid - (self.r - 0.5 * self.v)) / np.sqrt(self.v)
        return b_0 + b_1 * self.v + b_2 * self.v * np.square(e - c)

    def compute_associated_probability(self, eta):
        """  
        grid = gamma / n

        :param eta: 
        :return: 
            [p_u, p_m, p_d]
        """

        if eta == 1 and self.h == self.grid:
            factor_1 = 0.5
        else:
            factor_1 = self.v / (2 * eta ** 2 * self.grid ** 2)
        factor_2 = (self.r - self.v / 2) * np.sqrt(self.n) / (2 * eta * self.grid)

        return np.array([factor_1 + factor_2, 1 - 2 * factor_1, factor_1 - factor_2])

    def __str__(self):
        return "loc={loc}, prev={_in}, direction={d}, price = {p}, value={v}, h = {h}, grid={g}, eta={e}, o={o}, prob_list={pl} ".format(
            loc=self.loc,
            p=self.S,
            h=round(self.v * 100000, 4),
            d=self.d,
            v=self.value,
            o=self.out_node,
            g=self.grid,
            e=self.eta,
            pl=self.prob_list, _in=self.prev_loc)


def a_p_terminal_value_func(node, X):
    return max([0, node.S - X])


def a_p_value_func(node, X):
    return max([0, node.S - X])


class Tree:
    def __init__(self, E, r, gamma, n, K, b_0, b_1, b_2, c, root):

        self.E = E
        self.r = r
        self.gamma = gamma
        self.n = n
        self.b_0 = b_0
        self.b_1 = b_1
        self.b_2 = b_2
        self.c = c
        self.K = K

        self.min_idx = 0
        self.max_idx = self.K - 1

        self.layers = []
        for i in range(E + 1):
            self.layers.append(dict())

        self.layers[0][0] = dict()
        self.layers[0][0][self.min_idx] = root
        self.layers[0][0][self.max_idx] = root

    def make_a_node(self, src, d):
        """
        
        :param src: 
        :param d: in [-1, 0, 1] from trinomial
        :return: 
        """
        j = src.loc + d * src.eta
        return Node(src.get_log_price(d), src.get_conditional_var(d, self.b_0, self.b_1, self.b_2, self.c), self.r,
                    self.gamma, self.n, d=d, loc=j, prev_loc=src.loc)

    def add_to_layer(self, layer_index, src, d):

        j = src.loc + d * src.eta
        _node = self.make_a_node(src, d)

        # src.add_out(j)

        if j not in self.layers[layer_index]:
            self.layers[layer_index][j] = dict()

            self.layers[layer_index][j][self.min_idx] = _node
            self.layers[layer_index][j][self.max_idx] = _node
        else:
            if self.layers[layer_index][j][self.min_idx].h > _node.h:
                self.layers[layer_index][j][self.min_idx] = _node

            if self.layers[layer_index][j][self.max_idx].h < _node.h:
                self.layers[layer_index][j][self.max_idx] = _node

                # self.layers[layer_index][j].append(_node)

    def build_tree(self):
        for i in range(self.E):
            self.fulfill_layer(i)
            for j in self.layers[i]:
                for node_idx in self.layers[i][j]:
                    node = self.layers[i][j][node_idx]
                    for d in [-1, 0, 1]:
                        tree.add_to_layer(i + 1, node, d)

    def fulfill_layer(self, idx):

        layer = self.layers[idx]
        for j in sorted(layer):
            # self.y, self.h, self.r, self.gamma, self.n

            y, min_v, r, gamma, n = layer[j][self.min_idx].get_principle_parameter()
            y, max_v, r, gamma, n = layer[j][self.max_idx].get_principle_parameter()
            for i in range(1, self.K - 1):
                h = min_v + i * (max_v - min_v) / (self.K - 1)
                layer[j][i] = Node(y, h, r, gamma, n)


    def fulfill_tree(self):
        for layer in self.layers:
            for j in sorted(layer):
                # self.y, self.h, self.r, self.gamma, self.n

                y, min_v, r, gamma, n = layer[j][self.min_idx].get_principle_parameter()
                y, max_v, r, gamma, n = layer[j][self.max_idx].get_principle_parameter()
                for i in range(1, self.K - 1):
                    h = min_v + i * (max_v - min_v) / (self.K - 1)
                    layer[j][i] = Node(y, h, r, gamma, n)


    def back_induction(self):
        for j in self.layers[self.E]:
            group = self.layers[self.E][j]
            for _t in group:
                terminal_node = group[_t]
                terminal_node.evaluate(a_p_terminal_value_func, 100)

        for i in reversed(range(self.E)):
            print("i", i)
            for j in self.layers[i]:
                print(j)
                group = self.layers[i][j]
                for _t in group:
                    node = group[_t]

                    value = 0
                    v = node.v
                    o_node = node.out_node
                    prob_list = node.prob_list

                    for k in range(3):
                        print(i + 1, o_node[k])
                        value += prob_list[k] * self.get_interpolation(self.layers[i + 1][o_node[k]], v)

                    node.value = value
            print("---")


    def get_interpolation(self, group, v):
        _i = self.find_interval(group, v)

        if _i == 0 or _i == self.K - 1:
            return group[_i].value

        if group[_i - 1].v == v:
            return group[_i].value

        ratio = (group[_i].v - v) / (group[_i].v - group[_i - 1].v)
        return ratio * group[_i - 1].value + (1 - ratio) * group[_i].value


    def find_interval(self, group, v):
        _i = 0
        while _i < self.K:
            if v < group[_i].v:
                break
            _i += 1

        return min([_i, self.K - 1])


    def __str__(self):
        c_str = ""
        i = 0
        for layer in self.layers:
            for j in sorted(layer):
                c_str += "=====({i},{j})=====\n".format(i=i, j=j)
                for _t in layer[j]:
                    c_str += "[{_t}] {c} \n".format(_t=_t, c=layer[j][_t].__str__())
                c_str += "==================\n"
            i += 1

        return c_str


E = 3
K = 3
n = 1
gamma, r = 0.010469, 0
b_0, b_1, b_2, c = 0.000006575, 0.9, 0.04, 0

root = Node(4.60517, 0.0001096, r, gamma, n, h=0.010469)

tree = Tree(E, r, gamma, n, K, b_0, b_1, b_2, c, root)

tree.build_tree()
# tree.back_induction()
print(tree)


# gamma = 0.010469
# n = 1
# r = 0
# node_0_0 = Node(4.60517, 0.0001096, r, gamma, n, h=0.010469)
# # node_0_0.find_eta()
# print(node_0_0)
# print("-----------")
#
# b_0, b_1, b_2, c = 0.000006575, 0.9, 0.04, 0
# node_1_0 = Node(node_0_0.get_log_price(1), node_0_0.get_conditional_var(1, b_0, b_1, b_2, c), r, gamma, n)
# print(node_1_0)
# print("-----------")
#
# node_2_0 = Node(node_1_0.get_log_price(1), node_1_0.get_conditional_var(1, b_0, b_1, b_2, c), r, gamma, n)
# print(node_2_0)
# print("-----------")
#
# node_3_0 = Node(node_2_0.get_log_price(1), node_2_0.get_conditional_var(1, b_0, b_1, b_2, c), r, gamma, n)
# print(node_3_0)
# print("-----------")


# print(root.get_log_price(0))
# print(root.get_log_price(-1))

#
# def main(E, r, S, h0, b0, b1, b2, c, X, n1, n2):
#     """
#
#     :param E: (days before expiration)
#     :param r: (annual interest rate)
#     :param S: (stock price at time 0)
#     :param h0:
#     :param b0:
#     :param b1:
#     :param b2:
#     :param c:
#     :param X: (strike price)
#     :param n1: (number of partitions per day)
#     :param n2: (number of variances per node)
#     :return:
#     """
#     y = np.log(S)
#
#
# if __name__ == "__main__":
#     main(100, 0, 1, 0.0001096, 0.0049, 0.000006575, 0.9, 0.04, 0)
def get_min_eta(h, c_gamma):
    return int(np.sqrt(h) / c_gamma)


def find_successor_stock_prices(y, h, gamma, n_values=1):
    """

    :param y: 
    :param h: 
    :param eta: 
    :param n_values: [1;in this case, the tri-nomial tree]
        total 2n+1 points
    :return: 

    # price_list = get_derived_prices(4.60517, 0.000196, 0.010469, n_values=1)
    # print(price_list)
    # [ 4.594701  4.60517   4.615639]
    """
    _list = np.array([j for j in range(-n_values, n_values + 1)])
    return y + gamma * _list


def establish_the_normalized_innovation(eta, gamma, r_f, h, n_values=1):
    _list = np.array([j for j in range(-n_values, n_values + 1)])
    return (eta * gamma * _list - (r_f - h / 2)) / np.sqrt(h)


def calculate_variance_for_the_next_period(h, e, b0, b1, b2, c):
    return b0 + b1 * h + b2 * h * (e - c) ** 2


def compute_associated_probability(h, gamma, eta, r_f, n):
    """
    analytics solution for n_values=1 
    a.k.a 
    using 3 r.v. to approximate normal distribution
    :param n_values: 
    :return: 
    """
    factor_1 = h / (2 * eta ** 2 * gamma ** 2)
    factor_2 = (r_f - h / 2) * np.sqrt(n) / (2 * eta * gamma)

    return np.array([factor_1 + factor_2, 1 - factor_1, factor_1 - factor_2])

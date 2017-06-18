import numpy as np


class Node:
    __id = 0

    def __init__(self, log_price, conditional_var, risk_free_rate, gamma, n_split, d=0, loc=0, prev_loc=0,
                 h=None):

        self.loc = loc
        self.prev_loc = prev_loc

        self.id = Node.__id
        Node.__id += 1

        self.value = 0.0
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

        self.grid = float(gamma / np.sqrt(self.n))
        self.eta, self.prob_list = self.find_eta()

        self.in_node = []
        self.out_node = [self.loc + d * self.eta for d in range(-self.n, self.n + 1)]

    def get_principle_parameter(self):
        return self.y, self.loc, self.v, self.r, self.gamma, self.n

    def find_eta(self, eta=None):
        eta = int(np.ceil(self.h / self.gamma))

        prob_list = self.compute_associated_probability(eta)
        # print(prob_list)
        # print(eta)
        # exit()

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
        e = (j * self.eta * self.grid - (self.r - 0.5 * self.v)) / self.h
        return b_0 + b_1 * self.v + b_2 * self.v * (e - c) ** 2

    def compute_associated_probability(self, eta):
        """  
        grid = gamma / n

        :param eta: 
        :return: 
            [p_d, p_m, p_u]
        """

        if eta == 1 and self.h == self.grid:
            factor_1 = 0.5
        else:
            factor_1 = self.v / (2 * eta ** 2 * self.gamma ** 2)
        factor_2 = (self.r - self.v / 2) / (2 * eta * self.gamma * np.sqrt(self.n))

        return np.array([factor_1 - factor_2, 1 - 2 * factor_1, factor_1 + factor_2])

    def __str__(self):
        return "loc={loc}, prev={_in}, direction={d}, price = {p}, value={v}, h = {h}, grid={g}, eta={e}, o={o}, prob_list={pl}, pre_node={pre} ".format(
            loc=self.loc,
            p=self.S,
            h=self.v,
            d=self.d,
            v=self.value,
            o=self.out_node,
            g=self.grid,
            e=self.eta,
            pre=self.prev_loc,
            pl=self.prob_list, _in=self.prev_loc)


def a_p_terminal_value_func(node, X):
    return max([0, X - node.S])


def e_c_terminal_value_func(node, X):
    return max([0, node.S - X])


class Tree:
    def __init__(self, E, r, gamma, n, K, b_0, b_1, b_2, c, root, X):

        self.E = E
        self.r = r
        self.gamma = gamma
        self.n = n
        self.b_0 = b_0
        self.b_1 = b_1
        self.b_2 = b_2
        self.c = c
        self.K = K

        self.X = X

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

        # if np.abs(d) > 1:
        #     return

        j = src.loc + d * src.eta
        _node = self.make_a_node(src, d)

        # src.add_out(j)

        if j not in self.layers[layer_index]:
            self.layers[layer_index][j] = dict()

            self.layers[layer_index][j][self.min_idx] = _node
            self.layers[layer_index][j][self.max_idx] = self.make_a_node(src, d)
        else:
            if self.layers[layer_index][j][self.min_idx].v > _node.v:
                self.layers[layer_index][j][self.min_idx] = _node

            if self.layers[layer_index][j][self.max_idx].v < _node.v:
                self.layers[layer_index][j][self.max_idx] = _node

                # self.layers[layer_index][j].append(_node)

    def build_tree(self):
        for i in range(self.E):
            # print("build ", i)
            self.fulfill_layer(i)
            for j in self.layers[i]:
                for node_idx in self.layers[i][j]:
                    node = self.layers[i][j][node_idx]
                    for d in range(-self.n, self.n + 1):
                        self.add_to_layer(i + 1, node, d)
        self.fulfill_layer(self.E)

    def fulfill_layer(self, idx):

        layer = self.layers[idx]
        for j in sorted(layer):
            y, loc, min_v, r, gamma, n = layer[j][self.min_idx].get_principle_parameter()
            y, loc, max_v, r, gamma, n = layer[j][self.max_idx].get_principle_parameter()
            for i in range(1, self.K - 1):
                h = min_v + i * (max_v - min_v) / (self.K - 1)
                layer[j][i] = Node(y, h, r, gamma, n, loc=loc)

    def back_induction(self):
        for j in self.layers[self.E]:
            group = self.layers[self.E][j]
            for _t in group:
                terminal_node = group[_t]
                # terminal_node.evaluate(a_p_terminal_value_func, self.X)
                terminal_node.evaluate(e_c_terminal_value_func, self.X)

        for i in reversed(range(self.E)):
            # print("back induction", i)
            for j in self.layers[i]:
                group = self.layers[i][j]
                for _t in group:
                    node = group[_t]

                    value = 0
                    o_node = node.out_node
                    prob_list = node.prob_list

                    # [p_d, p_m, p_u]
                    # p_d = prob_list[0]
                    # p_m = prob_list[1]
                    # p_u = prob_list[2]

                    # prob_dict = dict()
                    #
                    # for k in range(-self.n, 1 + self.n):
                    #     prob_dict[k] = 0
                    #
                    # prob_dict[-1] = p_d = prob_list[0]
                    # prob_dict[0] = p_m = prob_list[1]
                    # prob_dict[1] = p_u = prob_list[2]
                    #
                    # for _ in range(1, self.n):
                    #
                    #     tmp = prob_dict.copy()
                    #     for k in range(-self.n, 1 + self.n):
                    #         prob_dict[k] = 0
                    #
                    #     for k in range(-1 * _, 1 + _):
                    #         prob_dict[k + 1] += tmp[k] * p_u
                    #         prob_dict[k - 1] += tmp[k] * p_d
                    #         prob_dict[k] += tmp[k] * p_m

                    print(i, j)
                    p_d = prob_list[0]
                    p_m = prob_list[1]
                    p_u = prob_list[2]
                    prob_arr = []
                    for m in range(2 * self.n + 1):
                        prob_arr.append(0)

                    prob_arr[2] = p_u
                    prob_arr[1] = p_m
                    prob_arr[0] = p_d

                    print("p_u", p_u)
                    print("p_m", p_m)
                    print("p_d", p_d)

                    l = self.n - 1
                    while l > 0:
                        for m in reversed(range(2 * (self.n - l) + 1)):
                            prob_arr[m + 2] = prob_arr[m + 2] + prob_arr[m] * p_u
                            prob_arr[m + 1] = prob_arr[m + 1] + prob_arr[m] * p_m
                            prob_arr[m] = prob_arr[m] * p_d

                        l -= 1
                    print("----p_all----")

                    print(prob_arr)
                    print("=============")
                    prob_dict = dict()
                    for _ in range(2 * self.n + 1):
                        prob_dict[_ - self.n] = prob_arr[_]

                    for k in range(-self.n, 1 + self.n):
                        if i == 1:
                            print("*********", i, j, "**********")

                        v = node.get_conditional_var(k, self.b_0, self.b_1, self.b_2, self.c)
                        # v = node.v
                        # print(i, j, "get_interpolation", i + 1, o_node[k])
                        value += prob_dict[k] * self.get_interpolation(self.layers[i + 1][o_node[k + self.n]], v,
                                                                       log=i == 1)

                    # discount
                    value = value / np.exp(self.r)

                    # value = max([value, self.X - node.S])

                    self.layers[i][j][_t].value = value
                    # print("---")

    def get_interpolation(self, group, v, log=False):

        _i = self.find_interval(group, v)

        if _i > self.K - 1:
            return group[self.K - 1].value

        if _i == 0:
            return group[0].value

        if group[_i - 1].v == v:
            return group[_i - 1].value

        ratio = (group[_i].v - v) / (group[_i].v - group[_i - 1].v)

        # print(ratio)
        return ratio * group[_i - 1].value + (1 - ratio) * group[_i].value

    def find_interval(self, group, v):
        _i = 0
        while _i < self.K:
            if v < group[_i].v:
                break
            _i += 1

        return _i

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


DEBUG = True


def Ritchken_Trevor_Algorithm(E, r, S, h, b_0, b_1, b_2, c, X, n, K):
    gamma = 0.01046900186264192
    y = np.log(S)
    r = r / 365

    root = Node(y, 0.0001096, r, gamma, n, h)
    print(root.grid)
    tree = Tree(E, r, gamma, n, K, b_0, b_1, b_2, c, root, X)
    tree.build_tree()
    tree.back_induction()
    if DEBUG:
        print(tree)
    print("The price is ", tree.layers[0][0][0].value)


Ritchken_Trevor_Algorithm(30, 0.01, 100, 0.01046900186264192, 0.000006575, 0.9, 0.04, 0, 100, 3, 3)

import numpy as np


def call_payoff(stock_price, strike_price):
    return max([stock_price - strike_price, 0])


def barrier_determine(stock_price, barrier):
    return 0 if stock_price > barrier else stock_price


def avg_split(_min, _max, k):
    step = (_max - _min) / k
    return np.arange(_min, _max + step, step)


def max_rav(j, i, u, d):
    return ((1 - np.power(u, j - i + 1)) / (1 - u) + np.power(u, j - i) * d * (
        1 - np.power(d, i)) / (1 - d)) / (1 + j)


def min_rav(j, i, u, d):
    return ((1 - np.power(d, i + 1)) / (1 - d) + np.power(d, i) * u * (
        1 - np.power(u, j - i)) / (1 - u)) / (1 + j)


def find_index(num, num_list):

    # print(num, num_list)

    if num <= num_list[0]:
        return 0

    max = 0
    max_idx = 0
    for i in range(len(num_list)):

        if num_list[i] > max:
            max = num_list[i]
            max_idx = i

        if num_list[i] >= num:
            return i - 1

    return max_idx


def european_style_asian_single_barrier_up_and_out(stock_price, strike_price, barrier,
                                                   maturity_in_years, annual_volatility,
                                                   risk_free_interest_rate,
                                                   number_of_periods,
                                                   number_of_states_per_node):
    """Drawing the payoff graph

        Args:
            stock_price: 
            strike_price: 
            barrier: 
            maturity_in_years:
            annual_volatility:
            risk_free_interest_rate:
            number_of_periods:
            number_of_states_per_node:

        Returns
            the pay off of the option
        """
    annual_volatility /= 100
    risk_free_interest_rate /= 100
    period_unit = maturity_in_years / number_of_periods
    u = np.exp(annual_volatility * np.sqrt(period_unit))
    d = 1 / u
    R = np.exp(risk_free_interest_rate * period_unit)
    p = (R - d) / (u - d)

    print(u, d)

    CRR_table = []

    for i in range(number_of_periods + 1):
        node_table = dict()
        node_table["rav"] = []
        node_table["val"] = []

        min_avg = stock_price * min_rav(number_of_periods + 1, i, u, d)
        max_avg = stock_price * max_rav(number_of_periods + 1, i, u, d)
        node_table["rav"] = [barrier_determine(x, barrier) for x in
                             avg_split(min_avg, max_avg, number_of_states_per_node)]
        for rav in node_table["rav"]:
            val = call_payoff(rav, strike_price)
            node_table["val"].append(val)

        CRR_table.append(node_table)


    # print(CRR_table)
    for j in reversed(range(number_of_periods)):
        print("${j}---------".format(j=j))
        for i in range(j):
            node_table = dict()
            node_table["rav"] = []
            node_table["val"] = []

            min_avg = stock_price * min_rav(j + 1, i, u, d)
            max_avg = stock_price * max_rav(j + 1, i, u, d)
            node_table["rav"] = [barrier_determine(x, barrier) for x in
                                 avg_split(min_avg, max_avg, number_of_states_per_node)]
            for rav in node_table["rav"]:
                if not rav == 0:
                    up_table = CRR_table[i]["rav"]
                    down_table = CRR_table[i + 1]["rav"]

                    up_idx = find_index(rav, up_table)
                    if up_idx + 1 in up_table and not up_table[up_idx + 1] == 0:
                        up_x = (rav - up_table[up_idx]) / (up_table[up_idx] - up_table[up_idx + 1])
                        up_val = up_x * up_table[up_idx] + (1 - up_x) * up_table[up_idx + 1]
                    else:
                        up_val = up_table[up_idx]
                    # print(up_val)

                    down_idx = find_index(rav, down_table)
                    if down_idx + 1 in down_table and not down_table[down_idx + 1] == 0:
                        down_x = (rav - down_table[down_idx]) / (down_table[down_idx] - down_table[down_idx + 1])
                        down_val = down_x * down_table[up_idx] + (1 - down_x) * down_table[up_idx + 1]
                    else:
                        down_val = down_table[down_idx]
                    # print(down_val)

                    val = (p * up_val + (1 - p) * down_val) / R
                else:
                    val = 0

                node_table["val"].append(call_payoff(barrier_determine(val, barrier), strike_price))

            CRR_table[i] = node_table
        print(CRR_table)


european_style_asian_single_barrier_up_and_out(100, 90, 110, 1, 30, 5, 120, 5)
# print(100 * max_rav(120, 1, 1.02776457471, 0.972985472168))

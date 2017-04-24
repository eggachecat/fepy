import numpy as np
from fepy import feplot

def evaluate_the_option_payoff(option_type, strike_price, stock_price):
    """[From 7.2][page-76] Payoff of options

    Args:
        option_type: 0 for call and 1 for put
        strike_price: the strike price of the option
        stock_price: the price of the stock corresponding to the option 

    Returns
        the pay off of the option
    """

    if option_type == 0:
        return max([0, stock_price - strike_price])
    else:
        return max([0, strike_price - stock_price])

def draw_option_hedge(options_arr,  min_stock_price = 0, max_stock_price = 100):
    """[From 7.4.1][page-79] Drawing the payoff graph

    Args:
        options_arr: objects: [(option_type, strike_price, long_or_short, value), ...]
                    long_or_short: -1 for long and 1 for short
        stock_price_range: (min, max)

    Returns
        the pay off of the option
    """

    intervals = 10000
    unit_stock_price = (max_stock_price - min_stock_price) / intervals

    stock_price_arr = np.arange(min_stock_price, max_stock_price, unit_stock_price)
    profit_arr = []

    for stock_price in stock_price_arr:
        profit = 0
        for option in options_arr:
            profit += option[2] * option[3]
            profit += -1 * option[2] * evaluate_the_option_payoff(option[0], option[1], stock_price)
        profit_arr.append(profit)
        stock_price += unit_stock_price

    canvas = feplot.FECanvas()
    canvas.draw_line_chart_2d(stock_price_arr, profit_arr)
    canvas.froze()
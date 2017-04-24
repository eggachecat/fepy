from fepy import ch7


def basic_test():
    # short a call
    ch7.draw_option_hedge([(0, 85, 1, 20)], max_stock_price=110)

    # long a call
    ch7.draw_option_hedge([(0, 85, -1, 20)], max_stock_price=110)

    # short a put
    ch7.draw_option_hedge([(1, 85, 1, 20)], max_stock_price=110)

    # long a call
    ch7.draw_option_hedge([(1, 85, -1, 20)], max_stock_price=110)

def ratio_hedge():
    # ratio hedge
    ch7.draw_option_hedge([(0, 85, 1, 20), (1, 85, 1, 20)], max_stock_price=110)

def bull_spread():
    # short a call and long a call
    ch7.draw_option_hedge([(0, 105, 1, 8), (0, 100, -1, 10)], max_stock_price=200)

def bull_spread_2():
    # short a call and long a call
    ch7.draw_option_hedge([(0, 105, 1, 8), (1, 100, 1, 10)], max_stock_price=200)

bull_spread()

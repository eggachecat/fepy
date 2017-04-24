import numpy as np
import prettytable  as pt


def evaluate_present_value(interest_rate, cash_flows):
    """[From 3.1.1][page-13] Efficient Algorithms for Present and Future Values 

    Args:
        interest_rate: also the discounting rate
        cash_flows: array who saves all future cash flows; whose length should equal to term of the investment

    Returns
        the present value given the future cash_flows and 
    """

    presentValue = 0
    d = 1 + interest_rate
    termOfTheInvestment = len(cash_flows)

    for i in range(0, termOfTheInvestment):
        presentValue += cash_flows[i] / d
        d = d * (1 + interest_rate)

    return presentValue


def calculate_effective_annual_interest_rate(annual_interest_rate, compounded_frequency):
    """[From 3.1][page-11]calculating effective annual interest_rate
    Args:
        annual_interest_rate: annual interest rate
        compounded_frequency: how many times the interest rate compounded per year

    """

    return float(np.power(float(1 + (annual_interest_rate / compounded_frequency)), compounded_frequency))


def evaluate_amortization(loan, term_of_the_amortization, interest_rate, annual_frequency_of_payment=12):
    """[From 3.3][page-15]Generate a table of an amortization schedule
    Args:
        loan: amount of money loaned from bank
        term_of_the_amortization: the period of the amortization
        interest_rate: the rate of the repayment from bank (per year)
        annual_frequency_of_payment: how many times should you pay to the bank anually

    Returns:
        A table recording the detail payment and its components for each preiod

    Example:
        (250000, 15, 0.08, 12) means taking out a 15-year $250,000 loan at an 8% rate paying 12 times per year (monthly).

    """
    effective_interest_rate = calculate_effective_annual_interest_rate(interest_rate, annual_frequency_of_payment)
    payment_per_preiod = loan * (interest_rate / annual_frequency_of_payment) / (
    1 - np.power(effective_interest_rate, -1 * term_of_the_amortization))

    remaining_principal = loan
    periodCtr = 0

    table = pt.PrettyTable(
        ["Period", "Payment", "Interest", "Principal", "Principal_Present_Value", "Remaining Principal"])

    while remaining_principal > 0:
        periodCtr += 1
        interest_per_preiod = remaining_principal * interest_rate / annual_frequency_of_payment

        principal_per_preiod = payment_per_preiod - interest_per_preiod
        remaining_principal -= principal_per_preiod
        principal_pv_per_preiod = principal_per_preiod / np.power(1 + interest_rate / annual_frequency_of_payment,
                                                                  periodCtr - 1)
        table.add_row(
            [periodCtr, payment_per_preiod, interest_per_preiod, principal_per_preiod, principal_pv_per_preiod,
             remaining_principal])

    total_payment = periodCtr * payment_per_preiod
    # table.add_row(["Total", total_payment, total_payment - loan, loan, "0"])
    print(table)


def calculating_internal_rate_of_return_bisection(present_value, cash_flows, epsilon=0.00000001):
    """[From 3.4.3][page-21]The biscection method for solving equation
    Args:
        present_value: the present value of a investment
        cash_flows: the cash flow that the investment will bring to us
        epsilon: the precision of the solution
    
    Returns:
        the internal rate of the project bisection method 
        detail in Args-example
    
    Example:
        (13000, [5000, 5000, 5000], 0.00000001) will give you the internal rate of a project 
        who has current investment = 13000 and will bring cash flow 5000 in the future 3 period with precision 0.00000001

    """

    def target_function(interest_rate):
        pv = 0
        d = 1 + interest_rate
        for cf in cash_flows:
            # print(pv)
            pv += cf / d
            d = d * (1 + interest_rate)

        return pv - present_value

    # usually 10 is big enough
    a = 0
    b = 10

    while b - a > epsilon:
        ir = (b + a) / 2
        if target_function(ir) == 0:
            return ir

        if target_function(ir) < 0:
            b = ir
        else:
            a = ir

    print(a, b)
    return (b + a) / 2


def calculating_internal_rate_of_return_newton(present_value, cash_flows, epsilon=0.000001):
    """[From 3.4.3][page-22]The Newton-Raphson method for solving equation
    Args:
        present_value: the present value of a investment
        cash_flows: the cash flow that the investment will bring to us
        epsilon: the precision of the solution
    
    Returns:
        the internal rate of the project with Newton-Raphson method 
        detail in Args-example
        
    Example:
        (13000, [5000, 5000, 5000], 0.00000001) will give you the internal rate of a project 
            who has current investment = 13000 and will bring cash flow 5000 in the future 3 period with precision 0.00000001


    """

    def target_function(interest_rate):
        pv = 0
        d = 1 + interest_rate
        for cf in cash_flows:
            # print(pv)
            pv += cf / d
            d = d * (1 + interest_rate)

        return pv - present_value

    def target_derived_function(interest_rate):
        df = 0
        d = (1 + interest_rate)
        for i in range(0, len(cash_flows)):
            d = d * (1 + interest_rate)
            cf = cash_flows[i]
            df += (i + 1) * cf / d

        return -df

    # usually enough
    x_old = float("inf")
    x_new = 0

    while np.abs(x_new - x_old) > epsilon:
        x_old = x_new
        x_new = x_new - target_function(x_new) / target_derived_function(x_new)

    return x_new


evaluate_amortization(250000, 15, 0.08)

# principal = 260000
# cfs = np.empty(15 * 12)
# cfs.fill(2000)


# print(calculating_internal_rate_of_return_bisection(260000, cfs) * 12)
# print(calculating_internal_rate_of_return_newton(260000, cfs) * 12)

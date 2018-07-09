#!/usr/bin/env python
# Robert Knight, 2017-2108
# This application is not licensed for use and redistribution.
# This is here for demonstration purposes.


def mean_of_sample(input_list):
    """ Calculate the mean of a sample """
    x_bar = float(sum(input_list)) / len(input_list)
    return x_bar


def sum_of_measures_Ex_or_Ey(x):
    """ Sum of measures, add up all values in the list """
    Ex = float(sum(x))
    return Ex


def sum_of_squares_ssx_or_ssy_Ex2_or_Ey2(x):
    """ Square each value in a list, and sum them """
    #  SSX  and SSY   (use this same thing for sum of squares of y)
    squares = []
    for i in x:
        squares.append(i ** 2)
    Ex2 = sum(squares)
    return Ex2


def standard_error_se(input_list):
    import math
    """ stand error - standard deviation of sampling distribution """
    se = (sample_standard_deviation(input_list)) / math.sqrt(float(len(input_list)))
    return se


def sum_of_squares_of_y_bar_sso(y_list):
    """ SSO """
    SSO = len(y_list) * mean_of_sample(y_list) ** 2
    return SSO


def mean_squared_errors_mse(x, y):
    """ MSE, mean squared errors """
    SSE = sum_of_squared_errors_sse(x, y)
    df = len(x) - 2
    MSE = SSE / df
    return MSE


def standard_deviation_of_errors_Se(x, y):
    import math
    """ standard deviation of errors, square root of MSE """
    Se = math.sqrt(mean_squared_errors_mse(x, y))
    return Se


def standard_deviation_of_b0_Sb0(x, y):
    """ standard deviation of intercept """
    Se = standard_deviation_of_errors_Se(x, y)
    exponent = float(.5)
    a = 1 / float(len(x))
    b_num = mean_of_sample(x) ** 2
    b_dena = sum_of_squares_ssx_or_ssy_Ex2_or_Ey2(x)
    b_denb = len(x) * (mean_of_sample(x) ** 2)
    #  print(Se, a, b_num, b_dena, b_denb, exponent)
    Sb0 = Se * ((a + (b_num / (b_dena - b_denb))) ** exponent)
    return Sb0


def standard_deviation_of_b1_Sb1(x, y):
    Se = standard_deviation_of_errors_Se(x, y)
    Ex2 = sum_of_squares_ssx_or_ssy_Ex2_or_Ey2(x)
    nx2 = len(x) * (mean_of_sample(x) ** 2)
    exponent = .5
    Sb1 = Se / (Ex2 - nx2) ** exponent
    return Sb1


def sum_of_squared_errors_sse(x, y):
    """ sum of squared errors """
    #  SSE = Ey2 - b0Ey - b1Exy
    Ey2 = sum_of_squares_ssx_or_ssy_Ex2_or_Ey2(y)
    line = simple_linear_regression(x, y)
    Ey = sum(y)
    Exy = sum_xy(x, y)
    b0 = line[0]
    b1 = line[1]
    SSE = Ey2 - (b0 * Ey) - (b1 * Exy)
    return SSE


def sum_of_squares_explained_by_regression_ssr(x, y):
    """sum of squares explained by regression """
    #  SSR = SST - SSE
    SSR = total_sum_of_squares_sst(y) - sum_of_squared_errors_sse(x, y)
    return SSR


def r_squared(x, y):
    """ r squared value """
    SSR = sum_of_squares_explained_by_regression_ssr(x, y)
    SST = total_sum_of_squares_sst(y)
    r2 = SSR / SST
    return r2


def sum_of_squared_deviations(x):
    """ Sum of Squared Deviations for variance and standard deviation """
    # (X-Xbar)^2
    a = []
    x_bar = mean_of_sample(x)
    for i in x:
        a.append((i - x_bar) ** 2)
    b = sum(a)
    return b


def confidence_interval(x, y, t_crit):
    """ confidence interval for b0 and b1
    returned in a 4 item list, with b0 first, and b1 second

    Uses parameters, not the same b0/b1 as the simple linear
    regression formula
    """
    line = simple_linear_regression(x, y)
    b0 = line[0]
    b1 = line[1]
    interval = []
    ci_b0_a = b0 - (t_crit * standard_deviation_of_b0_Sb0(x, y))
    ci_b0_b = b0 + (t_crit * standard_deviation_of_b0_Sb0(x, y))
    ci_b1_a = b1 - (t_crit * standard_deviation_of_b1_Sb1(x, y))
    ci_b1_b = b1 + (t_crit * standard_deviation_of_b1_Sb1(x, y))
    interval.append('intercept b0')
    interval.append(ci_b0_a)
    interval.append(ci_b0_b)
    interval.append('slope b1')
    interval.append(ci_b1_a)
    interval.append(ci_b1_b)
    return interval


def total_sum_of_squares_sst(y_list):
    """ total sum of squares  """
    #   SST = SSY - SSO
    #  E(y - y_bar)^2
    squared_errors = []
    y_bar = mean_of_sample(y_list)
    for i in y_list:
        squared_errors.append((i - y_bar) ** 2)
    SST = sum(squared_errors)
    return SST


def sum_of_squared_residuals(x_input_list, y_input_list):
    """ sum of squared residuals, used for confidence interval of slope
    for null hypothesis testing - H0=Variables are not related - meaning,
    the slope confidence interval contains zero.  If the slope does not
    confidence interval does not contain zero, the we can say the items are
    related, therefore rejecting the null hypothesis """
    x = x_input_list[:]
    y = y_input_list[:]
    y_predicted = []
    y_hat = []
    line = simple_linear_regression(x, y)
    squared_residuals = []
    for x_variable in x:
        y_predicted.append(predict_y(line[0], line[1], x_variable))
        #  print('y_predicted ', y_predicted)
    for y_variable in y_predicted:
        y_hat.append(y.pop(0) - y_variable)
        #  print('y_hat ', y_hat)
    for residual in y_hat:
        #  print('squared residuals ', squared_residuals)
        squared_residuals.append(residual * residual)
    return sum(squared_residuals)


def parameter_estimate_b1(x_list, y_list):
    """ calculate the parameter estimate for b1"""
    #  b1 = sxy/sxx
    x = x_list[:]
    y = y_list[:]
    Sxy = sum_of_product_of_difference_x_y(x, y)
    Sxx = sum_of_squares_of_difference_x(x)
    b1_hat = Sxy/Sxx
    return b1_hat


def parameter_estimate_b0(x_list, y_list):
    """ calculate the parameter estimate for b1"""
    #  b1 = sxy/sxx
    x = x_list[:]
    y = y_list[:]
    b1_hat = parameter_estimate_b1(x, y)
    b0_hat = mean_of_sample(y) - b1_hat * mean_of_sample(x)
    return b0_hat


def standard_error_of_the_estimate(x_list, y_list):
    """ calculate the standard error of the estimate """
    syx = sum_of_squared_residuals(x_list, y_list) / (len(x_list) - 2)
    return syx


def sum_of_squares_of_difference_x(x_list):
    """ sum of the squares of the difference between each x and x_bar """
    x = x_list[:]
    x_bar = mean_of_sample(x)
    a = []
    for i in x:
        a.append((i - x_bar) * (i - x_bar))
    Sxx = sum(a)
    return Sxx


def sum_of_product_of_difference_x_y(x_list, y_list):
    """ Sum of the product of the difference between x and x_bar and the
    difference between y and y_bar """
    x = x_list[:]
    x_bar = mean_of_sample(x)
    y = y_list[:]
    y_bar = mean_of_sample(y)
    a = []
    for i in x:
        a.append((i - x_bar) * (y.pop(0) - y_bar))
    Sxy = sum(a)
    return Sxy


def standard_error_of_regression(x_list, y_list):
    import math
    """ calculate the standard error of the regression called 's' """
    x = x_list[:]
    y = y_list[:]
    s = math.sqrt((sum_of_squared_residuals(x, y)) / (len(x) - 2))
    return s


def predict_y(b0, b1, x):
    """ predict a y value given the components of the slope equation """
    y = b0 + (b1 * x)
    return y


def sample_variance(x):
    """ Variance of the sample """
    s2 = sum_of_squared_deviations(x) / (len(x) - 1)
    return s2


def sample_standard_deviation(x):
    """ Standard Deviation of the sample """
    import math
    s2 = sum_of_squared_deviations(x) / (len(x) - 1)
    s = math.sqrt(s2)
    return s


def population_variance(x):
    """ Variance of the sample """
    s2 = sum_of_squared_deviations(x) / (len(x))
    return s2


def population_standard_deviation(x):
    """ Standard Deviation of the sample """
    import math
    s2 = sum_of_squared_deviations(x) / (len(x))
    sigma = math.sqrt(s2)
    return sigma


def sum_xy(x, y):
    """ Sum of X and Y for use with correlation coefficients """
    a = x[:]
    b = y[:]
    c = []
    for i in a:
        c.append(i * b.pop(0))
    d = sum(c)
    return d


def simple_moving_average(input_list, count):
    """ Simple moving average for n data points """
    n_size = float(count)
    x = []
    cut = count * -1
    for i in input_list:
        x.append(float(i))
    x_bar = sum(x[cut:]) / n_size
    return x_bar


def exponential_moving_averages(input_list, length, initial):
    """ Calculate an exponential moving average where the initial
    previous day EMA is set to the same value as input's list[0]"""
    exponent = float(2) / float((length + 1))
    x = input_list[:]
    ema = []
    initial_EMA = float(initial)
    ema.append(initial_EMA)
    for i in x:
        new_ema = (float(i) - ema[-1]) * exponent + ema[-1]
        ema.append(new_ema)
        print(str(i), str(new_ema), str(exponent))
    del ema[0]
    return ema


def exponential_moving_average(x, length, previous):
    """ Calculate an exponential moving average where the previous ema
    is given """
    exponent = float(2) / float((length + 1))
    yesterday = float(previous)
    ema = (x * exponent) + (yesterday * (1 - exponent))
    return ema


def correlation_coefficient(x, y):
    """ Correlation Coefficient - the r value """
    n = float(len(x))
    Ex = sum(x)
    Ey = sum(y)
    Exy = sum_xy(x, y)
    sx = sample_standard_deviation(x)
    sy = sample_standard_deviation(y)
    r = (Exy - ((1 / n * Ex) * Ey)) / ((n - 1) * sx * sy)
    return r


def correlation(x, y):
    """ Correlation Coefficient - the r value, the same as the
     correlation_coefficient, but with an easier name for other usages"""
    n = float(len(x))
    Ex = sum(x)
    Ey = sum(y)
    Exy = sum_xy(x, y)
    sx = sample_standard_deviation(x)
    sy = sample_standard_deviation(y)
    r = (Exy - ((1 / n * Ex) * Ey)) / ((n - 1) * sx * sy)
    return r


def coefficient_of_determination(r):
    """ R Squared  """
    # R-Squared
    a = r ** 2
    return a


def simple_linear_regression(x, y):
    """ Return the equation for a line through the data in two lists """
    line = []
    x_bar = mean_of_sample(x)
    y_bar = mean_of_sample(y)
    b1 = correlation_coefficient(x, y) * (
        sample_standard_deviation(y) / sample_standard_deviation(x))
    b0 = y_bar - (b1 * x_bar)
    line.append(b0)
    line.append(b1)
    return line


def mean_absolute_deviation(input_list, number_of_values):
    """ Calculate the mean absolute deviation - length is the size of the
     average - such as only the most recent 20 data points.  To use the
     entire list, use the length of the input list."""
    n = float(number_of_values)
    index = number_of_values * -1
    x = input_list[index:]
    y = []
    x_bar = sum(x) / n
    for i in x:
        y.append(abs(x_bar - i))
    mad = sum(y) / n
    return mad


def binomial(attempts, successes, proportion):
    """ Calculate the binomial probability """
    # n = attempts, x = successes, p = proportion
    import math
    bp = (math.factorial(attempts) / (
        math.factorial(successes) * math.factorial(attempts - successes))) * (
             proportion ** successes) * (
         (1 - proportion) ** (attempts - successes))
    print(str(bp))
    return bp


def z_area(z_value):
    import math
    """ return the area to the left of a z-value asymptote. -the P-Value """
    #  use "1 - zpercent(z_value)"  to find the area encompassing
    #  the space after the z-value if the value is above mean
    #  this returns the area to the left of the z-value.
    return 1. - 0.5 * math.erfc(z_value / (2 ** 0.5))


def p_value_t_dist_scipy(t_stat, df):
    from scipy.stats import t
    """ calculate the t area using scipy 
    This takes a T-Value, and the Degree of freedom, and gives the P-Value.
    For example, 12.71 at 1 degree of freedom, results in a p value of 
    almost .025, as does 2.306 at 8 df.    
    """
    results = []
    results.append(t.sf(t_stat, df))
    return results


def z_percent(z_value):
    """ return the percentage probability associated with a p-value. """
    x = 100 - (100 * z_area(z_value))
    if z_value < 0:
        y = 100 - x
    else:
        y = x
    return y


def t_value_significance_test(x, y):
    import math
    """ Calculate the t-value of a correlation coefficient 
    for a SIGNIFICANCE TEST """
    t = correlation_coefficient(x, y) * math.sqrt((len(x) - 2)/(
        1-coefficient_of_determination(correlation_coefficient(x, y))))
    return t


def standard_error_of_mean(list_x):
    import math
    """ returns the standard error of the mean for an input list """
    standard_error = sample_standard_deviation(list_x)/math.sqrt(len(list_x))
    return standard_error


def z_score(x, input_list):
    """ calculate a z-score """
    z = (x - mean_of_sample(input_list)) / sample_standard_deviation(input_list)
    return z


def t_score(x, input_list):
    """ T-scores transform a raw datum into standard form """
    t = (x - mean_of_sample(input_list)) / sample_standard_deviation(input_list)
    return t


def reference_range(x_list):
    """ Reference range - 95% confidence level for sample size >= 100 """
    ci = []
    ci.append(mean_of_sample(x_list) - (1.96 * sample_standard_deviation(x_list)))
    ci.append(mean_of_sample(x_list) + (1.96 * sample_standard_deviation(x_list)))
    return ci


def reference_range_900(x_list):
    """ Reference range - 90% confidence level for sample size >= 100 """
    ci = []
    ci.append(mean_of_sample(x_list) - (1.645 * sample_standard_deviation(x_list)))
    ci.append(mean_of_sample(x_list) + (1.645 * sample_standard_deviation(x_list)))
    return ci


def reference_range_950(x_list):
    """ Reference range - 95% confidence level for sample size >= 100 """
    ci = []
    ci.append(mean_of_sample(x_list) - (1.96 * sample_standard_deviation(x_list)))
    ci.append(mean_of_sample(x_list) + (1.96 * sample_standard_deviation(x_list)))
    return ci


def reference_range_990(x_list):
    """ Reference range - 99.0% confidence level for sample size >= 100 """
    ci = []
    ci.append(mean_of_sample(x_list) - (2.576 * sample_standard_deviation(x_list)))
    ci.append(mean_of_sample(x_list) + (2.576 * sample_standard_deviation(x_list)))
    return ci


def reference_range_995(x_list):
    """ Reference range - 99.5% confidence level for sample size >= 100 """
    ci = []
    ci.append(mean_of_sample(x_list) - (2.807 * sample_standard_deviation(x_list)))
    ci.append(mean_of_sample(x_list) + (2.807 * sample_standard_deviation(x_list)))
    return ci


def reference_range_999(x_list):
    """ Reference range - 99.9%% confidence level for sample size >= 100 """
    ci = []
    ci.append(mean_of_sample(x_list) - (3.291 * sample_standard_deviation(x_list)))
    ci.append(mean_of_sample(x_list) + (3.291 * sample_standard_deviation(x_list)))
    return ci


def null_hypothesis_test_95(x_list, y_list):
    """ Test the null hypothesis for two variables at the 95 CI """
    from scipy import stats

    # T has n-1 degrees of freedom
    df = len(x_list) - 1

    # print "95 CI t-crit: ", stats.t.ppf(1 - 0.025, df), "with df: ", df

    result = confidence_interval(x_list, y_list, stats.t.ppf(1 - 0.025, df))

    # print 'Line details:', test
    # print 'Slope confidence interval:', test[4], 'to', test[5]
    if result[4] < 0:
        # print 'Lower bound', result[4], 'is less than zero with an
        # upper bound of', result[5]
        if result[5] < 0:
            # print('Range is below than zero. Able to reject null hypothesis, H0, at 95% confidence interval.')
            print 'Related'
        else:
            # test[5] > 0
            # print('Range contains zero. Unable to reject null hypothesis, H0, at 95% confidence interval.')
            print 'Unrelated'
    else:
        # test[4] > 0
        # print('Lower bound', result[4], 'is greater than zero with an upper bound of', result[5])
        if result[5] < 0:
            # print('Range contains zero. Unable to reject null hypothesis, H0, at 95% confidence interval.')
            print 'Unrelated'
        else:
            # test[5] > 0
            # print('Range is above zero.  Able to reject null hypothesis, H0, at 95% confidence interval.')
            print 'Related'


def main():
    """ Main area of the program
    Here on the base library so it can be imported and used in other
    scripts"""


if __name__ == '__main__':
    main()

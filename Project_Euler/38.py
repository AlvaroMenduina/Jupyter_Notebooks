"""
Project Euler - Problem 38

Take the number 192 and multiply it by each of 1, 2, and 3:

    192 × 1 = 192
    192 × 2 = 384
    192 × 3 = 576

By concatenating each product we get the 1 to 9 pandigital, 192384576. We will call 192384576 the concatenated product of 192 and (1,2,3)

The same can be achieved by starting with 9 and multiplying by 1, 2, 3, 4, and 5, giving the pandigital, 918273645, which is the concatenated product of 9 and (1,2,3,4,5).

What is the largest 1 to 9 pandigital 9-digit number that can be formed as the concatenated product of an integer with (1,2, ... , n) where n > 1?

"""

import numpy as np

def concatenate_product(x, n):
    """
    Function that computes the concatenated product of X and (1, 2, ..., n)
    Example x = 192, n = 3
    192 × 1 = 192
    192 × 2 = 384
    192 × 3 = 576
    It returns '192384576'
    :return:
    """
    factors = list(np.arange(1, n+1))
    list_prod = []
    for f in factors:
        list_prod.append(str(x*f))
    s = ''
    for p in list_prod:
        s += p
    return s

def check_pandigital(number):
    """
    Function that checkes whether a number is PANDIGITAL
    i.e. it contains all digits from 1 to 9, only once
    Example: '192384576'
    """
    pandigital = True
    digits = list(np.arange(1, 10))
    if number.count('0') != 0:
        return False
    elif number.count('0') == 0:
        for d in digits:
            count = number.count(str(d))
            if count != 1:
                pandigital = False
                break
    return pandigital


if __name__ == "__main__":

    p = concatenate_product(192, 3)
    print(check_pandigital(p))

    N_max = 100000
    list_x = list(np.arange(1, N_max))
    valid_x, valid_n, valid_pan = [], [], []
    for k in np.arange(2, 10):
        print(k)
        for x in list_x:
            # print(x)
            number = concatenate_product(x, k)
            if check_pandigital(number):
                valid_x.append(x)
                valid_n.append(k)
                valid_pan.append(number)

    ### Solution
    valid_pan.sort()
    print("\nSolution: ", valid_pan[-1])


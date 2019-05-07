"""
Project Euler - Problem 56



A googol (10100) is a massive number: one followed by one-hundred zeros; 100100 is almost unimaginably large:
one followed by two-hundred zeros. Despite their size, the sum of the digits in each number is only 1.

Considering natural numbers of the form, ab, where a, b < 100, what is the maximum digital sum?


"""

import numpy as np

def sum_digits(number):
    s = 0
    for digit in str(number):
        s += int(digit)
    return s

if __name__ == "__main__":

    N = 100
    sums = []
    for a in range(N):
        for b in range(N):
            sums.append(sum_digits(a**b))

    print("\nMaximum digitial sum: ", np.max(sums))
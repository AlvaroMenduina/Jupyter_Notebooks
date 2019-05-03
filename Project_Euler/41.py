"""
Project Euler - Problem 41

We shall say that an n-digit number is pandigital if it makes use of all the digits 1 to n exactly once. For example, 2143 is a 4-digit pandigital and is also prime.

What is the largest n-digit pandigital prime that exists?

"""

import numpy as np
import itertools

def is_prime(n):
    prime = True
    if n % 2 == 0 and n != 2:
        return False
    elif n % 2 != 0 and n != 2:
        for factor in np.arange(3, int(np.sqrt(n)) + 1, 2):
            if n % factor == 0:
                return False
    return prime


def pandigitals_n(n):
    """
    Function that returns all n-pandigital numbers
    generated from the digits [1, 2, ..., n]
    """
    digits = list(np.arange(1, n+1))
    permutations = list(itertools.permutations(digits))
    pandigits = []
    for perm in permutations:
        s = ''
        for digit in perm:
            s += str(digit)
        # Ignore the even numbers, as they are not prime
        if s.endswith('2') == False and \
                s.endswith('4') == False and\
                s.endswith('6') == False and \
                s.endswith('8') == False:
            pandigits.append(int(s))
    return pandigits

if __name__ == "__main__":

    pandigitals_primes = []
    for n in np.arange(1, 10):
        print("\nCompute pandigital numbers up to %d" %n)
        pandigitals = pandigitals_n(n)
        for p in pandigitals:
            if is_prime(p):
                pandigitals_primes.append(p)

    print("\nLargest pandigital number that is prime: ", pandigitals_primes[-1])
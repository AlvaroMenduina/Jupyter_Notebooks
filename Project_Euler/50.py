"""
Project Euler - Problem 50

The prime 41, can be written as the sum of six consecutive primes:
41 = 2 + 3 + 5 + 7 + 11 + 13

This is the longest sum of consecutive primes that adds to a prime below one-hundred.

The longest sum of consecutive primes below one-thousand that adds to a prime, contains 21 terms, and is equal to 953.

Which prime, below one-million, can be written as the sum of the most consecutive primes?


"""

import numpy as np

def check_if_prime(n):
    # print("\nChecking if %d is prime" % n)
    if n <= 3:
        return n > 1
    elif n % 2 == 0 or n % 3 == 0:
        # print('False (Divisible)')
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            # print('False (Loop)')
            return False
        i += 6
    return True

def primes_up_to_n(n):
    primes = [2, 3]
    for i in np.arange(5, n+1, 2).astype(int):
        if check_if_prime(i):
            primes.append(i)
    return primes

if __name__ == "__main__":

    top = int(10**6)                # Primes that add up to a prime of less than TOP
    N = top/100                     # N has to be less than TOP because we are summing several primes
    primes = primes_up_to_n(N)
    n_primes = len(primes)

    factors = []                    # Save the consecutive prime factors
    length = []                     # Lengths of the sequence of consecutive primes
    sum_primes = []
    max_len = 1
    for i in range(n_primes):   # Loop over the starting prime for the chain of consecutive primes
        k = i + max_len         # k is the index of the sequence limit
        summ = 0
        while summ <= top and k < n_primes:

            n_factors = k - i + 1   # Length of the current sequence

            # Don't bother if the n factors is less than the current max
            try:
                l_max = max(length)
            except ValueError:
                l_max = 1
            if n_factors < l_max:
                k += 1
                continue

            # N factors longer than current maximum length
            fact = primes[i:k + 1]
            summ = np.sum(fact)

            # If the sum is small enough, check if it's prime
            if check_if_prime(summ) and summ <= top:
                print("\n")
                print(fact[:n_factors//10])
                print(summ)
                factors.append(fact)
                sum_primes.append(summ)
                length.append(n_factors)
            k += 1

    # The longest chain of consecutive prime numbers that add up to a prime below TOP
    i_long = np.argmax(length)
    print("\nThe (%d) prime factors: " %length[i_long], factors[i_long])
    print("add up to the prime: ", sum_primes[i_long])
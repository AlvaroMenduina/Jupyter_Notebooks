"""


"""
import itertools
import numpy as np

def check_if_prime(n):
    if n <= 3:
        return n > 1
    elif n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def primes_up_to_n(n):
    primes = [2, 3]
    for i in np.arange(5, n+1, 2).astype(int):
        if check_if_prime(i):
            primes.append(i)
    return primes

def prime_factors_of(x, primes_cut):
    factors = []
    if check_if_prime(x):
        return [x]
    else:
        for prime in primes_cut:
            if x % prime == 0:
                factors.append(prime)
    return factors

def cut_primes(primes, x):
    p = primes.copy()
    i = (np.argwhere(p > 3 * np.ceil(np.sqrt(x))))[0][0]
    return p[:i]


# def check_if_disjoint(list_factors):
#     disjoint = True
#     for (a, b) in itertools.combinations(list_factors, 2):
#         if not set(a).isdisjoint(set(b)):
#             return False
#     return disjoint

if __name__ == "__main__":

    N_consec = 4
    N = 150000

    primes = primes_up_to_n(N)

    i = 10
    while i < N:
    # for i in np.arange(1, N):
        list_num = [i + s for s in range(N_consec)]

        # Compute the prime factors
        list_factors = []
        for num in list_num:
            f = prime_factors_of(num, cut_primes(primes, num))
            list_factors.append(f)

        # Check if all have N_consec distinct factors
        if all([True if len(f)==N_consec else False for f in list_factors]):
            print("\nNumbes: ", list_num)
            print("List of distinct factors: ", list_factors)
            break
        i += 1
        # print(i)




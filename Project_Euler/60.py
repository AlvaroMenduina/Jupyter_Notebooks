"""
Project Euler - Problem 60


The primes 3, 7, 109, and 673, are quite remarkable. By taking any two primes and concatenating them in any order the result will always be prime. For example, taking 7 and 109, both 7109 and 1097 are prime. The sum of these four primes, 792, represents the lowest sum for a set of four primes with this property.

Find the lowest sum for a set of five primes for which any two primes concatenate to produce another prime.

"""

import numpy as np
import itertools

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

def concatenate_pair(num1, num2):

    return int(str(num1) + str(num2)), int(str(num2) + str(num1))

def check_pairs(list_nums):
    """
    Function to check whether all possible concatenations from a list of primes are prime
    :param list_nums: list containing some prime numbers to concatenate
    :return: True if all pairs are primes
    """
    pairs = []
    for p1, p2 in itertools.combinations(list_nums, 2):
        a, b = concatenate_pair(p1, p2)
        pairs.extend([check_if_prime(a), check_if_prime(b)])
    # print(pairs)
    return all(pairs)

def generate_prime_set(N):
    """
    Function that generates a dictionary.
    Each KEY is a prime p, and the VALUE is a list of primes
    that form prime pairs when concatenated with p.

    For example
    {3: [7, 11, 17, 31, 37, 59, 67, 73],
    7: [3, 19, 61, 97], ...}

    3 can form the prime pairs [37, 73] with 7, [311, 113] with 11
    :param N: use primes up to N
    :return: prime_set, primes up to N
    """

    primes = primes_up_to_n(N)
    # Both 2 and 5 won't work once you start concatenating
    primes.remove(2)
    primes.remove(5)

    print("\nGenerating set of primes that can form prime pairs:")

    prime_set = {}
    for p in primes:
        primes_c = primes.copy()
        primes_c.remove(p)
        candi_list = []

        ### For each prime check all the candidates that can form pairs with it
        for candidate in primes_c:
            pair1, pair2 = concatenate_pair(p, candidate)
            if check_if_prime(pair1) and check_if_prime(pair2):
                candi_list.append(candidate)
        prime_set[p] = candi_list

    print("Done")

    return prime_set, primes


if __name__ == "__main__":

    ### Check the given set [3, 7, 109, 673]
    N = 700
    prime_set, primes = generate_prime_set(N)

    for p1, p2 in list(itertools.combinations(primes, 2)):

        a, b = concatenate_pair(p1, p2)
        if check_if_prime(a) and check_if_prime(b):
            set1 = prime_set[p1]
            set2 = prime_set[p2]
            # print(p1, p2, set1, set2)

            inters_set = list(set(set1).intersection(set2))
            # print(inters_set)

            # if len(inters_set) >= 2:
            for p3, p4 in list(itertools.combinations(inters_set, 2)):
                a, b = concatenate_pair(p3, p4)
                if check_if_prime(a) and check_if_prime(b):
                    print("\nCandidate set: ", [p1, p2, p3, p4])
                    sum_candidates = p1 + p2 + p3 + p4
                    print(sum_candidates)

    ### Method for groups of 5
    N = 10000
    prime_set, primes = generate_prime_set(N)

    for p1, p2 in list(itertools.combinations(primes, 2)):

        set1 = prime_set[p1]
        set2 = prime_set[p2]

        # Check if they are in each others set
        if p1 in set2 and p2 in set1:
            inters_set = list(set(set1).intersection(set2))

            for p3, p4 in list(itertools.combinations(inters_set, 2)):

                set3 = prime_set[p3]
                set4 = prime_set[p4]

                # Check if they are in each others set
                if p3 in set4 and p4 in set3:
                    new_inters_set = list(set(set3).intersection(set4))
                    try:
                        new_inters_set.remove(p1)
                        new_inters_set.remove(p2)
                    except ValueError:
                        pass

                    if len(new_inters_set) >= 1:
                        for p5 in new_inters_set:
                            if check_pairs([p1, p2, p3, p4, p5]):
                                print([p1, p2, p3, p4, p5])
                                sum_candidates = np.sum([p1, p2, p3, p4, p5])
                                print("Sum of candidates")
                                print(sum_candidates)







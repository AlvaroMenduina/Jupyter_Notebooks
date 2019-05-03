"""
Project Euler - Problem 24

A permutation is an ordered arrangement of objects.
For example, 3124 is one possible permutation of the digits 1, 2, 3 and 4.
If all of the permutations are listed numerically or alphabetically, we call it lexicographic order.
The lexicographic permutations of 0, 1 and 2 are:

012 021 102 120 201 210

What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
"""

import numpy as np
import itertools
from math import factorial as fact

if __name__ == "__main__":

    """ (1) Brute Force Approach """
    # Generate all possible permutations with itertools.permutations
    # access the n-th case

    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]     # list of digits to use in the permutations
    perm = list(itertools.permutations(nums))

    n = int(10**6) - 1
    result_brute = perm[n]
    print("\n%d-th permutation: " %(n+1), result_brute)

    """ (1) Smart Approach """

    # By looking at the structure of the solution we can infer the pattern

    # There are a total of P_n = n! possible permutations.
    # As we only have n digits, the first digit of the permutations sorted
    # forms n groups of n! / n = (n-1)! digits

    # [0]12
    # [0]21
    # [1]02
    # [1]20    <-- k-th permutation
    # [2]01
    # [2]10

    # so from a list [0, 1, 2], the first digit of the k-th perm. has index:
    #   i1 = k // (n-1)!
    # We remove that digit from the list: [0, 2] and now we can form
    # (n-1) groups of (n-2)! digits, with an updated index k - i1 * (n-1)!
    #   i2 = (k - i1 * (n-1)!) / (n - 2)!

    # [0]12
    # [0]21
    # ---------------------- i1 * (n-1)!
    # [1]02    |   1[0]2
    # [1]20    |   1[2]0     <-- k-th permutation
    # [2]01    |
    # [2]10    |

    # Repeating that n types allows us to extract the digits of the k-th permutation
    # from the digit list directly, without creating the n! possibilities

    def compute_nth_permutation(d, k, message=False):
        """
        Function to calculate the digits of the k-th
        permutation of a list of digits (sorted)

        For example: digits = [0, 1, 2] (digits)
            [012]
            [021]
            [102]   <-- 3rd permutation (k)
            [120]
            [201]
            [210]
        :param digits: list of digits to use in the permutation
        :param k: k-th permutation
        :return:
        """
        digits = d.copy()
        n_digits = len(digits)      # Numbers of digits in the list
        k -= 1                      # -1 for proper indexing purposes. The "first" permutation is perm[0]

        # Brute force solution for validation
        perm = list(itertools.permutations(nums))
        solution = perm[k]
        if message:
            print("\nSolution: ", solution)

        digits.sort()               # Make sure the digits are sorted
        p = []                      # Save solution digits here

        for j in np.arange(1, n_digits):
            i = k // fact(n_digits - j)     # Relative index within the group
            d_j = digits[i]
            p.append(d_j)
            digits.remove(d_j)              # Remove the digit from the choices
            if message:
                print("\nIndex, Digit: (%d, %d)" % (i, d_j))
                print("Remaining: ", digits)
            k -= i * fact(n_digits - j)     # Update relative index
        p.extend(digits)            # Append the last remaining digit

        return p

    result_smart = compute_nth_permutation(nums, int(10**6))
    print("\nSmart approach: ", result_smart)


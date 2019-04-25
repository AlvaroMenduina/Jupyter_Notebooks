import numpy as np

if __name__ == "__main__":

    """
    (1)
    We form 4-digit numbers (with distinct digits) out of {1 2 3 4 5 6 7}
    What is the probability of the number being even?
    """

    # Total: ways in which we can form 4 digit numbers from a set of 7 digits: V_{7,4}
    # 7 * 6 * 5 * 4

    # Even: Last digit must be {2 4 6}, so 3 choices for last digit X 6 * 5 * 4 for the remaining 3 digits
    # Probability: 3/7

    def number(digits, length):
        choice = np.random.choice(digits, size=length, replace=False)
        return choice

    digits = [1, 2, 3, 4, 5, 6, 7]

    N_trials = 100000
    counts = 0
    for i in range(N_trials):
        n = number(digits, length=4)
        if n[-1] % 2 == 0:
            counts += 1

    prob_even = counts / N_trials
    print('\nProbability: %.3f | True (3/4): %.3f' %(prob_even, 3/7.))

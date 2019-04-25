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

    def number(digits, length, replace=False):
        choice = np.random.choice(digits, size=length, replace=replace)
        return choice

    digits = [1, 2, 3, 4, 5, 6, 7]

    N_trials = 100000
    counts = 0
    for i in range(N_trials):
        n = number(digits, length=4, replace=False)
        if n[-1] % 2 == 0:
            counts += 1

    prob_even = counts / N_trials
    print('\nProbability: %.3f | True (3/4): %.3f' %(prob_even, 3/7.))

    # ================================================================================================================ #

    """
    (2)
    We form 4-digit numbers (NOT necessarily distinct digits) out of {1 2 3 4 5 6 7}
    What is the probability of the number being even?
    """
    # Total: 7^4 ways to choose. Even: Last has to be drawn from [2 4 6] so 3 choices, times 7^3 for the rest
    # Probability: 3/7

    counts = 0
    for i in range(N_trials):
        n = number(digits, length=4, replace=True)
        if n[-1] % 2 == 0:
            counts += 1

    prob_even = counts / N_trials
    print('\nProbability: %.3f | True (3/4): %.3f' %(prob_even, 3/7.))

    # ================================================================================================================ #

    """
    (3)
    We form 4-digit numbers (with distinct digits) out of {1 2 3 4 5 6 7}
    What is the probability of the number being larger than 2500?
    """
    # Total: 7*5*6*4
    # If the first digit is 2: the second digit can be chosen from [5 6 7], and the two remaining out of 5 choices
    # So 3 * 5 * 4
    # If the first digits are [3 4 5 6 7] then the remaining can be taken from [1 2 and rest]
    # so 5 * 6 * 5 * 4
    # Prob: (3*5*4 + 5*6*5*4) / (7*6*5*4) = 11 / 14

    def number_string(digits, length, replace=False):
        choice = np.random.choice(digits, size=length, replace=replace)
        list_string = [str(digit) for digit in choice]
        number = ''
        for k in range(len(list_string)):
            number += list_string[k]
        return int(number)

    counts = 0
    for i in range(N_trials):
        n = number_string(digits, length=4, replace=False)
        if n >= 2500:
            counts += 1

    prob_2500 = counts / N_trials
    print('\nProbability: %.3f | True (11/14): %.3f' %(prob_2500, 11/14.))

    # ================================================================================================================ #

    """
    (4)
    We form 4-digit numbers (NOT necessarily distinct digits) out of {1 2 3 4 5 6 7}
    What is the probability of the number being larger than 2500?
    """
    # Total: 7^4
    # If the first digit is 2: the second digit can be chosen from [5 6 7], and the two remaining:
    # So 3 * 7^2
    # If the first digits are [3 4 5 6 7] then the remaining can be taken from any
    # so 5 * 7^3
    # Prob: (3*7^2 + 5*7^3) / (7^4) = 38/49

    counts = 0
    for i in range(N_trials):
        n = number_string(digits, length=4, replace=True)
        if n >= 2500:
            counts += 1

    prob_2500 = counts / N_trials
    print('\nProbability: %.3f | True (38/49.): %.3f' %(prob_2500, 38/49.))

    # ================================================================================================================ #

    """
    (5)
    6 people sit in a row of 6 seats. What's the probability of two friends sitting next to each other
    """
    # Total: 6!
    # If Alice and Bob sit next to each other, we can move that configuration along the row, and distribute the remaining
    # seat among the 4 other students
    # [A] [B] [-] [-] [-] [-]
    # [-] [A] [B] [-] [-] [-]
    # ...
    # [-] [-] [-] [-] [A] [B]
    # 5 possibilities. Times 2 for the permutation of {AB}
    # Probability: 2 * 5 * (4!) / 6! = 1 / 3

    def arrange_seats(seats):
        np.random.shuffle(seats)
        i_alice = np.argwhere(np.array(seats)==0)[0][0]
        i_bob = np.argwhere(np.array(seats)==1)[0][0]
        return seats, i_alice, i_bob

    seats = [0, 1, 2, 3, 4, 5]
    counts = 0
    for k in range(N_trials):
        new_seats, i, j = arrange_seats(seats)
        if np.abs(j - i) == 1:
            counts += 1

    prob_seats = counts / N_trials
    print('\nProbability: %.3f | True (1/3): %.3f' %(prob_seats, 1/3.))



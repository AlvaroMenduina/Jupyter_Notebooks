import numpy as np
import matplotlib.pyplot as plt

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

    # ================================================================================================================ #

    """
    (6)
    6 people sit in a circle. What's the probability of two friends sitting next to each other
    """
    # Total: 6!
    # 6 rotations of the pair, times 4! choices for the remaining friends, times 2 for the permutation
    # Prob: 2 * 6 * 4! / 6! = 2/ 5

    seats = [0, 1, 2, 3, 4, 5]
    counts = 0
    for k in range(N_trials):
        new_seats, i, j = arrange_seats(seats)
        if np.abs(j - i) == 1 or np.abs(j - i) == 5:
            counts += 1

    prob_seats = counts / N_trials
    print('\nProbability: %.3f | True (2/5): %.3f' %(prob_seats, 2/5.))

    # ================================================================================================================ #
    """
    (7)
    We throw a dice 3 times. Probability of only getting a 2 once
    """
    # Total: 6^3
    # If the first roll is a two, the remaining 2 rolls can only be drawn from [1, -, 3, 4, 5, 6] so 5^2
    # as that can happen for any of the rolls, we multiply by rolls
    # Prob: 3 * 5^2 / 6^3

    def roll_dice_n_times(n):
        choices = np.array([1, 2, 3, 4, 5, 6])
        rolls = []
        for k in range(n):
            i = np.random.choice(range(6), 1, replace=False)[0]
            rolls.append(choices[i])
        return rolls

    n = 3
    counts = 0
    for k in range(N_trials):
        rolls = roll_dice_n_times(n)
        two = rolls.count(2)
        if two == 1:
            counts += 1

    prob_rolls = counts / N_trials
    prob_true = n /6 * (5/6)**(n-1)
    print('\nProbability: %.3f | True: %.3f' % (prob_rolls, prob_true))

    def count_ocurrence(n_rolls, N_trials, c=2):
        counts = 0
        for k in range(N_trials):
            rolls = roll_dice_n_times(n_rolls)
            two = rolls.count(2)
            if two == c:
                counts += 1

        prob_rolls = counts / N_trials
        return prob_rolls

    n_rolls = np.arange(1, 30)
    p_true = []
    p = []
    for n_r in n_rolls:
        print(n_r)
        p.append(count_ocurrence(n_r, N_trials=5000))
        p_true.append(n_r/6 * (5/6)**(n_r - 1))

    plt.figure()
    plt.scatter(n_rolls, p)
    plt.plot(n_rolls, p_true, color='black', linestyle='--')
    plt.xlabel('Number of rolls')
    plt.ylabel('Probability of getting a number ONLY ONCE')
    plt.show()

    # So the peak occurs at approximately 5-6 rolls

    # ================================================================================================================ #

    """
    You roll a dice N times. What's the probability of getting the same number only R times?
    """
    # Total: 6^N
    # If you force a number to happen R times, the remaining choices are 5^(N - r)
    # times the number of ways you can rearrange the positions.
    # i.e. permutations with repetition n! / (r! (n-r)!) which is the same as a standard
    # combination of r elements out of n

    # Prob: 5^(n-r) / 6^n C(n, r)

    from scipy.special import comb
    n_rolls = np.arange(1, 30)
    n_counts = [1, 2, 3]
    p_true = []
    p = []
    plt.figure()
    for c in n_counts:
        print(c)
        pp = []
        pp_true = []
        for n in n_rolls:
            pp.append(count_ocurrence(n, N_trials=500, c=c))
            pp_true.append(5**(-c) * (5/6)**n * comb(n,c))
        p.append(pp)
        p_true.append(pp_true)

        plt.scatter(n_rolls, pp)
        plt.plot(n_rolls, pp_true, color='black', linestyle='--', label=c)
    plt.legend(title=r'Number of repetitions $R$')
    plt.xlabel('Number of rolls')
    plt.ylabel('Probability of getting a number R times')
    plt.show()







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
            pp.append(count_ocurrence(n, N_trials=1000, c=c))
            pp_true.append(5**(-c) * (5/6)**n * comb(n,c))
        p.append(pp)
        p_true.append(pp_true)

        plt.scatter(n_rolls, pp, label=c)
        plt.plot(n_rolls, pp_true, color='black', linestyle='--')
    plt.legend(title=r'Number of repetitions $R$')
    plt.xlabel('Number of rolls')
    plt.ylabel('Probability of getting a number R times')
    plt.show()

    # ================================================================================================================ #
    """
    (12)
    Spanish card deck (40 cards), you draw 5 cards.
        (a) Probability of getting the As de Oros
    """

    def draw_cards(n_cards, n_deck=40):
        cards = [c for c in range(n_deck)]
        aces = [0, 1, 2, 3]

        choice = np.random.choice(cards, size=n_cards, replace=False)

        # Is the As de Oros in the chosen cards?
        count_a = 1 if 0 in choice else 0

        s = 0
        for ace in aces:
            if ace in choice:
                s += 1

        # Is there only 1 Ace in the chosen cards?
        count_b = 1 if s == 1 else 0

        # Is there no Aces in the chosen cards?
        count_c = 1 if s == 0 else 0

        # Is there at least 1 Ace in the chosen cards?
        count_d = 1 if s >= 1 else 0

        return count_a, count_b, count_c, count_d

    pa, pb, pc, pd = np.zeros(4)
    for k in range(N_trials):
        count_a, count_b, count_c, count_d = draw_cards(5)
        pa += count_a
        pb += count_b
        pc += count_c
        pd += count_d

    pa /= N_trials
    pb /= N_trials
    pc /= N_trials
    pd /= N_trials

    print('\nProb of As de Oros: %.3f | True (1/8): %.3f' %(pa, 1/8.))
    print('\nProb of only 1 Ace: %.3f | True : %.3f' %(pb, 0.358))
    print('\nProb of no Aces: %.3f | True : %.3f' % (pc, 0.573))
    print('\nProb of at least 1 Ace: %.3f | True : %.3f' % (pd, 0.427))

    # ================================================================================================================ #

    def ball_boxes(n_balls, n_boxes):
        boxes = [n for n in range(n_boxes)]
        final = np.zeros(n_boxes)

        for i in range(n_balls):
            box = np.random.choice(boxes, size=1, replace=False)
            final[box] += 1

        total = np.sum(final)
        assert total == n_balls
        return final

    def check_boxes(boxes, boxes_with=1, value=1):
        # Check how many boxes with a certain value
        counts = 0
        for box in boxes:
            if box == value:
                counts += 1
        if counts == boxes_with:
            return 1
        else:
            return 0

    def theory(n_balls, n_boxes, p, value):
        """
        Probability of having [Value] balls in [p] boxes
        after putting n_balls in n_boxes at random
        """
        total = comb(n_boxes + n_balls - 1, n_balls)
        ways_to_chose_p_boxes = comb(n_boxes, p)

        n_boxes_new = n_boxes - p
        r_balls_new = n_balls - p*value
        ways_to_arrange_remaining = comb(n_boxes_new + r_balls_new - 1, r_balls_new)
        ways_to_arrange_remaining -= n_boxes_new * comb(n_boxes_new-1 + r_balls_new-value - 1, r_balls_new-value)

        ways_to_arrange_remaining = n_boxes_new * (r_balls_new -1)
        ways = ways_to_chose_p_boxes * ways_to_arrange_remaining
        prob = ways / total
        print(total)
        print(ways_to_arrange_remaining)
        return prob

    N_boxes = 10
    N_balls = 20
    N_trials = 100

    r = np.arange(1, N_boxes)
    v = 1
    prob = []
    for boxes_with in r:
        counts = 0
        for i in range(N_trials):
            counts += check_boxes(ball_boxes(N_balls, N_boxes), boxes_with, value=v)

        prob.append(counts / N_trials)

    plt.figure()
    plt.scatter(r, prob, s=4, label=v)
    plt.plot(r, theory(N_balls, N_boxes, r, v))
    plt.xlabel(r'Boxes with $n$ Ball(s)')
    plt.legend(title='$n$')
    plt.show()












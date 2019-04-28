import numpy as np
import matplotlib.pyplot as plt

""" 
### De Mere paradox ###
It is more likely to get at least an Ace in 4 rolls of a dice
than getting at least a double Ace in 24 rolls of 2 dices
"""

""" Numerical Approach """

def roll_dices(n_dices):
    roll = ''
    for i in range(n_dices):
        dice = np.random.choice(range(6), 1)[0]
        roll += str(dice)
    return roll

def roll_ndices_mtimes(n_dices, m_times):
    rolls = []
    for i in range(m_times):
        rolls.append(roll_dices(n_dices))
    return rolls

def check_aces(rolls, n_aces):
    aces = n_aces * '0'
    result = 0
    for r in rolls:
        if aces == r:
            result = 1
            break
    return result

def probability(n_dices, n_rolls, N_trials):
    counts = 0
    for i in range(N_trials):
        rolls = roll_ndices_mtimes(n_dices, n_rolls)
        counts += check_aces(rolls, n_dices)

    p = counts / N_trials
    return p

""" Theoretical Approach """

def theory(n_dices, n_rolls):
    p = 1. - (1. - (1/6.)**n_dices)**n_rolls
    return p

if __name__ == "__main__":

    n_dices = 1
    n_rolls = 4
    N_trials = 10000

    p_one = probability(n_dices=1, n_rolls=4, N_trials=N_trials)
    p_two = probability(n_dices=2, n_rolls=24, N_trials=N_trials)
    print('\nProbability of: | Numerical (Theory)')
    print('\nAt least 1 Ace after rolling 1 Dice(s) 4 times: %.4f (%.4f)' %(p_one, theory(1, 4)))
    print('\nAt least a Double Ace after rolling 2 Dice(s) 24 times: %.4f (%.4f)' %(p_two, theory(2, 24)))

    max_rolls = 100
    step = 5
    dices = [1, 2, 3]
    rolls = np.zeros((len(dices), max_rolls // step))

    plt.figure()
    for i, dice in enumerate(dices):
        print(dice)
        r = np.arange(1, max_rolls + 1, step)
        for j, n_roll in enumerate(r):
            rolls[i, j] = probability(dice, n_roll, N_trials=500)
        plt.scatter(r, rolls[i], label=dice)
        plt.plot(r, theory(dice, r), color='black', linestyle='--')

    plt.legend(title='Dice')
    plt.xlim([0, max_rolls])
    plt.ylim([0, 1])
    plt.xlabel('Number of Rolls')
    plt.show()







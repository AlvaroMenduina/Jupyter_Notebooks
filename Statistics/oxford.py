import numpy as np
import matplotlib.pyplot as plt

" PARAMETERS "
gain = 0.35
loss = 0.30

""" Numerical Approach - Simulating the games """
def game_round(capital):

    # Toss a coin
    toss = np.random.choice([0, 1], 1, replace=False)

    if toss == 0:
        capital *= 1. + gain
    if toss == 1:
        capital *= 1. - loss

    return capital

def play_n_games(initial_capital, n_rounds):
    capital = initial_capital
    capital_evolution = []
    for k in range(n_rounds):
        capital = game_round(capital)
        if capital <= 0.0:
            capital_evolution.append(0.0)
            break
        if capital > 0.0:
            capital_evolution.append(capital)

    return capital_evolution

""" Theoretical Approach """

def expectation(initial_capital, n_rounds):
    # Analytic formula for the Expectation

    exp = initial_capital *(1 + (gain - loss)/2) ** n_rounds
    return exp

def median_theor(initial_capital, n_rounds):
    # Analytic formula for the Median
    med = initial_capital / 2 * (1+gain)*(1-loss)*((1-loss)**(n_rounds-2) + (1+gain)**(n_rounds-2))
    return med


if __name__ == "__main__":

    """
    (1) A gambler makes a long sequence of bets against a rich friend. The gambler has initial
    capital C. On each round, a coin is tossed; if the coin comes up tails, he loses 30% of his
    current capital, but if the coin comes up heads, he instead wins 35% of his current capital.
    
        a) Let Cn be the gambler’s capital after n rounds. Write Cn as a product CY1Y2 . . . Yn
        where Yi are i.i.d. random variables. Find E Cn.
        
        b) Find the median of the distribution of C10 and compare it to E C10.
        
        c) Consider log Cn. What does the law of large numbers tell us about the behaviour of
        Cn as n → ∞? How is this consistent with the behaviour of E Cn?
    """
    initial_capital = 1.0
    N_trials = 10000
    N_games = 3
    rounds_played = []
    final_capital = []

    plt.figure()
    for k in range(N_trials):
        capital = play_n_games(initial_capital, N_games)
        rounds_played.append(len(capital))
        final_capital.append(capital[-1])
        plt.scatter(np.arange(1, len(capital)+1), capital, s=4)
    plt.yscale('log')

    # Expectation:
    num_exp = np.mean(final_capital)
    theo_exp = expectation(initial_capital, N_games)
    print('\nTheoretical Expectation after %d rounds: %.4f' %(N_games, theo_exp))
    print('Numerical Expectation: %.4f' %(num_exp))

    # Median
    num_med = np.median(final_capital)
    theo_med = median_theor(initial_capital, N_games)
    print('\nTheoretical Median after %d rounds: %.4f' %(N_games, theo_med))
    print('Numerical Median: %.4f' %(num_med))

    plt.show()

    y = []
    n = 10
    for i in np.arange(1, n):
        y.append(median_theor(1., i))
    plt.scatter(np.arange(1, n), median_theor(1., np.arange(1, n)))
    plt.scatter(np.arange(1, n), expectation(1., np.arange(1, n)))
    plt.show()



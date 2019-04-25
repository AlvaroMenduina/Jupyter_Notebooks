import numpy as np

""" 
6 cups and saucers come in pairs. Two cups and saucers are red, 2 are white, 2 have stars on.
We randomly place each cup onto a saucer. What is the probability that no cup is on a saucer 
of the same pattern
"""

def place_cups_onto_saucers():
    """
    Shuffle the CUPS, compare them to the SAUCERS
    if at least one CUP matches the pattern of its SAUCER: return failure
    if none of the CUPS matches the pattern of its SAUCER: return success
    :return:
    """

    cups = np.array([0, 0, 1, 1, 2, 2])
    saucers = cups.copy()
    np.random.shuffle(cups)

    result = 1
    for i in range(len(cups)):
        c, s = cups[i], saucers[i]

        if c == s:
            result = 0
            break

    # print('\nCups: ', cups)
    # print('Saucers: ', saucers)
    # print(result)

    return result


if __name__ == "__main__":

    counts = 0
    N_trials = 10000000
    for i in range(N_trials):
        counts += place_cups_onto_saucers()

    prob = counts / N_trials
    print(prob)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares as lsq

"""
Question:

Let R(n) be a random draw from [0, n-1]. What is the EXPECTED number iterations of R(n) needed to reach 0,
if we start at 10^100
"""

def R(n):
    """
    Draw a random number from [0, n-1],
    keep that number and use it for the next iteration
    """
    new_n = np.random.randint(0, n)
    return new_n

def iterations(N):
    """
    Simulates drawing samples from R(n) repeatedly
    N_1 = R(N_0)
    N_2 = R(N_1)
    until we reach 0
    :param N: starting point
    :return: the number of iterations needed to reach 0
    """
    i = N
    count = 1
    while i > 1:
        # print("\nDrawing a sample from [0, %d]" %(i-1))
        i = R(i)
        # print("Counts: ", count)
        # print("i: ", i)
        if i == 0:
            break
        else:
            count += 1
    return count

def log_model(nn, a, b):
    return a*np.log10(nn+1) + b

def residual(x, y_data, nn):
    a, b = x[0], x[1]
    return y_data - log_model(nn, a, b)

""" Theoretical Model for the Statistics """

"""
By setting up a decision tree with the different outcomes for small N,
we can infer the pattern of probabilities associated with the recursion
"""

def probabilities(n):
    """
    Computes both the ITERATIONS (it) need to reach 0 and the associated
    PROBABILITIES (prob) for each case of the decision tree for a given N

    Then the EXPECTED number of iterations is given by a simple expectation:
    EXPECTATION = \sum_k x_k P_k{x_k}
    :param n: starting point
    :return: PROBABILITIES and ITERATIONS
    """
    prob2 = np.array([1., 1.])
    it2 = [1, 2]

    if n == 2:
        p = 0.5*np.array(prob2)
        expect = np.dot(p, it2)
        return p, it2, expect

    if n > 2:
        prob = prob2
        it = it2
        k = 2
        while k < n:
            prob = np.concatenate([prob, prob / k])
            it = np.concatenate([it, it + np.ones_like(it)])
            k += 1
            # print(k)
        return (1/n) * np.array(prob), it

def expectation(n):
    """
    Recieves the PROBABILITIES and ITERATIONS from probabilities(n)
    and computes the EXPECTED number of iterations
    :param n: n: starting point
    :return: expected number of iterations needed to reach 0
    """
    print(n)
    p, i = probabilities(n)
    print('\nProbabilities: ', p[:10])
    print('Iterations: ', i[:10])
    expect = np.dot(p, i)
    return expect


if __name__ == "__main__":

    print(expectation(25))


    N = int(10**3)

    average = []
    nn = [i for i in np.arange(2, N, 10)]
    for n in nn:
        counts = []
        for i in range(100):
            counts.append(iterations(n))
        average.append(np.mean(counts))
    nn = np.array(nn)

    p, it = probabilities(50)

    x = np.arange(2, 25, 1)
    plt.figure()
    plt.scatter(nn, average, s=5)
    plt.plot(x, [expectation(y) for y in x], color='black')
    # plt.xscale('log')
    plt.xlabel(r'Starting point $N$')
    plt.ylabel('Iterations required')
    plt.show()

    #
    # x_solve = lsq(residual, x0=[1.0, 0.1], args=(average, nn,))
    # a, b = x_solve['x']
    # print(a, b)
    #





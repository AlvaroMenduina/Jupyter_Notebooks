import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares as lsq

"""
Question:

Let R(n) be a random draw from [0, n-1]. What is the EXPECTED number iterations of R(n) needed to reach 0,
if we start at 10^100
"""

def R(n):
    new_n = np.random.randint(0, n)
    return new_n

def iterations(N):
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

def n_iterations(n):
    it2 = [1, 2]
    if n == 2:
        return it2
    if n > 2:
        it = it2
        k = 2
        while k < n:
            it += [x + 1 for x in it]
            k += 1
        return it

def probabilities(n):
    prob2 = np.array([1., 1.])
    it2 = [1, 2]
    if n == 2:
        return 0.5*np.array(prob2), it2
    if n > 2:
        prob = prob2
        it = it2
        k = 2
        while k < n:
            # prob += [p / (k) for p in prob]
            prob = np.concatenate([prob, prob / k])
            # it += [x + 1 for x in it]
            it = np.concatenate([it, it + np.ones_like(it)])
            k += 1
            # print(k)
        return (1/n) * np.array(prob), it

def expectation(n):
    print(n)
    p, i = probabilities(n)
    expect = np.dot(p, i)
    return expect


if __name__ == "__main__":

    print(expectation(2))


    N = int(10**3)

    average = []
    nn = [i for i in np.arange(2, N, 10)]
    for n in nn:
        counts = []
        for i in range(100):
            counts.append(iterations(n))
        average.append(np.mean(counts))

    nn = np.array(nn)

    x_solve = lsq(residual, x0=[1.0, 0.1], args=(average, nn,))
    a, b = x_solve['x']
    print(a, b)

    x = np.arange(2, 100, 1)
    plt.figure()
    plt.scatter(nn, average, s=5)
    plt.plot(x, [expectation(y) for y in x], color='black')
    # plt.xscale('log')
    plt.xlabel(r'Starting point $N$')
    plt.ylabel('Iterations required')
    plt.show()





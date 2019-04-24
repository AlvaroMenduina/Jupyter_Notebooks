import numpy as np
import matplotlib.pyplot as plt

"""
Question:

If I split a stick of unit length into 3 random pieces, what is the expected
length of the largest piece?
"""

def split():
    points = np.sort(np.random.uniform(low=0.0, high=1.0, size=(2,)))

    lengths = [points[0], points[1] - points[0], 1 - points[1]]
    max_length = np.max(lengths)
    print(max_length)
    return max_length

if __name__ == "__main__":

    N_trials = 100000
    max_values = []
    for i in range(N_trials):
        max_values.append(split())

    print(np.mean(max_values))
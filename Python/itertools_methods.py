import numpy as np
import itertools

if __name__ == "__main__":

    ### Preliminaries

    # Zip(iter1, iter2)
    print('\nZip(iter1, iter2)')
    letters = ['A', 'B', 'C']
    numbers = [0, 1, 2]
    for l, n in zip(letters, numbers):
        print(l, n)

    # Map(fun, iter)
    print('\nmap(fun, iter) applies a 1-parameter function to each element in iter')
    words = ['Dog', 'Horse', 'Elephant']
    lengths = list(map(len, words))

    # Another map example
    result = list(map(sum, zip([1,2,3], [4,5,6])))

    ### Itertools

    # accumulate() -> itertools.accumulate(iterable[, func])
    def sum_funct(a, b):
        return a + b

    data = [1, 2, 3, 4, 5]
    result = itertools.accumulate(data, sum_funct)
    for r in result:
        print(r)

    # Combinations(iterable, r) returns iterator over combinations of r members
    colors = ['White', 'Black', 'Red', 'Blue']
    for c1, c2 in itertools.combinations(colors, 2):
        print(c1, c2)

    flags = list(itertools.combinations(colors, 2))

    # combinations with replacement: allows an element to be considered more than once
    for c1, c2, c3 in itertools.combinations_with_replacement(colors, 3):
        print(c1, c2, c3)

    flags = list(itertools.combinations_with_replacement(colors, 3))

    # count(start, step): it counts forever
    for i in itertools.count(10, 3):
        print(i)
        if i > 20:
            break

    # cycle: endlessly iterate over an iterable
    s = 0
    for i in itertools.cycle([0, 1, 2]):
        s += 2**i
        print('i: %d, sum: %d' %(i, s))
        if s > 500:
            break

    # Chain: takes a series of iterables and joins them into a long iterable
    even = np.arange(0, 10, 2)
    odd = np.arange(1, 10, 2)
    for i in itertools.chain(odd, even):
        print(i)


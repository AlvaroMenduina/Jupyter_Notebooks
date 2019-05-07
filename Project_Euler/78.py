

import numpy as np
import itertools

def arrange(n_coins):

    ways = 0
    for n_boxes in np.arange(1, n_coins + 1):
        # print("\nBoxes: ", n_boxes)

        boxes = [str(num) for num in np.arange(1, n_boxes+1)]
        counts = []
        for perm in list(itertools.combinations_with_replacement(boxes, n_coins)):
            p = ''
            for a in perm:
                p += a
            b = p.split('|')
            # print(b)
            count = [b[0].count(box) for box in boxes]
            # print(count)
            count.sort()
            if 0 not in count and count not in counts:
                counts.append(count)
        # print(len(counts))
        ways += len(counts)
    return ways




if __name__ == "__main__":

    n_coins = 8
    print(arrange(n_coins))
    # for n in np.arange()


        # print(np.unique(arrangements))

    # from math import factorial as fact
    #
    #
    # def combinations_with_rep(boxes, coins):
    #
    #     C = fact(boxes + coins - 1) / fact(coins) / fact(boxes - 1)
    #     return C
    #
    #
    # def p(n_coins):
    #     for n_boxes in np.arange(1, n_coins + 1):
    #         print("\nBoxes:", n_boxes)
    #         C = combinations_with_rep(n_boxes, n_coins - n_boxes)
    #
    #         print(C)

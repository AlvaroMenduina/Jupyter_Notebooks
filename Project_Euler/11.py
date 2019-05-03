"""
Project Euler - Problem 11

In the 20×20 grid below, four numbers along a diagonal line have been marked in red.

08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48

The product of these numbers is 26 × 63 × 78 × 14 = 1788696.

What is the greatest product of four adjacent numbers in the same direction
(up, down, left, right, or diagonally) in the 20×20 grid?

"""

import numpy as np

def product(array, n_terms, direction='Vertical'):
    """
    Function that computes the product of n_terms from a given array
    in different directions: horizontal, vertical and diagonal
    :param array:
    :param n_terms: number of terms in the product
    :param direction: axis for the product
    :return: a tensor containing the result of the product (and its terms for validation)
    """
    N, M = array.shape
    if direction == 'Vertical':
        n, m = (N + 1) - n_terms, M
    elif direction == 'Horizontal':
        n, m = N, (M + 1) - n_terms
    elif direction == 'Diagonal' or direction =='2nd Diagonal':
        n, m = (N + 1) - n_terms, (M + 1) - n_terms

    # Results are saved in an (n, m) tensor where the each (i,j) contains
    # [Prod(d), d1, ..., d_k, ] the product and the terms of the product
    result = np.zeros((n, m, n_terms + 1))
    if direction != '2nd Diagonal':
        for i in range(n):
            for j in range(m):
                if direction == 'Vertical':
                    digits = array[i:i+n_terms, j]
                    print(digits)
                elif direction == 'Horizontal':
                    digits = array[i, j:j+n_terms]
                    print(digits)
                elif direction == 'Diagonal':
                    digits = [array[i+a, j+a] for a in range(n_terms)]
                    print(digits)
                p = np.product(digits)
                print("Product: ", p)
                result[i, j, 0] = p
                result[i, j, 1:] = digits
    if direction == '2nd Diagonal':
        for i in np.arange(n, 0, -1):
            for j in range(m):
                digits = [array[i - a, j + a] for a in range(n_terms)]
                print(digits)
                p = np.product(digits)
                print("Product: ", p)
                result[i-n, j, 0] = p
                result[i-n, j, 1:] = digits
    return result


if __name__ == "__main__":

    ### Example
    A = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

    print("\nExample with a matrix A: ")
    print(A)

    print("\nProduct Horizontally")
    product(A, 2, 'Horizontal')

    print("\nProduct Vertically")
    print(A)
    product(A, 2, 'Vertical')

    print("\nProduct Diagonally")
    print(A)
    product(A, 2, 'Diagonal')

    print("\nProduct 2nd Diagonal")
    print(A)
    product(A, 2, '2nd Diagonal')

    h, v, d, dd = product(A, 2, 'Horizontal'), product(A, 2, 'Vertical'), product(A, 2, 'Diagonal'), product(A, 2, '2nd Diagonal')
    h_max = max([np.max(array[:,:,0]) for array in [h, v, d, dd]])

    ### Solution
    grid = np.loadtxt('11.txt')
    list_ways = ['Horizontal', 'Vertical', 'Diagonal', '2nd Diagonal']
    products = [product(grid, 4, way) for way in list_ways]

    # Compute the maximum for each method
    maxima = [np.max(array[:, :, 0]) for array in products]
    # Find the indices (i,j) of the max for each method (so we can retrieve the terms)
    arg_max = [np.argwhere(array[:, :, 0] == max_val) for (array, max_val) in zip(products, maxima)]
    index_sol = np.argmax(maxima)

    # Information on the solution
    solution = maxima[index_sol]
    i, j = arg_max[index_sol][0][0], arg_max[index_sol][0][1]
    solution_direction = list_ways[index_sol]

    print("\nSolution: %d " %int(solution))
    print("Direction: ", solution_direction)
    print("with terms: ", products[index_sol][i, j, 1:])



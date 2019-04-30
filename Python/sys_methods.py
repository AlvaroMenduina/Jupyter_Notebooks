import sys
import numpy as np
import os

"""
Pass the name of a txt file and N: how many of the numbers in the file to add together
"""

if __name__ == '__main__':

    print('\nsys.version and sys.version_info')
    print(sys.version)
    print(sys.version_info)

    print("\nsys.argv[0] is the name of the script")
    print(sys.argv[0])

    print("Number of arguments: ", len(sys.argv))
    # print("The arguments are: ", str(sys.argv))

    file_name, N = sys.argv[1], int(sys.argv[2])
    numbers = np.loadtxt(os.path.join(file_name))
    s = 0
    for number in numbers[:N]:
        s += number
    print('Summing the first %d numbers. Sum = %.f' %(N, s))
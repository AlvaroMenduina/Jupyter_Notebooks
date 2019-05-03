"""
Project Euler - Problem 13

Work out the first ten digits of the sum of the following one-hundred 50-digit numbers.

37107287533902102798797998220837590246510135740250
46376937677490009712648124896970078050417018260538
74324986199524741059474233309513058123726617309629
91942213363574161572522430563301811072406154908250
23067588207539346171171980310421047513778063246676
89261670696623633820136378418383684178734361726757
28112879812849979408065481931592621691275889832738
44274228917432520321923589422876796487670272189318
47451445736001306439091167216856844588711603153276
70386486105843025439939619828917593665686757934951
62176457141856560629502157223196586755079324193331
64906352462741904929101432445813822663347944758178
92575867718337217661963751590579239728245598838407
58203565325359399008402633568948830189458628227828
80181199384826282014278194139940567587151170094390
35398664372827112653829987240784473053190104293586
86515506006295864861532075273371959191420517255829
71693888707715466499115593487603532921714970056938
54370070576826684624621495650076471787294438377604
53282654108756828443191190634694037855217779295145
36123272525000296071075082563815656710885258350721

                    ...

"""

import numpy as np


def sum_numbers(number_list):
    """
    Function that computes the digits of the sum of a number_list
    Suitable for sums of large number that can't be represented
    as integers in Python
    :param number_list: list of long digits (in string format)
    :return: a list of the digits of the sum
    """
    n_digits = len(number_list[0])
    indices = list(range(n_digits))[::-1]  # Indices of the digits from last to first
    digits_of_sum = []
    tens = 0

    for i in indices:
        digits = [int(number[i]) for number in number_list]
        digits.append(tens)  # Add the tens from the previous iter
        sum_of_digits = sum(digits)
        # The sum will be something like 125 where 5(dig) 120(tens) that we carry to the next iter
        dig = sum_of_digits % 10
        digits_of_sum.append(dig)
        tens = (sum_of_digits - dig) // 10
    # Remember to flip the list digits_of_sum (we are starting from the last digit
    result = digits_of_sum[::-1]
    # Remember to add the last extra ten
    result[0] += tens * 10

    # Join the digits into a number
    num_str = [str(n) for n in result]
    lead = [s for s in num_str[0]]
    lead.extend(num_str[1:])

    return lead

if __name__ == "__main__":

    ### Example for validation

    N = 50
    rand_nums = np.random.randint(low=1000, high=9999, size=(N,))

    true_solution = np.sum(rand_nums)
    print("\nExample for Validation: True solution (directly summing): ", true_solution)

    nums = [str(n) for n in rand_nums]
    result = sum_numbers(nums)
    print("\nSolution with our method: ", result)

    ### Solution
    numbers = np.loadtxt('13.txt', dtype=str)
    sum_num = sum_numbers(numbers)

    print("\nProblem 13 - The first 10 digits of the sum are:")
    print(sum_num[:10])



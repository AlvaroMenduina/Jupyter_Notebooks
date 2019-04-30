"""
### Interesting Python things to know ###

Things that I don't normally use, related to LISTs



"""


if __name__ == "__main__":

    # List operations
    print('\nMethod append(x) appends a new ELEMENT at the end of list')
    l0 = [1, 2, 3, 4]
    print('Initial list: ', l0)
    l0.append(5)
    print('Append new element 5: ', l0)

    print('\nMethod extend(x) appends a new LIST x at the end')
    l0.extend([7, 8])
    print('Append new list [7, 8]: ', l0)

    print('\nMethod count(x) counts ocurrences of x in the list')
    print(l0.count(1))

    print('\nMethod index(x) returns the smallest index i were list[i] == x')
    print(l0.index(3))

    print('\nMethod insert(i, x) inserts a value x at a particular index')
    l0.insert(0, 10)
    print(l0)

    print('\nMethod pop([i]) pops the element of index i')
    l0.pop(0)
    print(l0)

    print('\nMethod remove(x) removes the first ocurrence of x')
    print(l0.remove(1))
    print(l0)

    print('\nPOP(index) and REMOVE(value) are equivalent, but one deals with index and the other with value')

    print('\nMethod sort([key], [reverse])')
    print('KEY must be a function such as LEN')
    l1 = ['AA', 'BBB', 'CCCC', 'D']
    print('Initial list: ', l1)
    print('Sort according to: ', len)
    l1.sort(key=len)
    print(l1)

    l2 = [[0, 1], [1, 1], [0, 0], [0.5, 0.5]]
    print('Initial list: ', l2)
    print('\nSort according to the norm of each vector')
    l2.sort(key=lambda x: np.sqrt(x[0]**2 + x[1]**2))
    print(l2)



    ### Accesing lists
    arr = [1, 20, 13, 47, 95]
    print('\nAccess list in reverse with REVERSED')
    for i in reversed(arr):
        print(i)

    print('\nAccess list but sorted with SORTED(reverse=Bool)')
    for i in sorted(arr, reverse=False):
        print(i)

    print('\nAccessing info from a list [name, grades, age]')
    student = ["Tom", 90, 95, 98, 30]
    name, *marks, age = student
    print('Student info: ', student)
    print('Marks', marks)

    y = (1, 'Two', 3, ('Five', 'Six', 'Seven'))
    a, *b, (*c, d) = y


    ### Looping over Dictionaries
    dic = {'Cats': 10, 'Dogs': 12, 'Birds': 1}
    print('\nDoing "for x in dic" prints the KEYS')
    for d in dic:
        print(d)

    print('\nDictionary.items() returns both KEY and VALUE')
    for key, item in dic.items():
        print(key, item)

    # Creating a dictionary from 2 lists with ZIP
    print('\nCreating a a dictionary from list of keys and items')
    keys = ['Cats', 'Dogs', 'Birds']
    values = [10, 45, 76]
    dc = dict(zip(keys, values))
    print(dc)

    # Create it with ZIP(ENUMERATE(list))
    print('\nCreating a a dictionary from ENUMERATE and a list of values')
    d = dict(enumerate(values))
    print(d)


    ### Logic
    arr = [1, 2, 3, 4, 5]
    print('\nFunction all() checks whether all statements fulfill condition')
    print('all(item > 0 for item in x)')
    all(item > 0 for item in arr)

    print('\nIf one condition is not fulfilled, it returns False')
    all(item < 4 for item in arr)

    print('\nFunction any() returns TRUE if at least one condition is met')
    any(item > 4 for item in arr)   # 5 is larger than 4





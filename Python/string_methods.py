"""
### Interesting Python things to know ###

Things that I don't normally use, related to STRINGS



"""
import numpy as np

if __name__ == "__main__":

    ### Capitalize a string
    s = 'alvaro'
    print(s, ' capitalize it: ', s.capitalize())

    ### Center
    s = 'ABC'
    print('\nMethod center(width, [pad]) centers the string in a field of length width. pad is the padding character')
    print(s.center(5, '0'))

    ### Count
    s = 'AABABBCABC'
    print('\nMethod count(sub) counts the occurrence of a specified substring')
    print(s.count('A'))

    ### Endswith
    s = 'Generalization'
    print('\nMethod endswith(suffix) checks for the end for a suffix')
    print(s + ' endswith(ation): ', s.endswith('ation'))

    ### Find
    letters = ['A', 'B', 'C']
    s = ''
    for i in range(1000):
        s += letters[np.random.choice(3, 1)[0]]
    sub_str = 'ABC'
    print('\nMethod find(substring) finds the first occurrence of a substring')
    print('Sequence: ', s[:10])
    i = s.find(sub_str)
    print('Find the first occurrence of ABC: ', i)
    print('s[i-1:i+4]', s[i-1:i+4])

    j = s.rfind(sub_str)
    print('Find the last occurrence of ABC: ', j)
    print('s[j-1:j+4]', s[j-1:j+4])

    print('\nMethod index(substring) is identical to find(substring) except that index returns exception if not found')

    ### Character types
    s = 'ABC123'
    ss = s + '%$£'
    print('\nMethod isalnum() checks whether all chars are ALPHANUMERICS')
    print('Is ' + s + ' alphanumeric? ', s.isalnum())
    print('Is ' + ss + ' alphanumeric? ', ss.isalnum())

    print('\nMethod isalpha() checks whether all chars are ALPHABETS')
    print('Is ' + s + ' alphabetic? ', s.isalpha())

    print('\nMethod isdigit() checks whether all chars are DIGITS')
    print('Is ' + s + ' digits? ', s.isdigit())

    title = 'The Lord Of The Rings'
    non_title = 'The lord of the rings'
    print('\nMethod istitle() checks whether all chars are TITLECASED')
    print('Is ' + title + ' titlecased?', title.istitle())
    print('Is ' + non_title + ' titlecased?', non_title.istitle())

    print('\nMethod isupper() checks whether all chars are UPPERCASE')
    title = 'ALIEN'
    print('Is ' + title + ' uppercase?', title.isupper())

    print('\nMethod swapcase() changes the case of the string')
    s = 'Alvaro'
    print('Initial string: ' + s)
    print('Swapped: ' + s.swapcase())

    print('\nMethod title() returns the title-case of string')
    s = 'how are you doing?'
    print(s)
    print(s.title())

    ### String operations

    ### Join
    sep = '_0_'
    joined = sep.join(letters)
    print('\nMethod sep.join(iterable) joins items in iterable with sep as separator')
    print('Items to join: ', letters)
    print('Separator: ', sep)
    print('Joined: ', joined)

    ### Just (ljust, rjust)
    s = '1234'
    print('\nMethod (l,r)just(width, [fill]) aligns the string with size width and pads with [fill]')
    print('ljust string ' + s + ' to size 10 and fill with zeros: ', s.ljust(10, '0'))
    print('rjust string ' + s + ' to size 10 and fill with zeros: ', s.rjust(10, '0'))

    ### lstrip
    s = ' ABC'
    print('\nMethod lstrip([chrs]) removes leading white space of [chrs] if provided')
    print(s)
    s2 = s.lstrip()
    print(s2)
    print('Strip A: ', s2.lstrip('A'))

    ### strip
    print('\nMethod strip([chars]) removes both leading and trailing chars')
    s = '00ABC00'
    print('Initial string: ' + s)
    print('Strip(0): ', s.strip('0'))

    ### partition
    s = '123456_ABCDE'
    print('\nMethod partition(sep) split the string based on the separator (if found)')
    print('Partition ' + s + ' separator=_', s.partition('_'))
    print('Partition ' + s + ' separator=0', s.partition('0'))

    ### replace
    s = 'ABCD_01234'
    print('\nMethod replace(old, new) replaces a substring for other')
    s2 = s.replace('AB', '01')
    print('Replace AB in ' + s + ' for 01', s2)

    s = 'AB_01_AB_01_AB_01'
    print('\nReplace(old, new, [maxreplace]) limits the number of times the string is replaced')
    print('Initial string: ', s)
    print('replace(AB, XX, 1 time): ', s.replace('AB', 'XX', 1))

    ### split
    print('\nMethod split(sep) splits the string given a separator')
    print('Initial string: ' + s + ' separator=_')
    print(s.split('_'))




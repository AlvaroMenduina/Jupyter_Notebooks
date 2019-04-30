"""
### Interesting Python things to know ###

Things that I don't normally use, related to DICTIONARIES

"""

if __name__ == "__main__":

    market = {'Apples': 10,
              'Pears': 12,
              'Melon': 2}
    m = market.copy()

    print('Len(dictionary) returns number of items: ', len(market))

    ### Deleting a key
    print('\nDeleting a KEY ')
    print(market)
    print('\nDeleting Melon (del dictionary[key])')
    del market['Melon']
    print(market)

    ### Clearing the whole dictionary
    print('\nMethod dictionary.clear() removes all items')
    market.clear()
    print(market)

    # Restore the dictionary
    market = m.copy()

    ### ITEMS, KEYS, VALUES
    print('\nDictionary.keys() returns keys')
    print(market.keys())
    print('Dictionary.values() returns values associated with keys')
    print(market.values())
    print('Dictionary.items() returns both keys and values')
    print(market.items())
    for key, item in market.items():
        print(key, item)

    # Creating a dictionary from another dictionary
    market = {'Apples': 10,
              'Pears': 12,
              'Melon': 2,
              'Ribs': 7,
              'Cod': 5}
    fruit_keys = ['Apples', 'Pears', 'Melon']
    values = {market[key] for key in fruit_keys}
    fruit = market.fromkeys(fruit_keys, values)
    # Pretty stupid because it assigns the SAME value to all keys

    ### POP(key) gets the value of a key and removes it from the dictionary
    # similar to delete but POP provides the value as output
    rem = market.pop('Apples')

    ### GET(key) same as POP but without removing the item
    pears = market.get('Pears')

    ### dictionary.UPDATE(another_dict) updates all objects of another_dict to m
    print('Initial Dictionary: ', market)
    market.update(fruit)
    print('Updated with FRUIT: ', market)








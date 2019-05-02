import numpy as np
import pandas as pd

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------- #
    #                                         SERIES                                      #
    # ----------------------------------------------------------------------------------- #

    ### Creating Series
    data_series = pd.Series(data=np.random.uniform(low=0, high=1, size=10),
                            index=list(range(10)), name='RandomValues')

    # You can access the Values (data) and the Indices
    print('DataSeries: \n')
    print(data_series)

    name = data_series.name
    indices = data_series.index
    values = data_series.values
    print('\nName of the Series: ', name)
    print('\nIndices: ', indices)
    print('\nValues: ', values)

    ### Creating a Series from a dictionary
    # It assigns the KEYS of the Dictionary to the INDICES of the Series

    fruit_dict = {'Apples': 15, 'Pears': 21, 'Oranges': 39}
    data_fruits = pd.Series(data=fruit_dict, name='Fruits')
    print('\nDataSeries from Dictionary (Fruits):')
    print(data_fruits)
    print('\nIndices: ', data_fruits.index)

    ### How Pandas deals with index when creating a DataSeries from a Dictionary
    age_dict = {'Amy': 25,
                'Adam': 23,
                'Aaron': 29,
                'Andrea': 30}
    # Create a Series from the dictionary with a list of indices
    age_series = pd.Series(age_dict, index=['Amy', 'Aaron', 'Noa'])

    # Pandas extracts the data from the dict for the Index that are present, and fill the rest with Nan
    print(age_series)

    ### Appending two Series together
    vegetable_dict = {'Onions': 13, 'Carrots': 45, 'Leeks': 2}
    vegetable_series = pd.Series(data=vegetable_dict)
    market = data_fruits.append(vegetable_series)

    # ----------------------------------------------------------------------------------- #
    #                                       DATAFRAMES                                    #
    # ----------------------------------------------------------------------------------- #

    ### Creating a simple DataFrame

    data = {'One': pd.Series([1., 2., 3.],
                             index=['a', 'b', 'c']),
            'Two': pd.Series([10., 20., 30., 40.],
                             index=['a', 'b', 'c', 'd'])}

    df = pd.DataFrame(data)
    print(df)
    print('\n')

    ### Create a DataFrame using a zip(list1, list2)
    names = ['Bob', 'Jessica', 'Mary', 'John']
    ages = [23, 34, 16, 49]
    age_data = list(zip(names, ages))

    df_ages = pd.DataFrame(data=age_data, columns=['Names', 'Ages'],
                           index=np.arange(1, len(age_data)+1))
    print(df_ages)
    print('\n')
    df_ages.to_csv('ages.csv', index=False, header=False)

    ### Load the CSV
    df_load = pd.read_csv('ages.csv', header=None, names=['Names', 'Ages'])
    print(df_load)
    print('\n')

    # df.head(n) shows the first n lines, df.tail(n) the last n lines

    print(df_load.values)   # df.values shows the data
    print(df_load.index)    # indices

    ### Operating with the data
    max_age = df_ages['Ages'].max()
    # Find the name of the oldest person
    oldest = df_ages['Names'][df_ages['Ages'] == max_age].values

    ### Adding an extra column
    df_ages['NewCol'] = 5
    print('\nAdding a new column with df[str] = value')
    print(df_ages)
    print('\n')

    ### Deleting a column
    del df_ages['NewCol']

    ### Editing the indices
    new_index = ['a', 'b', 'c', 'd']
    df_ages.index = new_index
    print('\nEditing the indices')
    print(df_ages)
    print('\n')

    ### Accessing by ROW index
    first_person = df_ages.loc['a']
    first = df_ages.iloc[0]
    print('Finding the first person')
    print('.loc[first_index]: ', first_person.values)
    print('\n.iloc[0]: ', first.values)

    ### Accessing by COLUMN
    people = df_ages['Names']
    print('\nPeople: ', people.values)

    ### Get Value of specific ROW / COLUMN pair
    first_name = df_ages.at['a', 'Names']
    first = df_ages.iat[0, 0]   # If you don't know the index labels

    ### Dealing with missing data
    df_ages.loc['b']['Ages'] = np.nan   # Assume we missed that data

    ### Find the missing spots
    mask_null = df.isnull()
    # Replace it with 0
    df.fillna(0)
    print(df_ages)
    print('\n')







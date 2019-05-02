import pandas as pd
import numpy as np

if __name__ == "__main__":

    ### Manipulating DataFrames with Pandas

    indices = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    eggs = pd.Series(data=[47, 110, 221, 77, 132, 205], name='eggs')
    salt = pd.Series(data=[12., 50., 89., 87., np.nan, 60.0], name='salt')
    spam = pd.Series(data=[17, 31, 72, 20, 52, 55], name='spam')

    df = pd.DataFrame(data=[eggs, salt, spam])
    df = df.transpose()
    df.index = indices

    print('\nInitial DataFrame')
    print(df)

    ### Indexing the DataFrame

    # You first access the Series by its name, and then the index
    salt_jan = df['salt']['Jan']
    print("\nAccessing the salt in January df['salt']['Jan']: ", salt_jan)

    # You can also access it by its name, as an Attribute of the DataFrame
    eggs_march = df.eggs['Mar']
    print("\nAccessing the eggs in March df.eggs['Mar']: ", eggs_march)

    # Using .loc access. Here you first access the Key and then the Column Name
    spam_may = df.loc['May', 'spam']
    print("\nAccessing the spam in May df.loc['May', 'spam']: ", spam_may)

    # Using .iloc access
    print("\nAccessing the spam in May df.iloc[4, 2]: ", df.iloc[4, 2])

    # ===================================================================================== #

    ### Slicing the DataFrame

    # df[list of column names]
    df_slice = df[['salt', 'eggs']]
    print("\nDataframe with only Salt and Eggs: ")
    print(df_slice)

    # Important difference: df["name"] is a Series (selecting column of "name")
    # whereas df[["name"]] is a DataFrame, slicing the original to a single column

    print("\ndf['name'] is a Series || df[['name']] is a DataFrame with only 1 column")
    print(type(df['eggs']))
    print(type(df[['eggs']]))

    # Slicing all. You can do a range with the column names
    df_slice = df.loc[:, 'eggs':'salt']
    print("\nDataframe sliced df.loc[:, 'eggs':'salt']: ")
    print(df_slice)

    # Slicing across all.
    df_slice = df.loc['Jan':'Mar', 'eggs':'salt']
    print("\nDataframe sliced df.loc['Jan':'Mar', 'eggs':'salt']: ")
    print(df_slice)

    # ===================================================================================== #

    ### Filtering the DataFrame
    salt_mask = df.salt > 60
    print('\nMonths when we had enough salt df.salt > 60')
    print(salt_mask)

    # Showing the DataFrame bits that fulfill the condition
    enough_salt = df[salt_mask]
    print('\ndf[df.salt > 60]')
    print(enough_salt)

    # Combining filters
    enough_salt_eggs = df[(df.salt >= 50) & (df.eggs < 200)]
    print("\nEnough salt and eggs: df[(condition1) & (condition2)]")

    # ===================================================================================== #

    ### Dealing with Zeros and NaNs

    df2 = df.copy()
    df2['bacon'] = [0, 0, 50, 60, 70, 80]

    # df.all() True if all elements within a series or along a dataframe axis are non-zero, not-empty or not-False
    print("\nAre the elements of each column / Series non-zero?")
    print(df2.all())

    # Select columns without zeros
    df_nonzero = df2.loc[:, df2.all()]
    print("\nColumns without zeros (df.loc[:, df.all()])")
    print(df_nonzero)

    # Select columns with ANY non-zero
    print("\nColumns with any non-zeros (df.loc[:, df.any()])")
    print(df2.loc[:, df2.any()])

    # Finding columns with ANY NaNs. df.isnull.any()
    print("\nColumns with any NaN (df.loc[:, df.isnull().any()])")
    print(df.loc[:, df.isnull().any()])

    # Finding columns without ANY NaNs. df.notnull().all()
    print("\nColumns without any NaN (df.loc[:, df.notnull().all()])")
    print(df.loc[:, df.isnull().any()])

    # Drop rows with any NaNs
    df_no_nans = df2.dropna(how='any')

    # Filtering a column based on another
    df_eggs_with_salt = df2.eggs[df2.salt > 55]
    print("\nEggs for the months where we had enough salt")
    print(df_eggs_with_salt)

    # Modifying a columns based on another
    df2.eggs[df2.salt > 55] += 5

    # ===================================================================================== #

    ### Applying functions to DataFrames

    # Transform the values into dozens
    df_dozen = df2.floordiv(12)
    print("\nDataframe in dozens:")
    print(df_dozen)

    # You can apply any function
    def dozens(x):
        return x//12

    print("\nDataframe in dozens: df.apply(function)")
    print(df2.apply(dozens))

    print("\nDataframe in dozens: df.apply(lambda x: x//12)")
    print(df2.apply(lambda x: x//12))

    # Storing a transformation
    df2['Dozens_eggs'] = df2.eggs.apply(dozens)
    print("\nAdding the transformation as extra column")
    print(df2)




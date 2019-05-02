import pandas as pd

if __name__ == "__main__":

    # Url of the Data
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data')
    ### This dataset contains information on abalone a type of sea snail

    cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
    print("\nDownloading data from URL")
    abalone = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)

    print("\nResulting pd.DataFrame")
    print(abalone.head())

    ### Group by quartiles of the number of rings
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    abalone['ring_quartile'] = pd.qcut(abalone.rings, q=4, labels=labels)
    grouped = abalone.groupby('ring_quartile')

    for q, qframe in grouped:   # q are the labels for each group
        print("\nRing quartile: %s" %q)
        print('-' * 20)
        print(qframe.head())
        print('-' * 20)
        print("\nTop 3 largest abalones:")
        print(qframe.nlargest(n=3, columns='length'))

    # You can also access the labels by calling grouped.groups.keys()
    keys = grouped.groups.keys()

    # Group.groups contains a Dictionary of the form {'label1: Index([i1, i2, i3...], 'label2': Index...}
    index1 = grouped.groups[labels[0]]  # The Index of the elements in the first quartile

    df_q1 = abalone.iloc[index1, :]

    print("\nAccessing Q1: abalone.iloc[grouped.groups[labels[0]], :]")
    print(df_q1.head())

    ### Aggregate the elements of each quartile
    df_agg = grouped['height', 'weight'].agg(['mean', 'median'])



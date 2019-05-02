import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Url of the Data
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data')
    ### This dataset contains information on abalone a type of sea snail

    cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
    print("\nDownloading data from URL")
    abalone = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)

    print("\nResulting pd.DataFrame")
    print(abalone.head())

    ### Indexing and slicing the Data

    # SEX: {M: male, F: female, I: infant
    # (1) Find out how the sex data is categorized
    sexes = abalone['sex'].unique()
    print("\nSex is categorized as: ", sexes)

    # (2) How many abolones of each sex
    male = abalone[abalone['sex'] == 'M']
    n_males = male.shape[0]

    # Compute all sexes at the same time
    n_sexes = [abalone[abalone['sex'] == sex].shape[0] for sex in sexes]
    print("\nHow many abalones of each sex?")
    for number, sex in zip(n_sexes, sexes):
        print("%s : %.d" %(sex, number))

    # (3) Group by sex
    sex_grouped = abalone.groupby('sex')
    print("\nGrouping them by sex")
    for label, group in sex_grouped:
        print("\nGroup: ", label)
        print(group.head())

    # (4) Show some data
    plt.figure()
    ring_min, ring_max = abalone['rings'].min(), abalone['rings'].max()
    for label, group in sex_grouped:
        plt.hist(group['rings'], histtype='step', label=label)
    plt.legend(title='Abalone Sex Group')
    plt.xlim([ring_min - 1, ring_max + 1])
    plt.title('Number of Rings')

    plt.figure()
    weight_min, weight_max = abalone['weight'].min(), abalone['weight'].max()
    for label, group in sex_grouped:
        plt.hist(group['weight'], histtype='step', label=label)
    plt.legend(title='Abalone Sex Group')
    plt.xlim([0.0, 1.1*weight_max])
    plt.title('Weight')
    plt.show()

    # Weight - Length correlation
    plt.figure()
    len_min, len_max = abalone['length'].min(), abalone['length'].max()
    for label, group in sex_grouped:
        plt.scatter(group['weight'], group['length'], label=label, s=10)
    plt.legend(title='Abalone Sex Group')
    plt.xlabel('weight')
    plt.ylabel('length')
    plt.xlim([0.0, weight_max])
    plt.ylim([0.0, len_max])
    plt.show()

    # Plot All correlations between the data columns
    column_names = abalone.columns[1:-1]
    import itertools
    comb = itertools.combinations(column_names, r=2)    # All possible combinations of (column1, column2)
    indices = itertools.product(range(2), range(3))     # Cartesian Product of indices for plot

    fig, axes = plt.subplots(nrows=2, ncols=3)
    for (column1, column2), (i, j) in zip(comb, indices):
        print(i, j)
        for label, group in sex_grouped:
            axes[i, j].scatter(group[column1], group[column2], label=label, s=10)
            axes[i, j].set_xlabel(column1)
            axes[i, j].set_ylabel(column2)
    axes[0, 0].legend(title='Sex Group')
    plt.show()

    # (5) Males with more than 10 rings and more than 1.0 of weight
    mask = (abalone['sex'] == 'M') & (abalone['weight'] >= 1.0) & (abalone['rings'] >= 10)
    large_males = abalone[mask]

    plt.figure()
    plt.hist(large_males['weight'], histtype='step')
    plt.xlim([0.0, 1.1 * weight_max])
    plt.title('Weight of Large Males (> 10 rings)')
    plt.show()

    # -------------------------------------------------------------------------------------- #

    ### Operating with Data

    # (6) Applying a Function to each column
    print("\nShowing Mean / Median values for each sex group")
    for label, group in sex_grouped:
        print("\nGroup: ", label)
        stats = group.agg(['mean', 'median'])
        print(stats)

    # (7) Showing the nlargest of each group
    print("\nShowing N largest for each sex group")
    for label, group in sex_grouped:
        large = group.nlargest(n=5, columns=['length'])
        print("\nGroup: ", label)
        print(large)

    # (8) Creating a new column with available data
    print("\nCreating a new column with some data")
    for label, group in sex_grouped:
        group['weight_length_ratio'] = group['weight'] / group['length']
        print("\nGroup: ", label)
        print(group.head())

    def mm_to_inches(x):
        return x * 0.03937

    for label, group in sex_grouped:
        group['length_in'] = group.length.apply(mm_to_inches)
        # Rearrange the column order
        old_columns = group.columns
        new_columns = old_columns.insert(2, old_columns[-1])
        new_columns = new_columns[:-1]
        print("\nGroup: ", label)
        print(group[new_columns].head())

    # -------------------------------------------------------------------------------------- #

    ### (***) Compare memory usage
    sex_dict = {sex: integer for integer, sex in enumerate(sexes)}
    # We can Categorize the Sex as integers to save Memory (see mapping_trick.py)
    print("\nCategorize sex as integers to save memory:")
    print(sex_dict)

    print("\nComparing memory usage for sex: string (M, F, I) vs integer (0, 1, 2)")
    print(abalone['sex'].memory_usage(index=False, deep=True))
    print(abalone['sex'].astype('category').memory_usage(index=False, deep=True))

    # -------------------------------------------------------------------------------------- #

    ### Group by quartiles of the number of rings
    print("\nGrouping data by Quartiles based on Number of Rings")
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



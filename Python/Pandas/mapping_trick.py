import pandas as pd
import sys

### Trick for Membership Binning

if __name__ == "__main__":

    ### ======================================================================= ###
    #         Trick 1 - Saving Memory by transforming to Categorical data         #
    ### ======================================================================= ###

    colors = pd.Series(['dark blue',
                        'light green',
                        'violet',
                        'dark blue',
                        'dark blue',
                        'light green',
                        'orange',
                        'orange'])

    print("\nA Pandas Series of colors: ")
    print(colors)

    # Compute how much memory (bytes) it takes as strings
    mem_str = colors.apply(sys.getsizeof)
    print("\nMemory: colors.apply(sys.getsizeof)")
    print(mem_str)

    # Array containing the unique values in colors
    unique_colors = colors.unique()

    # Create dictionary linking each unique color to an integer value
    print("\nUnique colors:")
    for i, color in enumerate(unique_colors):
        print(i, color)

    # The Dictionary associates an Integer to Each color
    color_map = {key: value for value, key in enumerate(unique_colors)}
    print("\nColor dictionary")
    print(color_map)

    # Apply a Map that substitutes the color names in the Series by the integers in dictionary
    as_int = colors.map(color_map)
    # keep track of everything in a df
    df_colors = pd.DataFrame({'Color':colors, 'Integer':as_int})
    print("\nAs integer: ")
    print(as_int)
    print("\nIn DataFrame format for comparison:")
    print(df_colors)

    # Show memory comparison
    print("\nMemory comparison: ")
    print("\nAs strings: ")
    print(mem_str)
    mem_int = as_int.apply(sys.getsizeof)
    print("\nAs integer: ")
    print(mem_int)

    # -------------------------------------------------------------------------------------- #

    ### This can be done much quicker with pd.factorize
    fact = pd.factorize(colors)
    print("\nMuch quicker to use pd.factorize(colors):")
    print(fact)

    # Converting to categorical saves a lot of memory when we have a few categories and lots of data
    mem_initial = colors.memory_usage(index=False, deep=True)
    mem_final = colors.astype('category').memory_usage(index=False, deep=True)

    # Let's repeat the colors many times
    n_repeat = 50
    many_colors = colors.repeat(n_repeat)
    print("\nMany colors (repeated %d times): " %n_repeat)
    print("Memory usage as string: ", many_colors.memory_usage(index=False, deep=True))
    print("Memory usage as integer: ", many_colors.astype('category').memory_usage(index=False, deep=True))

    # -------------------------------------------------------------------------------------- #
    ### Another way of doing it:
    print("\nAnother way of doing it: colors.astype('category')")
    cat_colors = colors.astype('category')
    print("Categories: cat_colors.cat.categories:")
    print(cat_colors.cat.categories)
    print("\nCodes: cat_colors.cat.codes:")
    print(cat_colors.cat.codes)

    # If you want to mimic the previous manual output you can re-order the categories according to our color map
    reordered = cat_colors.cat.reorder_categories(color_map)
    print("\Reordered to match color map")
    print(reordered.cat.codes)

    ### Caveats: Categorical data is less flexible
    # We can't just add a new color with cat_colors.iloc[5] = 'a new color'

    # Adding a new category
    cat_colors = cat_colors.cat.add_categories(['a new color'])
    cat_colors.iloc[5] = 'a new color'

    ### ======================================================================= ###
    #                    Trick 2 - Mapping for Membership Binning                 #
    ### ======================================================================= ###

    print("\n____________________________________________________________________________")

    ### Aside: Function Annotations in Python
    # An optional way of link function parameters and return value to arbitray metadata
    # in other words, a way of writing function documentation in the fly

    from typing import Any

    def membership_map(s: pd.Series, groups: dict, fillvalue: Any = -1) -> pd.Series:
        # Reverse & expand the dictionary key-value pairs
        groups = {x: k for k, v in groups.items() for x in v}
        return s.map(groups).fillna(fillvalue)

    # Here we are saying: parameters fillvalue: should be of type Any, with default value -1

    print("\nShowing function annotations: function_name.__annotations__")
    print(membership_map.__annotations__)

    ### Trick 2

    print("\nTrick (2) - Grouping Countries by Membership:")

    # A Series with different country names
    countries = pd.Series([
        'United States',
        'Canada',
        'Mexico',
        'Belgium',
        'United Kingdom',
        'Thailand',
        'China',
        'Spain',
        'France',
        'Senegal',
        'Croatia',
        'Italy'
    ])

    print("\nCountries:")
    print(countries)

    # A Dictionary with the countries grouped by Continent
    groups = {
        'North America': ('United States', 'Canada', 'Mexico'),
        'Europe': ('France', 'Belgium', 'Ireland', 'United Kingdom', 'Spain', 'Croatia', 'Italy', 'Poland')
    }

    print("\nContinent dictionary: ")
    print(groups)

    # Some of the Countries in the Series are not listed in the Dictionary
    # We want to map the countries to their associated continent:

    # What does membership_map do?
    # (1) It receives the Dictionary containing the Countries in each continent
    # (2) It loops over the Dictionaries key: values with
        # for continent, countries in group.items()
    # (3) For each pair (continent, countries) it loops over the countries with
        # for x in countries
    # (4) It populates a dictionary of Country: Continent
    grouped = {x: continent for continent, countries in groups.items() for x in countries}

    # (5) It applies a that Dictionary "Grouped" to an input Series of countries

    members = membership_map(countries, groups, fillvalue='Other')

    print("\nContinent of each country: ")
    print(members)

    # In DataFrame format for comparison
    df_members = pd.DataFrame({'Country': countries, 'Continent': members})
    print("\nIn the form of a DataFrame for comparison")
    print(df_members)
## Day 28

- dataframe.info() shows info like column names, data types and if a column contains missing values
- .describle() gives summary statistics like mean, std etc
- column labels can be accessed with .columns and row labels with .index (not .rows)

```jsx
# Print the head of the homelessness data
print(homelessness.head())

# Print information about homelessness
print(homelessness.info())

# Print the shape of homelessness
print(homelessness.shape)

# Print a description of homelessness
print(homelessness.describe())
```

- To better understand DataFrame objects, it's useful to know that they consist of three components, stored as attributes:
    - `.values`: A two-dimensional NumPy array of values.
    - `.columns`: An index of columns: the column names.
    - `.index`: An index for the rows: either row numbers or row names.
    
    ```jsx
    # Import pandas using the alias pd
    import pandas as pd
    
    # Print the values of homelessness
    print(homelessness.values)
    
    # Print the column index of homelessness
    print(homelessness.columns)
    
    # Print the row index of homelessness
    print(homelessness.index)
    ```
    
- .sort_values(column_name): to sort by based on a particular column, we can also give it a list of columns by which it should sort
    - giving another parameters `ascending=False` will make the order in descending. if the column names is passed as a list, this too can be passed as list, specifying for each column if it should be sorted in ascending or descending
- to filter on multiple values of a categorical variable, use `.isin()`
- **Subsetting rows by categorical variables**
    
    ```jsx
    # The Mojave Desert states
    canu = ["California", "Arizona", "Nevada", "Utah"]
    
    # Filter for rows in the Mojave Desert states
    mojave_homelessness = homelessness[homelessness["state"].isin(canu)]
    
    # See the result
    print(mojave_homelessness) 
    ```
    
- adding new columns (syntax is similar to adding new key to python dict)
    
    ```jsx
    # Add total col as sum of individuals and family_members
    homelessness["total"] = homelessness["individuals"] + homelessness["family_members"]
    
    # Add p_homeless col as proportion of total homeless population to the state population
    homelessness["p_homeless"] = homelessness["total"] / homelessness["state_pop"]
    
    # See the result
    print(homelessness)
    ```
    
- Which state has the highest number of homeless individuals per 10,000 people in the state?
```# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness["indiv_per_10k"] = 10000 * homelessness["individuals"] / homelessness["state_pop"]

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness["indiv_per_10k"]>20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values("indiv_per_10k", ascending=False)
# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[["state", "indiv_per_10k"]]

# See the result
print(result)
```
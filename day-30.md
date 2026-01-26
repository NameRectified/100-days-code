## Day 30

- Revised first four chapters from grokking algorithms.
- Revised the following problems:
    - Concatenation of array
    - Contains Duplicate
    - Valid Anagram
    - Two Sum
    - Longest Common Prefix
    - Pair Sum - Sorted
    - Triplet Sum
    - Is Palindrome Valid

---

### Pivoting dataframes

- Pivot tables are the standard way of aggregating data in spreadsheets.

```
# Pivot for mean weekly_sales for each store type
mean_sales_by_type = sales.pivot_table(values="weekly_sales", index="type")

# Print mean_sales_by_type
print(mean_sales_by_type)
```

```jsx
# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(values="weekly_sales", index="type", aggfunc=["mean", "median"])

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)
```

```jsx
# Pivot for mean weekly_sales by store type and holiday 
mean_sales_by_type_holiday = sales.pivot_table(values="weekly_sales", index="type", columns=["is_holiday"])

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)
```

```jsx
# Print mean weekly_sales by department and type; fill missing values with 0
print(sales.pivot_table(values="weekly_sales", index="department", columns=["type"], fill_value=0))

```

- `fill_value` replaces missing values with a real value (known as *imputation*).
- `margins` is a shortcut for when you pivoted by two variables, but also wanted to pivot by each of those variables separately: it gives the row and column totals of the pivot table contents.

```jsx
# Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
print(sales.pivot_table(values="weekly_sales", index="department", columns="type", fill_value=0, margins="All"))
```

- `All` returns an overall mean for each department, not `(A+B)/2`. `(A+B)/2` would be a **mean of means**, rather than an overall mean per department!

### Visualizing dataframes

```jsx
# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Look at the first few rows of data
print(avocados.head())

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby("size")["nb_sold"].sum()

# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind="bar")

# Show the plot
plt.show()
```

- Line plots are great for visualizing something over time.

```jsx
# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby("date")["nb_sold"].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot(kind="line")

# Show the plot
plt.show()
```

```jsx
# Scatter plot of avg_price vs. nb_sold with title
avocados.plot(x="nb_sold", y="avg_price", 
    title="Number of avocados sold vs. average price", kind="scatter"
)

# Show the plot
plt.show()
```

### Histogram

```jsx
# Histogram of conventional avg_price 
avocados[avocados["type"]=="conventional"]["avg_price"].hist(bins=20, alpha=0.5)

# Histogram of organic avg_price
avocados[avocados["type"]=="organic"]["avg_price"].hist(bins=20, alpha=0.5)
# Add a legend
plt.legend(["conventional", "organic"])

# Show the plot
plt.show()
```

### Handling Missing values

```jsx
# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Check individual values for missing values
print(avocados_2016.isna())

# Check each column for missing values
print(avocados_2016.isna().any())

# Bar plot of missing values by variable
avocados_2016.isna().sum().plot(kind="bar")

# Show plot
plt.show()

# Remove rows with missing values
avocados_complete = avocados_2016.dropna()

# Check if any columns contain missing values
print(avocados_complete.isna().any())
```

- Removing observations with missing values is a quick and dirty way to deal with missing data, but this can introduce bias to your data if the values are not missing at random.

```jsx
# From previous step
cols_with_missing = ["small_sold", "large_sold", "xl_sold"]
avocados_2016[cols_with_missing].hist()
plt.show()

# Fill in missing values with 0
avocados_filled = avocados_2016.fillna(0)

# Create histograms of the filled columns
avocados_filled[cols_with_missing].hist()

# Show the plot
plt.show()
```
## Day 27

- Numpy arrays are not good at handling different data types and hence we prefer pandas dataframe as a rectangular datastructure
- we can use dataframe_name.index to add labels to the rows

```jsx
import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars

cars.index = row_labels
# Print cars again
print(cars)
```

to fix index column while importing csv

```jsx
# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col=0)

# Print out cars
print(cars)
```

- loc: used to select based on labels
- iloc: position based accessing
- while accessiing elements, if you use a single pair of sq brackets, you get a pandas series, if you want a dataframe, use 2 pairs of sq brackets

```jsx
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])
```

- to loop over a dataframe:

```jsx
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for label, row in cars.iterrows():
    print(label)
    print(row)
```

- Hacker statistics: Statistical analysis that relies heavily on simulation or re-sampling for inference (finding probability distribution)
- For plt.plot() If you pass only one argument, Python will know what to do and will use the index of the list to map onto the `x` axis, and the values in the list onto the `y` axis.
- Why does the visualization look better after transposing?
    - The x-axis became the index (0 through 4), representing the five different simulations. The y-axis showed the position at that particular step number.
    - This visualization likely looked like a dense bundle of 101 lines clustered between x=0 and x=4.

```jsx
# numpy and matplotlib imported, seed set.

# initialize and populate all_walks
all_walks = []
for i in range(5) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to NumPy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()
# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()
```
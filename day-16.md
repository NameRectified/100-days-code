## Day 16

- scikit learn accepts only numerical data with no missing values.
- sklearn also does not accept categorical features and we need to convert them into numeric values
- To deal with categorical features we can use sklearn’s OneHotEncoder() or pandas’ get_dummies()
- Scikit-learn's scoring API follows a general convention: **higher return values are always better**.
- For metrics that are naturally scores (e.g., accuracy, R-squared), a higher value is better, so they are used as is.
- For metrics that are naturally errors or losses (e.g., MSE, Mean Absolute Error), a *lower* value is better.
- To find the number of missing values in each column:
    
    ```python
    print(music_df.isna().sum().sort_values())
    
    ```
    
- Then we drop empty values with:

```python
music_df = music_df.dropna()
```

## Imputation

- Instead of removing these columns with missing values we can `impute` values where we make educated guesses on what the missing values could be. we could use the mean or median, for categorical values we use the most common value(mode).
    - We must split data before imputing to avoid data leakage(leaking test set data to our model

```python
from sklearn.impute import SimpleImputer
```

- we can also impute with a pipeline(an object used to run a series of transformations and build a model in a single workflow)

```python
from sklearn.pipeline import Pipeline
```

- In a pipeline each step except the last must be a transformer(like imputer)

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

imputer = SimpleImputer()

knn = KNeighborsClassifier(n_neighbors = 3)

steps = [("imputer", imputer), 
         ("knn", knn)]
         
pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test, y_pred))
```

### Centering and scaling

- .describe() is used for having an overview of the dataset, like the ranges, mean, std of the columns
- Features on a larger scales can disproportionately influence the model
- We want our features to be on a similar scale and for this we normalize(scaling) and standardize(centering) the dataset
- How data can be scaled:
    - subtract the mean and divide by variance: all features are centered around zero and have a variance of 1. This is called `standardization`
    - subtract the minimum and divide by range: minimum zero and maximum 1
    - can also normalize so the data ranges from -1 to +1
- scaling in sklearn

```python
from sklearn.preprocessing import StandardScaler
```
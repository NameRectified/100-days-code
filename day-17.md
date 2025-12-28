## Day 17

- Lasso Regression: machine learning technique that enhances linear models by adding a penalty for large coefficients, effectively performing both regularization (preventing overfitting) and feature selection (shrinking less important feature coefficients to exactly zero)

### Evaluating multiple models

- Guiding principles:
    - Size of the dataset: if there are only few features, you can use a simpler model as it will have faster training time
        - Some models require large amounts of data to perform well
    - Interpretability: Some models are easier to explain which may be important to the stakeholders
    - Flexibility: The accuracy may improve by making fewer assumptions of the data
- Regression models can be evaluated using RMSE, R-squared
- Classification models can be evaluated using accuracy, confusion matrix, precision, recall, F1 score, ROC AUC
- Train several models and evaluate performance without any form of hyperparamter tuning(out of the box?)
- Models such as KNN,  Linear Regression(plus ridge or lasso), logistic regression, ANN are affected by scaling and so it is best to scale our data before evaluating models

```python
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
  
  # Append the results
  results.append(cv_scores)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()

from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for name, model in models.items():
  # Fit the model to the training data
  model.fit(X_train_scaled, y_train)
  
  # Make predictions on the test set
  y_pred = model.predict(X_test_scaled)
  
  # Calculate the test_rmse
  test_rmse = root_mean_squared_error(y_test, y_pred)
  print("{} Test Set RMSE: {}".format(name, test_rmse))
```

```python
# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))
```

- Identify the single feature that has the strongest predictive performance for classifying crop types:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")
# print(crops.isna().sum().sort_values()) # no null values
# print(crops['crop'].unique())

# # Write your code here
# split the data
X = crops.drop("crop", axis=1)
y = crops['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scores = {}
for feature in ["N", "P", "K", "ph"]:
    logreg = LogisticRegression(multi_class="multinomial")
    logreg.fit(X_train[[feature]], y_train)
    y_pred = logreg.predict(X_test[[feature]])
    # scores[feature] = metrics.accuracy_score(y_test, y_pred)
    scores[feature] = metrics.f1_score(y_test, y_pred, average="weighted")

print(scores)
max_key = max(scores, key=scores.get)
best_predictive_feature = {max_key: scores[max_key]}
print(best_predictive_feature)
```
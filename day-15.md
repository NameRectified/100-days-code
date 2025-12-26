## Day 15

- We cannot rely on accuracy(Number of correct predictions/total number of predictions) always. For example, we are trying to predict fraudulent transactions, generally very few percentage of transactions are fraudulent, so if a model predicts every transaction to be legitimate, it will still have a very high accuracy although it is serving no purpose.
- Hence we use metrics like precision, recall, and F1 score. (NOTE: The class of interest is taken as positive)
- Precision:  It is also called positive predictive value (PPV)
    - High precision: lower false positive rate i.e Not many legitimate transactions are predicted as fraudulent

$$
\frac{True Positive}{True Positive + False Positive}
$$

- Recall: It is also called sensitivity
    - High recall: Low false negative rate i.e the model predicted most fraudulent transactions correctly

$$
\frac{True Positive}{True Positive + False Negative}
$$

- F1 Score: It is the harmonic mean of precision and recall.

$$
F1  = 2 * \frac{precision * recall}{precision + recall}
$$

```python
fromm sklearn.metrics import classification_report, confusion_matrix
#classificatio report shows precision, recall and f1 score
```

- In `classificatio_report` the `support` represents the number of instances of each class within the true labels
- Which metric is more suitable for the following situations:
    - A model predicting the presence of cancer as the positive class : Recall as this model should minimize the number of false negatives
    - A classifier predicting the positive class of a computer program containing malware: Recall or F1 score as to avoid installing malware, false negatives should be minimized
    - A model predicting if a customer is a high-value lead for a sales team with limited capacity: Precision as with limited capacity, the sales team needs the model to return the highest proportion of true positives compared to all predicted positives, thus minimizing wasted effort.
    - A model for predicting if a transaction is fraudulent: Recall as missing an actual fraud (a false negative) is usually far costlier (financial loss, customer trust) than incorrectly flagging a legitimate transaction (a false positive).

`NOTE` Use precision when you want to minimize false positive rate and use recall when you want to minimize false negatives

### Logistic regression

- It is used for classification problems and it outputs probabilities
- Receiver Operating Characteristic(ROC curve): used to visualize how different threshholds affect true positive and false positive rates. After plotting fpr on x axis and tpr on y axis the curve if the curve is above the dotted line it means it is better than randomly guessing, if it is below it means it is worse than guessing randomly

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()
```

- ROC AUC refers to area under curve.

### Hyper Parameter tuning

- Parameters that we specify before fitting the model are called hyperparameters
- Choosing the correct hyperparameters:
    - try lots of hyperparameter values
    - fit all of them seperately
    - see how well they perform
    - choose the best performing values
    - this is called hyperparamter tuning
- It is essential to use cross validation(**repeatedly splitting data into training and testing sets, training the model on one part, and validating it on the other, then averaging the results**) to prevent overfitting to the test set
- We can split the data and perform cross validation on the training set
- We withhold the test set and use it for the final evaluation
- GridSearchCV (not good for scaling as total fits = number of cv(k fold) * no. of hyperparameters * total values

```python
from sklearn.model_selection import GridSearchCV
```

- RandomizedSearchCV: random values are selected
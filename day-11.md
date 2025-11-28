## Day 11

### K Nearest Neighbor

- `Classification`: Given a new data point, you compare it with `k` of its nearest neighbors and based on the majority of the nearest neighbors you classify it
- The distance formula for 2D system can be extended to any dimension
- `Regression`: You take the average of the K-Nearest Neighbors
- Classification is used for categorizing datapoints into groups, where as regression is used for continuous value output
- `Feature extraction` is converting an object into a list of numbers that can be compared.

- Cosine similarity is better than distance formula to find the distance between two points, rather than measuring the distance between two points, it compares the angles between two vectors
- Optical Character Recognition (OCR) algorithms measure lines points and curves and when a new character is given, these features are extracted and based on that we can use KNN to find the character

### Naive Bayes

- This classifier is used in spam filters
- The sentence can be broken down into words, and then for each word you see what is the probability of that word showing up in a spam email

### Exercises

10.1) If a user gives extreme side of rating we can use a normalizing factor, something like their 5 star rating is not as valuable as another person who rarely gives 5 star

10.2) A factor that makes influencer ratings have more value, like multiply theirs with a bigger number

10.3) Too low as we have lot of datapoints, we can consider more than 5 neighbors to give better suggestions
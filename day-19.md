## Day 19

- Cluster labels at any intermediate stage can be recovered can be used for cross tabulation
- Height of dendrogram is the difference between merging clusters
- the distance between clusters is defined by `linkage` method and `complete linkage` is where the distace between two clusters is the maximum distance between their samples and is specified by the method parameter
- `fcluster` (from scipy.cluster.hierarchy)function is used to extract the cluster labels
- To align cluster labels with country names:

```python
import pandas as pd
pairs = pd.DataFrame({'labels': labels, 'countries': country_names})
print(pairs.sort_values('labels'))
```

- extracting cluster labels

```python
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

```

### t-sne

- t-distributed stochastic neighbor embedding
- It maps samples from higher dimensional(dimensions = features) space to 2 or 3 dimensional space
- The mapping approximately preserves nearness of the samples

```python
from sklearn.manifold import TSNE
```

- it has only the `fit_transform` method but not fit and transform individually, this means it cant extend the map to include new samples
- the axes of a t-sne plot have no interpretable meaning, they are different everytime t-sne is applied, even on the same data, but the clusters have same position relative to one another

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)
# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feataure: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)
plt.show()

```

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)
# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points (Code to label each point with its company name)
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

```
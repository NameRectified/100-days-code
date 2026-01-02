## Day 18

### Unsupervised learning

- Supervised learning finds patterns for a prediction task
- It finds patterns in data without labels and without a specific prediction task in mind
- Dimension  = no. of features

### K-means clustering

- It finds clusters of samples( k or number of clusters must be specified)

```python
from sklearn.cluster import KMeans
```

- After fitting the K-means model new samples can be assigned to existing clusters as k-means remembers the mean of each cluster (”centroid”). The new class is assigned to the nearest centroid

```python
# Import pyplot
import  matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:, 0]
ys = new_points[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)
# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

```

- To align labels and species:

```python
import pandas as pd
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
```

- Cross tabulation

```python
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
```

- a good clustering has tight clusters
- Inertia measures clustering quality i.e how spread out the clusters are(lower is better), distance from each point to the centroid of its cluster
- after fitting you can see the a inertia attribute `model.inertia_`

### How many clusters to choose

- A good clusterig has tight clusters i.e low inertia but also not too many clusters
- so choose an ‘elbow’ in the inertia plot i.e the point where inertia begins to decrease slowly

### Plotting the inertia graph for various K

```python
ks = range(1, 6)
inertias = []

for k in ks:
    model = KMeans(n_clusters = k)
    model.fit(samples)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

```

- Using `.fit_predict()` is the same as using `.fit()` followed by `.predict()`

### Evaluating the cluster

```python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters = 3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

print(ct)

```

- Variance of a feature measures spread of its values
- To give every feature a chance, the data needs to be transformed so that the features have equal variance, this ca be done with `StandardScaler` that transforms each feature to have a mean 0 and variance 1

```python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters =4 )

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({"labels": labels, "species": species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)

```

- `StandardScaler()` standardizes **features** by removing the mean and scaling to unit variance(mean 0 and variance ), `Normalizer()` rescales **each sample**  independently of the other

### clustering stocks using kmeans

```python
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters = 10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Which stocks move together?
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))

```

## Visualizations

- Two of the unsupervised learning techniques for visualization: t-SNE and hierarchial clustering will be discussed
- t-sne: creates a 2d map of any dataset and conveys useful informationn about the proximity of the samples to one another

### hierarchcial clustering

- Hierarchical clustering arranges samples into a hierarchy of clusters, the tree like structure formed is called `dendrogram`
- How it works: each sample begins in a seperate cluster
    - at each step the two closest clusters are merged
    - it is continued till all the samples are in the same cluster
    - This is a particulate hierarchical clustering called `agglomerative clustering`
    - There is also `divisive clustering` that works the other way round
- It is done with scipy

```python
from scipy.cluster.hierarchy import linkage, dendogram
```

Q) If there are 5 data samples, how many merge operations will occur in a hierarchical clustering? : 4,  With 5 data samples, there would be 4 merge operations, and with 6 data samples, there would be 5 merges, and so on.

```python
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

```

### Plotting for the stock price movement

- SciPy hierarchical clustering doesn't fit into a sklearn pipeline, so you'll need to use the `normalize()` function from `sklearn.preprocessing` instead of `Normalizer`.

```python
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings,
        labels=companies,
        leaf_rotation=90,
        leaf_font_size = 6

)
plt.show()

```
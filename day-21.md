PCA and pearson correlation

```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```

- The principal components have to align with the axes of the point cloud

### Intrinsic dimension

- It is the number of features needed to approximate the dataset
- It is the essential idea behind dimension reduction as it tells what is the most compact representation of the dataset
- This can be detected with PCA
- PCA identifies intrinsic dimension when samples have any number of features
- Intrinsic dimension = number of pca features with significant variance
- PCA rotates and shifts samples to align them with the coordinate axes
- PCA features are ordered by various in descending order
- Intrinsic dimension is an idealization and there is not always one correct answer
- To find the intrinsic dimension by plotting the variances

```python
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()
# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

```

### Dimension reduction using PCA

- Dimension reduction represents the same data using less features
- Intrinsic dimension is a good choice for the number of features PCA should keep

### Word frequency arrays

- rows represent documents, columns represets words
- “sparse array”: most entries are zero, and we can use `scipy.sparse.csr_matrix` innsted of numpy array. csr matrices save space by remembering only the non-zero entries of the array
- scikit learn’s PCA does not support csr_matrices and we have to use TruncatedSVD(which performs the same transforms as pca but accepts csr matrices) instead
- We can measure the presence of words in each document using `tf-idf`
- tf = frequency of the word in the document, so if 10% of the words are “something” , the the tf of “something” is 0.1
- idf is a weightig scheme that reduces the influence of frequent words like “the”

Dimension reduction 

```python
# Import PCA
from sklearn.decomposition import PCA

	# Create a PCA model with 2 components: pca
pca = PCA(n_components = 2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)
# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)
```

tf-idf word-frequency array

```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names_out()

# Print words
print(words)

```

- Clustering wikipedia articles

```python
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)
# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)
# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))

```

### Non-negative matrix factorization (NMF)

- Another dimension reduction technique
- These models are interpretable unlike PCA, i.e they are easier to understand ourselves and to explain to others
- It cannot be applied to all datasets, all the sample features must be non-negative
- It achieves its interpretability by decomposing samples as the sum of their parts
- It decomposes documents as combination of common themes and images as combinations of common patterns
- In scikit learn NMF, the nnumber of componnents must be always specified
- It works with numpy arrays as well as sparse arrays in csr_matrix format
- NMF components and feature values are all non-negative
- The features and components of an NMF model can be combined to approximately reconstruct the original data samples
- if we multiply each NMF components by the corresponding nmf features we get values that are approximate of the original data samples (multiply feature values and components and add)
- They can also be expressed as a product of matrices, which is matrix factorization
- NMF applied to wikipedia article

```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components = 6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))

# Import pandas
import pandas as pd
# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

```

Q) NMF feature values of a sample are `[2, 1]` and components are  

[[1.  0.5 0. ]
[0.2 0.1 2.1]]

reconstruct approximate of original samples

Ans) multiply and add, we get `[2.2, 1.1, 2.1]`
## Day 22

### NMF for images

- In grayscale images there are no colors but only shades of gray so it can be encoded by the brightness of every pixel, representing the brightness as a number between 0 and 1 where 0 is totally black

### Building recommender systems using NMF

- for recommending articles:
    - strategy: apply nmf to word frequency array, as nmf feature valuess describe the topics, similar documents will have similar nmf feature values
    - to compare nmf features, use cosine similarity(uses angle b/w two linnes and higher value means more similar
    - use .dot() method of dataframe to calculate cosine similarity

### **Which articles are similar to 'Cristiano Ronaldo'?**

```python
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
```

### **Recommend musical artists**

```python
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components = 20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())

```

### Clusterig penguinn species

```python
# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()

## convert sex column to numerical
penguins_df = pd.get_dummies(penguins_df)
penguins_df.head()

## elbow analysis
inertia = []
ks = range(1,10)
for i in ks:
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(penguins_df)
    inertia.append(model.inertia_)
plt.plot(ks, inertia, '-o')
plt.show()
    
scaler = StandardScaler()
kmeans = KMeans(n_clusters = 4)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(penguins_df)
labels = pipeline.predict(penguins_df)
print(labels)
penguins_df['label'] = kmeans.labels_
plt.scatter(penguins_df['label'],penguins_df['culmen_length_mm'], c=kmeans.labels_)

# numeric_columns = list(penguins_df.select_dtypes(include=["number"]).columns.values)
numeric_columns = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g','label']

stat_penguins= penguins_df.groupby('label')[numeric_columns].mean()
stat_penguins

```
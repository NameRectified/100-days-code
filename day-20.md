## Day 20

### Dimension reduction

- Dimension reduction summarizes a dataset using its common occuring patterns
- dim. red. finds patterns in the data and uses these patterns to re-express it in a compressed form which makes computation efficient
- But a more important function of dimension reduction is to remove the less-informative or noisy features that cause problems in tasks like regression and classification

### Principle Component analysis (PCA)

- The first step is `decorrelation` and the next step reduces `dimension`

```python
from sklearn.decomposition import PCA
```

- It learns the principal components, the directions in which the samples vary the most. It is the principal components that PCA aligns with coordinate axes
- After a PCA model has been fit, the principal components are available as `.components_` attribute.

### Decorrelation

- In this step, PCA rotates the samples so that they are aligned with the coordinate axes
- It also shifts the data samples so that they have a mean zero
- As PCA aligns data with axes, the resulting PCA features are not linearly correlated(decorrelation)

### Pearson correlation

- Linear correlation ca be measured with `pearson` correlation
- values between -1 and 1
- value of 0 means no linear correlation
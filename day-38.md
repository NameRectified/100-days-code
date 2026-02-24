## Day 38

### Embeddings

- They are a numerical representation of text
- They map text to a multi-dimensional vector space
- Similar words are mapped closer to each other
- They are used in:
    - semantic search engines
    - recommendation systems: like a job portal can recommend based on the descriptions of jobs already viewed
    - classification
- Embedding models convert text into numerical representations that are able to capture the context and intent behind the text.
- Embeddings can be used for quite a few different tasks, but they really shine in recommendation, classification, and semantic search

```jsx
# Create an OpenAI client
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-3-small", input="testing this model"
)
# Convert the response into a dictionary
response_dict = response.model_dump()
print(response_dict)

# Extract the total_tokens from response_dict
print(response_dict['usage']['total_tokens'])

# Extract the embeddings from response_dict
print(response_dict['data'][0]['embedding'])
```

**Embedding product descriptions**

```jsx
# preview of structure of products list
products = [
    {
        "title": "Smartphone X1",
        "short_description": "The latest flagship smartphone with AI-powered features and 5G connectivity.",
        "price": 799.99,
        "category": "Electronics",
        "features": [
            "6.5-inch AMOLED display",
            "Quad-camera system with 48MP main sensor",
            "Face recognition and fingerprint sensor",
            "Fast wireless charging"
        ]
    },
    ...
]
```

```jsx
# Extract a list of product short descriptions from products
product_descriptions = [product['short_description'] for product in products]

# Create embeddings for each product description
response = client.embeddings.create(
    model = "text-embedding-3-small", input = product_descriptions
)
response_dict = response.model_dump()

# Extract the embeddings from response_dict and store in products
for i, product in enumerate(products):
    product['embedding'] = response_dict['data'][i]["embedding"]
    
print(products[0].items())
```

### **Visualizing the embedded descriptions using t-sne**

```jsx
# Create reviews and embeddings lists using list comprehensions
categories = [product['category'] for product in products]
embeddings = [product['embedding'] for product in products]

# Reduce the number of embeddings dimensions to two using t-SNE
tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))

# Create a scatter plot from embeddings_2d using first and second column
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

for i, category in enumerate(categories):
    plt.annotate(category, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()
```

### Measuring similarity

- By measuring the distance of the vector embedding we can know how similar they are to each other
- Cosine distance:
    - the distance ranges from 0-2
    - smaller numbers = greater similarity

```jsx
from scipy.spatial import distance
distance.cosine([0,1], [1,0])
```

## repeatable embedding

```jsx
# Define a create_embeddings function
def create_embeddings(texts):
  response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]

# Embed short_description using create_embeddings(), 
# and extract and print the embeddings in a single list.
print(create_embeddings(short_description)[0])

# Embed list_of_descriptions using create_embeddings() and print.

print(create_embeddings(list_of_descriptions))
```

### **Finding the most similar product**

```jsx
# Embed the search text
search_text = "soap"
search_embedding = create_embeddings(search_text)[0]

distances = []
for product in products:
  # Compute the cosine distance for each product description
  dist = distance.cosine(search_embedding, product['embedding'])
  distances.append(dist)

# Find and print the most similar product short_description    
min_dist_ind = np.argmin(distances)
print(products[min_dist_ind]['short_description'])
```

### Semantic search:

- uses embeddings to return results most similar to a search query. It consists of the following steps:
    - Embed the search query and other texts
    - compute the cosine distances
    - extract the text with the smallest cosine distance
- **Enriching embeddings**

```jsx
# Define a function to combine the relevant features into a single string
def create_product_text(product):
  return f"""Title: {product['title']}
Description: {product['short_description']}
Category: {product['category']}
Features: {product['features']}"""

# Combine the features for each product
product_texts = [create_product_text(product) for product in products]

# Create the embeddings from product_texts
product_embeddings = create_embeddings(product_texts)
```

- define a function called `find_n_closest()`, which computes the cosine distances between a query vector and a list of embeddings and returns the `n` smallest distances and their indexes.

```jsx
def find_n_closest(query_vector, embeddings, n=3):
  distances = []
  for index, embedding in enumerate(embeddings):
    # Calculate the cosine distance between the query vector and embedding
    dist = distance.cosine(query_vector, embedding)
    # Append the distance and index to distances
    distances.append({"distance": dist, "index": index})
  # Sort distances by the distance key
  distances_sorted = sorted(distances, key=lambda x:x['distance'])
  # Return the first n elements in distances_sorted
  return distances_sorted[0:n]

```

**Semantic search for products**

```jsx
# Create the query vector from query_text
query_text = "computer"
query_vector = create_embeddings(query_text)[0]

# Find the five closest distances
hits = find_n_closest(query_vector,product_embeddings, n = 5)

print(f'Search results for "{query_text}"')
for hit in hits:
  # Extract the product at each index in hits
  product = products[hit["index"]]
  print(product["title"])
```

### Recommendation system

- similar to semantic search
- process:
    - embed the potential recommendations and data point
    - calculate cosine distance
    - recommend closest item
- process for recommendations on multiple data points
    - combine multiple vectors into by taking the mean
    - compute cosine distances
    - compute the closest vector
        - ensure that it has not been already visited
- **Product recommendation system**

```jsx
# Combine the features for last_product and each product in products
last_product_text = create_product_text(last_product)
product_texts = [create_product_text(product) for product in products]

# Embed last_product_text and product_texts
last_product_embeddings = create_embeddings(last_product_text)[0]
product_embeddings = create_embeddings(product_texts)

# Find the three smallest cosine distances and their indexes
hits = find_n_closest(last_product_embeddings, product_embeddings, 3)

for hit in hits:
  product = products[hit['index']]
  print(product['title'])
```

- **Adding user history to the recommendation engine**

```jsx
# Prepare and embed the user_history, and calculate the mean embeddings
history_texts = [create_product_text(product) for product in user_history]
history_embeddings = create_embeddings(history_texts)[0]
mean_history_embeddings = np.mean(history_embeddings, axis=0)

# Filter products to remove any in user_history
products_filtered = [product for product in products if product not in user_history]

# Combine product features and embed the resulting texts
product_texts = [create_product_text(product) for product in products_filtered]
product_embeddings = create_embeddings(product_texts)

hits = find_n_closest(mean_history_embeddings, product_embeddings)

for hit in hits:
  product = products_filtered[hit['index']]
  print(product['title'])
```

### Embeddings for classification tasks

- classification tasks:
    - assigning labels to tasks
        - categorization: eg: assigning headlines into topics
        - sentiment analysis
    - zeroshot classification : not using labeled data
        - process:
            - embed class descriptions
            - embed the item to classify
            - compute cosine distances
            - assign the most similar label
    
    ```jsx
    # Create a list of class descriptions from the sentiment labels
    class_descriptions = [sentiment['label'] for sentiment in sentiments]
    
    # Embed the class_descriptions and reviews
    class_embeddings = create_embeddings(class_descriptions)
    review_embeddings = create_embeddings(reviews)
    ```
    
    ```jsx
    # Define a function to return the minimum distance and its index
    def find_closest(query_vector, embeddings):
      distances = []
      for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    
      return min(distances, key=lambda x: x["distance"])
    
    for index, review in enumerate(reviews):
      # Find the closest distance and its index using find_closest()
      closest = find_closest(review_embeddings[index], class_embeddings)
      # Subset sentiments using the index from closest
      label = sentiments[closest['index']]['label']
      print(f'"{review}" was classified as {label}')
    ```
    
    **Embedding more detailed descriptions**
    
    ```jsx
    
    # Extract and embed the descriptions from sentiments
    class_descriptions = [sentiment['description'] for sentiment in sentiments]
    class_embeddings = create_embeddings(class_descriptions)
    review_embeddings = create_embeddings(reviews)
    
    def find_closest(query_vector, embeddings):
      distances = []
      for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
      return min(distances, key=lambda x: x["distance"])
    
    for index, review in enumerate(reviews):
      closest = find_closest(review_embeddings[index], class_embeddings)
      label = sentiments[closest['index']]['label']
      print(f'"{review}" was classified as {label}')
    ```
    
    This time, you were able to correctly classify the second review! Even short and simple descriptions had a big impact on the results—adding more detail will make these zero-shot classifications even more accurate
    

### **Vector databases for embedding systems**

- limitations of the above approach
    - loading all the embeddings into memory (1536 floats(number of values openai api returns for the embedding) is nearly 13kb/embedding)
    - recalculation of embeddings for each new query
    - calculating cosine distances for every embedding and sorting is slow and scales linearly
- these limitations can be overcome by vector databases
- the majority of vector databases are NoSQL(not only sql)
- components to store:
    - embeddings
    - source text
    - metadata
        - IDs and references
        - additional data useful for filtering results
    - dont store source text as metadata as metadata must be small to be useful, so adding a large amount of text data will greatly decrease performance
- Getting started with chromadb

```jsx
# Create a persistant client
client = chromadb.PersistentClient()

# Create a netflix_title collection using the OpenAI Embedding function
collection = client.create_collection(
    name="netflix_titles",
    embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key="<OPENAI_API_TOKEN>")
)

# List the collections
print(client.list_collections())
```

### **Estimating embedding costs with tiktoken**

```jsx
# Load the encoder for the OpenAI text-embedding-3-small model
enc = tiktoken.encoding_for_model("text-embedding-3-small")

# Encode each text in documents and calculate the total tokens
total_tokens = sum(len(enc.encode(text)) for text in documents)

cost_per_1k_tokens = 0.00002

# Display number of tokens and cost
print('Total tokens:', total_tokens)
print('Cost:', cost_per_1k_tokens * total_tokens/1000)

```

### **Adding data to the collection**

```jsx
# Recreate the netflix_titles collection
collection = client.create_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key="<OPENAI_API_TOKEN>")
)

# Add the documents and IDs to the collection
collection.add(
  ids=ids,
  documents=documents
)

# Print the collection size and first ten items
print(f"No. of documents: {collection.count()}")
print(f"First ten documents: {collection.peek()}")
```

### Querying and updating the database

Querying the netflix collection

```jsx
# Retrieve the netflix_titles collection
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key="<OPENAI_API_TOKEN>")
)

# Query the collection for "films about dogs"
result = collection.query(query_texts=["films about dogs"], n_results=3)

print(result)
```

- It's important to remember to always specify the same embedding function when querying as was used to embed the documents; otherwise, your recommendations will likely be untrustworthy.
- **Updating and deleting items from a collection**

```jsx
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key="<OPENAI_API_TOKEN>")
)

# Update or add the new documents
collection.upsert(
    ids=[doc['id'] for doc in new_data],
    documents=[doc['document'] for doc in new_data]
)

# Delete the item with ID "s95"
collection.delete(ids=["s95"])

result = collection.query(
    query_texts=["films about dogs"],
    n_results=3
)
print(result)
```

**Querying with multiple texts**

```jsx
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key="<OPENAI_API_TOKEN>")
)

reference_ids = ['s999', 's1000']

# Retrieve the documents for the reference_ids
reference_texts = collection.get(ids=reference_ids)['documents']

# Query using reference_texts
result = collection.query(
  query_texts = reference_texts,n_results = 3
)

print(result['documents'])
```

**Filtering using metadata**
```
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key="<OPENAI_API_TOKEN>")
)

reference_texts = ["children's story about a car", "lions"]

# Query two results using reference_texts
result = collection.query(
  query_texts=reference_texts,
  n_results=2,
  # Filter for titles with a G rating released before 2019
  where={
    "$and": [
        {"rating": 
        	{"$eq":'G'}
        },
        {"release_year": 
         	{"$lt": 2019}
        }
    ]
  }
)

print(result['documents'])
```
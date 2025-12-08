## Day 14

### Inverted index

- A data structure used in search engines and information retrieval systems to efficiently retrieve documents or web pages containing a specific term.
- The index consists of words as keys and values are the documents or web pages containing that word
- Useful for large collection of documents
- Algorithm to build an inverted index
    - fetch the document and remove stop words, these are useless words such as , `I , a, an, the etc`
    - stemming of root word: derive the root word from the words like root word of swimming is swim. Use tools like Porter’s stemmer to do this
    - If the word is already present as a key in the index add that document also as a value that is mapped from that key, if the word is not already present, create a new entry.

### Basic version code

```jsx
document1 = "hey x, this is day 14 of 100 days of code"
document2 = "i love 100 days of code as it keeps me accountable"

# tokenize
doc1tokens = document1.lower().split()
doc2tokens = document2.lower().split()

# combine the unique tokens
terms = list(set(doc1tokens + doc2tokens))

inverted_index = {}

for term in terms:
    documents = []
    if term in doc1tokens:
        documents.append("Document 1")
    if term in doc2tokens:
        documents.append("Document 2")
    inverted_index[term] = documents
    
for word, presentIn in inverted_index.items():
    print(f"{word} is present in {presentIn}")
```

### Extended version

```jsx
documents = ["hey x, this is day 13 of 100 days of code", "i love 100 days of code as it keeps me accountable"]

wordsInDocs = {}
for idx, doc in enumerate(documents):
    wordsInDocs[f"Document {idx}"] = doc.lower().split()

print(wordsInDocs)
uniqueWords = list(set([item for sublist in wordsInDocs.values() for item in sublist]))
# or  uniqueWords = sum(wordsInDocs.values(), [])

invertedIndex = {}
for word in uniqueWords:
    presentIn = []
    # if word in
    for key, value in wordsInDocs.items():
        if word in value:
            presentIn.append(key)
    invertedIndex[word] = presentIn

for word, documents in invertedIndex.items():
    print(f"{word.title()} is present in {documents}")
    
```

### How are so many terms handled in real life?

- The number of terms in real world is so vast as each misspelled word, emojis, hashtags can become a new term
- Hence the following techniques are used for managing this problem:
    - Stop words removal: remove words that are frequent and provide no additional information, like a, an the, etc
    - Stemming(chop words based on heuristics) and lemmatization (chop words based on grammar rules and vocabulary): Derive the root word of the terms
    - Term dictionary compression: use data structures like tries, FSTs (finite state transducers), block trees
    - Sharding and partitioning: breaking the inverted index into pieces(shards) and storing each piece on different machines, instead of keeping one giant index on a server
        - two common ways to split are:
            - term-based partitioning
            - document-based partitioning
- Tiered indexing: Hot terms are in memory and cold terms are in storage
- `Scoring`: We rank queries that appear in multiple documents using TF-DIF (term frequency - inverse document frequency)
- 

$$
score = tf(term in doc) * idf(term in all docs)
$$

- tf(term frequency) : if a word appears more in a document it means it is more relevant
- idf(inverse document frequency): rare terms have more weight

References:

- https://en.wikipedia.org/wiki/Inverted_index
- https://www.geeksforgeeks.org/dbms/inverted-index/
- https://satyadeepmaheshwari.medium.com/inverted-index-the-backbone-of-modern-search-engines-8bfd19a9ff75
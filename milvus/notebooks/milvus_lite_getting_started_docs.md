### Initialization of Milvus Client:


```python

from pymilvus import MilvusClient 
client = MilvusClient("milvus_demo.db")
```



The code imports the `MilvusClient` class from the pymilvus library and initializes a client to connect to a Milvus instance or database named `milvus_demo.db`.

### Collection Handling



```python
if client.has_collection(collection_name="demo_collection"): 
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection", 
    dimension=768,  # The vectors we will use in this demo has 768 dimensions 
)

```
The code checks if a collection named "demo_collection" exists in the Milvus database.
If the collection exists, it drops the collection to start fresh.
A new collection named "demo_collection" is then created with a specified vector dimension of 768. This indicates that the vectors stored in this collection will have 768-dimensional embeddings.


```python
docs = [ 
    "Artificial intelligence was founded as an academic discipline in 1956.", 
    "Alan Turing was the first person to conduct substantial research in AI.", 
    "Born in Maida Vale, London, Turing was raised in southern England.", 
]
vectors = embedding_fn.encode_documents(docs)
```

### Document Preparation and Vector Encoding:

 list of text strings (docs) is prepared. These strings represent historical facts related to artificial intelligence and Alan Turing.
The embedding_fn.encode_documents(docs) function is used to convert the documents into vector embeddings, with each vector having 768 dimensions.
Print Vector Dimensions:


## Creating Data Entries:

```python
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} 
    for i in range(len(vectors))
]
```

A list of dictionaries (data) is created. Each dictionary represents an entity that contains:
id: The index of the document.
vector: The corresponding vector representation of the document.
text: The original document text.
subject: A label for metadata filtering, set to "history" for all entities in this example.


https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning#outcomes

data	Data to search with	list[list[Float]]	True
anns_field	Name of the vector field to search on	String	True
param	Specific search parameter(s) of the index on the vector field. For details, refer to Prepare search parameters.	Dict	True
limit	Number of nearest records to return. The sum of this value and offset should be less than 16384.	Integer	True
expr	Boolean expression to filter the data
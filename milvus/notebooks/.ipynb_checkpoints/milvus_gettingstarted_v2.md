```python
#!pip install -U pymilvus
```


```python
from pymilvus import MilvusClient
client = MilvusClient("milvus_demo.db")
```


```python
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)
```


```python
#!pip install "pymilvus[model]"
#!pip install torch torchvision
```


```python
from pymilvus import model

# If connection to https://huggingface.co/ failed, uncomment the following path
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created.
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))
```

    Dim: 768 (768,)
    Data has 3 entities, each with fields:  dict_keys(['id', 'vector', 'text', 'subject'])
    Vector dim: 768



```python
res = client.insert(collection_name="demo_collection", data=data)

print(res)
```

    {'insert_count': 3, 'ids': [0, 1, 2]}



```python
import json
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
# If you don't have the embedding function you can use a fake vector to finish the demo:
# query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

res_str = json.dumps(res, indent=4)

# Print the pretty-printed JSON string
print(res_str)


```

    [
        [
            {
                "id": 2,
                "distance": 0.5859944224357605,
                "entity": {
                    "text": "Born in Maida Vale, London, Turing was raised in southern England.",
                    "subject": "history"
                }
            },
            {
                "id": 1,
                "distance": 0.5118255615234375,
                "entity": {
                    "text": "Alan Turing was the first person to conduct substantial research in AI.",
                    "subject": "history"
                }
            }
        ]
    ]



```python
# Insert more docs in another subject.
docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
vectors = embedding_fn.encode_documents(docs)
data = [
    {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
    for i in range(len(vectors))
]

client.insert(collection_name="demo_collection", data=data)

# This will exclude any text in "history" subject despite close to the query vector.
res2 = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",
    limit=2,
    output_fields=["text", "subject"],
)
res2_str = json.dumps(res2, indent=4)

# Print the pretty-printed JSON string
print(res2_str)

```

    [
        [
            {
                "id": 4,
                "distance": 0.2703056335449219,
                "entity": {
                    "text": "Computational synthesis with AI algorithms predicts molecular properties.",
                    "subject": "biology"
                }
            },
            {
                "id": 3,
                "distance": 0.1642589271068573,
                "entity": {
                    "text": "Machine learning has been used for drug design.",
                    "subject": "biology"
                }
            }
        ]
    ]



```python
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
res
```




    data: ["{'id': 0, 'text': 'Artificial intelligence was founded as an academic discipline in 1956.', 'subject': 'history'}", "{'id': 1, 'text': 'Alan Turing was the first person to conduct substantial research in AI.', 'subject': 'history'}", "{'id': 2, 'text': 'Born in Maida Vale, London, Turing was raised in southern England.', 'subject': 'history'}"] 




```python
res = client.query(
    collection_name="demo_collection",
    ids=[0, 2],
    output_fields=["vector", "text", "subject"],
)

print(res[0].keys())
len(res[0]['vector'])
```

    0
    dict_keys(['id', 'text', 'subject', 'vector'])





    768




```python

```

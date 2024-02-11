import csv
import json
import random
import openai
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from milvus_util import embed

# REference: https://milvus.io/docs/integrate_with_openai.md
# Extract the book titles

FILE = './data/books.csv'  # Download it from https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks and save it in the folder that holds your script.
COLLECTION_NAME = 'jeopardy_db'  # Collection name
DIMENSION = 1536  # Embeddings size
COUNT = 10  # How many titles to embed and insert.
MILVUS_HOST = 'localhost'  # Milvus server URI
MILVUS_PORT = '19530'
OPENAI_ENGINE = 'text-embedding-ada-002'  # Which engine to use
#openai.api_key = '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'  # Use your own Open AI API Key here

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
has = utility.has_collection("jeopardy_db")
print(f"Does collection jeopardy_db exist in Milvus: {has}")


# # Create collection which includes the id, title, and embedding.
# fields = [
#     FieldSchema(name='id', dtype=DataType.INT64, descrition='Ids', is_primary=True, auto_id=False),
#     FieldSchema(name='title', dtype=DataType.VARCHAR, description='Title texts', max_length=200),
#     FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=DIMENSION)
# ]


fields = [
    FieldSchema(name='id', dtype=DataType.INT64, descrition='Ids', is_primary=True, auto_id=False),
    FieldSchema(name='Question', dtype=DataType.VARCHAR, description='Title texts', max_length=2000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=DIMENSION)
]
# schema = CollectionSchema(fields=fields, description='Title collection')
# collection = Collection(name=COLLECTION_NAME, schema=schema)

schema = CollectionSchema(fields, "jeopardy schema")
import pymilvus
j_milvus_collection = Collection("jeopardy_db")

# Create an index for the collection.
# Create an index for the collection.
# index_params = {
#     'index_type': 'IVF_FLAT',
#     'metric_type': 'L2',
#     'params': {'nlist': 1024}
# }
#collection.create_index(field_name="embedding", index_params=index_params)


# Load the collection into memory for searching
j_milvus_collection.load()

# Search the database based on input text
def search(text):
    # Search parameters for the index
    search_params={
        "metric_type": "L2"
    }

    results=j_milvus_collection.search(
        data=[embed(text)],  # Embeded search value
        anns_field="embedding",  # Search across embeddings
        param=search_params,
        limit=5,  # Limit to five results per search
        output_fields=['Question']  # Include title field in result
    )

    ret=[]
    for hit in results[0]:
        row=[]
        row.extend([hit.id, hit.score, hit.entity.get('Question')])  # Get the id, distance, and title for the results
        ret.append(row)
    return ret

search_terms=['glucose']

for x in search_terms:
    print('Search term:', x)
    for result in search(x):
        print(result)
    print()


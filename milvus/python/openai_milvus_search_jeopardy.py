import os

import requests
import json
from milvus_util import embed
# Download the data
resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')

data = json.loads(resp.text)  # Load data

# Parse the JSON and preview it
print(type(data), len(data))

def json_print(data):
    print(json.dumps(data, indent=2))

json_print(data[0])

from pymilvus import (
    Milvus,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import openai

COLLECTION_NAME = 'jeopardy_db'  # Collection name
DIMENSION = 1536  # Embeddings size
COUNT = 100  # How many titles to embed and insert.
MILVUS_HOST = 'localhost'  # Milvus server URI
MILVUS_PORT = '19530'
OPENAI_ENGINE = 'text-embedding-ada-002'  # Which engine to use
openai.api_key = os.environ.get('OPENAI_API_KEY')  # Use your own Open AI API Key here

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    print('WARNING: ' + COLLECTION_NAME + ' already exists')

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, descrition='Ids', is_primary=True, auto_id=False),
    FieldSchema(name='Question', dtype=DataType.VARCHAR, description='Title texts', max_length=2000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=DIMENSION)
]


schema = CollectionSchema(fields, "jeopardy schema")
import pymilvus
j_milvus_collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
try:
    index_params = {
        'index_type': 'IVF_FLAT',
        'metric_type': 'L2',
        'params': {'nlist': 1024}
    }

    j_milvus_collection.drop_index()
    #j_milvus_collection.create_index(field_name="Question", index_params=index_params)
except pymilvus.exceptions.IndexNotExistException as exception:
    print(exception)
finally:
    j_milvus_collection.create_index(field_name="Question", index_params=index_params)


q_a_dict = {}
for d in data:
    print(d)
    question = d['Question']
    answer = d['Answer']
    q_a_dict[question] = answer

import pickle

# try:
#     dict_file = open('./data/jeopardy_q_a', 'wb')
#     pickle.dump(q_a_dict, dict_file)
#     dict_file.close()

# except:
#     print("Something went wrong")
import time
count = 0
for d in data:
    question = d['Question']
    embedding = embed(question)
    ins = [[count],[d['Question']],[embedding]]
    count = count + 1

    j_milvus_collection.insert(ins)

    time.sleep(1)  # Free OpenAI account limited to 60 RPM

print("Completed")



# for idx, text in enumerate(random.sample(sorted(csv_load(FILE)), k=COUNT)):  # Load COUNT amount of random values from dataset
#     ins=[[idx], [(text[:198] + '..') if len(text) > 200 else text], [embed(text)]]  # Insert the title id, the title text, and the title embedding vector
#     collection.insert(ins)
#     time.sleep(1)  # Free OpenAI account limited to 60 RPM
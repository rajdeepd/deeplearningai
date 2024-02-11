
from pymilvus import (
    Milvus,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
connections.connect("default", host="localhost", port="19530")
utility.drop_collection("jeopardy")

print('after delete')

collections_response = utility.list_collections(timeout=None, using='default')
print(collections_response)

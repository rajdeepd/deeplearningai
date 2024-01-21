from pymilvus import (
    Milvus,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from pymilvus import list_collections
fmt = "\n=== {:30} ===\n"
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

collections_response = utility.list_collections(timeout=None, using='default')
print(collections_response)
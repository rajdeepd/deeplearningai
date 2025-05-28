---
layout: default
title: 11. Managing Data
nav_order: 11
description: ""
has_children: false
parent:  Milvus (U)
---


**Chapter X: Managing Data with PyMilvus**

In this se,ction we'll explore how to manage data in Milvus using PyMilvus, the Python SDK for Milvus. We'll cover the steps to connect to a Milvus server, create a collection, insert data, and delete entities.

---

### **Connecting to the Milvus Server**

To begin, we need to import the necessary modules:

```python
from pymilvus import MilvusClient, DataType
import string
import random
```

Here, we're importing `MilvusClient` from the `pymilvus` package, which will allow us to interact with the Milvus server. We also import `string` and `random` modules for any random data generation we might need.

Next, we establish a connection to the Milvus server by creating an instance of `MilvusClient`:

```python
client = MilvusClient(
    uri="http://localhost:19530"
)
```

In this line, we're connecting to a Milvus server running locally on port `19530`.

---

### **Creating a Collection**

With the client connected to the server, we can now create a new collection. A collection in Milvus is analogous to a table in a traditional database.

```python
client.create_collection(
    collection_name="quick_setup",
    dimension=5,
    metric_type="IP"
)
```

- **`collection_name="quick_setup"`**: Names the collection `quick_setup`.
- **`dimension=5`**: Specifies that the vectors we'll store have 5 dimensions.
- **`metric_type="IP"`**: Sets the similarity metric to Inner Product (IP), which is commonly used for measuring the similarity between vectors.

To verify that the collection has been created, we can list all collections in the Milvus server:

```python
client.list_collections()
```

This should output:

```python
['image_search', 'album1', 'quick_setup']
```

This confirms that our `quick_setup` collection has been successfully created.

---

### **Preparing Data for Insertion**

Before inserting data into the collection, we need to prepare it. We'll create a list of dictionaries, each representing an entity with an `id`, a `vector`, and a `color` attribute.

```python
data = [
    {"id": 0, "vector": [0.3580, -0.6023, 0.1841, -0.2628, 0.9029], "color": "pink_8682"},
    {"id": 1, "vector": [0.1988, 0.0602, 0.6976, 0.2614, 0.8387], "color": "red_7025"},
    {"id": 2, "vector": [0.4374, -0.5597, 0.6457, 0.7894, 0.2078], "color": "orange_6781"},
    {"id": 3, "vector": [0.3172, 0.9719, -0.3698, -0.4860, 0.9579], "color": "pink_9298"},
    {"id": 4, "vector": [0.4452, -0.8757, 0.8220, 0.4640, 0.3033], "color": "red_4794"},
    {"id": 5, "vector": [0.9858, -0.8144, 0.6299, 0.1206, -0.1446], "color": "yellow_4222"},
    {"id": 6, "vector": [0.8371, -0.0157, -0.3106, -0.5626, -0.8984], "color": "red_9392"},
    {"id": 7, "vector": [-0.3344, -0.2567, 0.8987, 0.9402, 0.5378], "color": "grey_8510"},
    {"id": 8, "vector": [0.3952, 0.4000, -0.5890, -0.8650, -0.6140], "color": "white_9381"},
    {"id": 9, "vector": [0.5718, 0.2407, -0.3737, -0.0672, -0.6980], "color": "purple_4976"}
]
```

**Note**: The vector values have been rounded for readability.

Each entity includes:

- **`id`**: A unique identifier.
- **`vector`**: A list of five floating-point numbers representing a 5-dimensional vector.
- **`color`**: A string attribute that combines a color name with a random number.

---

### **Inserting Data into the Collection**

With the data prepared, we can now insert it into our `quick_setup` collection:

```python
res = client.insert(
    collection_name="quick_setup",
    data=data
)
```

The `insert` method takes two arguments:

- **`collection_name`**: Specifies the target collection.
- **`data`**: The list of entities we want to insert.

After the insertion, we can check the result:

```python
res
```

Which should output:

```python
{'insert_count': 10, 'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'cost': 0}
```

This response indicates:

- **`insert_count`:** 10 entities have been inserted.
- **`ids`:** A list of the IDs of the inserted entities.
- **`cost`:** The time taken to perform the insertion (in milliseconds).

---

### **Deleting Entities by Primary Key**

Sometimes, we need to delete specific entities from a collection. Milvus allows us to delete entities by their primary keys.

For example, to delete the entities with IDs `0` and `2`, we execute:

```python
res = client.delete(collection_name="quick_setup", ids=[0, 2])
print(res)
```

The `delete` method requires:

- **`collection_name`**: The name of the collection from which to delete entities.
- **`ids`**: A list of primary keys identifying the entities to delete.

The output will be:

```python
{'delete_count': 2}
```

This confirms that two entities have been successfully deleted from the collection.

---

### **Summary**

In this chapter, we've learned how to:

- **Connect to a Milvus server** using the `MilvusClient`.
- **Create a new collection** with specified dimensions and metric type.
- **Prepare data** in the appropriate format for insertion.
- **Insert data** into a Milvus collection.
- **Delete entities** from a collection using their primary keys.

Understanding these operations is fundamental when working with vector databases like Milvus. It allows for efficient management of high-dimensional data, which is essential in applications like similarity search, recommendation systems, and machine learning.

---

**Next Steps**

In the following chapters, we'll delve deeper into querying the data, performing similarity searches, and optimizing performance. We'll also explore advanced features like indexing and data partitioning.
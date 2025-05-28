---
layout: default
title: 12. Dynamic collections
nav_order: 11
description: ""
has_children: false
parent:  Milvus (U)
---

## Dynamic Schema in Milvus Collections

### Introduction to Dynamic Data Model

In this section, we will explore the dynamic data model, also known as dynamic schema, within a Milvus collection. Previously, when creating a Milvus collection, we were required to explicitly declare the field schema. Consequently, any data inserted into that collection had to strictly adhere to this predefined schema.

### Enabling Dynamic Schema

Milvus has introduced support for a dynamic schema. This feature provides the flexibility to insert entities containing new fields into a collection without necessitating any modifications to the original schema. This offers significant adaptability when dealing with evolving data structures or fields that were not initially defined in the collection schema.

To activate the dynamic schema for a Milvus collection, you need to set a specific parameter, `enable_dynamic_field`, to `True` during the collection creation process.

### Practical Example: Creating a Collection with Dynamic Schema

Let's walk through a practical example of creating a Milvus collection with the dynamic schema enabled.

1.  **Import Necessary Modules:** We begin by importing the required modules for this demonstration.
2.  **Connect to the Server:** Next, we establish a connection to the running Milvus server before proceeding with collection creation.
3.  **Define Field Schema:** We then define the initial field schema for our collection. This might look familiar as it's based on a previous example. We define fields such as `name_id` (as the primary key), `count`, and `vector` (which will serve as the vector field).

    ```python
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections

    connections.connect(
        alias="default",
        host='localhost',
        port='19530'
    )

    name_id = FieldSchema(
        name="name_id",
        dtype=DataType.INT64,
        is_primary=True
    )
    count = FieldSchema(
        name="count",
        dtype=DataType.INT64
    )
    vector = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=128
    )
    ```

4.  **Create Collection Schema with Dynamic Field Enabled:** In this crucial step, we define the collection schema. We provide the list of initially defined fields and a description for the collection. Importantly, we set the `enable_dynamic_field` parameter to `True`.

    ```python
    collection_schema = CollectionSchema(
        fields=[name_id, count, vector],
        description="Collection with dynamic schema enabled",
        enable_dynamic_field=True
    )
    ```

5.  **Create the Collection:** We then proceed to create the Milvus collection using the defined schema. Let's name this collection `dynamic_schema_example`.

    ```python
    collection = Collection(
        name="dynamic_schema_example",
        schema=collection_schema,
        using='default'
    )
    ```

6.  **Create an Index (Optional):** For demonstration purposes, we might create an index on the vector field. This step is not directly related to the dynamic schema feature but is a common practice for vector collections.

    ```python
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="vector",
        index_params=index_params,
        index_name="vector_index"
    )
    ```

7.  **Insert Entities with a New Field:** Now, let's insert some entities into our `dynamic_schema_example` collection. Observe that the entities we are inserting contain the initially defined fields (`name_id`, `count`, `vector`). However, they also include a new field called `Rand`, which was **not** declared in the original field schema.

    ```python
    import random

    data_to_insert = [
        {"name_id": 1, "count": 10, "vector": [random.random() for _ in range(128)], "Rand": "value_1"},
        {"name_id": 2, "count": 25, "vector": [random.random() for _ in range(128)], "Rand": 123},
        {"name_id": 3, "count": 5, "vector": [random.random() for _ in range(128)], "Rand": {"key": "value"}}
    ]

    collection.insert(data_to_insert)
    ```

    Because we enabled the dynamic schema for this collection, the Milvus server will successfully accept this data, even though it contains the undeclared `Rand` field.

### Conclusion

This demonstrates the functionality of dynamic schema in Milvus collections. By setting the `enable_dynamic_field` parameter to `True` during collection creation, you gain the flexibility to insert data with new, previously undefined fields without altering the original schema.

Thank you for watching, and see you in the next section.
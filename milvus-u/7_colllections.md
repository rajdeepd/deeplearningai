---
layout: default
title: 7. Collections
nav_order: 7
description: ""
has_children: false
parent:  Milvus (U)
---
# Collections

In Milvus, collections and fields function similarly to tables and columns in a relational database. A Milvus collection is composed of one or more partitions, with data stored as segments within these partitions. Each Milvus collection must have a unique name and a collection schema. The collection schema defines the fields within the collection and may include an optional parameter to store a description of the collection. Another optional parameter, `auto_ID`, is used for automatic primary key allocation.

Additionally, there is another optional parameter called `enable_dynamic_field`, which allows you to enable a dynamic data model within Milvus collections. We will cover this particular setting and its usage in a separate session. For now, we'll focus on the fixed schema model, where the fields declared in the collection schema will constitute the final set of fields in that particular schema.

## Field Schemas and Parameters

Each field specified in the collection schema must have an associated field schema. The field schema includes the name and data type of the field as mandatory parameters. There are also several optional parameters, such as `is_primary`, `auto_ID`, `is_partition_key`, `max_length`, and `dim`.

The `is_primary` parameter determines whether a particular field will serve as the primary key. The `auto_I`D parameter allows the primary key field to automatically increment. If this setting is enabled, the primary key field should not be included in the data at the time of insertion. Depending on the data types associated with the field, you can specify dimensions or maximum lengths. The `is_partition_key` parameter determines whether the collection is partitioned based on this particular field.

Milvus has recently introduced custom partitioning keys. Now, let's examine the data types supported in Milvus collections.

## Supported Data Types

Within a collection, a field can be designated as a primary key, scalar field, or vector field. Milvus supports 64-bit integer and varchar data types for primary keys. A scalar field can be a boolean, an integer of 8, 16, 32, or 64 bits, a floating-point number, or a string/varchar data type, which is used to store strings. Recently, Milvus has introduced support for JSON and array types for scalar fields.

For vector fields, Milvus supports binary data types and floating-point data types. More recently, Milvus has added support for sparse float vectors. As the name suggests, binary vectors are supported through the `binary_vector` data type. Milvus also supports three different float vector types and a separate data type called sparse_float_vector.

The `float_vector` type can store 32-bit floating-point numbers, while `float16_vector` can store 16-bit half-precision floating-point numbers, which are common in deep learning and GPU computations. The `bfloat16_vector` type can store vectors with 16-bit floating-point numbers, offering reduced precision but the same exponent range as float32. This type is commonly used in deep learning to reduce memory and computational requirements without significantly impacting accuracy. The sparse_float_vector type is specifically used to efficiently store sparse vectors, where most elements are zero. This data type stores a list of non-zero elements in the vector and their respective indices. Support for sparse vectors in Milvus is currently in beta and is expected to be marked stable in version 3.

## Examples of Basic Operations
Let's explore some basic operations related to collections in Milvus. First, we'll import the necessary packages and modules for this example. Next, we'll connect to the running Milvus server with the required information.

Before creating a collection, we need to define the field schema for the fields that will be part of the collection. For instance, we have a field called song_name, which is of the varchar data type with a maximum allowed length of 200. The next field is `song_ID`, which is a 64-bit integer data type, and we've marked this field as the primary key. Another field, `listen_count`, is also a 64-bit integer data type.

We then have a field named `song_vec`, which is of the `float_vector` data type, making it a vector field. The dimension of these vectors should be 64. The next field is called `song_JSON,` and it is of the JSON data type. Finally, we have a field named song_array, which is of the array data type. The elements of the array are of the 64-bit integer data type. Note that array elements should be of homogeneous data types, and the maximum number of elements in the array is 900.

Once the field schema for all the fields has been declared, we can define the collection schema. The fields of this collection are passed as a list, and a description for this collection schema is also specified. With the collection schema ready, we can proceed to create the collection. The name of the collection will be `album_one`, and the collection schema will be specified accordingly, with the collection created using the default settings. As a result, the collection named `album_one` is successfully created.

To rename a collection, we can use the `rename_collection` method under the `utility` object. The first parameter is the existing name of the collection, and the next is the new name of the collection. After executing this, the collection is successfully renamed.

To delete or drop a collection from the Milvus server, we can use the drop_collection method under the utility object. Consequently, the collection named album_two is removed from the server.


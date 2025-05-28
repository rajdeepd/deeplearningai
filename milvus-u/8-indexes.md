---
layout: default
title: 8 Indexes
nav_order: 8
description: ""
has_children: false
parent:  Milvus (U)
---

## Indexing Data in Milvus

Let's explore indexing data within Milvus collections. Indexes are crucial data structures that significantly improve the speed of data retrieval. Milvus supports building indexes on both vector and scalar fields.

### Vector Indexes

Vector indexes are essential for accelerating vector similarity searches. Without them, searches would rely on brute-force methods, which can be slow for large datasets. To create a vector index, you need to specify:

* **Similarity Metric Type:** Determines how vector similarity is calculated (e.g., Euclidean distance, Cosine distance for float vectors; Jaccard distance, Hamming distance for binary vectors). Some older metrics have been deprecated.
* **Index Type:** The specific algorithm used to build the index (e.g., FLAT, IVF\_FLAT, HNSW).
* **Index Parameters:** Specific settings required by the chosen index type.
* **Vector Field:** The name of the vector field to index.

Vector indexes can be categorized as:

* **In-memory Indexes:** Stored in RAM for fast searching. Examples for float vectors include FLAT, IVF\_FLAT, IVF\_SQ8, IVF\_PQ, HNSW, and SCAN. For binary vectors: BIN\_FLAT and BIN\_IVF\_FLAT. For sparse vectors: SPARSE\_INVERTED\_INDEX and SPARSE\_BAND.
* **On-disk Indexes:** Used when indexes are too large for RAM. DISCANN is the currently supported algorithm.
* **GPU Indexes:** Leverage GPUs for accelerated similarity searches. Examples include GPU\_CAGRA, GPU\_IVF\_FLAT, GPU\_IVF\_PQ, and GPU\_BRUTE\_FORCE.

### Scalar Indexes

Indexes can also be built on scalar fields to improve query performance. By default, you don't need to specify index parameters for scalar fields, as Milvus can determine the appropriate index type based on the field's data type (e.g., Inverted Index for text/JSON/Boolean, STL Sort for numeric, Tree for Varchar). However, you can explicitly specify the index type:

* **Inverted Index**
* **STL Sort**
* **Tree**

### Working with Indexes 

The section will also demonstrate practical examples using the PyMilvus library:

1.  **Connecting to Milvus:** Establishing a connection to the Milvus server.
2.  **Creating a Vector Index:** Specifying the metric type (L2), index type (IVF\_SQ8), index parameters, and the vector field name using the `create_index` method of a collection object.
3.  **Creating a Scalar Index:** Using the `create_index` method with the scalar field name and the desired index type (e.g., Inverted Index).
4.  **Dropping an Index:** Removing an existing index from a collection using the `drop_index` method and specifying the index name.
.
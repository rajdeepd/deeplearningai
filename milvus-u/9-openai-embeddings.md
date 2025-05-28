---
layout: default
title: 9. OpenAI Embeddings
nav_order: 9
description: ""
has_children: false
parent:  Milvus (U)
---

### Working with Text Embeddings in PyMilvus: A Detailed Example

In this section, we'll walk through a practical example of generating text embeddings using the `PyMilvus` library. Specifically, we'll explore how to utilize the `OpenAIEmbeddingFunction` to encode text documents and queries into dense vectors, which are essential for various search and retrieval tasks.

#### Defining the Text Data

We begin by defining a list of text documents that we want to encode into embeddings. In this example, the documents contain a brief description of artificial intelligence:

```python
docs = [
    "Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines,",
    "particularly computer systems. It is a field of research in computer science that develops",
    "and studies methods and software that enable machines to perceive their environment and use",
    "learning and intelligence to take actions that maximize their chances of achieving defined goals.[1]",
    "Such machines may be called AIs."
]
```

Each entry in the `docs` list represents a separate text document that we will transform into a dense vector using an embedding function.

#### Initializing the OpenAI Embedding Function

Next, we initialize the `OpenAIEmbeddingFunction` from the `pymilvus` library. This function will be responsible for generating the embeddings:

```python
from pymilvus import model

# Initialize using 'text-embedding-3-large'
openai_ef = model.dense.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large",  # Specify the model name
    dimensions=512  # Set the embedding dimensionality according to MRL feature.
)
```

Here’s what happens in this step:

- **Model Selection**: We specify the `model_name` as `"text-embedding-3-large"`, which is the particular model we are using to generate the embeddings. This model is pre-trained to understand and convert text into high-dimensional vectors.
- **Dimensionality Setting**: The `dimensions` parameter is set to `512`, meaning that each generated embedding will be a vector with 512 dimensions. This dimensionality corresponds to the model's feature length.

#### Generating Embeddings

Once the embedding function is initialized, we can proceed to generate embeddings for our documents. The code demonstrates two ways to accomplish this:

1. **General Method**: Directly passing the documents or queries to the `openai_ef` function.
   
   ```python
   queries = docs
   queries_embeddings = openai_ef(queries)
   docs_embeddings = openai_ef(docs)
   ```

   In this method, both `queries` and `docs` are passed to the `openai_ef` function, which returns their respective embeddings.

2. **Specified Method**: Using specific functions to encode queries and documents separately.
   
   ```python
   queries_embeddings = openai_ef.encode_queries(queries)
   docs_embeddings = openai_ef.encode_documents(docs)
   ```

   Although the underlying process is the same, this method allows for clarity and specificity in use cases where queries and documents might require different handling or where clear function separation is needed.

#### Verifying the Embedding Dimensions

Finally, after obtaining the embeddings, we can verify their dimensions to ensure that they match our expectations:

```python
# Now we can check the dimension of embedding from results and the embedding function.
print("dense dim:", openai_ef.dim, queries_embeddings[0].shape)
print("dense dim:", openai_ef.dim, docs_embeddings[0].shape)
```

This code prints out the dimensionality of the generated embeddings. The `openai_ef.dim` should return `512`, and the `shape` of the first embedding vector should confirm this dimensionality, indicating that each text document has been successfully transformed into a dense vector of 512 dimensions.

### Summary

This example demonstrates how to encode text documents into dense embeddings using the `PyMilvus` library's `OpenAIEmbeddingFunction`. By specifying the model and dimensions, and then generating the embeddings either directly or through specific functions, you can prepare your text data for advanced search and retrieval operations. The final verification step ensures that the embeddings meet the expected dimensionality, making them ready for further use in your applications.
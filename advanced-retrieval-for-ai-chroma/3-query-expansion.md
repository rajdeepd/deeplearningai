---
layout: default
title: 3. Query Expansion 
nav_order: 3
description: "Chroma for RAG"
has_children: true
parent:  Advanced Retrieval for AI - Chroma
---

We can use LLMs to enhance our query results. Lets take a look. 

First type of query expansion is called expansion with generated results. Take a query and send it to LLM to generate an imagined answer to your query. Concatenate your query with imagined answer and send it back as the new query.

Let us take a look how this works in practice.


```python
from helper_utils import load_chroma, word_wrap, project_embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
```

We will import `helper_utils` and `embedding_functions`. Load the content of microsoft annual report into `chroma_collection` using `SentenceTransformerEmbeddingFunction`.


```python
embedding_function = SentenceTransformerEmbeddingFunction()

chroma_collection = load_chroma(filename='microsoft_annual_report_2022.pdf', collection_name='microsoft_annual_report_2022', embedding_function=embedding_function)
chroma_collection.count()
```
Instantiate openai_client

```python
import os
import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI()
```

Do a umap transformation.

```python
import umap

embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
```

Create a function to help get response based on content which asks model to hallucinate
We are passing two roles: system and user

```python
def augment_query_generated(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report. "
        },
        {"role": "user", "content": query}
    ] 

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content
```
Lets passon the original query to this function. Original query prepending the hypothetical response 


```python
original_query = "Was there significant turnover in the executive team?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))
```
Print the response


```
Was there significant turnover in the executive team? In the past
fiscal year, there was minimal turnover in the executive team. Only one
member, the Chief Financial Officer, retired after 15 years of
dedicated service to the company. The board immediately began a
comprehensive search for a suitable replacement, and we are pleased to
announce that a new CFO with extensive experience in the industry has
been appointed. This change has been seamless and we do not anticipate
any disruptions to our operations due to this transition.
```

Let us send this joint_query back to LLM


```python
results = chroma_collection.query(query_texts=joint_query, n_results=5, include=['documents', 'embeddings'])
retrieved_documents = results['documents'][0]

for doc in retrieved_documents:
    print(word_wrap(doc))
    print('')
```

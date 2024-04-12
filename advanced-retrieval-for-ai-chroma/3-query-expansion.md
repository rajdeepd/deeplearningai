---
layout: default
title: 3. Query Expansion 
nav_order: 3
description: "Chroma for RAG"
has_children: true
parent:  Advanced Retrieval for AI - Chroma
---

Field of information retrieval has been around for a while as a subfield of natural language processing, and there's many approaches to improving the relevancy of query results. But what's new is we have powerful large language models, and we can use those to augment and enhance the queries that we send to our vector-based retrieval system to get better results. Let's take a look.

## Expansion with Generated answers

First type of query expansion we're going to talk about is called expansion with generated answers.
Typically the way that this works is you take your query and you pass it over to an LLM which you prompt to generate a hypothetical or imagined answer to your query and then you concatenate your query with the imagined answer and use that as the new query which you pass to your retrieval system or vector database.
Then you return your query results as normal. Let's take a look at how this works in practice.


<img src="/deeplearningai/advanced-retrieval-for-ai-chroma/images/Screenshot_2024-04-09_at_12.00.21 PM.png" width="80%" />

So the first thing that we're going to do is once again and grab all the utilities that we need.
We're going to load everything we need from Chroma and create our embedding function.
We're going to set up our OpenAl client again, because we'll be using the LLM.


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
Instantiate `openai_client`

```python
import os
import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI()
```
To help with visualization, we're going to use UMAP and project our data set so that that's all ready to go for us.
Now that we're done setting up, let's take a look at expansion with generated answers and there's a reference here to the paper which demonstrates some of the empirical results that you can get by applying this method.

```python
import umap

embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
```

So to do expansion with generated answers we're yoing to use an LLM, in this case GPT, and just the same as last time we're going to prompt the model in a particular way.
Let's create this function called augment query generated and we're going to pass in a query, we're also going to pass in a model argument, in this case GPT 3.5 turbo by default, and we're going to prompt the model and in the system prompt we're going to say you're a helpful expert financial research assistant, provide an example answer to the given question that might be found in a document, like an annual report.
.

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

In other words, we're  asking the model to hallucinate, but we're gonna use that hallucination for something useful. In the user prompt, we're just going to pass the query as the content.
And then we'll do our usual unpacking of the response

That defines how we're going to prompt our model. Let's wire this together.Here's our original query, asking was there a significant turnover in the executive team.
We will generate a hypothetical answer and then we'll create our joint query which is basically the original query prepending the hypothetical answer. Let us take a look at what this actually looks like after we generate it.

```python
original_query = "Was there significant turnover in the executive team?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))
```
In the past fiscal year there was no significant turnover in the executive team, the core members of the executive team remained unchanged, etc. So let's send this query plus the hypothetical critical response to our retrieval system as a query.


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

And we'll querv the Chroma Collection the usual way and print out our results.
And we're sending the joint query as the query to our retrieval system. And we're retrieving the documents and the embeddings together. 


```python
results = chroma_collection.query(query_texts=joint_query, n_results=5, include=['documents', 'embeddings'])
retrieved_documents = results['documents'][0]

for doc in retrieved_documents:
    print(word_wrap(doc))
    print('')
```

You can inspect the output <a href="deeplearningai/advanced-retrieval-for-ai-chroma/3.1-output1.html" >here</a>.

```

```
So these are the documents that we get. Back, we see things here discussing leadership. We see how consultants and directors work together. Here we have an overview of the different directors that we had in Microsoft. And we talk about the different board committees. Let's visualize this. Let's see what sort of difference this made. 

So, to do that, we get our retrieved embeddings, we get the embedding for our original query, we get the embedding for our joint query, and then we project all three. I'm plotting the projection. And we see the red $X$ is our original query, the orange $X$ is our new query with the hypothetical answer. 

```python
retrieved_embeddings = results['embeddings'][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(original_query_embedding, umap_transform)
projected_augmented_query_embedding = project_embeddings(augmented_query_embedding, umap_transform)
projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)
```

    100%|██████████| 1/1 [00:01<00:00,  1.55s/it]
    100%|██████████| 1/1 [00:01<00:00,  1.46s/it]
    100%|██████████| 5/5 [00:05<00:00,  1.10s/it]



```python
import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')
plt.scatter(projected_original_query_embedding[:, 0], projected_original_query_embedding[:, 1], s=150, marker='X', color='r')
plt.scatter(projected_augmented_query_embedding[:, 0], projected_augmented_query_embedding[:, 1], s=150, marker='X', color='orange')

plt.gca().set_aspect('equal', 'datalim')
plt.title(f'{original_query}')
plt.axis('off')
```




And we see that we get this nice cluster of results, 

<img src="/deeplearningai/advanced-retrieval-for-ai-chroma/images/L3_output_10_2.png" width="80%" />

Original response was Red but the new response with augmented response is orange. We want to illustrate here is that using the hypothetical answer moves our query elsewhere in space hopefully producing better results for us.

## Query Expansion with Multiple Queries

So that was query expansion with generated queries, but there's another type of query expansion we can also try. This is called query expansion with multiple queries. And the way that you use this is to use the LLM to generate additional queries that might help answering the question. So what you do here is you take your original query, you pass it to the LLM, you ask the. LLM to generate several new related queries to the same original query, and then you pass those new queries along with your original query to the vector database or your retrieval system. That gives you results for the original and the new queries, and then you pass all of those results to the LLM to complete the RAG loop.
So let's take a look at how this works in practice.

<img src="/deeplearningai/advanced-retrieval-for-ai-chroma/images/Screenshot_2024-04-09_at_12.00.27 PM.png" width="80%" />


Once again, the starting point is a prompt in the model, and we see here that we have a system prompt, and the system prompt is a bit more detailed this time. We take in a query, which is our original query, and we ask the model. 





It's a helpful expert financial research assistant. The users are asking questions about an annual report, so this gives the model enough context to know what sorts of queries to generate. And then you say, suggest up to five additional related questions to help them find the information they need for the provided question. Suggest only short questions without compound sentences, and this makes sure that we get simple queries.
Suggest a variety of questions that cover different aspects of the topic, and this is very important because there are many ways to rephrase the same query, but what we're actually asking for is different but related queries. And finally, we want to make sure that they're complete questions, they're related to the original question, and we ask some formatting output.
One important thing to understand about these techniques in particular that bring an LLM into the loop of retrieval is prompt engineering becomes a concern. It's something that you have to think about and I really recommend that you as a student play with these prompts once you have a chance to try the lab.
See how they may change, see what different types of queries you can get the models to generate,
and the new queries, and then you pass all of those results to the LLM to complete the RAG loop.
So let's take a look at how this works in practice.

Once again, the starting point is a prompt in the model, and we see here that we have a system prompt, and the system prompt is a bit more detailed this time. We take in a query, which is our original query, and we ask the model. It's a helpful expert financial research assistant. The users are asking questions about an annual report, so this gives the model enough context to know what sorts of queries to generate. And then you say, suggest up to five additional related questions to help them find the information they need for the provided question. Suggest only short questions without compound sentences, and this makes sure that we get simple queries.
Suggest a variety of questions that cover different aspects of the topic, and this is very important because there are many ways to rephrase the same query, but what we're actually asking for is different but related queries.

And finally, we want to make sure that they're complete questions, they're related to the original question, and we ask some formatting output.

```python
def augment_multiple_query(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about an annual report. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content
```

One important thing to understand about these techniques in particular that bring an LLM into the loop of retrieval is prompt engineering becomes a concern. It's something that you have to think about and I really recommend that you play with these prompts once you have a chance to try the lab. See how they may change, see what different types of queries you can get the models to generate, array of augmented queries. So now we have one array where each entry is a query, our original query, plus the augmented queries.

We can grab the results. Chroma can do querying in batches. Let us look at the retrieved documents that we get.

One thing that's important here is because the queries are related, you might get the same document retrieved for more than one query. What we need to do is to deduplicate the retrieved and that's what we do here. Finally, let's just output the documents that we get. So we can see now the documents that we get for each query.

```python
queries = [original_query] + augmented_queries
results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents', 'embeddings'])

retrieved_documents = results['documents']

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

for i, documents in enumerate(retrieved_documents):
    print(f"Query: {queries[i]}")
    print('')
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print('')
    print('-'*100)
```
These are all to do with revenue, different aspects of revenue growth, which is exactly what we were hoping for. We have increases in Windows revenue.
We can see things that are coming from other components.
For example, what were the most important factors that contributed to decreases in revenue? So we see increased sales and marketing expenses, different types of investments, different types of tax breaks.
You can inspect the output <a href="deeplearningai/advanced-retrieval-for-ai-chroma/3.2-output2.html" >here</a>.

Essentially, each of these augmented queries are providing us with a slightly different set of results. And let's visualize that. What did we actually get in geometric space in response to these results?
We will take our original query embedding and our augmented query embeddings and project them.


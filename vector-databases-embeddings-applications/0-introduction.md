---
layout: default
title: 0. Introduction
nav_order: 1
description: ".."
has_children: false
parent:  Vector Databases
---

Hi, and welcome to this course, Vector Databases from Embeddings to Applications, this has been built in partnership with Weaviate. 
Large language models have enabled many new and exciting applications to be built. But a known shortcoming of LLMs is that a trained language model does not have knowledge of recent events, or knowledge available only in proprietary documents that it did not get to train on. 

To tackle this problem, you can use **retrieval augmented generation or RAG**. 
Key component of RAG is a vector database. 

Proprietary or recent data is first stored in this vector database. 
Then, when there's a query that concerns that information, that query is sent to the vector database which then retrieves the related text data. And finally, this retrieved text can be included in the prompt to the LLM to give it context with which to answer your question. 

**Vector databases** preceded recent generative AI explosion. They have long been a broader part of semantic search applications. These are applications that 
search on the meaning of words or phrases rather than keyword search that looks for exact matches, as well as in recommender systems where they've been used to find related items to recommend to a user. 

It would be really useful for you as an AI developer to understand how a vector 
database works, what really goes on under the hood. 
This will allow you to use vector databases more effectively in your own project. So for example, you know how to decide when to apply sparse search such as 
keyword search or dense search, which is what you get with vector similarities, or hybrid search which combines both sparse and dense search. 

Understanding how different similarity calculations work, will also help you to choose the best distance algorithm. 
Understanding this challenge of scaling vector databases and search will help you to choose between different embedding search algorithms.

Thanks, Andrew. It's a real privilege to be working 
with you on this course. By the end of this course, you'll understand 
and implement many of the elements that make 
up vector databases. Things like embeddings, dense vectors 
that represent the meaning of phrase, distance metrics, 
like dot product or cosine distance, different kinds 
of vector search, things like linear search, where you 
look at all the entries in a database, or approximate 
search, where you speed up search by allowing for 
results that are close and also different search paradigms 
like sparse, dense, and hybrid search. And finally, you build 
real-world applications of vector databases, creating a rack 
system with hybrid and multilingual search functionality. 
 
Let's go on to the next section to get started. 
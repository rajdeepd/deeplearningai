---
layout: default
title: 0. Introduction
nav_order: 1
description: "Chroma for RAG"
has_children: false
parent:  Advanced Retrieval for AI - Chroma
---

Retrieving relevant documents gives context to a language learning model (LLM), significantly enhancing its ability to answer queries and perform tasks. Many teams employ simple retrieval techniques based on semantic similarity or embeddings. However, through this course, I've learned more sophisticated techniques that allow for far superior outcomes. A common workflow in Retrieval-Augmented Generation (RAG) involves taking a query, embedding it, and then finding the most similar documents—those with similar embeddings—to provide context. Yet, this approach often leads to documents that discuss topics similar to the query but don't actually contain the answer.

To overcome this, I've learned to take the initial user query and rewrite it, a process known as query expansion, which helps pull in more directly related documents. This involves expanding the original query into multiple queries by rewording or rewriting it in different ways. Additionally, it involves guessing or hypothesizing what the answer might look like to see if anything in our document collection more directly resembles an answer rather than merely discussing the topics of the query.

I'm truly excited to be working with Andrew on this course and to share what I'm observing in the field regarding what does and doesn't work in RAG deployments. We'll start the course with a quick review of RAG applications and then delve into some of the pitfalls of retrieval where simple vector search falls short. We'll explore several methods to improve the results, including using an LLM to enhance the query itself and re-ranking query results with the help of a cross encoder, which assesses a pair of sentences to produce a relevancy score. Additionally, I'll learn how to adapt the query embeddings based on user feedback, further refining the retrieval process.

There's a lot of innovation happening in Retrieval-Augmented Generation (RAG) right now. In the final lesson, we'll also explore some of the cutting-edge techniques that aren't mainstream yet and are just now appearing in research. I believe these techniques will soon become much more widely adopted. We'd like to extend our gratitude to some of the individuals who have contributed to this course. From the Chroma team, we're thankful for Jeff Huber, Hammad Bashir, Liquan Pei, and Ben Eggers, as well as the support from Chroma's open-source developer community. Additionally, from the Deep Learning team, we have Geoff Ladwig and Esmael Gargari to thank.

The first lesson begins with an overview of RAG, and I hope you'll continue to watch that right after this. It's fascinating to see that, with these techniques, smaller teams than ever before have the capability to build effective systems. So, after completing this course, I might be in a position to build something really cool with an approach that previously would have been considered makeshift or "RAG tag.





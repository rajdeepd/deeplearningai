---
layout: default
title: 0. Introduction
nav_order: 1
description: "Introduction to Multimodal Search and RAG with Weaviate"
has_children: false
parent:  Building Multimodal Search and RAG - Weaviate
---

Welcome to this short course on building multi-modal search and RAG (Retrieval Augmented Generation) systems, using Weaviate. RAG systems provide a large language model (LLM) with context that includes information about your proprietary data, enabling the LLM to use that context when generating responses. A common way to build RAG applications is to use an active database to store your text with embeddings. Then, given a query, you retrieve relevant information from the vector database and add that as text context to your prompt.

But what if the context you need includes an image of a presentation, an audio clip, or even a video? This section teaches you the technical details behind implementing RAG with such multi-modal data. The first step is to find a way to compute embeddings so that data on related topics is embedded similarly, regardless of modality. For example, a text about a lion, an image showing a lion, and a video or audio of a lion roaring should be embedded close to each other so that a query about lions can retrieve all of this data. In other words, we want the embedding of concepts to be modality-independent. You will learn how this is done through a process called contrastive learning in the next video.

After developing a multi-modal retrieval model, you will use it to retrieve the context related to a user's query. This enables you to build a multi-modal search app where an image of a lion can be used to retrieve video, audio, and text related to that image. If your generative model supports multi-modal inputs, you can use it to retrieve results as context and provide it to the model to generate responses based on the relevant multi-modal contextual information.

In this section, you will first learn how to teach a computer to understand multi-modal data. Then, you will build a text-to-any as well as an any-to-any search system.

In the next step, you will learn how to combine language and multi-modal models into language-visual models that understand images as well as text. Next, you will focus on multi-modal RAG by mixing multi-modal search with multi-modal generation and reasoning.

As a final step, you will learn how multi-modality is used in the industry by implementing real-life examples, such as analyzing invoices and flowcharts. Many people have worked to create this course. I'd like to thank Zain Hasan from Weaviate, as well as Esmaeil Gargari from DeepLearning.AI, who contributed to this course.
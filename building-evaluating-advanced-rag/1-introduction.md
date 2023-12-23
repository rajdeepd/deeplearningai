---
layout: default
title: Introduction
nav_order: 1
description: "Computational challenges of training LLMs"
has_children: false
parent:  Deeplearning RAG
---

Retrieval, augmented generation or RAG has become a key method for getting MS answered questions over a users own data. But to actually build and productionize a high quality rag system costs a lot to have effective retrieval techniques to give the LM highly relevant context to generate his answer and also to have an effective evaluation framework to help you efficiently iterate and improve your rag system, both during initial development and during post deployment maintenance. 

This course covers two advanced retrieval methods, **sentence window retrieval** and **autoemerging retrieval** that deliver a significantly better context to LLM than simpler methods. It also covers how to evaluate your LLM question answering system with three evaluation metrics, context relevance, droughts and on relevance. 

Sentence window retrieval gives an LLM better context by retrieving not just the most relevant sentence but the window of sentences that occur before and after it in the document. Auto emerging retrieval organizes the document into a tree like structure where each parent nodes, text is divided among its child nodes. When enough child nodes are identified as relevant to a user's question, then the entire text of the parent node is provided as context for the L one. 

I know this sounds like a lot of steps but don't worry, we'll go over it in detail on code later. But the main takeaway is that this provides a way to dynamically retrieve more coherent chunks of text than simpler methods to evaluate rag based LLM apps, the rag triad, a triad of metrics for the three main steps of a rags execution is quite effective. For example, we'll cover in detail how to compute context relevance, which measures how relevant the retrieve chunks of text are to the user's question. This helps you identify and debug possible issues with how your system is retrieving context for the LM in the Q A system. But that's only part of the overall Q A system. 

We'll also cover additional evaluation metrics such as roundedness and answer relevance that let you systematically analyze what parts of your system are or are not yet working well, so that you can go in in a targeted way to improve whatever part needs the most work. If you're familiar with the concept of error analysis and machine learning, this has similarities and I've found that taking this sort of systematic approach helps you be much more efficient in building a reliable Q A system. Lets go of this course is to help you build production ready, right based OM apps and important parts of getting production ready is to iterate in a systematic way on the system. 

In the later half of this course, you gain hands on practice iterating using these retrieval methods and evaluation methods. And you also see how to use systematic experiment tracking to establish a baseline and then quickly improve on that. We'll also share some suggestions for tuning these two retrieval methods based on our experience, assisting partners who are building rag apps.  

The next lesson will give you an overview of what you'll see in the rest of the course. You'll try out question answering systems that use sentence window retrieval or auto merging retrieval and compare their performance on the Rag triad, Context Relevance and Grounded relevance. Sounds great. Let's get started.
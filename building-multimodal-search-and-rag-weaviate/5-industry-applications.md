---
layout: default
title: 5. Industry Applications
nav_order: 6
description: "Multimodal RAG"
has_children: false
parent:  Building Multimodal Search and RAG - Weaviate
---

In this lesson you will learn how multimodality is used in industry by implementing real life examples. You will analyze image content like invoices and flowcharts to generate structured output in different formats and stats.
All right. Let's get building.
In this lesson, you are going to work on three different applications of multimodality in industry.

In the first example, the input is an image of structured data, like a receipt or an invoice.
And then you extract structured fields and values from this image into a json format.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-27_at_8.27.36_PM.png"  width="80%" /> 


In the second example, you start over with a table from a company's investors deck and we'll extract out a markdown tabular representation that then can be processed and used. 

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-27_at_8.27.51_PM.png"  width="80%" /> 


In the third example, you'll get a language vision model to reason over logical flowchart and get it to output text or even python code that implements that logical flow.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-27_at_8.28.11_PM.png"  width="80%" /> 

In here you start with pretty much the same setup as you did in the previous lessons.
We need to ignore the warnings and also allow the vision API keys and then set up the, genai
library, and that's good.
Again, you're going to use pretty much the same helper functions.
So, one that converts an output to markdown, but there is a slight a modification to the call_LLM function, where you have plain text boolean.
The purpose of this is that depending of what we want, we may want to return the output as just plain text, or output it as markdown, which will come handy later on.
As the first example, you're going to use the vision model to analyze this invoice.
Let's now ask a question, given the invoice file, let's try to identify the items on the invoice and then output the results as json.
We're looking for quantity, description, unit price, and amount. So let's run this and let's see what the vision model can extract from this.

Okay. We see for the the second item, "new set of pedal arms", the unit price of 15 and a total amount for 30, which matches exactly what we have on the table.
It's pretty accurate. So, yeah, this is pretty awesome.
How about you ask now a reasoning question based on the input. Maybe you could check
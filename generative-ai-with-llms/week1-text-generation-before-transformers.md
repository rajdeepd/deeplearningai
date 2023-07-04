---
layout: default
title: Text generation before transformers
nav_order: 3
description: "Generative AI with Large Language Models"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---

# Text generation before transformers

It's important to note that generative algorithms are not new. Previous generations of language models made use of an architecture called recurrent neural networks or RNNs. RNNs while powerful for their time, were limited by the amount of compute and memory needed to perform well at generative tasks. 

## Generating text using RNNs

Let's look at an example of an RNN carrying out a simple next-word prediction generative task. With just one previous words seen by the model, the prediction can't be very good. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_8.20.41_PM.png" width="80%" />

As you scale the RNN implementation to be able to see more of the preceding words in the text, you have to significantly scale the resources that the model uses. As for the prediction, well, the model failed here. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_8.20.59_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_8.21.13_PM.png" width="80%" />


Even though you scale the model, it still hasn't seen enough of the input to make a good prediction. To successfully predict the next word, models need to see more than just the previous few words.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_8.21.23_PM.png" width="80%" />

Models needs to have an understanding of the whole sentence or even the whole document. 

## Understanding Languages can be challenging

The problem here is that language is complex. In many languages, one word can have multiple meanings. These are homonyms. In this case, it's only with the context of the sentence that we can see what kind of bank is meant. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_9.34.48_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_9.35.11_PM.png" width="80%" />


Words within a sentence structures can be ambiguous or have what we might call syntactic ambiguity. Take for example this sentence, "The teacher taught the students with the book." Did the teacher teach using the book or did the student have the book, or was it both? How can an algorithm make sense of human language if sometimes we can't? 

## Transformers

In 2017, after the publication of this paper, Attention is All You Need, from Google and the University of Toronto, everything changed.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_9.35.28_PM.png" width="80%" />


The transformer architecture had arrived. This novel approach unlocked the progress in generative AI that we see today.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-04_at_9.35.48_PM.png" width="80%" />

It can be scaled efficiently to use multi-core GPUs, it can parallel process input data, making use of much larger training datasets, and crucially, it's able to learn to pay attention to the meaning of the words it's processing. And attention is all you need. It's in the title.
---
layout: default
title: Model evaluation
nav_order: 6
description: "Instruction fine tuning"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---

## Model evaluation

### Introduction

Throughout this course, you've seen statements like the model demonstrated good performance on this task or this fine-tuned model showed a large improvement in performance over the base model. What do statements like this mean? How can you formalize the improvement in performance of your fine-tuned model over the pre-trained model you started with? Let's explore several metrics that are used by developers of large language models that you can use to assess the performance of your own models and compare to other models out in the world


 In traditional machine learning, you can assess how well a model is doing by looking at its performance on training and validation data sets where the output is already known. You're able to calculate simple metrics such as accuracy, which states the fraction of all predictions that are correct because the models are deterministic. But with large language models where the output is non-deterministic and language-based evaluation is much more challenging

### LLM evaluation challenges

Take, for example, the sentence, Mike really loves drinking tea. This is quite similar to Mike adores sipping tea. But how do you measure the similarity? Let's look at these other two sentences. 

Mike does not drink coffee, and Mike does drink coffee. 

There is only one word difference between these two sentences. However, the meaning is completely different. 

For humans like us with squishy organic brains, we can see the similarities and differences. But when you train a model on millions of sentences, you need an automated, structured way to make measurements.

### LLM Evaluation - Metrics

**ROUGE** and **BLEU**, are two widely used evaluation metrics for different tasks. ROUGE or recall oriented under study for jesting evaluation is primarily employed to assess the quality of automatically generated summaries by comparing them to human-generated reference summaries. On the other hand, BLEU, or bilingual evaluation understudy is an algorithm designed to evaluate the quality of machine-translated text, again, by comparing it to human-generated translations. Now the word BLEU is French for blue. You might hear people calling this blue but here I'm going to stick with the original BLEU. 

### LLM Evaluation - Metrics - Terminology

Before we start calculating metrics. Let's review some terminology. In the anatomy of language, a unigram is equivalent to a single word. A bigram is two words and n-gram is a group of n-words. Pretty straightforward stuff. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-31_at_7.08.32_PM.png" width="80%" />



### LLM Evaluation - Metrics - ROUGE-1

First, let's look at the ROUGE-1 metric. To do so, let's look at a human-generated reference sentence.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-31_at_7.10.04_PM.png" width="80%" />

It is cold outside and a generated output that is very cold outside. You can perform simple metric calculations similar to other machine-learning tasks using recall, precision, and F1. The recall metric measures the number of words or unigrams that are matched between the reference and the generated output divided by the number of words or unigrams in the reference. In this case, that gets a perfect score of one as all the generated words match words in the reference. Precision measures the unigram matches divided by the output size. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-31_at_7.10.30_PM.png" width="80%" />

The F1 score is the harmonic mean of both of these values.




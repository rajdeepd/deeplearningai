---
layout: default
title: Benchmarks
nav_order: 7
description: "Benchmarks"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---

As you saw in the last section, LLMs are complex, and simple evaluation metrics like the rouge and blur scores, can only tell you so much about the capabilities of your model. In order to measure and compare LLMs more holistically, you can make use of pre-existing datasets, and associated benchmarks that have been established by LLM researchers specifically for this purpose. Selecting the right evaluation dataset is vital, so that you can accurately assess an LLM's performance, and understand its true capabilities. You'll find it useful to select datasets that isolate specific model skills, like reasoning or common sense knowledge, and those that focus on potential risks, such as disinformation or copyright infringement. An important issue that you should consider is whether the model has seen your evaluation data during training

## Evaluation Benchmarks

You'll get a more accurate and useful sense of the model's capabilities by evaluating its performance on data that it hasn't seen before. Benchmarks, such as GLUE, SuperGLUE, or Helm, cover a wide range of tasks and scenarios. They do this by designing or collecting datasets that test specific aspects of an LLM.

## Glue

GLUE, or General Language Understanding Evaluation, was introduced in 2018. GLUE is a collection of natural language tasks, such as sentiment analysis and question-answering. GLUE was created to encourage the development of models that can generalize across multiple tasks, and you can use the benchmark to measure and compare the model performance.

## Super Glue

As a successor to GLUE, SuperGLUE was introduced in 2019, to address limitations in its predecessor. It consists of a series of tasks, some of which are not included in GLUE, and some of which are more challenging versions of the same tasks. SuperGLUE includes tasks such as multi-sentence reasoning, and reading comprehension.

## Glue and Super Glue Leaderboards

Both the GLUE and SuperGLUE benchmarks have leaderboards that can be used to compare and contrast evaluated models.

Disclaimer: metrics may not be up-to-date. Check [super.gluebenchmark.com](https://super.gluebenchmark.com) and [gluebenchmark.com](https://gluebenchmark.com/leaderboard) for the latest.

## Benchmarks for massive models

The results page is another great resource for tracking the progress of LLMs. As models get larger, their performance against benchmarks such as SuperGLUE start to match human ability on specific tasks. That's to say that models are able to perform as well as humans on the benchmarks tests, but subjectively we can see that they're not performing at human level at tasks in general. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.09.43_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.09.51_PM.png" width="80%" />



There is essentially an arms race between the emergent properties of LLMs, and the benchmarks that aim to measure them. Here are a couple of recent benchmarks that are pushing LLMs further. Massive Multitask Language Understanding, or MMLU, is designed specifically for modern LLMs. To perform well models must possess extensive world knowledge and problem-solving ability. Models are tested on elementary mathematics, US history, computer science, law, and more. In other words, tasks that extend way beyond basic language understanding. BIG-bench currently consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more. BIG-bench comes in three different sizes, and part of the reason for this is to keep costs achievable, as running these large benchmarks can incur large inference costs

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.10.48_PM.png" width="80%" />

Source: Hendrycks, 2021. "Measuring Massive Multitask Language Understanding"

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.11.19_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.19.02_PM.png" width="80%" />

Source: Suzgun et al. 2022. "Challenging BIG-Bench tasks and whether chain-of-thought can solve them"

## Holistic Evaluation of Language Models (HELM)

A final benchmark you should know about is the Holistic Evaluation of Language Models, or HELM. The HELM framework aims to improve the transparency of models, and to offer guidance on which models perform well for specific tasks

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.20.07_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_8.24.46_PM.png" width="80%" />

HELM takes a multimetric approach, measuring seven metrics across 16 core scenarios, ensuring that trade-offs between models and metrics are clearly exposed. One important feature of HELM is that it assesses on metrics beyond basic accuracy measures, like precision of the F1 score. The benchmark also includes metrics for fairness, bias, and toxicity, which are becoming increasingly important to assess as LLMs become more capable of human-like language generation, and in turn of exhibiting potentially harmful behavior. HELM is a living benchmark that aims to continuously evolve with the addition of new scenarios, metrics, and models. You can take a look at the results page to browse the LLMs that have been evaluated, and review scores that are pertinent to your project's needs.
---
layout: default
title: Pretraining Large language models
nav_order: 8
description: "Pretraining Large language models"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---

# Pretraining Large language models

## Generative Al project lifecycle

In the previous section, you were introduced to the generative AI project life cycle. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.38.59_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.39.07_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.39.33_PM.png" width="80%" />




As you saw, there are a few steps to take before you can get to the fun part, launching your generative AI app.

## Considerations for choosing a model

Once you have scoped out your use case, and determined how you'll need the LLM to work within your application, your next step is to select a model to work with. Your first choice will be to either work with an existing model, or train your own from scratch. There are specific circumstances where training your own model from scratch might be advantageous, and you'll learn about those later in this lesson. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.39.33_PM.png" width="80%" />

In general, however, you'll begin the process of developing your application using an existing foundation model. Many open-source models are available for members of the AI community like you to use in your application. The developers of some of the major frameworks for building generative AI applications like Hugging Face and PyTorch, have curated hubs where you can browse these models.

## Model hubs

A really useful feature of these hubs is the inclusion of model cards, that describe important details including the best use cases for each model, how it was trained, and known limitations. You'll find some links to these model hubs in the reading at the end of the week. The exact model that you'd choose will depend on the details of the task you need to carry out. Variance of the transformer model architecture are suited to different language tasks, largely because of differences in how the models are trained.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.39.45_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.40.10_PM.png" width="80%" />

To help you better understand these differences and to develop intuition about which model to use for a particular task, let's take a closer look at how large language models are trained. With this knowledge in hand, you'll find it easier to navigate the model hubs and find the best model for your use case. 

## Model architectures and pre-training objectives


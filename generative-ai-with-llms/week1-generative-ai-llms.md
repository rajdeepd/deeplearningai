---
layout: default
title: Week 1 Generative AI LLMs
nav_order: 2
description: "Generative AI with Large Language Models"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---
nav_order: 2
description: "Generative AI with Large Language Models"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---

# Generative AI & LLMs

In this lesson, we're going to set the scene. We'll talk about large language models, their use cases, how the models work, prompt engineering, how to make creative text outputs, and outline a project lifecycle for generative AI projects. Given your interest in this course, it's probably safe to say that you've had a chance to try out a generative AI tool or would like to. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-02_at_3.39.30_PM.png" width="80%" />

Whether it be a chat bot, generating images from text, 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-02_at_3.39.32_PM.png" width="80%" />

or using a plugin to help you develop code, what you see in these tools is a machine that is capable of creating content that mimics or approximates human ability. 

## Generative AI

Generative AI is a subset of traditional machine learning. 

And the machine learning models that underpin generative AI have learned these abilities by finding statistical patterns in massive datasets of content that was originally generated by humans. 

## Large Language Models

Large language models have been trained on trillions of words over many weeks and months, and with large amounts of compute power. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-02_at_3.40.24_PM.png" width="80%" />


These foundation models, as we call them, with billions of parameters, exhibit emergent properties beyond language alone, and researchers are unlocking their ability to break down complex tasks, reason, and problem solve. Here are a collection of foundation models, sometimes called base models, and their relative size in terms of their parameters.

You'll cover these parameters in a little more detail later on, but for now, think of them as the model's memory. And the more parameters a model has, the more memory, and as it turns out, the more sophisticated the tasks it can perform. Throughout this course, we'll represent LLMs with these purple circles, and in the labs, you'll make use of a specific open source model, flan-T5, to carry out language tasks. 

By either using these models as they are or by applying fine tuning techniques to adapt them to your specific use case, you can rapidly build customized solutions without the need to train a new model from scratch. Now, while generative AI models are being created for multiple modalities, including images, video, audio, and speech, in this course you'll focus on large language models and their uses in natural language generation. You will see how they are built and trained, how you can interact with them via text known as prompts. And how to fine tune models for your use case and data, and how you can deploy them with applications to solve your business and social tasks. The way you interact with language models is quite different than other machine learning and programming paradigms. In those cases, you write computer code with formalized syntax to interact with libraries and APIs. In contrast, large language models are able to take natural language or human written instructions and perform tasks much as a human would.

## Prompts and Completion

The text that you pass to an LLM is known as a prompt. The space or memory that is available to the prompt is called the context window, and this is typically large enough for a few thousand words, but differs from model to model.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-02_at_7.51.04_PM.png" width="80%" />

In this example, you ask the model to determine where Ganymede is located in the solar system. The prompt is passed to the model, the model then predicts the next words, and because your prompt contained a question, this model generates an answer.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-02_at_7.51.20_PM.png" width="80%" />

The output of the model is called a completion, and the act of using the model to generate text is known as inference.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-02_at_7.52.21_PM.png" width="80%" />


The completion is comprised of the text contained in the original prompt, followed by the generated text. You can see that this model did a good job of answering your question. It correctly identifies that Ganymede is a moon of Jupiter and generates a reasonable answer to your question stating that the moon is located within Jupiter's orbit. You'll see lots of examples of prompts and completions in this style throughout the course.
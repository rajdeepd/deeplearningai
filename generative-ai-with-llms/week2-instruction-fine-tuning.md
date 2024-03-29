---
layout: default
title: Instruction fine tuning
nav_order: 2
description: "Instruction fine tuning"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---

## Instruction fine tuning

Last week, you were introduced to the generative AI project lifecycle. You explored example use cases for large language models and discussed the kinds of tasks that were capable of carrying out. In this lesson, you'll learn about methods that you can use to improve the performance of an existing model for your specific use case. You'll also learn about important metrics that can be used to evaluate the performance of your finetuned LLM and quantify its improvement over the base model you started with. Let's start by discussing how to fine tune an LLM with instruction prompts. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.43.56_PM.png" width="80%" />


## Fine-tuning an LLM with instruction prompts

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.43.45_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.44.14_PM.png" width="80%" />
## In-context learning (ICL) - zero shot inference

Earlier in the course, you saw that some models are capable of identifying instructions contained in a prompt and correctly carrying out zero shot inference, while others, such as smaller LLMs, may fail to carry out the task.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.47.33_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.47.47_PM.png" width="80%" />


Like the example shown above.

## In-context learning (ICL) - one/few shot inference

You also saw that including one or more examples of what you want the model to do, known as one shot or few shot inference, can be enough to help the model identify the task and generate a good completion. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.48.12_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.56.19_PM.png" width="80%" />

However, this strategy has a couple of drawbacks. First, for smaller models, it doesn't always work, even when five or six examples are included. Second, any examples you include in your prompt take up valuable space in the context window, reducing the amount of room you have to include other useful information. 

## Fine tuning

Luckily, another solution exists, you can take advantage of a process known as fine-tuning to further train a base model. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.56.54_PM.png" width="80%" />

In contrast to pre-training, where you train the LLM using vast amounts of unstructured textual data via selfsupervised learning, fine-tuning is a supervised learning process where you use a data set of labeled examples to update the weights of the LLM. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.57.02_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_6.57.12_PM.png" width="80%" />

The labeled examples are prompt completion pairs, the fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task. One strategy, known as instruction fine tuning, is particularly good at improving a model's performance on a variety of tasks.


Here are a couple of example prompts to demonstrate this idea. The instruction in both examples is classify this review, and the desired completion is a text string that starts with sentiment followed by either positive or negative. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.17.35_PM.png" width="80%" />

The data set you use for training includes many pairs of prompt completion examples for the task you're interested in, each of which includes an instruction. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.18.05_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.18.18_PM.png" width="80%" />

For example, if you want to fine tune your model to improve its summarization ability, you'd build up a data set of examples that begin with the instruction summarize, the following text or a similar phrase. And if you are improving the model's translation skills, your examples would include instructions like translate this sentence. 



These prompt completion examples allow the model to learn to generate responses that follow the given instructions. Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.18.51_PM.png" width="80%" />

The process results in a new version of the model with updated weights. It is important to note that just like pre-training, full fine tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components that are being updated during training. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.19.09_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.19.09_PM.png" width="80%" />

So you can benefit from the memory optimization and parallel computing strategies that you learned about last section. 

How do you actually go about instruction, fine-tuning and LLM? The first step is to prepare your training data. There are many publicly available datasets that have been used to train earlier generations of language models, although most of them are not formatted as instructions. Luckily, developers have assembled prompt template libraries that can be used to take existing datasets, for example, the large data set of Amazon product reviews and turn them into instruction prompt datasets for fine-tuning. 

Prompt template libraries include many templates for different tasks and different data sets. Here are three prompts that are designed to work with the Amazon reviews dataset and that can be used to fine tune models for classification, text generation and text summarization tasks. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.20.41_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.24.20_PM.png" width="80%" />


You can see that in each case you pass the original review, here called review_body, to the template, where it gets inserted into the text that starts with an instruction like predict the associated rating, generate a star review, or give a short sentence describing the following product review. The result is a prompt that now contains both an instruction and the example from the data set. Once you have your instruction data set ready, as with standard supervised learning, you divide the data set into training validation and test splits. During fine tuning, you select prompts from your training data set and pass them to the LLM, which then generates completions. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.24.42_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.25.22_PM.png" width="80%" />

Next, you compare the LLM completion with the response specified in the training data. You can see here that the model didn't do a great job, it classified the review as neutral, which is a bit of an understatement. The review is clearly very positive. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.25.43_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.26.33_PM.png" width="80%" />


Remember that the output of an LLM is a probability distribution across tokens. So you can compare the distribution of the completion and that of the training label and use the standard crossentropy function to calculate loss between the two token distributions. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.26.41_PM.png" width="80%" />

And then use the calculated loss to update your model weights in standard backpropagation. You'll do this for many batches of prompt completion pairs and over several epochs, update the weights so that the model's performance on the task improves. 

As in standard supervised learning, you can define separate evaluation steps to measure your LLM performance using the holdout validation data set. This will give you the validation accuracy, and after you've completed your fine tuning, you can perform a final performance evaluation using the holdout test data set. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.27.01_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.27.26_PM.png" width="80%" />

This will give you the test accuracy. 

The fine-tuning process results in a new version of the base model, often called an instruct model that is better at the tasks you are interested in. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.27.40_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-24_at_7.27.55_PM.png" width="80%" />

Fine-tuning with instruction prompts is the most common way to fine-tune LLMs these days. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-26_at_10.24.01_PM.png" width="80%" />

From this point on, when you hear or see the term fine-tuning, you can assume that it always means instruction fine tuning.




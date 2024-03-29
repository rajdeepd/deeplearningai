---
layout: default
title: Fine tuning on a single task
nav_order: 3
description: "Instruction fine tuning"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---
While LLMs have become famous for their ability to perform many different language tasks within a single model, your application may only need to perform a single task. In this case, you can fine-tune a pre-trained model to improve performance on only the task that is of interest to you. For example, summarization using a dataset of examples for that task.


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-27_at_7.48.34_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-27_at_7.49.01_PM.png" width="80%" />


Interestingly, good results can be achieved with relatively few examples. Often just 500-1,000 examples can result in good performance in contrast to the billions of pieces of texts that the model saw during pre-training. 


## Catastrophic forgetting

However, there is a potential downside to fine-tuning on a single task. The process may lead to a phenomenon called catastrophic forgetting. Catastrophic forgetting happens because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on the single fine-tuning task, it can degrade performance on other tasks. For example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks.

For example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks. This model knew how to carry out named entity recognition before fine-tuning correctly identifying Charlie as the name of the cat in the sentence.


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-27_at_7.54.53_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-27_at_7.57.05_PM.png" width="80%" />

![](../images/Screenshot_2023-07-27_at_8.00.59_PM.png)


How to avoid catastrophic forgetting?

- First note that you might not have to!
- Fine-tune on multiple tasks at the same time
- Consider Parameter Efficient Fine-tuning (PEFT)

But after fine-tuning, the model can no longer carry out this task, confusing both the entity it is supposed to identify and exhibiting behavior related to the new task. What options do you have to avoid catastrophic forgetting? First of all, it's important to decide whether catastrophic forgetting actually impacts your use case. If all you need is reliable performance on the single task you fine-tuned on, it may not be an issue that the model can't generalize to other tasks. If you do want or need the model to maintain its multitask generalized capabilities, you can perform fine-tuning on multiple tasks at one time. Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train. Will discuss this option in more detail shortly. Our second option is to perform parameter efficient fine-tuning, or PEFT for short instead of full fine-tuning. PEFT is a set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters. PEFT shows greater robustness to catastrophic forgetting since most of the pre-trained weights are left unchanged. PEFT is an exciting and active area of research that we will cover later this week. In the meantime, let's move on to the next video and take a closer look at multitask fine-tuning.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-27_at_8.00.36_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-27_at_8.00.59_PM.png" width="80%" />


Question:Which of the following are true in respect to Catastrophic Forgetting? Select all that apply.

* Catastrophic forgetting only occurs in supervised learning tasks and is not a problem in unsupervised learning.
* Catastrophic forgetting occurs when a machine learning model forgets previously learned information as it learns new information.
* One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training.
* Catastrophic forgetting is a common problem in machine learning, especially in deep learning models.



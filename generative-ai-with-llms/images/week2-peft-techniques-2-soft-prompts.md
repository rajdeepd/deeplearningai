---
layout: default
title: PEFT techniques 1 - Soft prompts
nav_order: 9
description: "Benchmarks"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---

# PEFT techniques 2: Soft prompts

With LoRA, the goal was to find an efficient way to update the weights of the model without having to train every single parameter again. There are also additive methods within PEFT that aim to improve model performance without changing the weights at all. In this video, you'll explore a second parameter efficient fine tuning method called prompt tuning.  

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_5.11.47_PM.png" width="80%" />


## Prompt tuning is not prompt engineering!

Now, prompt tuning sounds a bit like prompt engineering, but they are quite different from each other. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_5.12.38_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_5.13.33_PM.png" width="80%" />


With prompt engineering, you work on the language of your prompt to get the completion you want. This could be as simple as trying different words or phrases or more complex, like including examples for one or Few-shot Inference. The goal is to help the model understand the nature of the task you're asking it to carry out and to generate a better completion. However, there are some limitations to prompt engineering, as it can require a lot of manual effort to write and try different prompts. You're also limited by the length of the context window, and at the end of the day, you may still not achieve the performance you need for your task. With prompt tuning, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values.

## Prompt tuning adds trainable "soft prompt" to inputs

The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent your input text. The soft prompt vectors have the same length as the embedding vectors of the language tokens. 

Including somewhere between 20 and 100 virtual tokens can be sufficient for good performance. The tokens that represent natural language are hard in the sense that they each correspond to a fixed location in the embedding vector space.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_5.13.49_PM.png" width="80%" />

However, the soft prompts are not fixed discrete words of natural language. Instead, you can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_5.14.08_PM.png" width="80%" />

And through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task

## Prompt tuning for multiple tasks




In full fine tuning, the training data set consists of input prompts and output completions or labels. The weights of the large language model are updated during supervised learning. In contrast with prompt tuning, the weights of the large language model are frozen and the underlying model does not get updated. Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-07_at_12.27.52_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-07_at_12.28.25_PM.png" width="80%" />

Prompt tuning is a very parameter efficient strategy because only a few parameters are being trained. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-07_at_12.28.32_PM.png" width="80%" />

In contrast with the millions to billions of parameters in full fine tuning, similar to what you saw with LoRA. You can train a different set of soft prompts for each task and then easily swap them out at inference time. You can train a set of soft prompts for one task and a different set for another. To use them for inference, you prepend your input prompt with the learned tokens to switch to another task, you simply change the soft prompt.

## Performance of prompt tuning

Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible. You'll notice the same LLM is used for all tasks, all you have to do is switch out the soft prompts at inference time. 





So how well does prompt tuning perform? In the original paper, Exploring the Method by Brian Lester and collaborators at Google. 

[The Power of Scale for Parameter-Efficient Prompt Tuning
Brian Lester, Rami Al-Rfou Noah Constant, Google Research]( https://arxiv.org/pdf/2104.08691.pdf)

## Interpretability of soft prompts

The authors compared prompt tuning to several other methods for a range of model sizes. In this figure from the paper, you can see the Model size on the X axis and the SuperGLUE score on the Y axis. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-07_at_4.35.03_PM.png" width="80%" />

This is the evaluation benchmark you learned about earlier this week that grades model performance on a number of different language tasks. The red line shows the scores for models that were created through full fine tuning on a single task. While the orange line shows the score for models created using multitask fine tuning. The green line shows the performance of prompt tuning and finally, the blue line shows scores for prompt engineering only. As you can see, prompt tuning doesn't perform as well as full fine tuning for smaller LLMs. However, as the model size increases, so does the performance of prompt tuning. And once models have around 10 billion parameters, prompt tuning can be as effective as full fine tuning and offers a significant boost in performance over prompt engineering alone.

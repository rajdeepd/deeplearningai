---
layout: default
title: PEFT techniques 1 - LoRA
nav_order: 8
description: "Benchmarks"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
For example in my thesis website I created an _includes/lib/mathjax.html I can use in any page with a Jekyll include like
# PEFT techniques 1: LoRA

## Low-Rank Adaptation of Large Language Models (LoRA)

Low-rank Adaptation, or LoRA for short, is a parameter-efficient fine-tuning technique that falls into the re-parameterization category. Let's take a look at how it works.

## Transformers: recap
 As a quick reminder, here's the diagram of the transformer architecture that you saw earlier in the course. The input prompt is turned into tokens, which are then converted to embedding vectors and passed into the encoder and/or decoder parts of the transformer.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_10.57.23_AM.png" width="80%" />

In both of these components, there are two kinds of neural networks; self-attention and feedforward networks. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_10.57.33_AM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_10.57.40_AM.png" width="80%" />

The weights of these networks are learned during pre-training.


## LoRA: Low Rank Adaption of LLMs

After the embedding vectors are created, they're fed into the self-attention layers where a series of weights are applied to calculate the attention scores. During full fine-tuning, every parameter in these layers is updated. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_10.57.47_AM.png" width="80%" />

LoRA is a strategy that reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_10.57.54_AM.png" width="80%" />

The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying. You then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process you saw earlier this week. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.04.31_AM.png" width="80%" />


For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. You then add this to the original weights and replace them in the model with these updated values.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.04.40_AM.png" width="80%" />

You now have a LoRA fine-tuned model that can carry out your specific task. Because this model has the same number of parameters as the original, there is little to no impact on inference latency. Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. However, in principle, you can also use LoRA on other components like the feed-forward layers. But since most of the parameters of LLMs are in the attention layers, you get the biggest savings in trainable parameters by applying LoRA to these weights matrices. 


## Concrete example using base Transformer as reference

Let's look at a practical example using the transformer architecture described in the Attention is All You Need paper. The paper specifies that the transformer weights have dimensions of 512 by 64. This means that each weights matrix has 32,768 trainable parameters. If you use LoRA as a fine-tuning method with the rank equal to eight, you will instead train two small rank decomposition matrices whose small dimension is eight. This means that Matrix A will have dimensions of 8 by 64, resulting in 512 total parameters. Matrix B will have dimensions of 512 by 8, or 4,096 trainable parameters. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.04.58_AM.png" width="80%" />

By updating the weights of these new low-rank matrices instead of the original weights, you'll be training 4,608 parameters instead of 32,768 and 86% reduction. Because LoRA allows you to significantly reduce the number of trainable parameters, you can often perform this method of parameter efficient fine tuning with a single GPU and avoid the need for a distributed cluster of GPUs.


## LoRA: Low Rank Adaption of LLMs

Since the rank-decomposition matrices are small, you can fine-tune a different set for each task and then switch them out at inference time by updating the weights. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.05.19_AM.png" width="80%" />

Suppose you train a pair of LoRA matrices for a specific task; let's call it Task A. To carry out inference on this task, you would multiply these matrices together and then add the resulting matrix to the original frozen weights. 

---
**NOTE**

A neural network contains many dense layers which perform matrix multiplication. The weight
matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al.
(2020) shows that the pre-trained language models have a low “instrisic dimension” and can still
learn efficiently despite a random projection to a smaller subspace. Inspired by this, we hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation. For a pre-trained weight matrix $$ W_0 \in \mathbb{R}^{d \times k} $$ We constrain its update by representing the latter with a low-rank decomposition $$ W_0+\Delta W=W_0+B A $$ ,  where $$ B ∈ {R}^{d \times k} $$ , $$ A ∈ {R}^{r \times k} $$ , and the rank $$r \ll \min (d, k)$$.
During training, $$W_0$$ is frozen and does not receive gradient updates, while $$A$$ and $$B$$ contain trainable parameters. 
Note both $$W_0$$ and $$∆W = BA$$ are multiplied with the same input, and their respectiveoutput vectors are summed coordinate-wise. For $$h = W_0 x$$, our modified forward pass yields:

$$
h=W_0 x+\Delta W x=W_0 x+B A x
$$

Reference: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)

---


You then take this new summed weights matrix and replace the original weights where they appear in your model. You can then use this model to carry out inference on Task A. If instead, you want to carry out a different task, say Task B, you simply take the LoRA matrices you trained for this task, calculate their product, and then add this matrix to the original weights and update the model again.


## Sample ROUGE metrics for full vs. LoRA fine-tuning

Remember, although FLAN-T5 is a capable model, it can still benefit from additional fine-tuning on specific tasks. With full fine-tuning, you update every way in the model during supervised learning. You can see that this results in a much higher ROUGE 1 score increasing over the base FLAN-T5 model by 0.19. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.05.31_AM.png" width="80%" />

The additional round of fine-tuning has greatly improved the performance of the model on the summarization task. Now let's take a look at the scores for the LoRA fine-tune model. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.05.37_AM.png" width="80%" />

You can see that this process also resulted in a big boost in performance. The ROUGE 1 score has increased from the baseline by 0.17. This is a little lower than full fine-tuning, but not much. However, using LoRA for fine-tuning trained a much smaller number of parameters than full fine-tuning using significantly less compute, so this small trade-off in performance may well be worth it.

You might be wondering how to choose the rank of the LoRA matrices. This is a good question and still an active area of research. In principle, the smaller the rank, the smaller the number of trainable parameters, and the bigger the savings on compute. However, there are some issues related to model performance to consider


## Choosing the LoRA rank

In the paper that first proposed LoRA, researchers at Microsoft explored how different choices of rank impacted the model performance on language generation tasks. You can see the summary of the results in the table here. The table shows the rank of the LoRA matrices in the first column, the final loss value of the model, and the scores for different metrics, including BLEU and ROUGE. The bold values indicate the best scores that were achieved for each metric. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.05.47_AM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.05.53_AM.png" width="80%" />


The authors found a plateau in the loss value for ranks greater than 16. In other words, using larger LoRA matrices didn't improve performance.
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-06_at_11.06.01_AM.png" width="80%" />

The takeaway here is that ranks in the range of 4-32 can provide you with a good trade-off between reducing trainable parameters and preserving performance. Optimizing the choice of rank is an ongoing area of research and best practices may evolve as more practitioners like you make use of LoRA. LoRA is a powerful fine-tuning method that achieves great performance. The principles behind the method are useful not just for training LLMs, but for models in other domains. The final path method that you'll explore this week doesn't change the LLM at all and instead focuses on training your input text. Join me in the next video to learn more.


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

<img src="/deeplearningai/generative-ai-with-llms/images/<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_7.38.59_PM.png" width="80%" />
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

### LLM pre-training at a high level

To begin, let's take a high-level look at the initial training process for LLMs. This phase is often referred to as pre-training. As you saw in Lesson 1, LLMs encode a deep statistical representation of language. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-14_at_7.36.56_PM.png" width="80%" />



This understanding is developed during the models pre-training phase when the model learns from vast amounts of unstructured textual data. This can be gigabytes, terabytes, and even petabytes of text. This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-14_at_7.37.01_PM.png" width="80%" />

In this self-supervised learning step, the model internalizes the patterns and structures present in the language.

The encoder generates an embedding or vector representation for each token. Pre-training also requires a large amount of compute and the use of GPUs. Note, when you scrape training data from public sites such as the Internet, you often need to process the data to increase quality, address bias, and remove other harmful content. As a result of this data quality curation, often only 1-3% of tokens are used for pre-training. You should consider this when you estimate how much data you need to collect if you decide to pre-train your own model.

## Transformers
Earlier this week, you saw that there were three variance of the transformer model; encoder-only encoder-decoder models, and decode-only. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-14_at_7.37.12_PM.png" width="80%" />

Each of these is trained on a different objective, and so learns how to carry out different tasks

## Autoencoding models (Encoder only models)

Encoder-only models are also known as Autoencoding models, and they are pre-trained using masked language modeling. Here, tokens in the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentence. This is also called a denoising objective. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-14_at_7.37.25_PM.png" width="80%" />

Autoencoding models are bi-directional representations of the input sequence, meaning that the model has an understanding of the full context of a token and not just of the words that come before.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.04.12_AM.png" width="80%" />


Encoder-only models are ideally suited to task that benefit from this bi-directional contexts. You can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.07.50_AM.png" width="80%" />

Some well-known examples of an autoencoder model are **BERT** and **RoBERTa**.
These patterns then enable the model to complete its training objective, which depends on the architecture of the model, as you'll see shortly. During pre-training, the model weights get updated to minimize the loss of the training objective. 

## Autoregressive models

Decoder-based autoregressive models, mask the input sequence and can only see the input tokens leading up to the token in question. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-14_at_7.44.43_PM.png" width="80%" />

The model has no knowledge of the end of the sentence. The model then iterates over the input sequence one by one to predict the following token. In contrast to the encoder architecture, this means that the context is unidirectional. By learning to predict the next token from a vast number of examples, the model builds up a statistical representation of language. Models of this type make use of the decoder component off the original architecture without the encoder

Decoder-only models are often used for text generation, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well. Well known examples of decoder-based autoregressive models are GBT and BLOOM.

## Sequence-to-sequence models

The final variation of the transformer model is the sequence-to-sequence model that uses both the encoder and decoder parts off the original transformer architecture. The exact details of the pre-training objective vary from model to model. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.22.23_AM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.24.44_AM.png" width="80%" />


A popular sequence-to-sequence model T5, pre-trains the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique Sentinel token, shown here as x. Sentinel tokens are special tokens added to the vocabulary, but do not correspond to any actual word from the input text. The decoder is then tasked with reconstructing the mask token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens.

### Use cases for Sequence-to-sequence models

You can use sequence-to-sequence models for translation, summarization, and question-answering. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.25.05_AM.png" width="80%" />

They are generally useful in cases where you have a body of texts as both input and output. Besides T5, which you'll use in the labs in this course, another well-known encoder-decoder model is <a href="https://huggingface.co/docs/transformers/model_doc/bart">BART</a>.

```text
The Bart model was proposed in BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, 
Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, 
Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.


Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a 
left-to-right decoder (like GPT).
The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling 
scheme, where spans of text are replaced with a single mask token.
BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. 
It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves 
new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, 
with gains of up to 6 ROUGE.

```


## Model architectures and pre-training objectives

To summarize, here's a quick comparison of the different model architectures and the targets off the pre-training objectives. Autoencoding models are pre-trained using masked language modeling. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.25.22_AM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.25.56_AM.png" width="80%" />

They correspond to the encoder part of the original transformer architecture, and are often used with sentence classification or token classification. Autoregressive models are pre-trained using causal language modeling. Models of this type make use of the decoder component of the original transformer architecture, and often used for text generation. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_10.27.44_AM.png" width="80%" />

Sequence-to-sequence models use both the encoder and decoder part of the original transformer architecture. The exact details of the pre-training objective vary from model to model. The T5 model is pre-trained using span corruption. Sequence-to-sequence models are often used for translation, summarization, and question-answering

## Significance of scale: task ability

Now that you have seen how this different model architectures are trained and the specific tasks they are well-suited to, you can select the type of model that is best suited to your use case. One additional thing to keep in mind is that larger models of any architecture are typically more capable of carrying out their tasks well. Researchers have found that the larger a model, the more likely it is to work as you needed to without additional in-context learning or further training. This observed trend of increased model capability with size has driven the development of larger and larger models in recent years.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-15_at_1.32.09_PM.png" width="80%" />

<font size="2px"><center>Model size vs time</center></font>

This growth has been fueled by inflection points and research, such as the introduction of the highly scalable transformer architecture, access to massive amounts of data for training, and the development of more powerful compute resources. This steady increase in model size actually led some researchers to hypothesize the existence of a new Moore's law for LLMs. Like them, you may be asking, can we just keep adding parameters to increase performance and make models smarter? Where could this model growth lead? While this may sound great, it turns out that training these enormous models is difficult and very expensive, so much so that it may be infeasible to continuously train larger and larger models. Let's take a closer look at some of the challenges associated with training large models in the next video.


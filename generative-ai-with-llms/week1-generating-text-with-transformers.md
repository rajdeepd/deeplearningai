---
layout: default
title: Generating text with transformers
nav_order: 5
description: "Generative AI with Large Language Models"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---
## Generating text with transformers

At this point, you've seen a high-level overview of some of the major components inside the transformer architecture. But you still haven't seen how the overall prediction process works from end to end. Let's walk through a simple example. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.11.39_PM.png" width="80%" />

In this example, you'll look at a translation task or a sequence-to-sequence task, which incidentally was the original objective of the transformer architecture designers. You'll use a transformer model to translate the French phrase into English. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.12.08_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.12.22_PM.png" width="80%" />

First, you'll tokenize the input words using this same tokenizer that was used to train the network. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.13.18_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.13.30_PM.png" width="80%" />

These tokens are then added into the input on the encoder side of the network, passed through the embedding layer, and then fed into the multi-headed attention layers.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.56.01_PM.png" width="80%" />



The outputs of the multi-headed attention layers are fed through a feed-forward network to the output of the encoder. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.57.02_PM.png" width="80%" />

At this point, the data that leaves the encoder is a deep representation of the structure and meaning of the input sequence. This representation is inserted into the middle of the decoder to influence the decoder's self-attention mechanisms. Next, a start of sequence token is added to the input of the decoder. This triggers the decoder to predict the next token, which it does based on the contextual understanding that it's being provided from the encoder. 

The output of the decoder's self-attention layers gets passed through the decoder feed-forward network and through a final softmax output layer.

At this point, we have our first tok

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.57.16_PM.png" width="80%" />



You'll continue this loop, passing the output token back to the input to trigger the generation of the next token, until the model predicts an end-of-sequence token. At this point, the final sequence of tokens can be detokenized into words, and you have your output. In this case, I love machine learning. There are multiple ways in which you can use the output from the softmax layer to predict the next token. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-15_at_10.57.31_PM.png" width="80%" />

These can influence how creative you are generated text is. You will look at these in more detail later this week. Let's summarize what you've seen so far. The complete transformer architecture consists of an encoder and decoder components. The encoder encodes input sequences into a deep representation of the structure and meaning of the input. The decoder, working from input token triggers, uses the encoder's contextual understanding to generate new tokens. It does this in a loop until some stop condition has been reached. 

While the translation example you explored here used both the encoder and decoder parts of the transformer, you can split these components apart for variations of the architecture. Encoder-only models also work as sequence-to-sequence models, but without further modification, the input sequence and the output sequence or the same length. Their use is less common these days, but by adding additional layers to the architecture, you can train encoder-only models to perform classification tasks such as sentiment analysis, BERT is an example of an encoder-only model

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-16_at_11.50.59_AM.png" width="80%" />


Encoder-decoder models, as you've seen, perform well on sequence-to-sequence tasks such as translation, where the input sequence and the output sequence can be different lengths. You can also scale and train this type of model to perform general text generation tasks. Examples of encoder-decoder models include BART as opposed to BERT and T5, the model that you'll use in the labs in this course.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-16_at_11.53.57_AM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-16_at_12.05.22_PM.png" width="80%" />

You'll learn more about these different varieties of transformers and how they are trained later this week. That was quite a lot. The main goal of this overview of transformer models is to give you enough background to understand the differences between the various models being used out in the world and to be able to read model documentation. I want to emphasize that you don't need to worry about remembering all the details you've seen here, as you can come back to this explanation as often as you need. Remember that you'll be interacting with transformer models through natural language, creating prompts using written words, not code. You don't need to understand all of the details of the underlying architecture to do this. This is called prompt engineering, and that's what you'll explore in the next part of this course. Let's move on to the next video to learn more
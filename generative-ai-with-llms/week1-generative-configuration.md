---
layout: default
title: Generative configuration
nav_order: 8
description: "Generative Configuration"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---
## Generative configuration - inference parameters


In this Section, you'll examine some of the methods and associated configuration parameters that you can use to influence the way that the model makes the final decision about next-word generation. If you've used LLMs in playgrounds such as on the Hugging Face website or an AWS, you might have been presented with controls like these to adjust how the LLM behaves. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-17_at_3.58.17_PM.png" width="80%" />

Each model exposes a set of configuration parameters that can influence the model's output during inference. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-17_at_4.00.30_PM.png" width="80%" />

Note that these are different than the training parameters which are learned during training time. Instead, these configuration parameters are invoked at inference time and give you control over things like the maximum number of tokens in the completion, and how creative the output is. Max new tokens is probably the simplest of these parameters, and you can use it to limit the number of tokens that the model will generate. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-17_at_4.01.06_PM.png" width="80%" />

You can think of this as putting a cap on the number of times the model will go through the selection process.

## Generative config - max new tokens

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-17_at_4.15.38_PM.png" width="80%" />

Here you can see examples of max new tokens being set to 100, 150, or 200.

## Generative config - greedy vs. random sampling 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-18_at_7.13.22_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-18_at_7.13.49_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-18_at_7.14.45_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-18_at_7.16.35_PM.png" width="80%" />


But note how the length of the completion in the example for 200 is shorter. This is because another stop condition was reached, such as the model predicting and end of sequence token. Remember it's max new tokens, not a hard number of new tokens generated. The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses. Here you can see a selection of words and their probability score next to them. Although we are only showing four words here, imagine that this is a list that carries on to the complete dictionary. Most large language models by default will operate with so-called greedy decoding. This is the simplest form of next-word prediction, where the model will always choose the word with the highest probability. This method can work very well for short generation but is susceptible to repeated words or repeated sequences of words. If you want to generate text that's more natural, more creative and avoids repeating words, you need to use some other controls. Random sampling is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection. For example, in the illustration, the word banana has a probability score of 0.02.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-18_at_7.17.37_PM.png" width="80%" />

With random sampling, this equates to a 2% chance that this word will be selected. By using this sampling technique, we reduce the likelihood that words will be repeated. However, depending on the setting, there is a possibility that the output may be too creative, producing words that cause the generation to wander off into topics or words that just don't make sense. Note that in some implementations, you may need to disable greedy and enable random sampling explicitly. For example, the Hugging Face transformers implementation that we use in the lab requires that we set do sample to equal true. 

## Generative configuration - top-k and top-p

Let's explore top k and top p sampling techniques to help limit the random sampling and increase the chance that the output will be sensible. Two Settings, top p and top k are sampling techniques that we can use to help limit the random sampling and increase the chance that the output will be sensible. T With top k, you specify the number of tokens to randomly choose from, and with top p, you specify the total probability that you want the model to choose from. One more parameter that you can use to control the randomness of the model output is known as temperature. This parameter influences the shape of the probability distribution that the model calculates for the next token. Broadly speaking, the higher the temperature, the higher the randomness, and the lower the temperature, the lower the randomness. The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability distribution of the next token. In contrast to the top k and top p parameters, changing the temperature actually alters the predictions that the model will make.

### Generative config - top-k sampling

o limit the options while still allowing some variability, you can specify a top k value which instructs the model to choose from only the k tokens with the highest probability. 

In this example here, k is set to three, so you're restricting the model to choose from these three options. The model then selects from these options using the probability weighting and in this case, it chooses donut as the next word. This method can help the model have some randomness while preventing the selection of highly improbable completion words. This in turn makes your text generation more likely to sound reasonable and to make sense. 

### Generative config - top-p sampling 

Alternatively, you can use the top p setting to limit the random sampling to the predictions whose combined probabilities do not exceed p. For example, if you set p to equal 0.3, the options are cake and donut since their probabilities of 0.2 and 0.1 add up to 0.3. The model then uses the random probability weighting method to choose from these tokens.


If you choose a low value of temperature, say less than one, the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words. You can see this here in the blue bars beside the table, which show a probability bar chart turned on its side. Most of the probability here is concentrated on the word cake. The model will select from this distribution using random sampling and the resulting text will be less random and will more closely follow the most likely word sequences that the model learned during training. If instead you set the temperature to a higher value, say, greater than one, then the model will calculate a broader flatter probability distribution for the next token. Notice that in contrast to the blue bars, the probability is more evenly spread across the tokens. This leads the model to generate text with a higher degree of randomness and more variability in the output compared to a cool temperature setting. This can help you generate text that sounds more creative. If you leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used. You've covered a lot of ground so far. You've examined the types of tasks that LLMs are capable of performing and learned about transformers, the model architecture that powers these amazing tools. You've also explored how to get the best possible performance out of these models using prompt engineering and by experimenting with different inference configuration parameters. In the next video, you'll start building on this foundational knowledge by thinking through the steps required to develop and launch an LLM -powered application.

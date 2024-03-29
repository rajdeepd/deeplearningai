---
layout: default
title: Multi task instruction fine tuning
nav_order: 4
description: "Instruction fine tuning"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---
## Multi task instruction fine tuning

Multitask fine-tuning is an extension of single task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks. Here, the dataset contains examples that instruct the model to carry out a variety of tasks, including summarization, review rating, code translation, and entity recognition. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.36.14_PM.png" width="80%" />

You train the model on this mixed dataset so that it can improve the performance of the model on all the tasks simultaneously, thus avoiding the issue of catastrophic forgetting.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.36.25_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.36.43_PM.png" width="80%" />

Over many epochs of training, the calculated losses across examples are used to update the weights of the model, resulting in an instruction tuned model that is learned how to be good at many different tasks simultaneously. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.37.47_PM.png" width="80%" />

One drawback to multitask fine-tuning is that it requires a lot of data. You may need as many as 50-100,000 examples in your training set.

 
### Instruction fine-tuning with FLAN

However, it can be really worthwhile and worth the effort to assemble this data. The resulting models are often very capable and suitable for use in situations where good performance at many tasks is desirable. Let's take a look at one family of models that have been trained using multitask instruction fine-tuning. Instruct model variance differ based on the datasets and tasks used during fine-tuning. One example is the FLAN family of models. FLAN, which stands for fine-tuned language net, is a specific set of instructions used to fine-tune different models. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.38.34_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.38.46_PM.png" width="80%" />

Because they're FLAN fine-tuning is the last step of the training process the authors of the original paper called it the metaphorical dessert to the main course of pre-training quite a fitting name. FLAN-T5, the FLAN instruct version of the T5 foundation model while FLAN-PALM is the flattening struct version of the palm foundation model. You get the idea, FLAN-T5 is a great general purpose instruct model. In total, it's been fine tuned on 473 datasets across 146 task categories.

### FLAN-T5: Fine-tuned version of pre-trained T5 model

hose datasets are chosen from other models and papers as shown here. Don't worry about reading all the details right now.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.39.04_PM.png" width="80%" />


**Source: Chung et al. 2022, "Scaling Instruction-Finetuned Language Models"**

If you're interested, you can access the original paper through a reading exercise after the section and take a closer look.

### SAMSum: A dialogue dataset

Three examples are shown here with the dialogue on the left and the summaries on the right. The dialogues and summaries were crafted by linguists for the express purpose of generating a high-quality training dataset for language models. The linguists were asked to create conversations similar to those that they would write on a daily basis, reflecting their proportion of topics of their real life messenger conversations. 




Although language experts then created short summaries of those conversations that included important pieces of information and names of the people in the dialogue. Here is a prompt template designed to work with this SAMSum dialogue summary dataset. 

### Sample FLAN-T5 prompt templates

The template is actually comprised of several different instructions that all basically ask the model to do this same thing. Summarize a dialogue. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_4.48.34_PM.png" width="80%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_10.05.35_PM.png" width="100%" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_10.05.45_PM.png" width="100%" />

For example, briefly summarize that dialogue. What is a summary of this dialogue? What was going on in that conversation? Including different ways of saying the same instruction helps the model generalize and perform better. Just like the prompt templates you saw earlier. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_10.05.53_PM.png" width="100%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_10.06.46_PM.png" width="100%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_10.07.03_PM.png" width="100%" />



You see that in each case, the dialogue from the SAMSum dataset is inserted into the template wherever the dialogue field appears. The summary is used as the label. After applying this template to each row in the SAMSum dataset, you can use it to fine tune a dialogue summarization task. ### Improving FLAN-T5's summarization capabilities

While FLAN-T5 is a great general use model that shows good capability in many tasks. 

You may still find that it has room for improvement on tasks for your specific use case. For example, imagine you're a data scientist building an app to support your customer service team, process requests received through a chat bot, like the one shown here. Your customer service team needs a summary of every dialogue to identify the key actions that the customer is requesting and to determine what actions should be taken in response. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-28_at_10.07.30_PM.png" width="100%" />

The SAMSum dataset gives FLAN-T5 some abilities to summarize conversations. However, the examples in the dataset are mostly conversations between friends about day-to-day activities and don't overlap much with the language structure observed in customer service chats. 

You can perform additional fine-tuning of the FLAN-T5 model using a dialogue dataset that is much closer to the conversations that happened with your bot. This is the exact scenario that you'll explore in the lab this week. You'll make use of an additional domain specific summarization dataset called dialogsum to improve FLAN-T5's is ability to summarize support chat conversations. This dataset consists of over 13,000 support chat dialogues and summaries. 






### Example support-dialog summarization

The **dialogsum dataset** is not part of the FLAN-T5 training data, so the model has not seen these conversations before. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-29_at_6.35.30_PM.png" width="100%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-29_at_6.36.13_PM.png" width="100%" />


Let's take a look at example from dialogsum and discuss how a further round of fine-tuning can improve the model. This is a support chat that is typical of the examples in the dialogsum dataset

The conversation is between a customer and a staff member at a hotel check-in desk. The chat has had a template applied so that the instruction to summarize the conversation is included at the start of the text. Now, let's take a look at how FLAN-T5 responds to this prompt before doing any additional fine-tuning, note that the prompt is now condensed on the left to give you more room to examine the completion of the model. Here is the model's response to the instruction. You can see that the model does as it's able to identify that the conversation was about a reservation for Tommy.

source:
https://huggingface.co/datasets/knkarthick/dialogsum/viewer/knkarthick--dialogsum/


## Summary before fine-tuning FLAN-T5 with our dataset

The conversation is between a customer and a staff member at a hotel check-in desk. The chat has had a template applied so that the instruction to summarize the conversation is included at the start of the text. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-29_at_6.37.31_PM.png" width="100%" />


Now, let's take a look at how FLAN-T5 responds to this prompt before doing any additional fine-tuning, note that the prompt is now condensed on the left to give you more room to examine the completion of the model. Here is the model's response to the instruction. You can see that the model does as it's able to identify that the conversation was about a reservation for Tommy.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-29_at_6.42.44_PM.png" width="100%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-29_at_6.42.54_PM.png" width="100%" />

However, it does not do as well as the human-generated baseline summary, which includes important information such as Mike asking for information to facilitate check-in and the models completion has also invented information that was not included in the original conversation. Specifically the name of the hotel and the city it was located in.

## Summary after fine-tuning FLAN-T5 with our dataset

Now let's take a look at how the model does after fine-tuning on the dialogue some dataset, hopefully, you will agree that this is closer to the human-produced summary. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-29_at_7.15.12_PM.png" width="100%" />

There is no fabricated information and the summary includes all of the important details, including the names of both people participating in the conversation. 

## Fine tuning with your own data

This example, use the public dialogue, some dataset to demonstrate fine-tuning on custom data. In practice, you'll get the most out of fine-tuning by using your company's own internal data. For example, the support chat conversations from your customer support application. This will help the model learn the specifics of how your company likes to summarize conversations and what is most useful to your customer service colleagues. I know there's a lot to take in here. But don't worry, this example is going to be covered in the lab. You'll get a chance to see this in action and try it out for yourself. One thing you need to think about when fine-tuning is how to evaluate the quality of your models completions. In the next section, you'll learn about several metrics and benchmarks that you can use to determine how well your model is performing and how much better you're fine-tuned version is than the original base model.


**Question**

What is the purpose of fine-tuning with prompt datasets?

1. To increase the computational resources required for training a language model.
2. To decrease the accuracy of a pre-trained language model by introducing new prompts.
3. To eliminate the need for instructions and prompts in training a language model.
4. To improve the performance and adaptability of a pre-trained language model for specific tasks.


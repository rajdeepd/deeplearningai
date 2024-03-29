---
layout: default
title: Pre-training for domain adaptation
nav_order: 15
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---

## Pre-training for domain adaptation

So far, I've emphasized that you'll generally work with an existing LLM as you develop your application. This saves you a lot of time and can get you to a working prototype much faster. 

However, there's one situation where you may find it necessary to pretrain your own model from scratch. If your target domain uses vocabulary and language structures that are not commonly used in day to day language. You may need to perform domain adaptation to achieve good model performance. 

## Law Domain

For example, imagine you're a developer building an app to help lawyers and paralegals summarize legal briefs. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-08_at_8.19.43_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-08_at_8.20.30_PM.png" width="80%" />


Legal writing makes use of very specific terms like mens rea in the first example and res judicata in the second. These words are rarely used outside of the legal world, which means that they are unlikely to have appeared widely in the training text of existing LLMs. As a result, the models may have difficulty understanding these terms or using them correctly. Another issue is that legal language sometimes uses everyday words in a different context, like consideration in the third example. Which has nothing to do with being nice, but instead refers to the main element of a contract that makes the agreement enforceable. For similar reasons, you may face challenges if you try to use an existing LLM in a medical application. Medical language contains many uncommon words to describe medical conditions and procedures. And these may not appear frequently in training datasets consisting of web scrapes and book texts. 

## Medical Domain

Some domains also use language in a highly idiosyncratic way. This last example of medical language may just look like a string of random characters, but it's actually a shorthand used by doctors to write prescriptions. This text has a very clear meaning to a pharmacist, take one tablet by mouth four times a day, after meals and at bedtime. Because models learn their vocabulary and understanding of language through the original pretraining task.

## BloombergGPT: domain adaptation for finance

Pretraining your model from scratch will result in better models for highly specialized domains like law, medicine, finance or science. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-09_at_6.27.34_PM.png" width="80%" />

Now let's return to BloombergGPT, first announced in 2023 in a paper by Shijie Wu, Steven Lu, and colleagues at Bloomberg. BloombergGPT is an example of a large language model that has been pretrained for a specific domain, in this case, finance. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-09_at_6.27.51_PM.png" width="80%" />

The Bloomberg researchers chose to combine both finance data and general purpose tax data to pretrain a model that achieves Best in class results on financial benchmarks. While also maintaining competitive performance on general purpose LLM benchmarks. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-09_at_6.28.14_PM.png" width="80%" />

As such, the researchers chose data consisting of 51% financial data and 49% public data. In their paper, the Bloomberg researchers describe the model architecture in more detail. They also discuss how they started with a chinchilla scaling laws for guidance and where they had to make tradeoffs

## BloombergGPT relative to other LLMs

These two graphs compare a number of LLMs, including BloombergGPT, to scaling laws that have been discussed by researchers. On the left, the diagonal lines trace the optimal model size in billions of parameters for a range of compute budgets. On the right, the lines trace the compute optimal training data set size measured in number of tokens. The dashed pink line on each graph indicates the compute budget that the Bloomberg team had available for training their new model. The pink shaded regions correspond to the compute optimal scaling loss determined in the Chinchilla paper. In terms of model size, you can see that BloombergGPT roughly follows the Chinchilla approach for the given compute budget of 1.3 million GPU hours, or roughly 230,000,000 petaflops. The model is only a little bit above the pink shaded region, suggesting the number of parameters is fairly close to optimal. However, the actual number of tokens used to pretrain BloombergGPT 569,000,000,000 is below the recommended Chinchilla value for the available compute budget. The smaller than optimal training data set is due to the limited availability of financial domain data. Showing that real world constraints may force you to make trade offs when pretraining your own models. Congratulations on making it to the end of week one, you've covered a lot of ground, so let's take a minute to recap what you've seen. Mike walked you through some of the common use cases for LLMs, such as essay writing, dialogue summarization and translation. He then gave a detailed presentation of the transformer architecture that powers these models. And discussed some of the parameters you can use at inference time to influence the model's output. He wrapped up by introducing you to a generative AI project lifecycle that you can use to plan and guide your application development work. Next, you saw how models are trained on vast amounts of text data during an initial training phase called pretraining. This is where models develop their understanding of language. You explored some of the computational challenges of training these models, which are significant. In practice because of GPU memory limitations, you will almost always use some form of quantization when training your models. You finish the week with a discussion of scaling laws that have been discovered for LLMs and how they can be used to design compute optimal models. If you'd like to read more of the details, be sure to check out this week's reading exercises.
---
layout: default
title: Model evaluation
nav_order: 6
description: "Instruction fine tuning"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---

## Model evaluation

### Introduction

Throughout this course, you've seen statements like the model demonstrated good performance on this task or this fine-tuned model showed a large improvement in performance over the base model. What do statements like this mean? How can you formalize the improvement in performance of your fine-tuned model over the pre-trained model you started with? Let's explore several metrics that are used by developers of large language models that you can use to assess the performance of your own models and compare to other models out in the world


 In traditional machine learning, you can assess how well a model is doing by looking at its performance on training and validation data sets where the output is already known. You're able to calculate simple metrics such as accuracy, which states the fraction of all predictions that are correct because the models are deterministic. But with large language models where the output is non-deterministic and language-based evaluation is much more challenging

### LLM evaluation challenges

Take, for example, the sentence, Mike really loves drinking tea. This is quite similar to Mike adores sipping tea. But how do you measure the similarity? Let's look at these other two sentences. 

Mike does not drink coffee, and Mike does drink coffee. 

There is only one word difference between these two sentences. However, the meaning is completely different. 

For humans like us with squishy organic brains, we can see the similarities and differences. But when you train a model on millions of sentences, you need an automated, structured way to make measurements.

### LLM Evaluation - Metrics

**ROUGE** and **BLEU**, are two widely used evaluation metrics for different tasks. ROUGE or recall oriented under study for jesting evaluation is primarily employed to assess the quality of automatically generated summaries by comparing them to human-generated reference summaries. On the other hand, BLEU, or bilingual evaluation understudy is an algorithm designed to evaluate the quality of machine-translated text, again, by comparing it to human-generated translations. Now the word BLEU is French for blue. You might hear people calling this blue but here I'm going to stick with the original BLEU. 

### LLM Evaluation - Metrics - Terminology

Before we start calculating metrics. Let's review some terminology. In the anatomy of language, a unigram is equivalent to a single word. A bigram is two words and n-gram is a group of n-words. Pretty straightforward stuff. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-31_at_7.08.32_PM.png" width="80%" />



### LLM Evaluation - Metrics - ROUGE-1

First, let's look at the ROUGE-1 metric. To do so, let's look at a human-generated reference sentence.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-31_at_7.10.04_PM.png" width="80%" />

It is cold outside and a generated output that is very cold outside. You can perform simple metric calculations similar to other machine-learning tasks using recall, precision, and F1. The recall metric measures the number of words or unigrams that are matched between the reference and the generated output divided by the number of words or unigrams in the reference. In this case, that gets a perfect score of one as all the generated words match words in the reference. Precision measures the unigram matches divided by the output size. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-31_at_7.10.30_PM.png" width="80%" />

The F1 score is the harmonic mean of both of these values.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-03_at_8.34.00_PM.png" width="80%" />

### LLM Evaluation - Metrics - ROUGE-2


You can get a slightly better score by taking into account bigrams or collections of two words at a time from the reference and generated sentence. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-03_at_8.34.55_PM.png" width="80%" />

By working with pairs of words you're acknowledging in a very simple way, the ordering of the words in the sentence. By using bigrams, you're able to calculate a ROUGE-2. Now, you can calculate the recall, precision, and F1 score using bigram matches instead of individual words. You'll notice that the scores are lower than the ROUGE-1 scores. With longer sentences, they're a greater chance that bigrams don't match, and the scores may be even lower.

### LLM Evaluation - Metrics - ROUGE-L 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_1.54.46_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_1.55.40_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_1.58.13_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.28.23_PM.png" width="80%" />





Rather than continue on with ROUGE numbers growing bigger to n-grams of three or fours, let's take a different approach. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.28.55_PM.png" width="100%" />

Instead, you'll look for the longest common subsequence present in both the generated output and the reference output. In this case, the longest matching sub-sequences are, it is and cold outside, each with a length of two. You can now use the LCS value to calculate the recall precision and F1 score, where the numerator in both the recall and precision calculations is the length of the longest common subsequence, in this case, two. Collectively, these three quantities are known as the Rouge-L score. As with all of the rouge scores, you need to take the values in context. You can only use the scores to compare the capabilities of models if the scores were determined for the same task. For example, summarization. Rouge scores for different tasks are not comparable to one another. 

### LLM Evaluation - Metrics - ROUGE clipping

As you've seen, a particular problem with simple rouge scores is that it's possible for a bad completion to result in a good score. Take, for example, this generated output, cold, cold, cold, cold. As this generated output contains one of the words from the reference sentence, it will score quite highly, even though the same word is repeated multiple times. The Rouge-1 precision score will be perfect. One way you can counter this issue is by using a clipping function to limit the number of unigram matches to the maximum count for that unigram within the reference. 

### LLM Evaluation - Metrics

In this case, there is one appearance of cold and the reference and so a modified precision with a clip on the unigram matches results in a dramatically reduced score. However, you'll still be challenged if their generated words are all present, but just in a different order. For example, with this generated sentence, outside cold it is. This sentence was called perfectly even on the modified precision with the clipping function as all of the words and the generated output are present in the reference. Whilst using a different rouge score can help experimenting with a n-gram size that will calculate the most useful score will be dependent on the sentence, the sentence size, and your use case. Note that many language model libraries, for example, Hugging Face, which you used in the first week's lab, include implementations of rouge score that you can use to easily evaluate the output of your model. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.29.16_PM.png" width="100%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.30.44_PM.png" width="100%" />

You'll get to try the rouge score and use it to compare the model's performance before and after fine-tuning in this week's lab

### LLM Evaluation - Metrics - BLEU

The other score that can be useful in evaluating the performance of your model is the BLEU score, which stands for bilingual evaluation under study. Just to remind you that BLEU score is useful for evaluating the quality of machine-translated text. The score itself is calculated using the average precision over multiple n-gram sizes. Just like the Rouge-1 score that we looked at before, but calculated for a range of n-gram sizes and then averaged. Let's take a closer look at what this measures and how it's calculated. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.32.36_PM.png" width="100%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.34.53_PM.png" width="100%" />

The BLEU score quantifies the quality of a translation by checking how many n-grams in the machine-generated translation match those in the reference translation. To calculate the score, you average precision across a range of different n-gram sizes. If you were to calculate this by hand, you would carry out multiple calculations and then average all of the results to find the BLEU score


### LLM Evaluation - Metrics

For this example, let's take a look at a longer sentence so that you can get a better sense of the scores value. The reference human-provided sentence is, I am very happy to say that I am drinking a warm cup of tea.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-04_at_4.35.10_PM.png" width="100%" />

Now, as you've seen these individual calculations in depth when you looked at rouge, I will show you the results of BLEU using a standard library. Calculating the BLEU score is easy with pre-written libraries from providers like Hugging Face and I've done just that for each of our candidate sentences. The first candidate is, I am very happy that I am drinking a cup of tea. The BLEU score is 0.495. As we get closer and closer to the original sentence, we get a score that is closer and closer to one. Both rouge and BLEU are quite simple metrics and are relatively low-cost to calculate. You can use them for simple reference as you iterate over your models, but you shouldn't use them alone to report the final evaluation of a large language model. Use rouge for diagnostic evaluation of summarization tasks and BLEU for translation tasks. For overall evaluation of your model's performance, however, you will need to look at one of the evaluation benchmarks that have been developed by researchers. Let's take a look at some of these in more detail in the next video.
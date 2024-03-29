---
layout: default
title: Lab 2 walkthrough
nav_order: 11
description: "Lab 2 walkthrough"
has_children: false
parent: Week2
grand_parent: Coursera - GenAI with LLMs 
---

# Lab 2 walkthrough

This week's lab, let you try out fine-tuning using PEFT with LoRA for yourself by improving the summarization ability of the Flan-T5 model. My colleague Chris, is going to walk you through this week's notebook. I'll pass you over to him. Hey, thanks, Shelby. Now, let's take a look at Lab 2. In Lab 2, you will get hands-on with full fine-tuning and Parameter-Efficient Fine-Tuning, also called PEFT with prompt instructions. You will tune the Flan-T5 model further with your own specific prompts for your specific summarization task. 



Let's jump right into the notebook. Lab 2, we are going to actually fine-tune a model. Lab 1, we were doing the zero-shot inference, the in-context learning. 





Now we are actually going to modify the weights of our language model, specific to our summarization task and specific to our dataset. 

Real quick, just double-check that you have the eight CPU, 32 gigabyte, that's the instance type here. This is an AWS instance type from SageMaker, ml.m5.2xl. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_7.23.20_PM.png" />

Let's do this pip installs. While the pip installs are happening, let me explain torch and torchdata the same as Lab 1 where we are going to use PyTorch, we are then pip installing the torchdata library to help with the PyTorch data loading. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_7.24.02_PM.png" />

There's also a library called evaluates, and this is what we're going to use with our rouge score to calculate rouge. You learned about rouge in the lessons as a way to measure how well does a summary encapsulate what was in the original conversation or the original text. 


You learned about rouge in the lessons as a way to measure how well does a summary encapsulate what was in the original conversation or the original text. Now, these two libraries, LoRA and PEFT, you heard about a bit in the lessons.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_7.24.24_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_7.24.34_PM.png" />



This is what we will use to do the parameter efficient fine-tuning. Now, I'm going to do some imports here from those pip installs. If you do see this, sometimes this clean data in minutes thing shows up here, you don't need this for the lab. If you see it, I think this comes up whenever we import pandas, just click the "X" and click "Don't Show Again" because we're not using that part of SageMaker. Once again, we have the `AutoModelForSeq2Seq``. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_9.44.31_PM.png" />

This is what's going to give us access to **Flan-T5** through the transformers python library, the tokenizer, we use generation config in the previous lab. Now we're going to see two new classes, one called `TrainingArguments`, one called `Trainer`. These are all from transformers, these are always we can use that simplifies our code when we're trying to train our language model or fine-tune our language model.


 We see that we are going to import PyTorch and the evaluate, and we will use I believe pandas and numpy later on. Let's load the dataset just like we did in the first lab.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_7.24.48_PM.png" />

Let's load the model just like we did in the first lab and the tokenizer, and this is called the original model and this will be useful later when we compare all the different fine-tuning strategies to the original model that is not fine-tuned. Here is a convenience function that prints out all of the parameters that are in the model and specifically the trainable parameters.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-09_at_7.25.28_PM.png" />

This will become useful when we introduce the PEFT version of the model which does not train all of the parameters. Here we see there are approximately 250 million parameters being trained when we do the full fine-tuning, which is the first part of this lab where we full fine-tune. The second part of the lab will be where we do the parameter efficient fine-tuning specifically with LoRA, where we will only train very small number. So keep that in mind, this is a kind of a lot of messy code but it's pretty useful for the comparison. Just like we did in the first lab, we're going to show a sample input. We're going to show the human baseline. We're going to do the zero shot.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_8.33.29_PM.png" />



This is not one shot, not few shot, we're pass that, that was Lab 1. Here, we are trying to gets the point where one simple call into our model can give us a decent summary without having to pass in the one shot and few shot examples, that's the goal. The first way that we're going to do is we are going to do full fine-tuning.

Here is a convenience function that can tokenize and wrap our dataset in a prompt.

As we saw in the first lab where we had a prompt that said summarize the following conversation, and then we're actually going to give it the dialogue, and then we're going to end the prompt with those summary colon. This function will let us map over all of the elements of our dataset and convert them into prompts with instruction.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_8.53.05_PM.png" />

That's what we're going to do here, which is full fine-tuning with instruction prompts. Here, we're just going to take a sample just to keep the resource requirements law for this particular lab, speed things up a little bit. Let's take a look at the size. Here we have about 125 training examples. We're going to use five for validation. We're going to use 15 to actually do our holdout test later on when we compare. 





## Fine Tune the model with Preprocessed Dataset

We're going to fine tune with the training and we're going to validate with the validation. Then when all of that said and done, we're then going to use the 15 test examples to then compare the different strategies for fine-tuning with instruction. Here we see training arguments and we see some defaults here for the learning rate. We see some pretty low values for the max steps and the number of the epochs.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_8.53.18_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_8.54.03_PM.png" />" width="150%"/>

That's because we do want to try to minimize the amount of compute that's needed for this lab. If you have more time, you can certainly change these values and bump them up to maybe five epochs, maybe max steps 100. In a bit, I'll show you how we actually work around that.

We have trained offline a much larger model with much higher max steps and training epochs and in a bit, we will actually pull that in and then continue from there. But this is what the code looks like. Here's the training dataset, there's the evaluation validation dataset, here's where we call train. Actually let me just do Shift Enter, get this started.


This will take a few minutes, even with the low max steps and the low epochs, this still does take a few minutes to run. Then here's that step where we actually pull in from the cloud objects storage, a model that we trained outside of this lab that is a little bit better. So we'll actually start with that. Let's give the train a few minutes to complete. What we're doing here is we are actually instruction fine-tuning our Flan-T5 language model with our specific dataset on a very specific summarization task. Then later we'll see how the ROUGE metric compares between the original model and the instruction fine-tune model that we have here. Let's pull in that model from S3 object storage that we trained offline that's a little bit better accuracy and lower loss that we were able to train for longer outside of this specific lab

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.27.07_PM.png" /> 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.27.41_PM.png" /> 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.27.57_PM.png" /> 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.28.09_PM.png" /> 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.29.35_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.40.06_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-10_at_9.40.26_PM.png" />

 I do want to keep an eye on the size of this model. This is a fully fine-tuned instruction model, and you'll see it's close to one gigabyte, and that will come in handy later when we compare it to PEFT, which is on the order of 10 megabytes. Here we see 945 megabytes, so we pulled that model down into a directory here called flan dialogue summary checkpoint. Now we're going to load that instruction model, so now this becomes our new model that we are then going to use to compare here in a bit.
 
 
### Evaluate the Model Qualitatively (Human Evaluation)

Now that we've loaded what we're calling instruct model, let's actually try from our test dataset using the human eye, 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_9.57.35_PM.png" />


let's qualitatively test and see how does this look. The baseline summary Person 1 teaches Person 2 how to upgrade in Person 2's system. The original model without any instruction fine-tuning, just zero-shot. This time it's giving us Person 1, you'd like to upgrade your computer, Person 2, you'd like to upgrade your computer, so not very good. The instruction fine tune model that we just got done training is Person 1 suggests Person 2 should upgrade their system, hardware, and CD ROM, Person 2 thinks that's a great idea. 
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_9.57.54_PM.png" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_9.58.47_PM.png" />

That's qualitatively, that's just looking. Now, we only took a look at one example, but this is why we have quantitative techniques to do this comparison, to do the evaluation. 

## Rouge Metric

Specifically, let's load ROUGE and we're going to take a look, I think we're just going to do maybe the first 10 here, and let's compare them.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.03.59_PM.png" />




Let's compare the ROUGE metrics for both the original Flan-T5 and the instruction fine-tune model that we tuned up above. Here we see that the instruction fine-tuned model score is much higher on the ROUGE evaluation metric than the original Flan-T5 model. This is showing that with a little bit of fine-tuning using our dataset and a specific prompt, we were actually able to improve the ROUGE metric. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.03.15_PM.png" />

One other thing that we did offline was we did this much longer with a much larger test dataset

It wasn't just the 10 or the 15 examples, this actually was the full dataset, and let's take a look. That's what this file is. The CSV file that came along in this data directory with this lab. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.16.01_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.16.44_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.17.06_PM.png" />

Here we see with a much larger dataset, the scores are still pretty similar, where we're getting close to double, not quite double in some cases, but pretty significant improvement upon the original Flan-T5. Here we see the percentage improvements specifically. If we actually do the calculation, we see rouge1 is 18% higher, rouge2 10%, rougeL 13, rougeLsum 13.7 as well.


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.17.16_PM.png" />

## Perform Parameter Efficient Fine-Tuning (PEFT)

Now let's get into parameter efficient fine-tuning. This is one of my favorite topics. This makes such a big difference, especially when you're constrained by how much compute resources that you have, you can lower the footprint both memory, disk, GPU, CPU, all of the resources can be reduced just by introducing PEFT into your fine-tuning process. In the lessons you learned about LoRA, you learned about the rank. Here we're going to choose rank of 32, which is actually relatively high. 



But we are just starting with that. Here it's the SEQ_2_SEQ_LM, this is FLAN-T5. With just a few extra lines of code here to configure our LoRA fine-tuning.




In a lot of cases you can fine-tune very large models on a single GPU. Here's some more of those training arguments. This is really back to the original hugging face training and training arguments, except instead of using just the regular model, we are actually using the PEFT model.

Here this is a convenience function offered by the PEFT library : `get_peft_model(..)` and we give it the original model, which is the FLAN-T5. We give it the LoRA configuration which we defined above with the Rank 32. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.43.14_PM.png" />

We say get me a PEFT version of that model. That's what comes out as 1.4 percent. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_11.55.01_AM.png" />

Now we do the training arguments. Again, small number of steps, small number of epochs here. We do have a version that was trained offline. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.43.32_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-11_at_10.43.43_PM.png" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_11.55.01_AM.png" />

That's a little bit better than the one that is in this lab specifically, and that's what we're going to download here in a sec.

### PEFT model stored in S3

Let's do that. Here's the other model that was stored in S3 cloud storage. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_12.22.22_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_12.22.34_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_12.22.42_PM.png" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_1.07.25_PM.png" />

Now we see this is only 14 megabytes. 

These are called the PEFT adapters or LoRA adopters. 



These get merged or combined with the original LLM. When you go to actually serve this model, which we will hear in a bit, you have to take the original LLM and then merge in this LoRA PEFT adapter. These are much smaller and you can reuse the same base LLM and swap in different PEFT adapters when needed. Now that we have the PEFT adapter copied down from S3, we're going to merge that with the original LLM, which is FLAN-T5 and use that to actually perform summarization.


Now one thing to call out that's not entirely obvious is that when we do this, I can actually set the `is_trainable` flag to `false`.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_1.17.34_PM.png" />
By setting the is_trainable flag to false, we are telling PyTorch that we're not interested in training this model. All we're interested in doing is the forward pass just to get the summaries. This is significant because we can tell PyTorch to not load any of the update portions of these operators and to basically minimize the footprint needed to just perform the inference with this model. This is a pretty neat flag. 



<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_1.18.36_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_1.20.13_PM.png" />

Wanted to show it here because this is a pattern that you want to try to find when you're doing your own modeling. When you know that you're ready to deploy the model for inference, there are usually ways that you can hint to the framework, such as PyTorch that you're not going to be training. This can then further reduce the resources needed to make these predictions. Here, just to emphasize it, I do print out the number of trainable parameters. Keep in mind at this point we are only planning to do inference, and let's move on to that. Zero percent of these trainable parameters. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_1.18.15_PM.png" />

Here, we're going to build some sample prompts from our test data set. We're just going to pick something randomly here, essentially Index 200. We're going to see the instruction model. Got it. Mostly right I think, the PEFT model gets a little bit, starts to find a little bit more nuance here. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_2.29.24_PM.png" />


But really, as we'll see qualitatively when we run the rouge metrics. Here we're going to compare the human baseline to the original FLAN-T5 to the instruction full fine-tuned, and then to the PEFT fine-tuned

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_2.34.32_PM.png" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_2.47.04_PM.png" />

For the most part, just glancing here, it looks like these are pretty similar. But let's take a look at the rouge metrics and see what's going on. Here we see the instruction fine-tuned was a pretty drastic improvement over the original FLAN-T5. We see that the PEFT model does suffer a little bit of a degradation from the full fine-tuned. It's pretty close in some cases. It's not too bad. But we use much less resources during fine-tuning, than we would have if we did the full instruction. 

Load the values for the PEFT model now and check its performance compared to other models.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_2.35.03_PM.png" />

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_2.47.42_PM.png" />

Up above I was just looking at maybe 10, 15 examples. Here we see larger. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_4.26.26_PM.png" />

Looks like I think I have it here rouge one, PEFT loses about one to maybe 1.7 percent across all for of these rouge metrics. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_4.32.23_PM.png" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-12_at_4.32.33_PM.png" />

That's not bad relative to the savings that you get when you use PEFT.
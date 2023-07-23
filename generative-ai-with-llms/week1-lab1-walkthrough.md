---
layout: default
title: Lab 1 walkthrough
nav_order: 11
description: "Lab 1 walkthrough"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---



This is Lab 1, and like I said, we are going to grab a data-set of these conversations that are happening between people. What we plan to do is to summarize these dialogues and so think of a support dialogue between you and your customers maybe the end of the month you want to summarize all of the issues that your customer support team has dealt with that month. Some other things to note now, I'm zoomed in a little bit much here, but you can see that we have eight CPUs, we have 32 gigs of RAM.

Code Reference Download: <a href="./Lab_1_summarize_dialogue.ipynb">Lab 1 Jupyter notebook</a>
<br/>
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-21_at_11.11.42_PM.png" width="150%" />


We're using Python three and these are some of the pip install. So if I do a Shift Enter here, this is going to start doing the Python library installs and we see that we're going to be using PyTorch. 

```
%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.27.2 \
    datasets==2.11.0  --quiet
```


We are installing a library called Torch data, which helps with the data loading and some other aspects for PyTorch specific to data-sets. Here we see Transformers. This is a library from Hugging Face, a really cool company who has built a whole lot of open source tooling for large language models. 

```python
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
```
They also had built this library, this Python library called data-sets, that can load in many of the common public data-sets that people use to either train models, fine tune models, or just experiment with. If you click Shift Enter there, this will run for a bit. Now keep in mind this does take a few minutes to load. This whole notebook will depend on these libraries. So make sure that these do install. Just ignore these errors, these warnings. We always try to do things to mitigate these errors and warnings and they always show up, things will still work. 


Just trust me, these libraries and these notebooks do run. We've pinned all of the Python library versions so that as new versions come out, it will not potentially break these notebooks so just keep that in mind. This does say to restart the kernel, I don't think you have to do that. Let's just keep on going. Now we're going to actually do the imports here. This is going to import functions called load data-set, this is going to import some of the models and tokenizers that are needed to accomplish our lab here.

## Load the Dataset

We're going to use this dataset called Dialogue sum and this is a public dataset that transformers, and specifically the data-sets library does expose and does give us access to, so all we do is call load data-set that was imported up above and we pull in this data-set

```python
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
```
We're going to explore some of the data, we're going to actually try to summarize with just the **flat T5** based model. Before we get there though, let me load the data-set. Let's take a look at some of the examples of this dataset.


```python
example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()

```
Example output

```
---------------------------------------------------------------------------------------------------
Example  1
---------------------------------------------------------------------------------------------------
INPUT DIALOGUE:
#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
---------------------------------------------------------------------------------------------------


```
Here's a sample dialogue between person 1 and person 2. Person 1 says, what time is it, Tom? It looks like person 2 's name is Tom actually. Just a minute, it's 10.00 to 9.00 by my watch and on and on. Here's the baseline human summary. This is what a human has labeled this conversation to be, a summary of that conversation. Now we will try to improve upon that summary by using our model.

Again, no model has even been loaded yet. This is purely just the actual data. Here's the conversation and then think of this like this is the training sample and then this is what a human has labeled it and then we will compare the human summary, which is what we're considering to be the baseline, we will compare that to what the model predicts is the summary. The model will actually generate a summary. Here's a second example. You can see it's got some familiar terms here that a lot of us are familiar with, CD ROM painting program for your software.

```
---------------------------------------------------------------------------------------------------
Example  2
---------------------------------------------------------------------------------------------------
INPUT DIALOGUE:
#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
---------------------------------------------------------------------------------------------------
```

Now, here's where we're actually going to load the model. FLAN-T5, we spoke about in the videos. This is a very nice general purpose model that can do a whole lot of tasks and today we'll be focused on FLAN-T5's, ability to summarize conversations. After loading the model, we have to load the tokenizer. Now, these are all coming from the Hugging Face Transformers library. To give you an example, before transformers came along, we had to write a lot of this code ourselves. Depending on the type of model, there's now many different language models and some of them do things very differently than some of the other models. There was a lot of bespoke ad hoc libraries out there that were all trying to do similar things. Then Hugging Face came along and really has a very well optimized implementation of all of these. 

Load the FLAN-T5 model, creating an instance of the AutoModelForSeq2SeqLM class with the .from_pretrained() method.

```python
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

Here is the tokenizer. This is what's going to be used to convert the raw text from our conversation into our vector space that can then be processed by our Flan-T5 model.

To perform encoding and decoding, you need to work with text in a tokenized form. Tokenization is the process of splitting texts into smaller units that can be processed by the LLM models.

Download the tokenizer for the FLAN-T5 model using `AutoTokenizer.from_pretrained()` method. Parameter `use_fast` switches on fast tokenizer. At this stage, there is no need to go into the details of that, but you can find the tokenizer parameters in the documentation.


```python
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```


This toeknizer is going to be used to convert the raw text from our conversation into our vector space that can then be processed by our Flan-T5 model. Just to give you an idea, let's just take a sample sentence here. What time is it, Tom? The first sentence from our conversation up above, we see the encoded sentence is actually these numbers here. Then if you go to decode it, we see that this decodes right back to the original.

```python
sentence = "What time is it, Tom?"
sentence_encoded = tokenizer(sentence, return_tensors='pt')

sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0], 
        skip_special_tokens=True
    )

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)
```
The tokenizer's job is to convert raw text into numbers. 

```
ENCODED SENTENCE:
tensor([ 363,   97,   19,   34,    6, 3059,   58,    1])

DECODED SENTENCE:
What time is it, Tom?
```


Those numbers point to a set of vectors or the embeddings as they're often called, that are then used in mathematical operations like our deep learning, back-propagation, our linear algebra, all that fun stuff. Now, let's run this cell here and continue to explore.

Now, that we've loaded our model and we've loaded our tokenizer, we can run through some of these conversations through the Flan-T5 model and see what does this model actually generate as a summary for these conversations. Here again, we have the conversation. Here again is the baseline summary

```python
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')

```
We have the conversation. Here again is the baseline summary of example 1.


```
---------------------------------------------------------------------------------------------------
Example  1
---------------------------------------------------------------------------------------------------
INPUT PROMPT:
#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - WITHOUT PROMPT ENGINEERING:
Person1: It's ten to nine.

```

Then we see without any prompt engineering at all, just taking the actual conversation, passing it to our Flan-T5 model, it doesn't do a very good job summarizing. We see it's 10 to nine.

Let us look at the second example


```
---------------------------------------------------------------------------------------------------
Example  2
---------------------------------------------------------------------------------------------------
INPUT PROMPT:
#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - WITHOUT PROMPT ENGINEERING:
#Person1#: I'm thinking of upgrading my computer.


```

Same with the conversation about our CD-ROM, baseline summary as Person 1 teaches Person 2 how to upgrade the software and hardware in Person 2's system. The model generated Person 1 is thinking about upgrading their computer. Again, lots of details in this original conversation that do not come through the summary. Let's see how we can improve on this.

Here's an example. This is called in-context learning and specifically zero shots inference with an instruction. Here's the instruction, which is **summarize the following conversation**. 

```python
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
    """

    # Input constructed prompt instead of the dialogue.
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)    
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
```

Here is the actual conversation, and then we are telling the model where it should print the summary, which is going to be after this word summary. Now this seems very simple and let's see how it does. Let's see if things do get better. Not much better here. The baseline is still Person 1 is in a hurry, Tom tells Person 2 there's plenty of time.


```
---------------------------------------------------------------------------------------------------
Example  1
---------------------------------------------------------------------------------------------------
INPUT PROMPT:

Summarize the following conversation.

#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.

Summary:
    
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
The train is about to leave.

---------------------------------------------------------------------------------------------------
Example  2
---------------------------------------------------------------------------------------------------
INPUT PROMPT:

Summarize the following conversation.

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

Summary:
    
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
#Person1#: I'm thinking of upgrading my computer.

```

Then the zero shot in context learning with a prompt, it just says the train is about to leave. Again, not the greatest. And then here is the zero-shot for the computer sample. It's still thinking that Person 1 is trying to upgrade, so not much better.

## Zero Shot Inference with the Prompt Template from FLAN-T5

There is a different prompt that we can use here, which is where we just say **Dialogue** . Now these are really up to you. This is the prompt engineering side of these large language models where we're trying to find the best prompt and in this case just zero-shot inference. No fine-tuning of the model, no nothing. All we're doing is just finding different instructions to pass and seeing if the model does any better with slightly different phrases. Let's see how this does. Really this is the inverse of before where here we're just saying here's the dialogue, and then afterward we're saying what was going on up in that dialogue. Let's see if this does anything better.


```python
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
        
    prompt = f"""
Dialogue:

{dialogue}

What was going on?
"""

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')
```


Output of the model for two examples:
```
---------------------------------------------------------------------------------------------------
Example  1
---------------------------------------------------------------------------------------------------
INPUT PROMPT:

Dialogue:

#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.

What was going on?

---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.

---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
Tom is late for the train.

---------------------------------------------------------------------------------------------------
Example  2
---------------------------------------------------------------------------------------------------
INPUT PROMPT:

Dialogue:

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

What was going on?

---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
#Person1#: You could add a painting program to your software. #Person2#: That would be a bonus. #Person1#: You might also want to upgrade your hardware. #Person1#

```

Tom is late for the train, so it's picking that up, but still not great. Here we see Person 1. You could add a painting program. Person 2 that would be a bonus. A little bit better. It's not exactly right, but it's getting better. It's at least picking up some of the nuance.

## Summarize Dialogue with One Shot and Few Shot Inference

One shot and few shot inference are the practices of providing an LLM with either one or more full examples of prompt-response pairs that match your task - before your actual prompt that you want completed. This is called "in-context learning" and puts your model into a state that understands your specific task. You can read more about it in this blog from HuggingFace.

### One Shot Inference

Let's build a function that takes a list of example_indices_full, generates a prompt with full examples, then at the end appends the prompt which you want the model to complete (example_index_to_summarize). You will use the same FLAN-T5 prompt template from the previous section. Here we are giving the samples which are correct.


```python
def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        
        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""
    
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    
    prompt += f"""
Dialogue:

{dialogue}

What was going on?
"""
        
    return prompt
```
Here we are giving first example with summary, and second example without the summary.

Construct the prompt to perform one shot inference

```python
example_indices_full = [40]
example_index_to_summarize = 200

one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(one_shot_prompt)
```
Output of the one shot prompt


```
Dialogue:

#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.

What was going on?
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.



Dialogue:

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

What was going on?
```
`
Pass the prompt to the model:


```python
summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(one_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')
```

Compare the expected baseline human summary and the actual output.

```text
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

---------------------------------------------------------------------------------------------------
MODEL GENERATION - ONE SHOT:
#Person1 wants to upgrade his system. #Person2 wants to add a painting program to his software. #Person1 wants to add a CD-ROM drive.
```

Let's see how we do here. Here we're just going to do the upgrade software. Person1 wants to upgrade, Person2 wants to add painting program, Person1 wants to add a CD ROM. I think it's a little better and let's keep going. There's something called few-shot inference as well.

## Few Shot Inference

Let's explore few shot inference by adding two more full dialogue-summary pairs to your prompt.

Few shot means that we're giving three full examples, including the human baseline summary, 1, 2, 3, and then a fourth but without the human summary. Yes, even though we have it, we're just exploring our model right now. We're saying, tell us what that forth dialogue is. That summary. Just ignore some of these errors. Some of these sequences are a bit larger than the 512 context length of the model. 

```python
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200

few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(few_shot_prompt)
```


```
Dialogue:

#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.

What was going on?
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.



Dialogue:

#Person1#: May, do you mind helping me prepare for the picnic?
#Person2#: Sure. Have you checked the weather report?
#Person1#: Yes. It says it will be sunny all day. No sign of rain at all. This is your father's favorite sausage. Sandwiches for you and Daniel.
#Person2#: No, thanks Mom. I'd like some toast and chicken wings.
#Person1#: Okay. Please take some fruit salad and crackers for me.
#Person2#: Done. Oh, don't forget to take napkins disposable plates, cups and picnic blanket.
#Person1#: All set. May, can you help me take all these things to the living room?
#Person2#: Yes, madam.
#Person1#: Ask Daniel to give you a hand?
#Person2#: No, mom, I can manage it by myself. His help just causes more trouble.

What was going on?
Mom asks May to help to prepare for the picnic and May agrees.



Dialogue:

#Person1#: Hello, I bought the pendant in your shop, just before. 
#Person2#: Yes. Thank you very much. 
#Person1#: Now I come back to the hotel and try to show it to my friend, the pendant is broken, I'm afraid. 
#Person2#: Oh, is it? 
#Person1#: Would you change it to a new one? 
#Person2#: Yes, certainly. You have the receipt? 
#Person1#: Yes, I do. 
#Person2#: Then would you kindly come to our shop with the receipt by 10 o'clock? We will replace it. 
#Person1#: Thank you so much. 

What was going on?
#Person1# wants to change the broken pendant in #Person2#'s shop.



Dialogue:

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

What was going on?
```

Pass on the test prompt example and list the output


```python
summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0], 
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')

```

```
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

---------------------------------------------------------------------------------------------------
MODEL GENERATION - FEW SHOT:
#Person1 wants to upgrade his system. #Person2 wants to add a painting program to his software. #Person1 wants to upgrade his hardware.

```

This is where you can actually play with some of these configuration parameters that you learn during the lessons. Things like the sampling, temperature. You can play with these try out and gain your intuition on how these things can impact what's actually generated by the model. In some cases, for example, by raising the temperature up above, towards one or even closer to two, you will get very creative type of responses. If you lower it down I believe 0.1 is the minimum for the hugging face implementation anyway, of this generation config class here that's used when you actually generate. I can pass generation config right here. If you go down to 0.1, that will actually make the response more conservative and will oftentimes give you the same response over and over. If you go higher, I believe actually 2.0 is the highest. If you try to 2.0, that will start to give you some very wild responses.
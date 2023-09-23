---
layout: default
title: "Evaluation Part ii"
nav_order: 10
description: "Evaluation"
parent: ChatGPT
---
In the last video, you saw how to evaluate an LLM output in an example where it had the right answer. 
And so we could write down a function to explicitly just tell us if the LLM outputs the right categories and list of products. 
But what if the LLM is used to generate text and there isn't just one right piece of text? Let's take a look at an approach for how to evaluate that type of LLM output. 
Here's my usual helper functions, 


```python
import os
import openai
import sys
sys.path.append('../..')
import utils
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```


```python
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]
```

Given a customer message, "tell me about the smartx pro phone and the fotosnap camera.".

```python
customer_msg = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs or TV related products do you have?"""

products_by_category = utils.get_products_from_query(customer_msg)
category_and_product_list = utils.read_string_to_list(products_by_category)
product_info = utils.get_mentioned_product_info(category_and_product_list)
assistant_answer = utils.answer_user_msg(user_msg=customer_msg,
                                                   product_info=product_info)
```

Output


```
Sure! Let me provide you with some information about the SmartX ProPhone and the FotoSnap DSLR Camera.

The SmartX ProPhone is a powerful smartphone with advanced camera features. It has a 6.1-inch display, 128GB storage, a 12MP dual camera, and supports 5G connectivity. The SmartX ProPhone is priced at $899.99 and comes with a 1-year warranty.

The FotoSnap DSLR Camera is a versatile camera that allows you to capture stunning photos and videos. It features a 24.2MP sensor, 1080p video recording, a 3-inch LCD screen, and supports interchangeable lenses. The FotoSnap DSLR Camera is priced at $599.99 and also comes with a 1-year warranty.

As for TVs and TV-related products, we have a variety of options available. Some of our popular TV models include the CineView 4K TV, CineView 8K TV, and CineView OLED TV. We also have home theater systems like the SoundMax Home Theater and SoundMax Soundbar. Each product has its own unique features and price points. Is there a specific TV or TV-related product you are interested in?
```

And so here's the assistant answer, "Sure, I'd be happy to help!". Smartphone, the smartx pro phone, and so on and so forth. So, how can you evaluate if this is a good answer or not? 
Seems like there are lots of possible good answers. One way to evaluate this is to write a rubric, meaning a set of guidelines, to evaluate this answer on different dimensions, and then use that to decide whether or not you're satisfied with this answer.

But this prompt says in the system message, "You are an assistant that evaluates how well the customer service agent answers a user question by looking at the context that the customer service agent is using to generate its response.". 
This response is what we had further up in the notebook, that was the assistant answer. And we're going to specify the data in this prompt, what was the customer message, what was the context, that is, what was the product and category information that was provided, and then what was the output of the LLM. 
 
This is a rubric. So, we want the LLM to, "Compare the factual content of the submitted answer with the context. 
Ignore differences in style, grammar, or punctuation. 
We wanted to check a few things, like, "Is the assistant response based only on the context provided? Does the answer include information that is not provided in the context? Is there any disagreement between the response and the context?" 
Finally, we wanted to print out yes or no, and so on. 

```python
cust_prod_info = {
    'customer_msg': customer_msg,
    'context': product_info
}

def eval_with_rubric(test_set, assistant_answer):

    cust_msg = test_set['customer_msg']
    context = test_set['context']
    completion = assistant_answer
    
    system_message = """\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by looking at the context that the customer service \
    agent is using to generate its response. 
    """

    user_message = f"""\
You are evaluating a submitted answer to a question based on the context \
that the agent uses to answer the question.
Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Context]: {context}
    ************
    [Submission]: {completion}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the context. \
Ignore any differences in style, grammar, or punctuation.
Answer the following questions:
    - Is the Assistant response based only on the context provided? (Y or N)
    - Does the answer include information that is not provided in the context? (Y or N)
    - Is there any disagreement between the response and the context? (Y or N)
    - Count how many questions the user asked. (output a number)
    - For each question that the user asked, is there a corresponding answer to it?
      Question 1: (Y or N)
      Question 2: (Y or N)
      ...
      Question N: (Y or N)
    - Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = get_completion_from_messages(messages)
    return response
```

If we were to run this evaluation. This is what you get:"the assistant response is based only on the context provided.". It does not, in this case, seem to make up new information. There isn't disagreements. User asked two questions. Answered question one and answered question two. So answered both questions. 
We would look at this output and maybe conclude that this is a pretty good response.

```
- Is the Assistant response based only on the context provided? (Y or N)
Y

- Does the answer include information that is not provided in the context? (Y or N)
N

- Is there any disagreement between the response and the context? (Y or N)
N

- Count how many questions the user asked. (output a number)
2

- For each question that the user asked, is there a corresponding answer to it?
Question 1: Y
Question 2: Y

- Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
2
```
Note:  We are using the ChatGPT 3.5 Turbo model for this evaluation. 
For a more robust evaluation, it might be worth considering using GPT-4 because even if you deploy 3.5 Turbo in production and generate a 
lot of text, if your evaluation is a more sporadic exercise, then it may be prudent to pay for the somewhat more expensive GPT-4 API call to get a more rigorous evaluation of the output. 

Evaluate against an ideal answer:


```python
test_set_ideal = {
    'customer_msg': """\
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs or TV related products do you have?""",
    'ideal_answer':"""\
Of course!  The SmartX ProPhone is a powerful \
smartphone with advanced camera features. \
For instance, it has a 12MP dual camera. \
Other features include 5G wireless and 128GB storage. \
It also has a 6.1-inch display.  The price is $899.99.

The FotoSnap DSLR Camera is great for \
capturing stunning photos and videos. \
Some features include 1080p video, \
3-inch LCD, a 24.2MP sensor, \
and interchangeable lenses. \
The price is 599.99.

For TVs and TV related products, we offer 3 TVs \


All TVs offer HDR and Smart TV.

The CineView 4K TV has vibrant colors and smart features. \
Some of these features include a 55-inch display, \
'4K resolution. It's priced at 599.

The CineView 8K TV is a stunning 8K TV. \
Some features include a 65-inch display and \
8K resolution.  It's priced at 2999.99

The CineView OLED TV lets you experience vibrant colors. \
Some features include a 55-inch display and 4K resolution. \
It's priced at 1499.99.

We also offer 2 home theater products, both which include bluetooth.\
The SoundMax Home Theater is a powerful home theater system for \
an immmersive audio experience.
Its features include 5.1 channel, 1000W output, and wireless subwoofer.
It's priced at 399.99.

The SoundMax Soundbar is a sleek and powerful soundbar.
It's features include 2.1 channel, 300W output, and wireless subwoofer.
It's priced at 199.99

Are there any questions additional you may have about these products \
that you mentioned here?
Or may do you have other questions I can help you with?
    """
}
```

## Check is LLMs response varies from the expert answer


One design pattern that I hope you can take away 
from this is that when you can specify a rubric, 
meaning a list of criteria by which to 
evaluate an LLM output, then you can actually 
use another API call to evaluate your first LLM output. 
There's one other design pattern that could be useful 
for some applications, which is if you can 
specify an ideal response. So here, I'm going to specify a test 
example where the customer message is, "tell me 
about the smartx pro phone", and so on. 
And here's an ideal answer. So this is if you have an expert human 
customer service representative write a really good answer.

The expert says, this would be a great answer., "Of course! The SmartX ProPhone is a.". It goes on to give a lot of helpful information. 
Now, it is unreasonable to expect any LLM to generate this exact answer word for word. And in classical natural language processing 
techniques, there are some traditional metrics for measuring if the LLM output is similar to this expert human written outputs. For example, there's something called the BLEU score, BLEU, that you can search online to read more about. They can measure how similar one piece of text is from another. But it turns out there's an even better way, which is you can use a prompt, which I'm going to specify here, to ask an LLM to compare how well the automatically generated customer service agent output corresponds to the ideal expert response that was written by a human that I just showed up above. 
Here's the prompt we can use, which is. We're going to use an LLM and tell it to be an assistant that evaluates how well the 
customer service agent answers a user question by comparing the response, that was the automatically generated one, to the ideal (expert) human written response. 

So we're going to give it the data, which is what was the customer request, what 
is the expert written ideal response, and then what did our 
LLM actually output. 
And this rubric comes from the OpenAI open source evals framework, 
which is a fantastic framework with many evaluation methods 
contributed both by OpenAI developers and 
by the broader open source community. 
In fact, if you want you could contribute an eval to 
that framework yourself to help others evaluate their Large 
Language Model outputs. 
So in this rubric, we tell the LLM to, 
"Compare the factual content of the submitted answer 
with the expert answer. Ignore any differences in style, 
grammar, or punctuation.". 
And feel free to pause the video and 
read through this in detail, but the key is we ask it to 
carry the comparison and output a score from A to E, 
depending on whether the "submitted answer is a 
subset of the expert answer and is fully consistent", 
versus the "submitted answer is a superset of the expert answer 
and is fully consistent with it". This might mean it hallucinated or 
made up some additional facts. 
"Submitted answer contains all the details as the expert 
answer.", whether there's disagreement or whether "the 
answers differ, but these differences don't matter 
from the perspective of factuality". 
And the LLM will pick whichever of these seems to be the 
most appropriate description. So here's the assistant answer that 
we had just now. I think it's a pretty good answer, but now let's see what 
the things when it compares the assistant answer to test 
set ID. Oh, looks like it got an A. 
And so it thinks "The submitted answer is 
a subset of the expert answer and is 
fully consistent with it", and that sounds right to me. 
This assistant answer is much shorter than the 
long expert answer up top, but it does hopefully is consistent. 
Once again, I'm using GPT-3.5 Turbo in this example, but to get 
a more rigorous evaluation, it might make sense to use GPT-4 in your own 
application. 
Now, let's try something totally different. I'm 
going to have a very different assistant answer, "life is 
like a box of chocolates", quote from a movie called "Forrest Gump". 
 

```python
def eval_vs_ideal(test_set, assistant_answer):

    cust_msg = test_set['customer_msg']
    ideal = test_set['ideal_answer']
    completion = assistant_answer
    
    system_message = """\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by comparing the response to the ideal (expert) response
    Output a single letter and nothing else. 
    """

    user_message = f"""\
You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Expert]: {ideal}
    ************
    [Submission]: {completion}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
    The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
    (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
    (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
    (C) The submitted answer contains all the same details as the expert answer.
    (D) There is a disagreement between the submitted answer and the expert answer.
    (E) The answers differ, but these differences don't matter from the perspective of factuality.
  choice_strings: ABCDE
"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = get_completion_from_messages(messages)
    return response
```


```python
print(assistant_answer)
```


```
Sure! Let me provide you with some information about the SmartX ProPhone and the FotoSnap DSLR Camera.

The SmartX ProPhone is a powerful smartphone with advanced camera features. It has a 6.1-inch display, 128GB storage, a 12MP dual camera, and supports 5G connectivity. The SmartX ProPhone is priced at $899.99 and comes with a 1-year warranty.

The FotoSnap DSLR Camera is a versatile camera that allows you to capture stunning photos and videos. It features a 24.2MP sensor, 1080p video recording, a 3-inch LCD screen, and supports interchangeable lenses. The FotoSnap DSLR Camera is priced at $599.99 and also comes with a 1-year warranty.

As for TVs and TV-related products, we have a variety of options available. Some of our popular TV models include the CineView 4K TV, CineView 8K TV, and CineView OLED TV. We also have home theater systems like the SoundMax Home Theater and SoundMax Soundbar. Each product has its own unique features and price points. Is there a specific TV or TV-related product you are interested in?
```

And if we were to evaluate that it outputs D and it concludes that, 
"there is a disagreement between the submitted answer", life is like a box of chocolate and the expert answer. 
So it correctly assesses this to be a pretty terrible answer. And so that's it. I hope you take away from 
this section two design patterns. 
First is, even without an expert provided ideal answer, if you can write a rubric, you can use one 
LLM to evaluate another LLM's output. And second, if you can provide an expert provided ideal answer, 
then that can help your LLM better compare if, and if a specific assistant output is similar to the expert provided ideal answer. I hope that helps you to evaluate your LLM systems 
outputs. 
So that both during development as well as when the system is running and you're getting responses, you can continue to monitor its performance and also have these tools to continuously evaluate and keep on improving the performance of your system. 

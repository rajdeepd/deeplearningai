---
layout: default
title: "Evaluation"
nav_order: 8
description: "Evaluation"
parent: ChatGPT
---

# Evaluation

In this section we'll put together everything we've learned 
in the previous videos to create an end-to-end example of a 
customer service assistant. We'll go through the following 
steps. First, we'll check the input to see 
if it flags the moderation API. Second, if it doesn't, we'll extract the list of products. Third, if the products are found, we'll try to look them up. 
Four, we'll answer the user question with the model. 
finally, we'll put the answer through the moderation API.  If 
it is not flagged, we'll return it to the user. 
Let us start with setup where we have an addition import
We have this additional import.

## Setup

Additional import is panel.

```python
import os
import openai
import sys
sys.path.append('../..')
import utils

import panel as pn  # GUI
pn.extension()

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
This is a Python package we'll use for a chatbot UI.  We are to paste in a function "process_user_message". 
We will run an example first and  then we'll talk through the function. 

The third step is looking up the product information. 
 
With this product information, the model is trying to answer the question as we've seen in the previous sections. Finally it sends the response to the moderation API again to make sure it's  safe to show to the user. This is the response that we're now familiar with. Let us talk about what actually is happening.
 
We have a helper function "process_user_message". It takes in the user input, which is the current message, and an array of all of the messages so far and when we build the chatbot UI. The first step, checking to see if the input flags the moderation API. We have covered this in a previous video. 
 
If the input is flagged, then we tell the user that we can't process the request. If it is not flagged, we try to extract the list of products as we did in the previous video. Next we try to look up the products. if no products are found, this will just be an empty string. 
Then we answer the user question, so we give the conversation history and the new messages with the relevant product information. 
We get the response, and then we run this response through the moderation API. 
If it is flagged, we tell the user that we can't provide this information. Maybe we'll say something like, let me connect 
you, and you could take some subsequent step. Let us put this all together with a nice UI as shown below: 

Let us try having a conversation; We have a function that will just accumulate the messages as we interact with the assistant. 

```python
def process_user_message(user_input, all_messages, debug=True):
    delimiter = "```"
    
    # Step 1: Check input to see if it flags the Moderation API or is a prompt injection
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]

    if moderation_output["flagged"]:
        print("Step 1: Input flagged by Moderation API.")
        return "Sorry, we cannot process this request."

    if debug: print("Step 1: Input passed moderation check.")
    
    category_and_product_response = utils.find_category_and_product_only(user_input, utils.get_products_and_category())
    #print(print(category_and_product_response)
    # Step 2: Extract the list of products
    category_and_product_list = utils.read_string_to_list(category_and_product_response)
    #print(category_and_product_list)

    if debug: print("Step 2: Extracted list of products.")

    # Step 3: If products are found, look them up
    product_information = utils.generate_output_string(category_and_product_list)
    if debug: print("Step 3: Looked up product information.")

    # Step 4: Answer the user question
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers. \
    Make sure to ask the user relevant follow-up questions.
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]

    final_response = get_completion_from_messages(all_messages + messages)
    if debug:print("Step 4: Generated response to user question.")
    all_messages = all_messages + messages[1:]

    # Step 5: Put the answer through the Moderation API
    response = openai.Moderation.create(input=final_response)
    moderation_output = response["results"][0]

    if moderation_output["flagged"]:
        if debug: print("Step 5: Response flagged by Moderation API.")
        return "Sorry, we cannot provide this information."

    if debug: print("Step 5: Response passed moderation check.")

    # Step 6: Ask the model if the response answers the initial user query well
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question?
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    evaluation_response = get_completion_from_messages(messages)
    if debug: print("Step 6: Model evaluated the response.")

    # Step 7: If yes, use this answer; if not, say that you will connect the user to a human
    if "Y" in evaluation_response:  # Using "in" instead of "==" to be safer for model output variation (e.g., "Y." or "Yes")
        if debug: print("Step 7: Model approved the response.")
        return final_response, all_messages
    else:
        if debug: print("Step 7: Model disapproved the response.")
        neg_str = "I'm unable to provide the information you're looking for. I'll connect you with a human representative for further assistance."
        return neg_str, all_messages

user_input = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"
response,_ = process_user_message(user_input,[])
print(response)
```


If we run this, now let's try and have a conversation with 
the customer service assistant. So let's ask, *"what TVs do you have"*. And under the hood, the assistant is going through all of the steps in the `process_user_message` function. 

It listed a variety of different TVs. Let us look at next question, *"which is the cheapest"*.
It us going through all the same steps, but this time it's passing the conversation history as well into the prompts. 
It is telling us that this speaker is the cheapest  TV related product we have. Let us see what the most expensive is. The most expensive TV is the *CineView 8K TV*.
Let us ask for more information about it, *"tell me more about it"*. We have received some more information about this television. 
In this example, we've combined the techniques we've learned  throughout the course to create a comprehensive system with a chain of steps that evaluates user inputs, processes them, and then checks the output. 
By monitoring the quality of the system across a larger number of inputs, you can alter the 
steps and improve the overall performance of your system. 
We might find that our prompts could be better for some of the steps, maybe some of the steps aren't even necessary. Maybe we'll find a better retrieval method. We'll discuss this more in the next section. 
 
---
layout: default
title: "Process Inputs: Chain of Thoughts"
nav_order: 4
description: "Language Models for Prompts"
parent: ChatGPT
---

# Process Inputs Chain of Thoughts

## Overview


## Setup

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
```

### Create a get Completion from message method

This method uses `openai.ChatCompletion()` API with `gpt-3.5-turbo` model

```python
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]
```
We will send the user and system prompts using this method to OpenAI

## Chain of thought Prompting

First create a system_message with 4 Steps of reasoning.

1. Step 1: Find out if the customer is asking about specific product
2. Step 2: If the customer is asking about specific product, whether it is from the following list
3. Step 3: List the assumptions
4. Step 4: Figure out if the assumption is true
5. Step 5: Correct customers assumption and answer based on info in the list


```python
delimiter = "####"
system_message = f"""
Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags,\
i.e. {delimiter}. 

Step 1:{delimiter} First decide whether the user is \
asking a question about a specific product or products. \
Product cateogry doesn't count. 

Step 2:{delimiter} If the user is asking about \
specific products, identify whether \
the products are in the following list.
All available products: 
1. Product: TechPro Ultrabook
   Category: Computers and Laptops
   Brand: TechPro
   Model Number: TP-UB100
   Warranty: 1 year
   Rating: 4.5
   Features: 13.3-inch display, 8GB RAM, 256GB SSD, Intel Core i5 processor
   Description: A sleek and lightweight ultrabook for everyday use.
   Price: $799.99

2. Product: BlueWave Gaming Laptop
   Category: Computers and Laptops
   Brand: BlueWave
   Model Number: BW-GL200
   Warranty: 2 years
   Rating: 4.7
   Features: 15.6-inch display, 16GB RAM, 512GB SSD, NVIDIA GeForce RTX 3060
   Description: A high-performance gaming laptop for an immersive experience.
   Price: $1199.99

3. Product: PowerLite Convertible
   Category: Computers and Laptops
   Brand: PowerLite
   Model Number: PL-CV300
   Warranty: 1 year
   Rating: 4.3
   Features: 14-inch touchscreen, 8GB RAM, 256GB SSD, 360-degree hinge
   Description: A versatile convertible laptop with a responsive touchscreen.
   Price: $699.99

4. Product: TechPro Desktop
   Category: Computers and Laptops
   Brand: TechPro
   Model Number: TP-DT500
   Warranty: 1 year
   Rating: 4.4
   Features: Intel Core i7 processor, 16GB RAM, 1TB HDD, NVIDIA GeForce GTX 1660
   Description: A powerful desktop computer for work and play.
   Price: $999.99

5. Product: BlueWave Chromebook
   Category: Computers and Laptops
   Brand: BlueWave
   Model Number: BW-CB100
   Warranty: 1 year
   Rating: 4.1
   Features: 11.6-inch display, 4GB RAM, 32GB eMMC, Chrome OS
   Description: A compact and affordable Chromebook for everyday tasks.
   Price: $249.99

Step 3:{delimiter} If the message contains products \
in the list above, list any assumptions that the \
user is making in their \
message e.g. that Laptop X is bigger than \
Laptop Y, or that Laptop Z has a 2 year warranty.

Step 4:{delimiter}: If the user made any assumptions, \
figure out whether the assumption is true based on your \
product information. 

Step 5:{delimiter}: First, politely correct the \
customer's incorrect assumptions if applicable. \
Only mention or reference products in the list of \
5 available products, as these are the only 5 \
products that the store sells. \
Answer the customer in a friendly tone.

Use the following format:
Step 1:{delimiter} <step 1 reasoning>
Step 2:{delimiter} <step 2 reasoning>
Step 3:{delimiter} <step 3 reasoning>
Step 4:{delimiter} <step 4 reasoning>
Response to user:{delimiter} <response to customer>

Make sure to include {delimiter} to separate every step.
"""
```
Let us try with a user message

```python
user_message = f"""
by how much is the BlueWave Chromebook more expensive \
than the TechPro Desktop"""

messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
] 

response = get_completion_from_messages(messages)
print(response)
```

You will get output similar to the listing below
```
Step 1:#### The user is asking a question about two specific products, the BlueWave Chromebook and the TechPro Desktop.
Step 2:#### The prices of the two products are as follows:
- BlueWave Chromebook: $249.99
- TechPro Desktop: $999.99
Step 3:#### The user is assuming that the BlueWave Chromebook is more expensive than the TechPro Desktop.
Step 4:#### The assumption is incorrect. The TechPro Desktop is actually more expensive than the BlueWave Chromebook.
Response to user:#### The BlueWave Chromebook is actually less expensive than the TechPro Desktop. The BlueWave Chromebook costs $249.99 while the TechPro Desktop costs $999.99.

```

Let us try another example of product not in the list

```python
user_message = f"""
do you sell tvs"""
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)

```
Response from the bot

```
Step 1:#### The user is asking if the store sells TVs.
Step 2:#### The list of available products does not include any TVs.
Response to user:#### I'm sorry, but we do not sell TVs at this store. Our available products include computers and laptops.

```

## Inner Monologue
How to hide chain of reasoning and show the final thought

```python
try:
    final_response = response.split(delimiter)[-1].strip()
except Exception as e:
    final_response = "Sorry, I'm having trouble right now, please try asking another question."
    
print(final_response)
```

Output will be the final response

```
I'm sorry, but we do not sell TVs at this store. Our available products include computers and laptops.
```

In the next section we look at how to split one task into simpler sub-tasks instead of trying to do everything in one prompt.







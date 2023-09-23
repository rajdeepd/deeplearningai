---
layout: default
title: "Check Inputs"
nav_order: 7
description: "Check Inputs"
parent: ChatGPT
---
In this video we will show checking output of the LLM model. It is important to do this for quality, safety and relevance of the output
We will use moderation API to check the quality of the output but with additional prompts before displaying them.

Let us dive into examples.

## Setup

We have already discussed moderation APIs in terms of evaluating inputs.

```python
import os
import openai
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

Let us use Moderation API to check for harmful content. Let us look a the input below and send it to `openai.Moderation.create`


```python

```final_response_to_customer = f"""
The SmartX ProPhone has a 6.1-inch display, 128GB storage, \
12MP dual camera, and 5G. The FotoSnap DSLR Camera \
has a 24.2MP sensor, 1080p video, 3-inch LCD, and \
interchangeable lenses. We have a variety of TVs, including \
the CineView 4K TV with a 55-inch display, 4K resolution, \
HDR, and smart TV features. We also have the SoundMax \
Home Theater system with 5.1 channel, 1000W output, wireless \
subwoofer, and Bluetooth. Do you have any specific questions \
about these products or any other products we offer?
"""
response = openai.Moderation.create(
    input=final_response_to_customer
)
moderation_output = response["results"][0]
print(moderation_output)

```
If you look at the output you will notice it is not flagged

```json
{
  "categories": {
    "hate": false,
    "hate/threatening": false,
    "self-harm": false,
    "sexual": false,
    "sexual/minors": false,
    "violence": false,
    "violence/graphic": false
  },
  "category_scores": {
    "hate": 4.3336598e-07,
    "hate/threatening": 5.640859e-10,
    "self-harm": 2.8951988e-10,
    "sexual": 2.2455274e-06,
    "sexual/minors": 1.2425416e-08,
    "violence": 6.0600332e-06,
    "violence/graphic": 4.518234e-07
  },
  "flagged": false
}
```
## Check if the output is factually based on Product Information

Another approach for checking outputs is to ask the model itself if the generated was satisfactory and if it follows a certain rubric that you define. This can be done by providing the generated output as part of the input to the model and asking it to rate the quality of the output. You can do this in various different ways.

You could also use a chain of thought reasoning prompt for this. 
This might be a little bit difficult for the model to validate both in one step so you could play around with this. You could also add some other kind of guidelines.  You could ask, give a rubric, like a rubric for an exam or grading an essay. 
You could use that kind of format and say, does this use a friendly tone in line with our brand guidelines and maybe outline some of your brand guidelines if that's something that's very important to you. So let's add our customer message.

```python
system_message = f"""
You are an assistant that evaluates whether \
customer service agent responses sufficiently \
answer customer questions, and also validates that \
all the facts the assistant cites from the product \
information are correct.
The product information and user and customer \
service agent messages will be delimited by \
3 backticks, i.e. ```.
Respond with a Y or N character, with no punctuation:
Y - if the output sufficiently answers the question \
AND the response correctly uses product information
N - otherwise

Output a single letter only.
"""
customer_message = f"""
tell me about the smartx pro phone and \
the fotosnap camera, the dslr one. \
Also tell me about your tvs"""
product_information = """{ "name": "SmartX ProPhone", "category": "Smartphones and Accessories", "brand": "SmartX", "model_number": "SX-PP10", "warranty": "1 year", "rating": 4.6, "features": [ "6.1-inch display", "128GB storage", "12MP dual camera", "5G" ], "description": "A powerful smartphone with advanced camera features.", "price": 899.99 } { "name": "FotoSnap DSLR Camera", "category": "Cameras and Camcorders", "brand": "FotoSnap", "model_number": "FS-DSLR200", "warranty": "1 year", "rating": 4.7, "features": [ "24.2MP sensor", "1080p video", "3-inch LCD", "Interchangeable lenses" ], "description": "Capture stunning photos and videos with this versatile DSLR camera.", "price": 599.99 } { "name": "CineView 4K TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-4K55", "warranty": "2 years", "rating": 4.8, "features": [ "55-inch display", "4K resolution", "HDR", "Smart TV" ], "description": "A stunning 4K TV with vibrant colors and smart features.", "price": 599.99 } { "name": "SoundMax Home Theater", "category": "Televisions and Home Theater Systems", "brand": "SoundMax", "model_number": "SM-HT100", "warranty": "1 year", "rating": 4.4, "features": [ "5.1 channel", "1000W output", "Wireless subwoofer", "Bluetooth" ], "description": "A powerful home theater system for an immersive audio experience.", "price": 399.99 } { "name": "CineView 8K TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-8K65", "warranty": "2 years", "rating": 4.9, "features": [ "65-inch display", "8K resolution", "HDR", "Smart TV" ], "description": "Experience the future of television with this stunning 8K TV.", "price": 2999.99 } { "name": "SoundMax Soundbar", "category": "Televisions and Home Theater Systems", "brand": "SoundMax", "model_number": "SM-SB50", "warranty": "1 year", "rating": 4.3, "features": [ "2.1 channel", "300W output", "Wireless subwoofer", "Bluetooth" ], "description": "Upgrade your TV's audio with this sleek and powerful soundbar.", "price": 199.99 } { "name": "CineView OLED TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-OLED55", "warranty": "2 years", "rating": 4.7, "features": [ "55-inch display", "4K resolution", "HDR", "Smart TV" ], "description": "Experience true blacks and vibrant colors with this OLED TV.", "price": 1499.99 }"""
q_a_pair = f"""
Customer message: ```{customer_message}```
Product information: ```{product_information}```
Agent response: ```{final_response_to_customer}```

Does the response use the retrieved information correctly?
Does the response sufficiently answer the question

Output Y or N
"""
messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': q_a_pair}
]

response = get_completion_from_messages(messages, max_tokens=1)
print(response)
```
Response will be a `Y` in this case.

Let us look at another example

This response is, "life is like a box of chocolates". 
Let us add our message to do with the output checking. 
The model has determined that this does not sufficiently answer the question or use the retrieved information. 
This question, "does it use the retrieved information correctly?". This is a good prompt to use if you want to make sure that the model isn't hallucinating, which is making up things that aren't true.



```python
another_response = "life is like a box of chocolates"
q_a_pair = f"""
Customer message: ```{customer_message}```
Product information: ```{product_information}```
Agent response: ```{another_response}```

Does the response use the retrieved information correctly?
Does the response sufficiently answer the question?

Output Y or N
"""
messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': q_a_pair}
]

response = get_completion_from_messages(messages)
print(response)
```

You will get `N` as a response.

## Summary

In general, checking outputs using the moderation API is good practice, but while asking the model to  evaluate its own output might be useful for immediate feedback to ensure the quality of responses in a very small number of cases. 
I think it's probably unnecessary most of the time, especially if you're using a more advanced model like GPT-4. 
 

---
layout: default
title: "Evaluation Part 1"
nav_order: 9
description: "Evaluation Part 1"
parent: ChatGPT
---
# Evaluation Part 1

In this section we will share with you best practices for evaluating the outputs of an LLM and we want to share 
with you specifically what it feels like to build one of these 
systems. 
One key distinction between what you Learn in this section and what you may have seen in more traditional machine learning supervised learning applications is because you can build such an application so quickly, 
the methods for evaluating it, it tends not to start off with a test set. 

Instead, you often end up gradually building up a set of test examples. 
Let me share with you what I mean by that. 
You may remember this diagram from the second video about how prompt-based development speeds up the core parts of model development from maybe months 
to just minutes or hours or at most a very small number of days. 
In the traditional supervised learning approach, if you needed to collect say, 10,000 labeled examples anyway, then the incremental cost of collecting another 1,000 test examples isn't that bad. So, in the traditional supervised learning setting, 
it was not unusual to collect a training set, collect a development set or holdout cross-validation set in the test set and then tap those at hand throughout this development process. 
 
<img src="https://cdn.mathpix.com/snip/images/oEjzgosNskK3r9UYo069nXr2JoSjeIWPpqCAhZG3Th4.original.fullsize.png" />
 
But if you're able to specify a prompt in just minutes and get something working in hours, then it would seem like a huge pain if you had to pause for a long time to collect 1,000 test examples because you cannot get this working with zero training examples. 
So, when building an application using an LLM, this is what it often feels like. First, you would tune the prompts on just a small handful of examples, maybe one to three to five examples and try to 
get a prompt that works on them. 
And then as you have the system undergo additional testing, you occasionally run into a few examples that are tricky. 
The prompt doesn't work on them or the algorithm doesn't work 
on them. 
And in that case, you can take these additional one or two or three or five examples and add them to the set that you're testing on to just add additional tricky examples opportunistically. 
Eventually, you have enough of these examples you've added to your slowly growing development set that it becomes a bit inconvenient to manually run every example through the prompt every time you change the prompt. 
 
Then you start to develop metrics to measure performance on this small set of examples, such as average accuracy. One interesting aspect of this process is, if you decide at any moment in time, your system is working well enough, you can stop right there and not go on to the next bullet. 
In fact, there are many deploy applications that stops at maybe the first or the second bullet and are running actually just fine. 

If your hand-built development set that you're evaluating the model on isn't giving you sufficient confidence yet in the performance of your system, then that's when you may go to the next step of collecting a randomly sampled set of examples to tune the model to. 
This would continue to be a development set or a hold-out 
cross-validation set, because it'd be quite common to continue to tune your prompt to this. 
And only if you need even higher fidelity estimate of the performance of your system, then might you collect and use a hold-out test sets that you don't even look at yourself when you're tuning the model. And so step four tends to be more important if, your system is getting the right answer 91% of the time, and you want to tune it to get it to give the right answer 92% or 93% of the time, then you do need a larger set of examples to measure those differences between 91% and 93% performance. 

Only if you really need an unbiased, fair estimate of how was the system doing, then do you need to go beyond the development set to also collect a hold-out test set. One important caveat, I've seen a lot applications of large language models where there isn't meaningful meaning risk of harm if it gives not quite the right answer. 
 
But obviously, for any high-stakes applications, if there's a risk of bias or an inappropriate output causing harm to someone, then the responsibility to collect a test set to rigorously evaluate the performance of 
your system to make sure it's doing the right thing before you 
use it, that becomes much more important. For example, if you are using it to summarize articles just for yourself to read and no one else, then maybe the risk of harm is more modest, and you can stop early in this process without going to the expense of bullets four and five and collecting larger data sets on which to evaluate the algorithm.

In this example, let us start with the usual helper functions.

## Setup

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


## Get Relevant Product and Categories

Use the utils function to get a list of products and categories. 
In the computers and laptops category, there's a list of computers and laptops, in the smartphones and accessories category, here's a list of smartphones and accessories, and so on for other categories. 
Now, let's say, the task we're going to address is, given a user input, such as, "what TV can I buy if I'm on a budget?", to retrieve the relevant categories and products, so that we 
have the right info to answer the user's query. 


```python
products_and_category = utils.get_products_and_category()
products_and_category
```

Output

```text
{'Computers and Laptops': ['TechPro Ultrabook',
  'BlueWave Gaming Laptop',
  'PowerLite Convertible',
  'TechPro Desktop',
  'BlueWave Chromebook'],
 'Smartphones and Accessories': ['SmartX ProPhone',
  'MobiTech PowerCase',
  'SmartX MiniPhone',
  'MobiTech Wireless Charger',
  'SmartX EarBuds'],
 'Televisions and Home Theater Systems': ['CineView 4K TV',
  'SoundMax Home Theater',
  'CineView 8K TV',
  'SoundMax Soundbar',
  'CineView OLED TV'],
 'Gaming Consoles and Accessories': ['GameSphere X',
  'ProGamer Controller',
  'GameSphere Y',
  'ProGamer Racing Wheel',
  'GameSphere VR Headset'],
 'Audio Equipment': ['AudioPhonic Noise-Canceling Headphones',
  'WaveSound Bluetooth Speaker',
  'AudioPhonic True Wireless Earbuds',
  'WaveSound Soundbar',
  'AudioPhonic Turntable'],
 'Cameras and Camcorders': ['FotoSnap DSLR Camera',
  'ActionCam 4K',
  'FotoSnap Mirrorless Camera',
  'ZoomMaster Camcorder',
  'FotoSnap Instant Camera']}
```
## Find relevant product and category names (version 1)

Here is a prompt,the prompt specifies a set of instructions, and it actually gives the language model one example of a good output. This is sometimes called a few-shot or technically one-shot prompting, because we're actually using a 
user message and a system message to give it one example of a good output. If someone says, "I want the most expensive computer.". 

Here is the list of products and categories that are in the product catalog.

```python
def find_category_and_product_v1(user_input,products_and_category):

    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    AND
        'products': <a list of products that must be found in the allowed products below>


    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.
    

    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}
    

    """
    
    few_shot_user_1 = """I want the most expensive computer."""
    few_shot_assistant_1 = """ 
    [{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_1 },
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ] 
    return get_completion_from_messages(messages)


```

Yeah, let's just return all the computers, because we don't have pricing information. Now, let's use this prompt on the customer message, "Which TV can I buy if I'm on a budget?"

 ## Evaluate

### Question 1
Now, let's use this prompt on the customer message, "Which TV can I buy if I'm on a budget?". 
And so we're passing in to this both the prompt, customer message zero, as well as the products and category. This is the information that we have retrieved up top using the utils function.

```python
customer_msg_0 = f"""Which TV can I buy if I'm on a budget?"""
products_by_category_0 = find_category_and_product_v1(customer_msg_0,
                                                      products_and_category)
print(products_by_category_0)
```

Output recieved is listed blow. It returns all the TVs and Home theatre systems.

```text

[{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]
```

### Question 2

To see how well the prompt is doing, you may evaluate it on a second prompt. The customer says, "I need a charger for my smartphone.". 

```python
customer_msg_1 = f"""I need a charger for my smartphone"""
products_by_category_1 = find_category_and_product_v1(customer_msg_1,
                                                      products_and_category)
print(products_by_category_1)

```

```text
[{'category': 'Smartphones and Accessories', 'products': ['MobiTech PowerCase', 'MobiTech Wireless Charger', 'SmartX EarBuds']}]
```

It looks like it's correctly retrieving this data. Category, smartphones, accessories, and it lists the relevant products.

### Question 3

And here's another one. "What computers do you have?". 
Hopefully you'll retrieve a list of the computers. 
```python
customer_msg_2 = f"""
What computers do you have?"""
​
products_by_category_2 = find_category_and_product_v1(customer_msg_2,
                                                      products_and_category)
products_by_category_2
```

```text
[{'category': 'Computers and Laptops', 'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
```

### Question 4

```python
customer_msg_3 = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs do you have?"""
​
products_by_category_3 = find_category_and_product_v1(customer_msg_3,
                                                      products_and_category)
print(products_by_category_3)
```

```text
[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']},{'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}]
    
[{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]
```

Here we have three prompts, and if you are developing this prompt for the first time, it would be quite reasonable to have one or two or three examples like this, and to keep on tuning the prompt until it gives appropriate outputs, until the prompt is retrieving the relevant products and categories to the customer request for all of your prompts, all three of them in this example. And if the prompt had been missing some products or something, then what we would do is probably go back to edit the prompt a few times until it gets it right on all three of these prompts.

After you've gotten the system to this point, you 
might then start running the system in testing. Maybe 
send it to internal test users or try using it yourself, and just run it for a while to see what happens. 

And sometimes you will run across a prompt that it fails on. So here's an example of a prompt, "tell me about the smartx pro phone and the fotosnap camera. Also, what TVs do you have?". 
When I run it on this prompt, it looks like it's outputting the right data, but it also outputs a bunch of text here, this extra junk. It makes it harder to parse this into a Python list of dictionaries. 

We don't like that it's outputting this extra junk. So when you run across one example that the system 
fails on, then common practice is to just note down that this is a somewhat tricky example, so let's add this to our set of examples that we're going to test the system on systematically. 

If you keep on running the system for a while longer, maybe it works on those examples. We did tune the 
prompt to three examples, so maybe it will work on many examples, but just by chance you might runacross another example where it generates an error. 

## Harder Test Cases

Custom message 4 below also causes the system to output a bunch of junk text at the end that we don't want. 

Trying to be helpful to give all this extra text, we actually don't want this.

```python
customer_msg_4 = f"""
tell me about the CineView TV, the 8K one, Gamesphere console, the X one.
I'm on a budget, what computers do you have?"""

products_by_category_4 = find_category_and_product_v1(customer_msg_4,
                                                      products_and_category)
print(products_by_category_4)
```

Output

```text
[{'category': 'Televisions and Home Theater Systems', 'products': ['CineView 8K TV']},
     {'category': 'Gaming Consoles and Accessories', 'products': ['GameSphere X']},
     {'category': 'Computers and Laptops', 'products': ['BlueWave Chromebook']}]
     
    Note: The CineView TV mentioned is the 8K one, and the Gamesphere console mentioned is the X one. 
    For the computer category, since the customer mentioned being on a budget, we cannot determine which specific product to recommend. 
    Therefore, we have included all the products in the Computers and Laptops category in the output.
```

And so at this point, you may have run this prompt, maybe on hundreds of examples, maybe you have test users, but you would just take the examples, the tricky ones is doing poorly on, and now I have this set of five examples, index from 0 to 4, have this set of five examples that you use to further fine-tune the prompts.

## Prompt modified for harder test cases

After a little bit of trial and error, you might decide to modify the prompts as follows. 
So here's a new prompt, this is called prompt v2. But what we did here was we added to the prompt, 
"Do not output any additional text that's not in JSON format.", just to emphasize, please don't output this JSON stuff. And added a second example using the user and assistant message for few-shot prompting where the user asked for the cheapest computer. 
And in both of the few-shot examples, we're demonstrating to the system a response where it gives 
only JSON outputs. So here's the extra thing that we just added to the prompt, "Do not output any additional text that's not in JSON formats.", and we use `few_shot_user_1`, `few_shot_assistant_1`, and `few_shot_user_2`, `few_shot_assistant_2` to give it two of these few shot prompts

```python
def find_category_and_product_v2(user_input,products_and_category):
    """
    Added: Do not output any additional text that is not in JSON format.
    Added a second example (for few-shot prompting) where user asks for 
    the cheapest computer. In both few-shot examples, the shown response 
    is the full list of products in JSON only.
    """
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    AND
        'products': <a list of products that must be found in the allowed products below>
    Do not output any additional text that is not in JSON format.
    Do not write any explanatory text after outputting the requested JSON.


    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.
    

    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}
    

    """
    
    few_shot_user_1 = """I want the most expensive computer. What do you recommend?"""
    few_shot_assistant_1 = """ 
    [{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    
    few_shot_user_2 = """I want the most cheapest computer. What do you recommend?"""
    few_shot_assistant_2 = """ 
    [{'category': 'Computers and Laptops', \
'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook']}]
    """
    
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_1 },
    {'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"},  
    {'role':'assistant', 'content': few_shot_assistant_2 },
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ] 
    return get_completion_from_messages(messages)
```

#### Evaluation

```python
customer_msg_3 = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs do you have?"""

products_by_category_3 = find_category_and_product_v2(customer_msg_3,
                                                      products_and_category)
print(products_by_category_3)
```

So let me hit Shift-Enter to find that prompt. If you were to go back and manually rerun this 
prompt on all five of the examples of user inputs, including this one that previously had given a broken 
output, you'll find that it now gives a correct output. And if you were to go back and rerun this new prompt, this is prompt version v2, on that customer message example that had results in the broken output with extra junk after the JSON output, then this will generate a better output. 

Please rerun customer message 4 and prompt v2 and check if it generates correct value.
Make sure on modifying the prompts do some regression testing.
When fixing the incorrect outputs on prompts 3 and 4, it didn't break the output on prompt 0 either. Now
can tell that if I had to copy-paste 5 prompts, customers such as 0, 1, 2, 3, and 4, into my Jupyter notebook and run them and then manually look at them to see if they output in the right categories and 
products. You can kind of do it. I can look at this and go, "Yep, category, TV and home theater systems, products.


```text
[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']}]
```

Yep, looks like you  got all of them.".  But it's actually a little bit painful to do this manually, to  manually inspect or to look at this output to make sure with  your eyes that this is exactly the right output.  So when the development set that you're tuning to becomes more  than just a small handful of examples, it  then becomes useful to start to automate the testing process. 
So here is a set of 10 examples where I'm specifying 10 customer messages. 


```python
msg_ideal_pairs_set = [
    
    # eg 0
    {'customer_msg':"""Which TV can I buy if I'm on a budget?""",
     'ideal_answer':{
        'Televisions and Home Theater Systems':set(
            ['CineView 4K TV', 'SoundMax Home Theater', 'CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV']
        )}
    },

    # eg 1
    {'customer_msg':"""I need a charger for my smartphone""",
     'ideal_answer':{
        'Smartphones and Accessories':set(
            ['MobiTech PowerCase', 'MobiTech Wireless Charger', 'SmartX EarBuds']
        )}
    },
    # eg 2
    {'customer_msg':f"""What computers do you have?""",
     'ideal_answer':{
           'Computers and Laptops':set(
               ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook'
               ])
                }
    },

    # eg 3
    {'customer_msg':f"""tell me about the smartx pro phone and \
    the fotosnap camera, the dslr one.\
    Also, what TVs do you have?""",
     'ideal_answer':{
        'Smartphones and Accessories':set(
            ['SmartX ProPhone']),
        'Cameras and Camcorders':set(
            ['FotoSnap DSLR Camera']),
        'Televisions and Home Theater Systems':set(
            ['CineView 4K TV', 'SoundMax Home Theater','CineView 8K TV', 'SoundMax Soundbar', 'CineView OLED TV'])
        }
    }, 
    
    # eg 4
    {'customer_msg':"""tell me about the CineView TV, the 8K one, Gamesphere console, the X one.
I'm on a budget, what computers do you have?""",
     'ideal_answer':{
        'Televisions and Home Theater Systems':set(
            ['CineView 8K TV']),
        'Gaming Consoles and Accessories':set(
            ['GameSphere X']),
        'Computers and Laptops':set(
            ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', 'TechPro Desktop', 'BlueWave Chromebook'])
        }
    },
    
    # eg 5
    {'customer_msg':f"""What smartphones do you have?""",
     'ideal_answer':{
           'Smartphones and Accessories':set(
               ['SmartX ProPhone', 'MobiTech PowerCase', 'SmartX MiniPhone', 'MobiTech Wireless Charger', 'SmartX EarBuds'
               ])
                    }
    },
    # eg 6
    {'customer_msg':f"""I'm on a budget.  Can you recommend some smartphones to me?""",
     'ideal_answer':{
        'Smartphones and Accessories':set(
            ['SmartX EarBuds', 'SmartX MiniPhone', 'MobiTech PowerCase', 'SmartX ProPhone', 'MobiTech Wireless Charger']
        )}
    },

    # eg 7 # this will output a subset of the ideal answer
    {'customer_msg':f"""What Gaming consoles would be good for my friend who is into racing games?""",
     'ideal_answer':{
        'Gaming Consoles and Accessories':set([
            'GameSphere X',
            'ProGamer Controller',
            'GameSphere Y',
            'ProGamer Racing Wheel',
            'GameSphere VR Headset'
     ])}
    },
    # eg 8
    {'customer_msg':f"""What could be a good present for my videographer friend?""",
     'ideal_answer': {
        'Cameras and Camcorders':set([
        'FotoSnap DSLR Camera', 'ActionCam 4K', 'FotoSnap Mirrorless Camera', 'ZoomMaster Camcorder', 'FotoSnap Instant Camera'
        ])}
    },
    
    # eg 9
    {'customer_msg':f"""I would like a hot tub time machine.""",
     'ideal_answer': []
    }
    
]

```


So  here's a customer message, "Which TV can I buy if I'm on a  budget?" as well as what's the ideal answer. Think of this as  the right answer in the test set, or really, I should  say development set, because we're actually tuning  to this. 

We have collected here  10 examples indexed from 0 through 9, where  the last one is if the user says, "I would like hot tub time  machine.". We have no relevant products to that, really sorry, so the ideal answer  is the empty set.
 
```python
import json
def eval_response_with_ideal(response,
                              ideal,
                              debug=False):
    
    if debug:
        print("response")
        print(response)
    
    # json.loads() expects double quotes, not single quotes
    json_like_str = response.replace("'",'"')
    
    # parse into a list of dictionaries
    l_of_d = json.loads(json_like_str)
    
    # special case when response is empty list
    if l_of_d == [] and ideal == []:
        return 1
    
    # otherwise, response is empty 
    # or ideal should be empty, there's a mismatch
    elif l_of_d == [] or ideal == []:
        return 0
    
    correct = 0    
    
    if debug:
        print("l_of_d is")
        print(l_of_d)
    for d in l_of_d:

        cat = d.get('category')
        prod_l = d.get('products')
        if cat and prod_l:
            # convert list to set for comparison
            prod_set = set(prod_l)
            # get ideal set of products
            ideal_cat = ideal.get(cat)
            if ideal_cat:
                prod_set_ideal = set(ideal.get(cat))
            else:
                if debug:
                    print(f"did not find category {cat} in ideal")
                    print(f"ideal: {ideal}")
                continue
                
            if debug:
                print("prod_set\n",prod_set)
                print()
                print("prod_set_ideal\n",prod_set_ideal)

            if prod_set == prod_set_ideal:
                if debug:
                    print("correct")
                correct +=1
            else:
                print("incorrect")
                print(f"prod_set: {prod_set}")
                print(f"prod_set_ideal: {prod_set_ideal}")
                if prod_set <= prod_set_ideal:
                    print("response is a subset of the ideal answer")
                elif prod_set >= prod_set_ideal:
                    print("response is a superset of the ideal answer")

    # count correct over total number of items in list
    pc_correct = correct / len(l_of_d)
        
    return pc_correct
```

If you want to evaluate automatically,  what the prompt is doing on any of these 10 examples,  here is a function to do so. It's kind of a long function. Feel  free to pause the video and read through  it if you wish. But let me just demonstrate what  it is actually doing.  So let me print out the customer message,  for customer message 0.  So the customer message is,"Which TV can I buy if I'm on  a budget?".  And let's also print out the ideal answer. So  the ideal answer is here are all the TVs that we want the prompt to retrieve. 

```python
print(f'Customer message: {msg_ideal_pairs_set[7]["customer_msg"]}')
print(f'Ideal answer: {msg_ideal_pairs_set[7]["ideal_answer"]}')
```

Output of the print function above

```text

Customer message: What Gaming consoles would be good for my friend who is into racing games?
Ideal answer: {'Gaming Consoles and Accessories': {'GameSphere X', 'ProGamer Racing Wheel', 'GameSphere VR Headset', 'ProGamer Controller', 'GameSphere Y'}}
```


And let me now call the prompt. This  is prompt V2 on this customer message with that  user products and category information. Let's print  it out and then we'll call the eval.  We'll call the eval response of ideal function to  see how well the response matches the ideal  answer.  

```python
response = find_category_and_product_v2(msg_ideal_pairs_set[7]["customer_msg"],
                                         products_and_category)
print(f'Resonse: {response}')

```

And in this case, it did output the category that we wanted,  and it did output the entire list of products.  And so this gives it a score of 1.0.  Just to show you one more example, it turns out that I know  it gets it wrong on example 7.  

So if I change this from 0 to 7 and run it,  this is what it gets. Oh, let me update this to 7 as well.  So under this customer message, this is the ideal answer where  it should output under gaming consoles  and accessories. So list of gaming consoles and accessories.  But whereas the response here has three outputs,  it actually should have had 1, 2, 3, 4, 5 outputs. And so it's  missing some of the products.  So what I would do if I'm tuning the prompt now is I would then  use a fold to loop over all 10 of the development set examples, where we repeatedly pull out the customer message, get  the ideal answer, the right answer, call  the arm to get a response, evaluate it, and then, you know, accumulate  it in average. And let me just run this. 


```python
# Note, this will not work if any of the api calls time out
score_accum = 0
for i, pair in enumerate(msg_ideal_pairs_set):
    print(f"example {i}")
    
    customer_msg = pair['customer_msg']
    ideal = pair['ideal_answer']
    
    # print("Customer message",customer_msg)
    # print("ideal:",ideal)
    response = find_category_and_product_v2(customer_msg,
                                                      products_and_category)

    
    # print("products_by_category",products_by_category)
    score = eval_response_with_ideal(response,ideal,debug=False)
    print(f"{i}: {score}")
    score_accum += score
    

n_examples = len(msg_ideal_pairs_set)
fraction_correct = score_accum / n_examples
print(f"Fraction correct out of {n_examples}: {fraction_correct}")
```

So this will take a while to run, but when it's done running, this  is the result. We're running through the 10 examples. 


```text
example 0
0: 1.0
example 1
1: 1.0
example 2
2: 1.0
example 3
3: 1.0
example 4
4: 1.0
example 5
5: 1.0
example 6
6: 1.0
example 7
incorrect
prod_set: {'GameSphere VR Headset', 'ProGamer Racing Wheel', 'ProGamer Controller'}
prod_set_ideal: {'GameSphere X', 'ProGamer Controller', 'GameSphere Y', 'ProGamer Racing Wheel', 'GameSphere VR Headset'}
response is a subset of the ideal answer
7: 0.0
example 8
8: 1.0
example 9
9: 1
Fraction correct out of 10: 0.9

```


Looks like  example 7 was wrong. And so the fraction correct of 10 was 90% correct.
If you were to tune the prompts,  you can rerun this to see if the percent correct goes up or down.



What you just saw in this notebook is going through steps 1,  2, and 3 of this bulleted list, and this already gives a pretty  good development set of 10 examples with which to tune and validate  the prompts is working.  If you needed an additional level of rigor,  then you now have the software needed to  collect a randomly sampled set of maybe 100  examples with their ideal outputs, and maybe even go beyond  that to the rigor of a holdout test set that you don't  even look at while you're tuning the prompt. But for  a lot of applications, stopping at bullet 3, but there  are also certainly applications where you could do  what you just saw me do in this Jupyter notebook, and it  gets a pretty performance system quite quickly.


With again, the important caveat that if you're  working on a safety critical application or an  application where there's non-trivial risk of harm, then of course,  it would be the responsible thing to do to actually get  a much larger test set to really verify  the performance before you use it anywhere.
And so that's it. I find that the workflow of building  applications using prompts is very different than a workflow  of building applications using supervised learning,  and the pace of iteration feels much faster. And  if you have not yet done it before, you might be surprised at  how well an evaluation method built on just  a few hand-curated tricky examples. You think  with 10 examples, and this is not statistically valid  for almost anything. But you might be surprised  when you actually use this procedure, how effective  adding a handful, just a handful of tricky examples.
into development sets might be in terms of  helping you and your team get to an  effective set of prompts and effective system. In this  video, the outputs could be evaluated quantitatively, as in  there was a desired output and you could  tell if it gave this desired output or not. So the  next section, let's take a look at how you can evaluate output in that setting where what is the right answer is a bit more ambiguous.

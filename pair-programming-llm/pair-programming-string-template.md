---
layout: default
title: Using a String template
nav_order: 2
description: ""
has_children: false
parent:  Using a String template 
---

#### Setup
Set the MakerSuite API key with the provided helper function.


```python
import os
from utils import get_api_key
import google.generativeai as palm
from google.api_core import client_options as client_options_lib

palm.configure(
    api_key=get_api_key(),
    transport="rest",
    client_options=client_options_lib.ClientOptions(
        api_endpoint=os.getenv("GOOGLE_API_BASE"),
    )
)
```

#### Pick the model that generates text


```python
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison
```




    Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40)



#### Helper function to call the PaLM API

Note: temperature is set to 0.0

```python
from google.api_core import retry
@retry.Retry()
def generate_text(prompt, 
                  model=model_bison, 
                  temperature=0.0):
    return palm.generate_text(prompt=prompt,
                              model=model,
                              temperature=temperature)
```

#### Prompt template

1. priming: getting the LLM ready for the type of task you'll ask it to do.
2. question: the specific task.
3. decorator: how to provide or format the output.


```python
prompt_template = """
{priming}

{question}

{decorator}
Your solution:
"""
```


```python
priming_text = "You are an expert at writing clear, concise, Python code."
```


```python
question = "create a doubly linked list"
```

#### Observe how the decorator affects the output
- In other non-coding prompt engineering tasks, it's common to use "chain-of-thought prompting" by asking the model to work through the task "step by step".
- For certain tasks like generating code, you may want to experiment with other wording that would make sense if you were asking a developer the same question.

In the code cell below, try out option 1 first, then try out option 2.


```python
# option 1
# decorator = "Work through it step by step, and show your work. One step per line."

# option 2
decorator = "Insert comments for each line of code."
```


```python
prompt = prompt_template.format(priming=priming_text,
                                question=question,
                                decorator=decorator)
```

#### review the prompt


```python
print(prompt)
```

#### Call the API to get the completion


```python
completion = generate_text(prompt)
print(completion.result)
```
```
```python
class Node:

    """Node in a doubly linked list."""

    def __init__(self, data):
        """Initialize a node with the given data."""
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:

    """Doubly linked list."""

    def __init__(self):
        """Initialize an empty doubly linked list."""
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self):
        """Return the number of nodes in the list."""
        return self.size

    def is_empty(self):
        """Return True if the list is empty."""
        return self.size == 0

    def add_first(self, data):
        """Add a new node with the given data to the front of the list."""
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def add_last(self, data):
        """Add a new node with the given data to the end of the list."""
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def remove_first(self):
        """Remove the first node in the list."""
        if self.is_empty():
            raise ValueError("List is empty")
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        else:
            self.head.prev = None
        self.size -= 1

    def remove_last(self):
        """Remove the last node in the list."""
        if self.is_empty():
            raise ValueError("List is empty")
        self.tail = self.tail.prev
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.size -= 1

    def __iter__(self):
        """Iterate over the nodes in the list in order."""
        node = self.head
        while node is not None:
            yield node.data
            node = node.next

    def __str__(self):
        """Return a string representation of the list."""
        return "[" + ", ".join(str(node.data) for node in self) + "]"

```
```
#### Try another question


```python
question = """create a very large list of random numbers in python, 
and then write code to sort that list"""
```


```python
prompt = prompt_template.format(priming=priming_text,
                                question=question,
                                decorator=decorator)
```


```python
print(prompt)
```


```python
completion = generate_text(prompt)
print(completion.result)
```

#### Try out the generated code
- Debug it as needed.  For instance, you may need to import `random` in order to use the `random.randint()` function.


```python
# copy-paste some of the generated code that generates random numbers
random_numbers = [random.randint(0, 100) for _ in range(100000)]
print(random_numbers)
```

Screenshot_2023-10-08_at_5.00.18_PM.png

<img src="/deeplearningai/pair-programming-llm/images/Screenshot_2023-10-07_at_10.38.47_PM.png" width="80%"/>
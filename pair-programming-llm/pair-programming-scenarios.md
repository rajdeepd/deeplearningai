---
layout: default
title: pair-programing-scenarios
nav_order: 4
description: ""
has_children: false
parent:  Pair Programming LLM 
---

# pair-programing-scenarios

**Setup**
Set the MakerSuite API key with the provided helper function.
```
import os
from utils import get_api_key
import google.generativeai as palm
from google.api_core import client_op
```
**Pick the model that generates text**

```
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison
```


```
Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40)
```

#### Helper function to call the PaLM API


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

### Scenario 1: Improve existing code
- An LLM can help you rewrite your code in the way that's recommended for that particular language.
- You can ask an LLM to rewrite your Python code in a way that is more 'Pythonic".


```python
prompt_template = """
I don't think this code is the best way to do it in Python, can you help me?

{question}

Please explain, in detail, what you did to improve it.
"""
```


```python
question = """
def func_x(array)
  for i in range(len(array)):
    print(array[i])
"""
```


```python
completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```
Output is listed below with the explaination

```
def func_x(array):
  print(*array)

I improved the code by using the `*` operator to unpack the array into individual arguments for the `print()` function. This is more concise and efficient than using a `for` loop.
```

#### Ask for multiple ways of rewriting your code

There might be multiple ways of doing this, let us update the prompt template. We are also adding a new decorator : `Please explore multiple ways of solving the problem, and explain each`.

```python
prompt_template = """
I don't think this code is the best way to do it in Python, can you help me?

{question}

Please explore multiple ways of solving the problem, and explain each.
"""
```


```python
completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```


```
There are a few ways to improve the code.

1. Use a list comprehension. This will allow you to iterate over the array and print each element in a single line of code.


def func_x(array):
  print([element for element in array])


1. Use the `enumerate() function. This function returns an iterator that yields the index and value of each element in the array. You can use this to print each element in the array with its index.

`
def func_x(array):
  for i, element in enumerate(array):
    print(f"{i}: {element}")


1. Use the map() function. This function applies a function to each element in an iterable. You can use this to print each element in the array with a custom formatting.


def func_x(array):
  print(map(str, array))


Here is a table comparing the three methods:

| Method | Pros | Cons |
|---|---|---|
| List comprehension | Concise | Can be difficult to read for complex code |
| `enumerate()` | Easy to read | Requires an extra variable to store the index |
| `map()` | Flexible | Requires a custom function to format the output |

Ultimately, the best way to solve the problem depends on the specific requirements of your code. If you need a concise solution, the list comprehension is a good option. If you need a solution that is easy to read, the `enumerate()` function is a good option. If you need a solution that is flexible, the map() function is a good option.
```

#### Paste markdown into a markdown cell

If the model outputs what looks like a table in markdown, you can copy-paste markdown into a markdown cell to make it easier to view:

For example:

| Method | Pros | Cons |
|---|---|---|
| List comprehension | Concise | Can be difficult to read for complex code |
| `enumerate()` | Easy to read | Requires an extra variable to store the index |
| `map()` | Flexible | Requires a custom function to format the output |


#### Ask the model to recommend one of the methods as most 'Pythonic'


```python
prompt_template = """
I don't think this code is the best way to do it in Python, can you help me?

{question}

Please explore multiple ways of solving the problem, 
and tell me which is the most Pythonic
"""
```


```python
completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```
### Scenario 2: Simplify code
- Ask the LLM to perform a code review.
- Note that adding/removing newline characters may affect the LLM completion that gets output by the LLM.


```python
# option 1
prompt_template = """
Can you please simplify this code for a linked list in Python?

{question}

Explain in detail what you did to modify it, and why.
"""
```

After you try option 1, you can modify it to look like option 2 (in this markdown cell) and see how it changes the completion.
```Python
# option 2
prompt_template = """
Can you please simplify this code for a linked list in Python? \n
You are an expert in Pythonic code.

{question}

Please comment each line in detail, \n
and explain in detail what you did to modify it, and why.
"""
```


```python
question = """
class Node:
  def __init__(self, dataval=None):
    self.dataval = dataval
    self.nextval = None

class SLinkedList:
  def __init__(self):
    self.headval = None

list1 = SLinkedList()
list1.headval = Node("Mon")
e2 = Node("Tue")
e3 = Node("Wed")
list1.headval.nextval = e2
e2.nextval = e3

"""
```


```python
completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```
Output


```
There are a few ways to improve this code.

First, we can use the `collections.deque` class instead of a linked list. This will make the code more concise and efficient.

from collections import deque

def create_linked_list(data):
  """Creates a linked list from a list of data."""
  list1 = deque()
  for item in data:
    list1.append(item)
  return list1

list1 = create_linked_list(["Mon", "Tue", "Wed"])


Second, we can use the `list.extend()` method to add multiple items to a list at once. This will make the code more concise.

def create_linked_list(data):
  """Creates a linked list from a list of data."""
  list1 = []
  for item in data:
    list1.extend([item])
  return list1

list1 = create_linked_list(["Mon", "Tue", "Wed"])


Finally, we can use the `list.insert()` method to insert an item at a specific index in a list. This will make the code more flexible.


def create_linked_list(data):
  """Creates a linked list from a list of data."""
  list1 = []
  for i, item in enumerate(data):
    list1.insert(i, item)
  return list1

list1 = create_linked_list(["Mon", "Tue", "Wed"])


Of the three solutions, I would say that the most Pythonic is the first one, using the `collections.deque` class. This is because it is the most concise and efficient solution.
```

### Scenario 3: Write test cases

- It may help to specify that you want the LLM to output "in code" to encourage it to write unit tests instead of just returning test cases in English.


```python
prompt_template = """
Can you please create test cases in code for this Python code?

{question}

Explain in detail what these test cases are designed to achieve.
"""
```


```python
# Note that the code I'm using here was output in the previous
# section. Your output code may be different.
question = """
class Node:
  def __init__(self, dataval=None):
    self.dataval = dataval
    self.nextval = None

class SLinkedList:
  def __init__(self):
    self.head = None

def create_linked_list(data):
  head = Node(data[0])
  for i in range(1, len(data)):
    node = Node(data[i])
    node.nextval = head
    head = node
  return head

list1 = create_linked_list(["Mon", "Tue", "Wed"])
"""
```


```python
completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```

```
import unittest

class TestSLinkedList(unittest.TestCase):

    def test_create_linked_list(self):
        """Test that a linked list is created with the correct data."""
        data = ["Mon", "Tue", "Wed"]
        head = create_linked_list(data)
        self.assertEqual(head.dataval, "Mon")
        self.assertEqual(head.nextval.dataval, "Tue")
        self.assertEqual(head.nextval.nextval.dataval, "Wed")

    def test_insert_into_linked_list(self):
        """Test that a new node can be inserted into a linked list."""
        data = ["Mon", "Tue", "Wed"]
        head = create_linked_list(data)
        new_node = Node("Thu")
        new_node.nextval = head.nextval
        head.nextval = new_node
        self.assertEqual(head.nextval.dataval, "Thu")
        self.assertEqual(head.nextval.nextval.dataval, "Tue")

    def test_delete_from_linked_list(self):
        """Test that a node can be deleted from a linked list."""
        data = ["Mon", "Tue", "Wed"]
        head = create_linked_list(data)
        del_node = head.nextval
        head.nextval = del_node.nextval
        self.assertEqual(head.nextval.dataval, "Wed")
        self.assertIsNone(del_node.nextval)

if __name__ == "__main__":
    unittest.main()


The test cases are designed to achieve the following:

* Test that a linked list is created with the correct data.
* Test that a new node can be inserted into a linked list.
* Test that a node can be deleted from a linked list.
```

### Scenario 4: Make code more efficient
- Improve runtime by potentially avoiding inefficient methods (such as ones that use recursion when not needed).


```python
prompt_template = """
Can you please make this code more efficient?

{question}

Explain in detail what you changed and why.
"""
```


```python
question = """
# Returns index of x in arr if present, else -1
def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        return -1

# Test array
arr = [ 2, 3, 4, 10, 40 ]
x = 10

# Function call
result = binary_search(arr, 0, len(arr)-1, x)

if result != -1:
    print("Element is present at index", str(result))
else:
    print("Element is not present in array")

"""
```


```python
completion = generate_text(
    prompt = prompt_template.format(question=question)
)
print(completion.result)
```

I made the following changes to the code to make it more efficient:

* I used the `bisect` function to find the index of the middle element of the array. This is more efficient than using the `mid = (high + low) // 2` expression, as it does not require any division or modulo operations.
* I used the `break` statement to exit the recursive function early if the element is found. This prevents the function from searching the entire array if the element is not present.

The following is the improved code:

```python
# Returns index of x in arr if present, else -1
def binary_search(arr, x):
    # Find the index of the middle element of the array
    mid = bisect.bisect_left(arr, x)

    # Check if the element is found
    if mid < len(arr) and arr[mid] == x:
        return mid
    else:
        return -1

# Test array
arr = [ 2, 3, 4, 10, 40 ]
x = 10

# Function call
result = binary_search(arr, x)

if result != -1:
    print("Element is present at index", str(result))
else:
    print("Element is not present in array")
```

This code is significantly faster than the original code, as it does not perform any unnecessary calculations.
#### Try out the LLM-generated code
- If it uses `bisect`, you may first need to `import bisect`
- Remember to check what the generated code is actually doing.  For instance, the code may work because it is calling a predefined function (such as `bisect`), even though the rest of the code is technically broken.


```python
# Paste the LLM-generated code to inspect and debug it

```

### Scenario 5: Debug your code


```python
prompt_template = """
Can you please help me to debug this code?

{question}

Explain in detail what you found and why it was a bug.
"""
```


```python
# I deliberately introduced a bug into this code! Let's see if the LLM can find it.
# Note -- the model can't see this comment -- but the bug is in the
# print function. There's a circumstance where nodes can be null, and trying
# to print them would give a null error.
question = """
class Node:
   def __init__(self, data):
      self.data = data
      self.next = None
      self.prev = None

class doubly_linked_list:
   def __init__(self):
      self.head = None

# Adding data elements
   def push(self, NewVal):
      NewNode = Node(NewVal)
      NewNode.next = self.head
      if self.head is not None:
         self.head.prev = NewNode
      self.head = NewNode

# Print the Doubly Linked list in order
   def listprint(self, node):
       print(node.data),
       last = node
       node = node.next

dllist = doubly_linked_list()
dllist.push(12)
dllist.push(8)
dllist.push(62)
dllist.listprint(dllist.head)

"""
```

Notice in this case that we are using the default temperature of `0.7` to generate the example that you're seeing in the lecture video.  
- Since a temperature > 0 encourages more randomness in the LLM output, you may want to run this code a couple times to see what it outputs.


```python
completion = generate_text(
    prompt = prompt_template.format(question=question),
    temperature = 0.7
)
print(completion.result)
```

I found a bug in the `listprint` method. The bug is that the `last` variable is not being initialized before it is used. This causes the program to crash when it tries to print the first node in the list.

To fix the bug, I added the following line to the beginning of the `listprint` method:

```
last = node
```

This initializes the `last` variable to the value of the `node` variable, which is the first node in the list. This allows the program to successfully print the first node in the list without crashing.
---
layout: default
title: Getting Started
nav_order: 2
description: ""
has_children: false
parent:  Pair Programming LLM 
---


# Getting Started

#### Setup
Set the MakerSuite API key with the provided helper function.


```python
from utils import get_api_key
```

In this classroom, we've installed the relevant libraries for you.

If you wanted to use the PaLM API on your own machine, you would first install the library:
```Python
!pip install -q google.generativeai
```
The optional flag `-q` installs "quietly" without printing out details of the installation.



```python
import os
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

### Explore the available models


```python
for m in palm.list_models():
    print(f"name: {m.name}")
    print(f"description: {m.description}")
    print(f"generation methods:{m.supported_generation_methods}\n")
```


    name: models/chat-bison-001
    description: Chat-optimized generative language model.
    generation methods:['generateMessage', 'countMessageTokens']
    
    name: models/text-bison-001
    description: Model targeted for text generation.
    generation methods:['generateText', 'countTextTokens', 'createTunedTextModel']
    
    name: models/embedding-gecko-001
    description: Obtain a distributed representation of a text.
    generation methods:['embedText']

#### Filter models by their supported generation methods

- `generateText` is currently recommended for coding-related prompts.
- `generateMessage` is optimized for multi-turn chats (dialogues) with an LLM.


```python
models = [m for m in palm.list_models() 
          if 'generateText' 
          in m.supported_generation_methods]
models
```

```
[Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40)]
```
`


```python
model_bison = models[0]
model_bison
```

```
Model(name='models/text-bison-001', base_model_id='', version='001', display_name='Text Bison', description='Model targeted for text generation.', input_token_limit=8196, output_token_limit=1024, supported_generation_methods=['generateText', 'countTextTokens', 'createTunedTextModel'], temperature=0.7, top_p=0.95, top_k=40)
```



#### helper function to generate text

- The `@retry` decorator helps you to retry the API call if it fails.
- We set the temperature to 0.0 so that the model returns the same output (completion) if given the same input (the prompt).


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

#### Ask the LLM how to write some code

<img src="/deeplearningai/pair-programming-llm/images/Screenshot_2023-10-07_at_10.36.53_PM.png" width="50%"/>

```python
prompt = "Show me how to iterate across a list in Python."
```


```python
completion = generate_text(prompt)
```


```python
print(completion.result)
```


<img src="/deeplearningai/pair-programming-llm/images/Screenshot_2023-10-07_at_10.38.47_PM.png" width="80%"/>

### Try out the code

* Try copy-pasting some of the generated code and running it in the notebook.
* Remember to test out the LLM-generated code and debug it make sure it works as intended.

```python
# paste the LLM's code here
list = ["apple", "banana", "cherry"]

# Iterate over the list using a for loop
for item in list:
    print(item)

```

Output of the code

```
apple
banana
cherry
```


---
layout: default
title: 4. Multimodal Retrieval Augmented Generation (MM-RAG)¶
nav_order: 5
description: "Multimodal RAG"
has_children: false
parent:  Building Multimodal Search and RAG - Weaviate
---

## Introduction

In this lesson, you learn the concept of multimodal retrieval augmented generation by mixing what you want to search together with language vision models. Then, you implement the full multimodal RAG process using Weaviate and the large multi-model model?

Even if all the objective knowledge, the problem of large language models is that they have no information about data that wasn't presented to them during training.
So if you try to prompt them for information that they don't have, though, it is said that they don't know or what is even more likely they would just hallucinate and make up an answer, which is probably even worse. 

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-23_at_12.23.38_PM.png"  width="80%" /> 

A potential solution to this, is retrieval augmented generation or RAG. Here, what you do is instead of just providing the language model with a question in the form of a prompt, you give it a question along with retrieved relevant information.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-23_at_12.23.46_PM.png"  width="80%" /> 


Now the model can perform this retrieve, then generate operation where it can read relevant information before you has to answer your question. The output is customized to the information that you provide it. 

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-23_at_12.23.56_PM.png"  width="80%" /> 

Typically, if you want to scale up your RAG resolution, or want to put all your documents into a vector database like Weaviate. 

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-23_at_12.24.30_PM.png"  width="80%" /> 

Then you can retrieve the most relevant documents from the vector database using the prompt and pass those relevant documents into the context window of the large language model together with the prompt.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-23_at_3.44.59_PM.png"  width="80%" /> 

This way, you help your language model to generate a response based on the provided context. But as you already saw that a vector database is capable of retrieving a lot more than just text.
So let's take advantage of the multi-modal knowledge base with Weaviate. To store and search through images, video and text.


## Multi-modal RAG

In this diagram, you can see retrieval of an image from our multimodal vector database. 

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-23_at_3.45.08_PM.png"  width="80%" /> 

Passing that image along with text instruction to a large multi-model model, you will get a response that is grounded in the multimodal understanding of the world. This process is known as multimodal retrieval augmented generation. Because you augmented the generation with retrieval of multimodal data. Let's now see all of this in practice.

2:13


* In this classroom, the libraries have been already installed for you.
* If you would like to run this code on your own machine, you need to install the following:
```
    !pip install -U weaviate-client
    !pip install google-generativeai
```


```python
import warnings
warnings.filterwarnings("ignore")
```


s.
## Setup
### Load environment variables and API keys

In his lab, you use images and text as input, then you get alarms to reason over. It does completely the full RAG workflow. So, like in the
previous lessons, let's start with, a little command that will ignore all the unnecessary warnings, and we're good to go.
```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
```

> Note: learn more about [GOOGLE_API_KEY](https://ai.google.dev/) to run it locally.

Load the necessary API keys, and we will actually going to use the combination of the two keys that we used in the previous lesson.
W
So we can execute thi
### Connect to Weaviate

We will have the embedding API key and the key that we use for our vision model. Now that you have the required API keys, it's time to connect to the Weaviate instance.
This time, what are we going to do is, use this special backup system. That's because we created a data set with 30,000 images pre vectorized.
You can import them, really fast without actually having to wait for that.


```python
import weaviate

client = weaviate.connect_to_embedded(
    version="1.24.4",
    environment_variables={
        "ENABLE_MODULES": "backup-filesystem,multi2vec-palm",
        "BACKUP_FILESYSTEM_PATH": "/home/jovyan/work/backups",
    },
    headers={
        "X-PALM-Api-Key": EMBEDDING_API_KEY,
    }
)

client.is_ready()
```

### Restore 13k+ prevectorized resources

In order to restore those images that we promised, we basically have to run this little command here : `resources-img-and-vid`, where we specify the resources, where we want to go.
But the most important thing is that the collection name ( `Resources` in this case), where we will load the new data set is called resources.

And in order to restore those images that we promised, we basically have to run this little command here, where we specify the resources, where we want to go. But the most important thing is that the collection name, where we will load the new data set is called resources.
So we can execute this.


```python
client.backup.restore(
    backup_id="resources-img-and-vid",
    include_collections="Resources",
    backend="filesystem"
)

# It can take a few seconds for the "Resources" collection to be ready.
# We add 5 seconds of sleep to make sure it is ready for the next cells to use.
import time
time.sleep(5)
```

### Preview data count

Now what we can do is very quickly preview, the number of objects that we have in the collection.

```python
from weaviate.classes.aggregate import GroupByAggregate

resources = client.collections.get("Resources")

response = resources.aggregate.over_all(
    group_by=GroupByAggregate(prop="mediaType")
)

# print rounds names and the count for each
for group in response.groups:
    print(f"{group.grouped_by.value} count: {group.total_count}")
```

```
image count: 13394
video count: 200
```

## Multimodal RAG

### Step 1 – Retrieve content from the database with a query


So we are getting the collections object and running an aggregate function that basically counts all the objects inside and grouping them by media type.
And then now we can go basically printed based on, what we get per group.
And running this, we can see that we have over 13,000 images and then 200 videos. Won't necessarily use the videos in this lesson. but you can try to query them later if you want.
And now we're getting to the fun
part of running the full multimodal RAG in two steps.
First step will be to send a query and retrieve content from the database space in a query. We will do it as a function called retrieve image.
Given a query we want to get an image. In the first part we basically have to grab our resources collection.
Now that we have our resources collection, basically we're calling a new text query. Given a query from the function, we are also providing a filter because in this case we only want to get images, that will pass into the vision model later.
We are only interested in a path to the image, and we'll return just one object. Once we get the result back, we'll grab the first

```python
from IPython.display import Image
from weaviate.classes.query import Filter

def retrieve_image(query):
    resources = client.collections.get("Resources")
# ============
    response = resources.query.near_text(
        query=query,
        filters=Filter.by_property("mediaType").equal("image"), # only return image objects
        return_properties=["path"],
        limit = 1,
    )
# ============
    result = response.objects[0].properties
    return result["path"] # Get the image path
```

### Run image retrieval

AWe are only interested in a path to the image, and we'll return just one object.
Once we get the result back, we'll grab the first object, grab the properties, and we'll return from this function ust the path to the image that we're able to find, with the new text query. So in summary, if we run this given a query, we should get back an image URL. that matched our query.
Now you can test the retrieve image function.
How about trying a query like "fishing with my buddies." and then if you run this, you should get something like this. like, you see in here, we have a man holding a fish, and probably the dog was recognized as the body of the man in the picture. If you run a different query and your query is not exactly represented in a data set, as the query, you may get surprising results.
But don't get discouraged by this.


```python
# Try with different queries to retreive an image
img_path = retrieve_image("fishing with my buddies")
display(Image(img_path))
```

<img src="deeplearningai/building-multimodal-search-and-rag-weaviate/images/4-output-1.png" width="50%"/>

5.05
Feel free to play with different types of queries. Just a little caveat. The kind of things that you will get back
depends on what's already in the data set.

If you're searching for something that's not in there or not exactly represent, you might get some surprising results.

### Step 2 - Generate a description of the image




Now for the generative part, you are going to follow the same steps as you did in the previous lesson. So you need to set up the API key for the generative model.
Also like you did it in the previous lesson, to set up the helper function to convert output to markdown and the call the LLM function which, given an image path and a prompt, can generate a nice description of the image.

```python
import google.generativeai as genai
from google.api_core.client_options import ClientOptions

# Set the Vision model key
genai.configure(
        api_key=GOOGLE_API_KEY,
        transport="rest",
        client_options=ClientOptions(
            api_endpoint=os.getenv("GOOGLE_API_BASE"),
        ),
)
```

Finally, to complete the loop, you're going to call the LLM function. Given the image path from the retrieval segment.
So that was from step one, and the description.
You should be able to execute this, which takes about a few secondsYou should get back a description of the image of a man holding a fish with a dog next to him.
You can see in here the description I got, it talks about a man with a green hat and a khaki vest holding a large fish in his hand, etc., etc., and even talks about the dog standing next to the man.
You probably will get a different description than I did. But that's part of how the lens generate, responses token by token.
Now you can combine all of it together.

So let's create an MM RAG function
where the first step will be to call the retrieve image function.
And then the output will be set inside the source image variable.
And the second step will be
to call LLM with the source image from the previous step and the prompt, that should return the description.
Let's execute this. Finally you can call a full mirror function.
You can search for something like paragliding through the mountains, that should both grab an image.
Also at the end provide a description of the image just like this.

```python
# Helper function
import textwrap
import PIL.Image
from IPython.display import Markdown, Image

def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

def call_LMM(image_path: str, prompt: str) -> str:
    img = PIL.Image.open(image_path)

    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()

    return to_markdown(response.text)  

```




### Run vision request

```python
call_LMM(img_path, "Please describe this image in detail.")
```

And just like this, you are able to combine, two different parts from lesson two and three, the retrieval and the generative part, in order to actually get a multimodal RAG function.
And now the widget instance is closed.
Cool.
So in this lesson, you learned how to combine the retrieval together with generative models.
Even though these two were two completely different models, you were able to actually build something that combines into one big functionality, which gives you a lot of power.
And in the next lesson, you learned how to take this into industry applications and try this on many different real life use cases.
See you there.
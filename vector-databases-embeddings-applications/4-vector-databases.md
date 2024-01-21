---
layout: default
title: 4. Vector Databases Details
nav_order: 4
description: ".."
has_children: true
parent:  Vector Databases
---

## Introduction

In this lesson, we'll introduce Weaviate, an open-source vector database, and discuss how you can use it 
to perform semantic search, and how it supports CRUD operations, which stands for Create, Read, Update, and Delete. You'll also inspect the objects and vectors that are stored in a database. This lesson will provide you with the fundamentals of how to get started with a vector database and even some advanced topics such as performing filtered search.



### Step 1 - Download sample data

In this project, we'll use this sample dataset, which contains a set of Jeopardy questions. So, the idea is that we'll have something like a category, question and an answer, and we'll load it 
into a vector database and perform semantic search queries on it.

```python
import requests
import json

# Download the data
resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

# Parse the JSON and preview it
print(type(data), len(data))

def json_print(data):
    print(json.dumps(data, indent=2))

json_print(data[0])
```

### Step 2 - Create an embedded instance of Weaviate vector database

We will set up an embedded instance of Weaviate, and use OpenAI to generate our vector embeddings and for this we need to load an OpenAI API key. 

```python
import weaviate, os
from weaviate import EmbeddedOptions
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

client = weaviate.Client(
    embedded_options=EmbeddedOptions(),
    additional_headers={
        "X-OpenAI-BaseURL": os.environ['OPENAI_API_BASE'],
        "X-OpenAI-Api-Key": openai.api_key  # Replace this with your actual key
    }
)
print(f"Client created? {client.is_ready()}")
```

    Binary /home/jovyan/.cache/weaviate-embedded did not exist. Downloading binary from https://github.com/weaviate/weaviate/releases/download/v1.22.3/weaviate-v1.22.3-Linux-amd64.tar.gz
    Started /home/jovyan/.cache/weaviate-embedded: process ID 128
    Client created? True

If you're curious what sort of things are available inside an embedded instance, basically, Weaviate offers this modular system which allows you to use something like generative search with OpenAI, or you could run textual vectorization with Cohere, HangingPlace or OpenAI. 

```python
json_print(client.get_meta())
```


```json
{
  "hostname": "http://127.0.0.1:8079",
  "modules": {
    "generative-openai": {
      "documentationHref": "https://platform.openai.com/docs/api-reference/completions",
      "name": "Generative Search - OpenAI"
    },
    "qna-openai": {
      "documentationHref": "https://platform.openai.com/docs/api-reference/completions",
      "name": "OpenAI Question & Answering Module"
    },
    "ref2vec-centroid": {},
    "reranker-cohere": {
      "documentationHref": "https://txt.cohere.com/rerank/",
      "name": "Reranker - Cohere"
    },
    "text2vec-cohere": {
      "documentationHref": "https://docs.cohere.ai/embedding-wiki/",
      "name": "Cohere Module"
    },
    "text2vec-huggingface": {
      "documentationHref": "https://huggingface.co/docs/api-inference/detailed_parameters#feature-extraction-task",
      "name": "Hugging Face Module"
    },
    "text2vec-openai": {
      "documentationHref": "https://platform.openai.com/docs/guides/embeddings/what-are-embeddings",
      "name": "OpenAI Module"
    }
  },
  "version": "1.22.3"
}
```
This feature is power behind it, because it allows you to skip the manual vectorization and let the database take care of it for you. And this 
is what we want to show you in the next few steps. 

## Step 3 - Create Question collection

We need to start with creating a new collection. So, we'll call it a question. And 
this time we'll use Text2Vec OpenAI Vectorizer. And this 
is actually a pretty powerful module, which allows 
you to automatically generate vector embeddings 
as you import the data, and at the time of each query, the vector 
database will grab the necessary input and then 
send it to OpenAI to vectorize. 

```python
# resetting the schema. CAUTION: This will delete your collection 
if client.schema.exists("Question"):
    client.schema.delete_class("Question")
class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai",  # Use OpenAI as the vectorizer
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text",
            "baseURL": os.environ["OPENAI_API_BASE"]
        }
    }
}
client.schema.create_class(class_obj)
```

## Step 4 - Load sample data and generate vector embeddings

Just as a reminder let's print one of the data objects so that we know the data structure for it. And then, we can take that data object and then import it into our questions collection. 

This is what we're gonna do here. We are going to import the data in batches of five and then basically for each object we will say hey we're importing this question, construct our objects with the answer question in category and then pass it into the database with the collection name question.

```python
# reminder for the data structure
json_print(data[0])
```


```python
with client.batch.configure(batch_size=5) as batch:
    for i, d in enumerate(data):  # Batch import data
        
        print(f"importing question: {i+1}")
        
        properties = {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        }
        
        batch.add_data_object(
            data_object=properties,
            class_name="Question"
        )
```


    importing question: 1
    importing question: 2
    importing question: 3
    importing question: 4
    importing question: 5
    importing question: 6
    importing question: 7
    importing question: 8
    importing question: 9
    importing question: 10



You notice here that we are not passing a vector embedding this time because that's basically what the Text2Vec OpenAI module is supposed to do, and it will generate a vector embedding for every object. 


So, if you run this the vectorization is complete. To verify we can run this quick aggregate 
query on the question collection, and we can see that we do have 10 objects inside. 

```python
count = client.query.aggregate("Question").with_meta_count().do()
json_print(count)
```

## Let's Extract the vector that represents each question!

Let us grab one of the objects from our question collection. 
Let's see what category, question and answer it has. But 
more importantly, let's have a look what vector embeddings was generated for that specific object. So, if 
we run this, we can see a whole vector embedding, which is 
pretty long.

```python
# write a query to extract the vector for a question
result = (client.query
          .get("Question", ["category", "question", "answer"])
          .with_additional("vector")
          .with_limit(1)
          .do())

json_print(result)
```

It should be about 1500-dimensional embedding. And now, let's try to run a vector query using SemanticSearch. We'll use with near text operator, and we'll pass in our query as concepts and the query itself is biology so that's what we're looking for. And then, to add some extra info. 

```json
{
  "data": {
    "Get": {
      "Question": [
        {
          "_additional": {
            "vector": [
              -0.003712297,
              0.014701234,
              -0.0030918994,
              -0.009980161,
              ..
              -0.025125256
            ]
          },
          "answer": "species",
          "category": "SCIENCE",
          "question": "2000 news: the Gunnison sage grouse isn't just another northern sage grouse, but a new one of this classification"
        }
      ]
    }
  }
}
```        

## Query time

Let's also display additional property, which is the distance.

What is the distance between the `query`: `biology` and the returned objects?

If we run this query, we should get two objects that match the query for biology. And this is our result.

```python
response = (
    client.query
    .get("Question",["question","answer","category"])
    .with_near_text({"concepts": "biology"})
    .with_additional('distance')
    .with_limit(2)
    .do()
)

json_print(response)
```


```json
{
  "data": {
    "Get": {
      "Question": [
        {
          "_additional": {
            "distance": 0.19693345
          },
          "answer": "DNA",
          "category": "SCIENCE",
          "question": "In 1953 Watson & Crick built a model of the molecular structure of this, the gene-carrying substance"
        },
        {
          "_additional": {
            "distance": 0.20143658
          },
          "answer": "species",
          "category": "SCIENCE",
          "question": "2000 news: the Gunnison sage grouse isn't just another northern sage grouse, but a new one of this classification"
        }
      ]
    }
  }
}
```


 
The other model uses a cosine distance, then smaller numbers indicate better matches. In this case, 0.19 and 0.2, indicates a pretty strong match to our biology query. We 
can also run a query to return all the objects that we have in the database and then kind of like look 
at all the available distances that we get here so you can see that as we go down the distances increase since we use a cosine distance metric that kind of means that the worst matches are at the bottom and the best ones are at the top. 

## We can let the vector database know to remove results after a threshold distance!

Let us run the same query again, we don't always know how many objects are the best matches. I can say hey this my distance is should be at least 0.24 and anything that is above that distance should be rejected and this is a really good method of kind of say like. 

I have certain requirements for the quality of my results and anything that goes beyond that should be ignored. So, like you can see here the final result was cut off at 0.23. 


```python
response = (
    client.query
    .get("Question", ["question", "answer"])
    .with_near_text({"concepts": ["animals"], "distance": 0.24})
    .with_limit(10)
    .with_additional(["distance"])
    .do()
)

json_print(response)
```


```json
{
  "data": {
    "Get": {
      "Question": [
        {
          "_additional": {
            "distance": 0.18963969
          },
          "answer": "Elephant",
          "question": "It's the only living mammal in the order Proboseidea"
        },
        {
          "_additional": {
            "distance": 0.19144487
          },
          "answer": "the nose or snout",
          "question": "The gavial looks very much like a crocodile except for this bodily feature"
        },
        {
          "_additional": {
            "distance": 0.20419747
          },
          "answer": "Antelope",
          "question": "Weighing around a ton, the eland is the largest species of this animal in Africa"
        },
        {
          "_additional": {
            "distance": 0.21438634
          },
          "answer": "species",
          "question": "2000 news: the Gunnison sage grouse isn't just another northern sage grouse, but a new one of this classification"
        },
        {
          "_additional": {
            "distance": 0.23649544
          },
          "answer": "the diamondback rattler",
          "question": "Heaviest of all poisonous snakes is this North American rattlesnake"
        },
        {
          "_additional": {
            "distance": 0.24689794
          },
          "answer": "DNA",
          "question": "In 1953 Watson & Crick built a model of the molecular structure of this, the gene-carrying substance"
        },
        {
          "_additional": {
            "distance": 0.25260252
          },
          "answer": "wire",
          "question": "A metal that is ductile can be pulled into this while cold & under pressure"
        },
        {
          "_additional": {
            "distance": 0.25561416
          },
          "answer": "Liver",
          "question": "This organ removes excess glucose from the blood & stores it as glycogen"
        },
        {
          "_additional": {
            "distance": 0.2580055
          },
          "answer": "Sound barrier",
          "question": "In 70-degree air, a plane traveling at about 1,130 feet per second breaks it"
        },
        {
          "_additional": {
            "distance": 0.26835108
          },
          "answer": "the atmosphere",
          "question": "Changes in the tropospheric layer of this are what gives us weather"
        }
      ]
    }
  }
}
```
## Vector Databases support for CRUD operations

### Create

Sincewe are working with a vector database, that means that we 
can also perform various CRUD operations, like create, read, update 
or delete. 
So, to create a single object, all we need to do 
is call client data object create, and then we can pass 
in the data object inside, and then provide the 
collection name that we'll insert it into. And again, the 
Text2Vec OpenAI module will generate the vector embedding 
for this object. So, let's add the object. Now we 
can print its UUID. Now, let's see a read example to read the object that we 
just created in the previous block, and we are 
going to grab it by this object ID and then 
if we print it this is our object. 

```python
#Create an object
object_uuid = client.data_object.create(
    data_object={
        'question':"Leonardo da Vinci was born in this country.",
        'answer': "Italy",
        'category': "Culture"
    },
    class_name="Question"
 )
```
If you are curious to see what is the vector embedding that was generated 
for it all we have to do is just add this with vector true and then running that will give 
us the object with all the information and its vector embedding. 

```python
print(object_uuid)
```

    d3a021e6-e963-4a26-8466-93833ef471f3
### Read


```python
data_object = client.data_object.get_by_id(object_uuid, class_name="Question")
json_print(data_object)
```

```python
{
  "class": "Question",
  "creationTimeUnix": 1705819795723,
  "id": "d3a021e6-e963-4a26-8466-93833ef471f3",
  "lastUpdateTimeUnix": 1705819795723,
  "properties": {
    "answer": "Italy",
    "category": "Culture",
    "question": "Leonardo da Vinci was born in this country."
  },
  "vectorWeights": null
}
```


```python
data_object = client.data_object.get_by_id(
    object_uuid,
    class_name='Question',
    with_vector=True
)

json_print(data_object)
```

Output json will have `vector` filled with appropriate values
```
{
  "class": "Question",
  "creationTimeUnix": 1705819795723,
  "id": "d3a021e6-e963-4a26-8466-93833ef471f3",
  "lastUpdateTimeUnix": 1705819795723,
  "properties": {
    "answer": "Italy",
    "category": "Culture",
    "question": "Leonardo da Vinci was born in this country."
  },
  "vector": [
    0.022491472,
    -0.013062453,
    -0.0031088893,
    ...
       -0.030705081
  ],
  "vectorWeights": null
}
```

And now, let's grab that object and maybe update it in the next section

### Update

Lets update the object. Previously the answer was set to just Italy but let's set it to Florence in Italy. So, if we run this the object will get updated and then we could grab it again by its ID and we can 
see the answer indeed got updated.

```python
client.data_object.update(
    uuid=object_uuid,
    class_name="Question",
    data_object={
        'answer':"Florence, Italy"
    })
```


```python
data_object = client.data_object.get_by_id(
    object_uuid,
    class_name='Question',
)

json_print(data_object)
```

Output shown below confirms the same.


```json
{
  "class": "Question",
  "creationTimeUnix": 1705831632199,
  "id": "e4643167-abe9-442a-b6f2-52fca545c030",
  "lastUpdateTimeUnix": 1705831632646,
  "properties": {
    "answer": "Florence, Italy",
    "category": "Culture",
    "question": "Leonardo da Vinci was born in this country."
  },
  "vectorWeights": null
}
```
Lets look at the delete operation

### Delete

And finally, we go to the stage where we want to delete our sample object so in this case 
what we want to do first is check how many objects we have before. Then, we can delete the object based 
on its ID and then finally we print the aggregate just to verify that we have one object left. 




```python
json_print(client.query.aggregate("Question").with_meta_count().do())
```


```json
{
  "data": {
    "Aggregate": {
      "Question": [
        {
          "meta": {
            "count": 11
          }
        }
      ]
    }
  }
}
```
`

```python
client.data_object.delete(uuid=object_uuid, class_name="Question")
```
Before we had 11, now back to 10. And this concludes this lesson. 

```json
{
  "data": {
    "Aggregate": {
      "Question": [
        {
          "meta": {
            "count": 10
          }
        }
      ]
    }
  }
}
```


```python
json_print(client.query.aggregate("Question").with_meta_count().do())
```

In here, you learn how to use a vector database to automatically vectorize all your data with OpenAI and also use the same mechanism to vectorize your queries and perform various searches including vector search 
and filtered search. And we also went over how to use various CRUD operations, so that you could 
maintain your data as you go throughout the life cycle of your applications. And in the next lesson, we'll 
introduce the concept of sparse and dense vector, but we'll also look at the hybrid search, which allows us to combine both of those methods to provide better results. 
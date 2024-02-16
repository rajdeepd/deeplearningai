---
layout: default
title: 3. Approximate Nearest Neighbors
nav_order: 3
description: ".."
has_children: true
parent:  Vector Databases and Embeddings - Weaviate
---

In this lesson you will get a theoretical and 
practical understanding of how approximate nearest neighbors' algorithms, also known as ANN, trade a little bit of accuracy or recall for a lot of performance. Specifically, you'll explore the hierarchical navigable small words ANN algorithm to understand how this powers the world's most powerful vector databases. We'll also demonstrate the scalability of HNSW and how it solves the runtime complexity issues of brute-force KNN. Let's get coding! 


## Navigable Small World

So, if you look at this example with 20 vectors, searching for the nearest one is not necessarily a big issue.

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_12.11.52 PM.png" width="80%" />

However, as soon as we get to thousands or millions of vectors, this is becoming a way bigger problem and definitely a big no-go. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_12.12.05 PM.png" width="80%" />


There are many algorithms that allow us to actually find the nearest vectors in a more efficient way. One of the algorithms that solve this problem is HNSW, which is based on small world phenomenon in human social networks. And the whole idea is that on average we are all connected from each other by six degrees of separation. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_12.12.19 PM.png" width="80%" />

So, it's very possible that you have a friend, who has a friend, who has a friend, who has a friend, who is connected to Andrew. And the whole idea is that you probably know somebody that knows everyone. And that person probably also knows somebody that's really well connected and through that you could actually through within six steps find the connection to the person you're looking for. And then, automatically we could apply the same concepts to vector embeddings if we build those kinds of connections. 

Lets have a look at navigable small world which is an algorithm that allows us to construct these connections between different nodes. So, for example, in here we have eight randomly generated vectors and let's have a look how we can connect all of them together to their nearest neighbors. 



If we start with vector 0, there's nothing to connect just yet. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.47.48 PM.png" width="80%" />

Then, we add vector 1 and the only possible connection is to a vector 0. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.48.32 PM.png" width="80%" />

And now ,we can add vector 2 and we can build two connections between 1 and 0. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.48.37 PM.png" width="80%" />

Following the same method, we can go to vector 3 and the two nearest connections will be to vectors 2 and 0. 


<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.48.46 PM.png" width="80%" />

Cool! Next, let's go to number four and the two nearest connections probably two and zero. Great! 
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.48.54 PM.png" width="80%" />


Now, let's go to five is well connected to two and zero. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.49.00 PM.png" width="80%" />

Now, we can go to six connects to two and four. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.49.05 PM.png" width="80%" />

And then, finally, seven connects to vectors five and three and just like that we build a Navigo small world, and just as a note for you in this example 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.49.09 PM.png" width="80%" />

 
Now, let's see how the search of the Navigable small world can be performed. 

## Search 1 : Nearest Neighbour

In this example we have a query vector which is on the left and we could already 
kind of guess that number six will be the nearest vector and usually how you work with NSW, and we start with a random entry node and we try to move across towards the nearest neighbors. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2023-12-31_at_4.49.09 PM.png" width="80%" />

So, starting from node `7`, which is connected to nodes `3` and `5`, we can see that `5` is closer than node `7`. So, we can move there. 


<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-13_at_6.08.29 PM.png" width="80%" />


From node `5` we can see that we are also connected to `0` and `2` and `2` is definitely way closer. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-13_at_6.08.42 PM.png"  width="80%" />


Great! And then, from 2 we have multiple candidates with the best option being node `6` and then from `6` there are no longer better candidates and therefore our query concludes here and that's how we found our best match. Which happens to be our nearest neighbor vector that we're looking 
for. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-13_at_6.08.49 PM.png"  width="80%" />

So, the search with NSW doesn't always result in finding the best match. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-13_at_6.09.45 PM.png"  width="80%" />

The search with NSW doesn't always result 
in finding the best match. 

## NSW Search 2

Let's have a look at this from this example. If we start from node 0, in this case our potential candidates are here and the best possible version from this step is vector number 1. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.10.14 PM.png"  width="80%" />

And then, from vector number 1 there are no longer any better candidates and therefore our search concludes here. In this case we 
didn't find the best possible result, however we found approximately nearest 
neighbor, which is still a pretty 
good result, but it's not necessarily the perfect result. 



<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.11.18 PM.png"  width="80%" />
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.11.24 PM.png"  width="80%" />
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.11.38 PM.png"  width="80%" />
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.18.42 PM.png"  width="80%" />



### Hierarchical Navigable Small World

Now, it's time to learn hierarchical navigable small world, which puts several layers of navigable small worlds on top of each other. And the way you can imagine it's a 
bit like if you were to travel to some place 
in the world. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.10.59 PM.png"  width="80%" />

First, you probably you would take a 
plane to the nearest airport to where you're trying to 
get to. Then, maybe you would go to and catch a 
train that would take you to the town where you 
wanna get to. And then finally, once you are 
at the bottom layer, you would walk or maybe take a taxi to the destination 
that you're going to. And we're pointing out that the 
construction of each layer of HNSW is very 
similar to how it's done in NSW, so we 
won't be diving into that. And the way the querying works with 
HNSW is that, again, we're starting with a random node, and we 
only can choose from those available at the highest level, 
and then we move to the nearest one 
within that layer. Once we are there, we can find the 
best possible match at that and the next layer, and eventually, once 
we are at the bottom layer, we 
can go to the object which is the closest to the query vector, which will help us to the last mile of the search. 

And the way nodes are assigned to each different layer. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-14_at_4.19.07 PM.png"  width="80%" />


It's done by randomly generating a number which assigns 
that node to that layer and all the ones below and it's worth pointing that the chances of landing in a higher layer is logarithmically lower versus the one below. And as a result we'll have a lot fewer nodes in the top layer versus the one towards the bottom. 
So, for example if the random number is 0 then that node would only exist on the bottom layer, layer 0. If the random number is 2, then that node will exist on layer 0, 1 and 2, etc. So, here are some characteristics of HNSW. 

There is a lot lower likelihood for a node to exist in higher levels and query time 
increases logarithmically, which means that as the number of data points increases, the number of comparisons to perform vector search only goes up logarithmically. In computer science, this 
is known as **O(logn)** runtime complexity, which 
visualizes very nicely that as the number of data points grows, the speed doesn't necessarily 
suffer that much over time. And as you can see in this graph, if we go from half a million vectors to a million vectors, the increase of runtime is minimal. 

Lets see how this all works in code. 

So, in this 
notebook we'll start with 40 vectors with two dimensions.
for each and we will set the nearest 
neighbor connections to 2. And we can construct 
those randomly. So, now let's add a query vector which is located at 0.5, 
0.5. So, in here, what we do is create a list of nodes with the query vector, 
which is that one single point. 

```python
from random import random, randint
from math import floor, log
import networkx as nx
import numpy as np
import matplotlib as mtplt
from matplotlib import pyplot as plt
from utils import *

vec_num = 40 # Number of vectors (nodes)
dim = 2 ## Dimention. Set to be 2. All the graph plots are for dim 2. If changed, then plots should be commented. 
m_nearest_neighbor = 2 # M Nearest Neigbor used in construction of the Navigable Small World (NSW)

vec_pos = np.random.uniform(size=(vec_num, dim))
```
Then, we use the Network X library 
for illustration purposes, which we'll use later on in the 
next block again. Finally, we print our nodes, and this part 
creates the position of the query for plotting in the 
next box. 

```python
## Query
query_vec = [0.5, 0.5]

nodes = []
nodes.append(("Q",{"pos": query_vec}))

G_query = nx.Graph()
G_query.add_nodes_from(nodes)

print("nodes = ", nodes, flush=True)

pos_query=nx.get_node_attributes(G_query,'pos')
```

Output from the print statement above

```console
nodes =  [('Q', {'pos': [0.5, 0.5]})]
```
Next, we will run a brute-force algorithm to find the best possible 
vector embedding for our search and then plot it 
on the graph. So, in this case we can see our query over here and the best 
possible match right next to it. 


```python
G_lin, G_best) = nearest_neigbor(vec_pos,query_vec)

pos_lin=nx.get_node_attributes(G_lin,'pos')
pos_best=nx.get_node_attributes(G_best,'pos')

fig, axs = plt.subplots()

nx.draw(G_lin, pos_lin, with_labels=True, node_size=150, node_color=[[0.8,0.8,1]], width=0.0, font_size=7, ax = axs)
nx.draw(G_query, pos_query, with_labels=True, node_size=200, node_color=[[0.5,0,0]], font_color='white', width=0.5, font_size=7, font_weight='bold', ax = axs)
nx.draw(G_best, pos_best, with_labels=True, node_size=200, node_color=[[0.85,0.7,0.2]], width=0.5, font_size=7, font_weight='bold', ax = axs)
```

## Brute Force

Method nearest_neighbor is implemented in `utils.py` file. The details of the implementation are explained in the section <a href="/deeplearningai/vector-databases-embeddings-applications/3.1-utils-nearest-neighbour.html">3.1 Nearest Neightbour Listing</a>

We will run a brute-force algorithm to find the best possible 
vector embedding for our search and then plot it 
on the graph.
```python
(G_lin, G_best) = nearest_neigbor(vec_pos,query_vec)

pos_lin=nx.get_node_attributes(G_lin,'pos')
pos_best=nx.get_node_attributes(G_best,'pos')

fig, axs = plt.subplots()

nx.draw(G_lin, pos_lin, with_labels=True, node_size=150, node_color=[[0.8,0.8,1]], width=0.0, font_size=7, ax = axs)
nx.draw(G_query, pos_query, with_labels=True, node_size=200, node_color=[[0.5,0,0]], font_color='white', width=0.5, font_size=7, font_weight='bold', ax = axs)
nx.draw(G_best, pos_best, with_labels=True, node_size=200, node_color=[[0.85,0.7,0.2]], width=0.5, font_size=7, font_weight='bold', ax = axs)
```


    
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/output_7_0.png"  width="80%" />
    
In this case we can see our query in the figure above and the best 
possible match right next to it.

### HNSW Construction

In this step, we construct our HNSW layers. And then, in loop we go one by one. And then, print the layer ID. And then, we show all the nodes and the connections at each layer. We will be using the method `construct_HNSW(vec_pos,m_nearest_neighbor)`. Please refer to link for more details.

<a href="/deeplearningai/vector-databases-embeddings-applications/3-utils-construct-HNSW.html">construct_HNSW</a>

```python
GraphArray = construct_HNSW(vec_pos,m_nearest_neighbor)

for layer_i in range(len(GraphArray)-1,-1,-1):
    fig, axs = plt.subplots()

    print("layer_i = ", layer_i)
        
    if layer_i>0:
        pos_layer_0 = nx.get_node_attributes(GraphArray[0],'pos')
        nx.draw(GraphArray[0], pos_layer_0, with_labels=True, node_size=120, node_color=[[0.9,0.9,1]], width=0.0, font_size=6, font_color=(0.65,0.65,0.65), ax = axs)

    pos_layer_i = nx.get_node_attributes(GraphArray[layer_i],'pos')
    nx.draw(GraphArray[layer_i], pos_layer_i, with_labels=True, node_size=150, node_color=[[0.7,0.7,1]], width=0.5, font_size=7, ax = axs)
    nx.draw(G_query, pos_query, with_labels=True, node_size=200, node_color=[[0.8,0,0]], width=0.5, font_size=7, font_weight='bold', ax = axs)
    nx.draw(G_best, pos_best, with_labels=True, node_size=200, node_color=[[0.85,0.7,0.2]], width=0.5, font_size=7, font_weight='bold', ax = axs)
    plt.show()
```

    layer_i =  3



At the top layer we can see that we have nodes 20, 34, 28 and 
39 already connected to each other.

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-15_at_12.36.25 PM.png"  width="80%" />

    
    layer_i =  2


When we get to layer 2, we get more nodes with more connections. Then, at layer 1 we have pretty much most of the nodes already reconnected. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-15_at_12.36.45 PM.png"  width="80%" />
  

    layer_i =  1


First thing that we will get is search path graph array, which contains the graph with the travel paths across all layers. 
    
    layer_i =  0

Finally, at layer 0 we have all the nodes present and connected 
to their nearest neighbors.

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-15_at_12.36.55 PM.png"  width="80%" />


Now, that we have the whole network built up across all the layers, we can run an actual HNSW search query. 

<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/Screenshot_2024-01-15_at_12.37.09 PM.png"  width="80%" /> 
    


### HNSW Search

Next, we have entry graph array, which gives us the entry point to the graph. And then, we run across all the layers in a loop to plot all the results layer by layer for the visual purposes.

```python
(SearchPathGraphArray, EntryGraphArray) = search_HNSW(GraphArray,G_query)

for layer_i in range(len(GraphArray)-1,-1,-1):
    fig, axs = plt.subplots()

    print("layer_i = ", layer_i)
    G_path_layer = SearchPathGraphArray[layer_i]
    pos_path = nx.get_node_attributes(G_path_layer,'pos')
    G_entry = EntryGraphArray[layer_i]
    pos_entry = nx.get_node_attributes(G_entry,'pos')

    if layer_i>0:
            pos_layer_0 = nx.get_node_attributes(GraphArray[0],'pos')
            nx.draw(GraphArray[0], pos_layer_0, with_labels=True, node_size=120, node_color=[[0.9,0.9,1]], width=0.0, font_size=6, font_color=(0.65,0.65,0.65), ax = axs)

    pos_layer_i = nx.get_node_attributes(GraphArray[layer_i],'pos')
    nx.draw(GraphArray[layer_i], pos_layer_i, with_labels=True, node_size=100, node_color=[[0.7,0.7,1]], width=0.5, font_size=6, ax = axs)
    nx.draw(G_path_layer, pos_path, with_labels=True, node_size=110, node_color=[[0.8,1,0.8]], width=0.5, font_size=6, ax = axs)
    nx.draw(G_query, pos_query, with_labels=True, node_size=80, node_color=[[0.8,0,0]], width=0.5, font_size=7, ax = axs)
    nx.draw(G_best, pos_best, with_labels=True, node_size=70, node_color=[[0.85,0.7,0.2]], width=0.5, font_size=7, ax = axs)
    nx.draw(G_entry, pos_entry, with_labels=True, node_size=80, node_color=[[0.1,0.9,0.1]], width=0.5, font_size=7, ax = axs)
    plt.show()
```

    layer_i =  3



    
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/output_11_1.png"  width="80%" />
    
 So, we start at the top layer with the node 39, which from 39, 
we can move to node 20, which gets us closer to the query. 
And once we are on 20, there are no longer 
any nodes that are neighbors to 20 that 
will get us closer to the query. Then, we can move to the layer 2. 


    layer_i =  2


<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/output_11_3.png"  width="80%" />

From layer 2, node 20 can take us to node 16, 
but node 16 doesn't have any other candidates that could get us 
closer to the query, which then takes us to the next layer.


    layer_i =  1

    
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/output_11_5.png"  width="80%" />
    
From layer 1, we can move from node 16 to 2. And 
from 2, there are no longer any other candidates that would get us 
closer to the query. 

    layer_i =  0

So, we can finally move to 
the bottom layer. And then, from node 2, we can actually go all the 
way to node 25, which as it happens, is 
the perfect match to our query. 

    
<img src="/deeplearningai/vector-databases-embeddings-applications/l3_images/output_11_7.png"  width="80%" />
    
And just like that, we 
made HNSW query across all the layers and 
return the newest possible match

## Pure Vector Search - with a vector database

In this section lets have a look how we can perform vector search with a vector database, which pretty much encompasses all that functionality inside. So, for this we'll use Weaviate, an open-source vector database. And one of the modes that Weaviate offers is an embedded option, which allows us to run the vector database inside the notebook. 



```python
import weaviate, json
from weaviate import EmbeddedOptions

client = weaviate.Client(
    embedded_options=EmbeddedOptions(),
)

client.is_ready()
```

    Binary /home/jovyan/.cache/weaviate-embedded did not exist. Downloading binary from https://github.com/weaviate/weaviate/releases/download/v1.22.3/weaviate-v1.22.3-Linux-amd64.tar.gz
    Started /home/jovyan/.cache/weaviate-embedded: process ID 178


    {"action":"startup","default_vectorizer_module":"none","level":"info","msg":"the default vectorizer modules is set to \"none\", as a result all new schema classes without an explicit vectorizer setting, will use this vectorizer","time":"2024-01-14T15:18:07Z"}
    {"action":"startup","auto_schema_enabled":true,"level":"info","msg":"auto schema enabled setting is set to \"true\"","time":"2024-01-14T15:18:07Z"}
    {"level":"warning","msg":"Multiple vector spaces are present, GraphQL Explore and REST API list objects endpoint module include params has been disabled as a result.","time":"2024-01-14T15:18:07Z"}
    {"action":"grpc_startup","level":"info","msg":"grpc server listening at [::]:50060","time":"2024-01-14T15:18:07Z"}
    {"action":"restapi_management","level":"info","msg":"Serving weaviate at http://127.0.0.1:8079","time":"2024-01-14T15:18:07Z"}





    True


As a first step, we need to create our data schema, or what I like to call it, data collection. In this case, what we'll do is call it myCollection or yourCollection with the vectorizer set to null, which basically means that we just want to use pure vector search. And the distance metric they want to use is set 
to cosine. 

```python
# resetting the schema. CAUTION: This will delete your collection 
# if client.schema.exists("MyCollection"):
#     client.schema.delete_class("MyCollection")

schema = {
    "class": "MyCollection",
    "vectorizer": "none",
    "vectorIndexConfig": {
        "distance": "cosine" # let's use cosine distance
    },
}

client.schema.create_class(schema)

print("Successfully created the schema.")
```
And then, once we run that, we'll get a new empty collection in our database. 

### Import the Data

And just in case, if you want to rerun 
the same example again later on, and you need to recreate their 
collection, I'm going to leave you a piece of code that allows you to 
just delete the collection if it exists and 
then recreate it, but don't necessarily feel like you have 
to rerun it over and over again. And now, it's time to 
import some data into the vector database. So, let's say we 
have these five random objects with a title and 
a full value and a vector embedding. And here's the code that will 
help us actually get our data object and 
load them into the database. We kind of set it 
up, this is basically a best practice although we only 
work with five objects. Usually if you're loading like 
tens of thousands or millions it's actually 
good to use a batch loading process. And then, 
what we do is actually run in a loop through all 
our data items and construct like let's call 
it a properties object. And then, we run this client batch add 
data object which insert the object into our database. So, what 
we need to do is add the collection name, so we call 
it myCollection. Then, the data object is properties that we had, 
and the vector actually exists inside itemVector, and 
that's how we actually pass the vectors 
into the database. 


```python
data = [
   {
      "title": "First Object",
      "foo": 99, 
      "vector": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
   },
   {
      "title": "Second Object",
      "foo": 77, 
      "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
   },
   {
      "title": "Third Object",
      "foo": 55, 
      "vector": [0.3, 0.1, -0.1, -0.3, -0.5, -0.7]
   },
   {
      "title": "Fourth Object",
      "foo": 33, 
      "vector": [0.4, 0.41, 0.42, 0.43, 0.44, 0.45]
   },
   {
      "title": "Fifth Object",
      "foo": 11,
      "vector": [0.5, 0.5, 0, 0, 0, 0]
   },
]
```


```python
client.batch.configure(batch_size=10)  # Configure batch

# Batch import all objects
# yes batch is an overkill for 5 objects, but it is recommended for large volumes of data
with client.batch as batch:
  for item in data:

      properties = {
         "title": item["title"],
         "foo": item["foo"],
      }

      # the call that performs data insert
      client.batch.add_data_object(
         class_name="MyCollection",
         data_object=properties,
         vector=item["vector"] # your vector embeddings go here
      )

```
And now, let's check how many objects we 
have inside our database. So, we can run this query on 
our collection and then we could just ask for the 
count of the objects inside and if you run. That 
we can see that our collection contains five objects.

```python
# Check number of objects
response = (
    client.query
    .aggregate("MyCollection")
    .with_meta_count()
    .do()
)

print(response)
```

    {'data': {'Aggregate': {'MyCollection': [{'meta': {'count': 5}}]}}}


### Query Weaviate: Vector Search (vector embeddings)

Let us have some fun with actually querying the database.

#### Basic Querying

The query 
is as follows. So, what we say like hey. I want 
to search to myCollection and I want to 
get the title back and I want to run it with this vector. So, this is 
like just a random vector across six dimensions 
just to match our original data, and by 
saying this we are telling Weaviate to only 
get two best matches. 

```python
response = (
    client.query
    .get("MyCollection", ["title"])
    .with_near_vector({
        "vector": [-0.012, 0.021, -0.23, -0.42, 0.5, 0.5]
    })
    .with_limit(2) # limit the output to only 2
    .do()
)

result = response["data"]["Get"]["MyCollection"]
print(json.dumps(result, indent=2))
```

    [
      {
        "title": "Second Object"
      },
      {
        "title": "Fourth Object"
      }
    ]

And if I run this it kind of tells us 
that the second object, the fourth object matches our results.

#### Vector Embeddings of Matched Objects

If you want to see the vector embeddings for all the matched objects 
we can copy the code and add the folliwing line

```python
.with_additional(["distance", "vector, id"])

```

This additional method basically tells us to get also the distance the vector and the ID for our data and now we can see that the first object 
the distance was calculated as 0.65 and this is the vector that got matched and the same thing for this second matched vector

```python
response = (
    client.query
    .get("MyCollection", ["title"])
    .with_near_vector({
        "vector": [-0.012, 0.021, -0.23, -0.42, 0.5, 0.5]
    })
    .with_limit(2) # limit the output to only 2
    .with_additional(["distance", "vector, id"])
    .do()
)

result = response["data"]["Get"]["MyCollection"]
print(json.dumps(result, indent=2))
```

    [
      {
        "_additional": {
          "distance": 0.6506307,
          "id": "c7f64e67-b4be-4491-b402-001e15616663",
          "vector": [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7
          ]
        },
        "title": "Second Object"
      },
      {
        "_additional": {
          "distance": 0.8072029,
          "id": "c7d69e17-0af0-4802-8eb7-e3d7355eb07e",
          "vector": [
            0.4,
            0.41,
            0.42,
            0.43,
            0.44,
            0.45
          ]
        },
        "title": "Fourth Object"
      }
    ]


### Vector Search with filters

Since we are working with a vector database, we can do all the additional things like filtered on specific properties. 
In this case what we can do is add an extra bit of code which will 
tell the database to search on foo for values that are greater than 44 and only search on the pre-filtered objects like this. And you can see that the only object that got matched are those where the foo value indeed is greater than 44. 


```python
response = (
    client.query
    .get("MyCollection", ["title", "foo"])
    .with_near_vector({
        "vector": [-0.012, 0.021, -0.23, -0.42, 0.5, 0.5]
    })
    .with_additional(["distance, id"]) # output the distance of the query vector to the objects in the database
    .with_where({
        "path": ["foo"],
        "operator": "GreaterThan",
        "valueNumber": 44
    })
    .with_limit(2) # limit the output to only 2
    .do()
)

result = response["data"]["Get"]["MyCollection"]
print(json.dumps(result, indent=2))
```

    [
      {
        "_additional": {
          "distance": 0.6506307,
          "id": "c7f64e67-b4be-4491-b402-001e15616663"
        },
        "foo": 77,
        "title": "Second Object"
      },
      {
        "_additional": {
          "distance": 0.8284496,
          "id": "c3536c09-9aec-455e-b52f-24b2640c0bdc"
        },
        "foo": 99,
        "title": "First Object"
      }
    ]


### nearObject Example

One other thing that we can do here with a vector search we can look for other similar objects based on a provided ID of an object. So, in here, we're just grabbing the first result from the previous query and we're looking for three objects that match this object. And in this case, you of course find itself along with the fourth and the first object too.

```python
response = (
    client.query
    .get("MyCollection", ["title"])
    .with_near_object({ # the id of the the search object
        "id": result[0]['_additional']['id']
    })
    .with_limit(3)
    .with_additional(["distance"])
    .do()
)

result = response["data"]["Get"]["MyCollection"]
print(json.dumps(result, indent=2))
```

    [
      {
        "_additional": {
          "distance": 0
        },
        "title": "Second Object"
      },
      {
        "_additional": {
          "distance": 0.051573694
        },
        "title": "Fourth Object"
      },
      {
        "_additional": {
          "distance": 0.06506646
        },
        "title": "First Object"
      }
    ]


In this lesson you learn how HNSW works, how you can construct 
HNSW layers and search across all of them, but also you 
learn how to use similar algorithms in a production ready database. 
And in the next lesson, you will learn how to use 
the vector database with machine learning 
models like OpenAI, how to vectorize the data 
and how to vectorize the queries, but also you will dive 
into CRUD operations for creating, reading, updating, 
deleting objects. 

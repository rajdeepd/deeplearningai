---
layout: default
title: 2. Multimodal Search
nav_order: 3
description: "Multimodal Search"
has_children: false
parent:  Building Multimodal Search and RAG - Weaviate
---

In this lesson, you learn how a concept is understood across multiple modalities and then implement a multimodal retrieval using Weaviate, an open source vector database. You build a text to any search as well as any to any search.

Even though you can't hear this video, you can probably hear the lion roaring in your mind's ear.


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_8.53.49_PM.png"  width="80%" /> 




This is because humans are very good at inferring information across modalities.
So good, in fact, that we can even do it in the absence of the other modality. Here, being the missing sound.
In the case of this video of a train passing by, you can probably imagine the choo choo sound or even feel the wind or the ground shaking.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_8.53.56_PM.png"  width="80%" /> 

This is because we are very good at understanding information from all of our senses.

What did we see, hear, feel or smell? We can understand the very same data point using multiple senses from multiple modalities. Same goes when you hear the sound like this one.

You know, straight away that this is a cash register. This is because your multimodal reasoning works in all directions.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_8.54.13_PM.png"  width="80%" /> 


The way a machine can gain a similar understanding of multimodal data is by creating a shared multimodal vector space where similar data points, regardless of their modalities, are placed close to each other, And with the unified multimodal embedding space, you can perform
text to any modality search.

For example, you can search for "lions roam the savannas" and you response get back multimodal data that represents the most similar content, whether this is text, images, audio, or even videos.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_8.54.29_PM.png"  width="80%" /> 


You can even perform any to any search. Your query can be in any modality and your retrieved objects in all available modalities as well.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_10.33.44_PM.png"  width="80%" /> 


For example, you could use an image of a lion or a video of a lion to match all those matching objects.
Now let's go over step by step how multimodal search works.
First, if we take a king of the jungle, run it in a MM model, we'll get a vector embedding back.
We could also take a video, pass it to MM the model, and we get another vector embedding path.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_10.34.02_PM.png"  width="80%" /> 

And you can already see that this vector embeddings are pretty similar to each other.
And as we go, we could be loading millions of objects, if not billions, and eventually create a whole vector embedding space of all our vectors.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_10.34.14_PM.png"  width="80%" /> 

And for example, we could take a picture of a lion and pass it through an MM model, which will give us a new vector embedding.
We can then use that vector embedding to point into our vector space, would it show us where all the similar objects are and response get another object like this bunch of lions running.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-20_at_10.34.37_PM.png"  width="80%" /> 


Let's now see all of this in practice.
In this lab, you add multimodal data to a vector database, then perform any to any search.
All right.
Let's code.

<p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

* In this classroom, the libraries have been already installed for you.
* If you would like to run this code on your own machine, you need to install the following:
```
    !pip install -U weaviate-client
```


```python
import warnings
warnings.filterwarnings('ignore')
```

## Setup

So here again, let's have the function to ignore all the unnecessary warnings.
So now let's load the necessary API keys
o run our large multimodal model embeddings ike this.
And now let's connect to Weaviate.
And for this we are going to use the embedded version of Weaviate, which allows us to run them the database in memory.
And we need two things.
Ne need our multi to like module that does
the multi-modal victimization and also helps us with search.
And then in here we need to pass in the necessary key for that model to work.
So let's run the connection,
and we are connected.
And don't worry about these messages that you see here.
Those are just information messages to us.
We are connecting to the database
and about everything that is happening underneath.
And now let's create the collection that we'll use
for storing our vector embeddings and all our images, etc..
And we'll it animals.
And what do we need to add next is a vector razor.
So this is a multiple vector razor, so that can work with both text
and all the different modalities and we are telling Weaviate that for images,
ve will use the image property and for vectorizing video
fields will use video property. And then finally,
we need to add additional information like where the project is.
And the most important one is what's the model that we are using.
So for this lesson we'll be using the multimodal embedding 001
and we want to get all our embeddings of 1400 dimensions.
And then before we run it, I'm going to add
this little conditional function so that if you ever need to rerun it,
basically we will check if the collection animals exists
so we can delete it and then recreate the whole thing.
The only thing is that be careful
because if you delete a collection, you lose all the data in that collection.
But let's run it now.
### Load environment variables and API keys

So here again, let's have the function to ignore all the unnecessary warnings.
So now let's load the necessary API keys
o run our large multimodal model embeddings ike this.

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
```

## Connect to Weaviate

Let's connect to Weaviate, we are going to use the embedded version of Weaviate, which allows us to run them the database in memory.
We need two things.
 need our multi to like module that does
the multi-modal victimization and also helps us with search.
And then in here we need to pass in the necessary key for that model to work.
So let's run the connection,



```python
import weaviate, os

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

Don't worry about these messages that you see here. Those are just information messages to us. We are connecting to the database and everything that is happening underneath.

## Create the Collection

Let us create the collection that we'll use for storing our vector embeddings and all our images, etc and we'll it animals.

What do we need to add next is a vectorizer.
This is a multiple vectorizer, so that can work with both text and all the different modalities and we are telling Weaviate that for images, we will use the image property and for vectorizing video fields will use video property. And then finally, we need to add additional information like where the project is. And the most important one is what's the model that we are using. So for this lesson we'll be using the `multimodalembedding001` and we want to get all our embeddings of 1400 dimensions.
Before we run it, I'm going to add this little conditional function so that if you ever need to rerun it, basically we will check if the collection animals exists so we can delete it and then recreate the whole thing.
The only thing is that be careful because if you delete a collection, you lose all the data in that collection. But let's run it now.

```python
from weaviate.classes.config import Configure

# Just checking if you ever need to re run it
if(client.collections.exists("Animals")):
    client.collections.delete("Animals")
    
client.collections.create(
    name="Animals",
    vectorizer_config=Configure.Vectorizer.multi2vec_palm(
        image_fields=["image"],
        video_fields=["video"],
        project_id="semi-random-dev",
        location="us-central1",
        model_id="multimodalembedding@001",
        dimensions=1408,        
    )
)
```


Now we have an empty collection. Let's add the helper function which we've given path will give us a basically for representation of the file there. 

## Helper functions

We need this function because that's how we pass any image or video file into an embedding for vectorization.
Now that we have an empty collection and a function to convert images to base64, 

```python
import base64

# Helper function to convert a file to base64 representation
def toBase64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')
```

## Insert Images into Weaviate

We can go and start inserting the images. We need to first grab our animals collection and then these animals.

```python
animals = client.collections.get("Animals")

source = os.listdir("./source/animal_image/")

with animals.batch.rate_limit(requests_per_minute=100) as batch:
    for name in source:
        print(f"Adding {name}")
        
        path = "./source/image/" + name
    
        batch.add_object({
            "name": name,            # name of the file
            "path": path,            # path to the file to display result
            "image": toBase64(path), # this gets vectorized - "image" was configured in vectorizer_config as the property holding images
            "mediaType": "image",    # a label telling us how to display the resource 
        })
```


```python
# Check for failed objects
if len(animals.batch.failed_objects) > 0:
    print(f"Failed to import {len(animals.batch.failed_objects)} objects")
    for failed in animals.batch.failed_objects:
        print(f"e.g. Failed to import object with error: {failed.message}")
else:
    print("No errors")
```

## Insert Video Files into Weaviate

> Note: the input video must be at least 4 seconds long.


```python
animals = client.collections.get("Animals")

source = os.listdir("./source/video/")

for name in source:
    print(f"Adding {name}")
    path = "./source/video/" + name    

    # insert videos one by one
    animals.data.insert({
        "name": name,
        "path": path,
        "video": toBase64(path),
        "mediaType": "video"
    })
```


```python
# Check for failed objects
if len(animals.batch.failed_objects) > 0:
    print(f"Failed to import {len(animals.batch.failed_objects)} objects")
    for failed in animals.batch.failed_objects:
        print(f"e.g. Failed to import object with error: {failed.message}")
else:
    print("No errors")
```
Let's check if there are any errors. No errors.


## Check count

Just to summarize, we should be able to see how many objects we have and of what type. We can see that we have nine images and six videos.
that's exactly what I expected.

> Total count should be 15 (9x image + 6x video)


```python
agg = animals.aggregate.over_all(
    group_by="mediaType"
)

for group in agg.groups:
    print(group)
```

<p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Utils File and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>



## Build MultiModal Search

### Helper Functions

Now that we have all the data inside the database, we can start looking at search.
But before we get to that, we need a few helper function.
So this helper function, allows us to actually print our results in a nice matter, especially this one display media.
So given an object, depending on the label, whatever we labeled it as an image one or a video one will display it in an image or a video.
So let's run this.
And the other set of helper functions are those that allow us to convert an image from url or again, the same one for converting to base64.
And now this is the moment you are waiting for and where the fun begins a cleric.


```python
# Helper functions to display results
import json
from IPython.display import Image, Video

def json_print(data):
    print(json.dumps(data, indent=2))

def display_media(item):
    path = item["path"]

    if(item["mediaType"] == "image"):
        display(Image(path, width=300))

    elif(item["mediaType"] == "video"):
        display(Video(path, width=300))
```


```python
import base64, requests

# Helper function – get base64 representation from an online image
def url_to_base64(url):
    image_response = requests.get(url)
    content = image_response.content
    return base64.b64encode(content).decode('utf-8')

# Helper function - get base64 representation from a local file
def file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')
```

## Text to Media Search

> Where the fun begins!


```python
animals = client.collections.get("Animals")

response = animals.query.near_text(
    query="dog playing with stick",
    return_properties=['name','path','mediaType'],
    limit=3
)
```
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-23_at_7.41.09_PM.png"  width="80%" /> 


```python
for obj in response.objects:
    json_print(obj.properties)
    display_media(obj.properties)
```

> Note: Please be aware that the output from the previous cell may differ from what is shown in the video. This variation is normal and should not cause concern.

If you grab our animals collection, what we can do is running query of type near text, which basically means that you can use text to search to our multimodal collection.
We are looking for dog playing with a stick and want to get back name path and media type and we want to just get the best results and run it. That runs pretty fast and we can then display all the results.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-23_at_7.41.41_PM.png"  width="80%" /> 

I want to iterate over the objects in there and we use our helper functions from before, like the display media.
We should get some results.
You can see straight away that the first result is a video of a dog running of a stick, which is cool. The but also, we have a picture of a dog and another video of a dog giving a high five.

## Image to Media Search

So now we've done text search and I think you've done it before, probably. So let's try something harder.
How about we use this image as an input for a query?


```python
# Use this image as an input for the query
Image("./test/test-cat.jpg", width=300)
```



<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-24_at_11.26.24_AM.png"  width="100%"  />

And a query this, is
actually very similar, except this time we are calling new image
and we are converting this file to base 64 format.
So we are using the test cat and again return the same properties and iterate over all the objects. And let's execute it.

```python
# The query
response = animals.query.near_image(
    near_image=file_to_base64("./test/test-cat.jpg"),
    return_properties=['name','path','mediaType'],
    limit=3
)

for obj in response.objects:
    json_print(obj.properties)
    display_media(obj.properties)
```

Response shows how well the search works

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-24_at_11.26.58_AM.png"  width="50%"  />


## Image search - from web URL

We are going to run a very similar query, except this time we'll call your URL to base64. But the rest of the query is pretty much the same. And if we execute this, we'll get these pictures of one meerkat that looks kind of angry but is all right. Don't worry about him. We have another one. Chilling.
But the fun thing is that we also were able to match a video of a meerkat.
So with that, we were able o actually do a multimodal search that we used images to find both other images and videos and this is very powerful.


```python
Image("https://raw.githubusercontent.com/weaviate-tutorials/multimodal-workshop/main/2-multimodal/test/test-meerkat.jpg", width=300)
```

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/2-test_meerkat.png"  width="50%"  />



```python
# The query
response = animals.query.near_image(
    near_image=url_to_base64("https://raw.githubusercontent.com/weaviate-tutorials/multimodal-workshop/main/2-multimodal/test/test-meerkat.jpg"),
    return_properties=['name','path','mediaType'],
    limit=3
)

for obj in response.objects:
    json_print(obj.properties)
    display_media(obj.properties)
```

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-24_at_11.13.27_PM.png"  width="50%"  />

Now for something that is probably the hardest task or an LLM, which is to make a video search.
So let's try to run the third with this video of these two meerkats

## Video to Media Search

This time what we're going to call is a new media function. We again, converting that video to a base64 representation.
I will tell the collection that this time this is a video that we're searching for.
If we execute that, in response, we get two videos and a picture of a meerkat.
Just like that, we're able to actually perform any to any from text to images to video and getting all kind of modalities in response.'

> Note: the input video must be at least 4 seconds long.


```python
Video("./test/test-meerkat.mp4", width=400)
```


```python
from weaviate.classes.query import NearMediaType

response = animals.query.near_media(
    media=file_to_base64("./test/test-meerkat.mp4"),
    media_type=NearMediaType.VIDEO,
    return_properties=['name','path','mediaType'],
    limit=3
)

for obj in response.objects:
    # json_print(obj.properties)
    display_media(obj.properties)
```


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-25_at_11.17.34_PM.png"  width="50%"  />
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-25_at_11.17.48_PM.png"  width="50%"  />

## Visualizing a Multimodal Vector Space

In this part, what you want to see is actually how this vector space looks like when we are loading both video and image embeddings and how they actually live on the same space.
For content there are similar. They should be next to each other.




Let's start by loading some of the necessary libraries. And probably the most important one here is the UMAP, which allows us to reduce the dimensionality of the vector.
We'll go from 1400 to 2 dimensions, which will allow us to actually plot it as an image, as a two dimensional image.

> To make this more exciting, let's loadup a large dataset!


```python
import numpy as np
import sklearn.datasets
import pandas as pd
import umap
import umap.plot
import matplotlib.pyplot as plt
```

## Load vector embeddings and mediaType from Weaviate 

Now what we want to do is load the vector embeddings in a media type from Weaviate.
We are using this iterator function which basically will go to all the objects that we have in our collection and then give us back vector embeddings, but also we can access all the properties like the media type.
You can see here that we are not calling the animals collection anymore.
We have a for your benefit because if we just try
to visualize a vector spatial 15 objects, that's not going to be very exciting.
So we pre-loaded this database with 14,000 images and videos
so that we can actually get some better results.

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

So let's run this and then quickly pull all the vector embeddings together with some of the properties that we need.
Now what we are going to do is set up a data frame with our embeddings together with the labels, which will act as a series.
This line is what does the conversion from the 1400 dimension to two dimensions.


```python
# Collection named "Resources"
collection = client.collections.get("Resources")

embs = []
labs = []
for item in collection.iterator(include_vector=True):
    #print(item.properties)\
    labs.append(item.properties['mediaType'])
    embs.append(item.vector)
```


```python
embs2 = [emb['default'] for emb in embs]

emb_df = pd.DataFrame(embs2)
labels = pd.Series(labs)

labels[labels=='image'] = 0
labels[labels=='video'] = 1
```

>Note: this might take some minutes to complete the execution.


```python
%%time
mapper2 = umap.UMAP().fit(emb_df)
```

## Plot the embeddings

Don't worry, this actually should take a little while.
Could be up to half a minute, but after that we should be good to go.
Now that we have all the embeddings pre calculated and drop down to two dimensions, we could actually plot them.
So this is the function that performs the plotting.
If we run this, we get a nice vector space.
Something that I haven't mentioned earlier, the data set that we prevent to write for this exercise actually came from ten different categories.
So you can see how, like we said, a similar vector embeddings are always stored very close to each other.


```python
plt.figure(figsize=(10, 8))
umap.plot.points(mapper2, labels=labels, theme='fire')

# Show plot
plt.title('UMAP Visualiztion of Embedding Space')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show();
```

## Interactive plot of vectors

>Note: Once you run the following cell, please be aware that on the right-hand side,  there are special buttons available. These buttons enable you to perform various functions such as highlighting and more.


```python
umap.plot.output_notebook()

p = umap.plot.interactive(mapper2, labels=labels, theme='fire')

umap.plot.show(p)
```

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-25_at_11.18.09_PM.png"  width="100%"  />


## Close the connection to Weaviate

And then the final step that we have to do, and that's something that you always need to remember when you're done with this instance.
What we have to do is just close it. So you can open it from another notebook.
In this lesson, you learn how you could use
a vector database with multimodal models, how you can
vector write them and stored metadata together with the vector embeddings, and then use text and image and video search across all the modalities.
But also we run a nice test and plotted
14,000 different vector embeddings to show how similar vectors
are grouped together, even if they come from different modalities.
And in the next lesson, you learn about large
multimodal models and how they work and how they get trained.
So I will see you there.

```python
client.close()
```


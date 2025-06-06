---
layout: default
title: 3. Preparing for Text RAG
nav_order: 3
description: "Knowlege Graph fundamentals"
has_children: false
parent:  Knowledge Graph for RAG (deeplearning.ai)
---



<!--
0:01 RAG systems start by using vector representations of text to match
0:05 your prompt to relevant sections within the unstructured data.
0:09 So, in order to be able to find relevant text in a knowledge graph in the same way, 0:13 you'll need to create embeddings of the text fields in your graph.
0:17 Let's take a look at how to do this.
0:20 So, to get started, you'll import some packages as we did in the last notebook, 0:24 and we'll also set up Neo4j.
0:30 You'll load the same environment variables that we have in the previous notebook, $\underline{0: 34}$ but now including a new variable called OpenAI API Key,
0:37 which we'll use for calling the OpenAI Embeddings model.
0:41 And finally, as before, we'll use the Neo4j Graph class for creating
0:44 a connection to the Knowledge Graph so we can send it some queries.
0:49 The first step for enabling vector search is to create a vector index.
0:54 Okay, in this very first line, we're creating a vector index,
0:57 we're going to give it a name, movie tagline embeddings.
1:00 And we're going to add that we should create
1:02 this index only if it doesn't already exist.
1:04 We're going to create the index for nodes that we're
1:08 going to call m that have the label movie,
$1: 10$ and on those nodes for the tagline property of the movies.
1:14 We're going to create embeddings and store those.
1:16 We have some options while we're setting up the index as well
$1: 19$ that we're passing in as this index config object right here.
1:23 There's two things of course that are important.
$1: 25$ It's how big are the vectors themselves, what are the dimensions of the vectors.
1:28 Here it's 1536, which is the default size for OpenAl's embedding model.
1:34 And OpenAI also recommends using cosine similarity,
1:36 so we're specifying that here as the similarity function.
1:48 That cipher query is nice and straightforward. Looks like this.
1:51 We can see that there's the name that we specified before.
1:53 We can see that it's ready to go and that it's a vector index. So fantastic.
2:07 We're going to match movies that have the label movie,
2:10 and where the movie.tagline is not null.
2:15 In this next line, we're going to take the movie
2:18 and also calculate an embedding for the tagline.
$\underline{2: 20}$ We're going to do that by calling this function that's called genai.vector.encode.
2:27 We're passing in the parameter which is the value we want to encode.
2:30 Here that's movie.tagline.
2:32 We're going to specify what embedding model we want to use. That's OpenAl.
2:36 And because OpenAI requires a key,
2:37 we're also going to pass in a little bit of configuration here.
$\underline{2: 40}$ It says here's the token for OpenAI.
2:43 It's going to be this OpenAI API key. Now,
$2: 49$ This value here is what we call a query parameter.
3:25 Okay.
3:28 This query may take a few seconds to run because it calls out to the OpenAI API
-->

### Introduction to Vector Embeddings in Knowledge Graphs

RAG systems start by using vector representations of text to match your prompt to relevant sections within the unstructured data. So, in order to be able to find relevant text in a knowledge graph in the same way, you'll need to create embeddings of the text fields in your graph. Let's take a look at how to do this.

### Setting Up the Environment for Embeddings

To get started, you'll import some packages as we did in the last notebook, and we'll also set up Neo4j. You'll load the same environment variables that we have in the previous notebook, but now including a new variable called OpenAI API Key, which we'll use for calling the OpenAI Embeddings model. 


```python
from dotenv import load_dotenv
import os

from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings
warnings.filterwarnings("ignore")
```


```python
# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Note the code below is unique to this course environment, and not a 
# standard part of Neo4j's integration with OpenAI. Remove if running 
# in your own environment.
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'
```


```python
# Connect to the knowledge graph instance using LangChain
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
```


And finally, as before, we'll use the Neo4j Graph class for creating a connection to the Knowledge Graph so we can send it some queries.

### Creating a Vector Index

The first step for enabling vector search is to create a vector index. Okay, in this very first line, we're creating a vector index, we're going to give it a name, movie tagline embeddings. And we're going to add that we should create this index only if it doesn't already exist. We're going to create the index for nodes that we're going to call M that have the label movie, and on those nodes for the tagline property of the movies. We're going to create embeddings and store those. We have some options while we're setting up the index as well that we're passing in as this index config object right here. There's two things of course that are important. It's how big are the vectors themselves, what are the dimensions of the vectors. Here it's 1536, which is the default size for OpenAI's embedding model. And OpenAI also recommends using cosine similarity, so we're specifying that here as the similarity function. 


```python
kg.query("""
  CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
  FOR (m:Movie) ON (m.taglineEmbedding) 
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }}"""
)

```

```python
kg.query("""
  SHOW VECTOR INDEXES
  """
)
```


    [{'id': 3,
      'name': 'movie_tagline_embeddings',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'VECTOR',
      'entityType': 'NODE',
      'labelsOrTypes': ['Movie'],
      'properties': ['taglineEmbedding'],
      'indexProvider': 'vector-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': None}]



That Cypher query is nice and straightforward. Looks like this. We can see that there's the name that we specified before. We can see that it's ready to go and that it's a vector index. So fantastic.

### Generating and Storing Embeddings

We're going to match movies that have the label movie, and where the `movie.tagline` is not null. In this next line, we're going to take the movie and also calculate an embedding for the tagline. We're going to do that by calling this function that's called `genai.vector.encode`. We're passing in the parameter which is the value we want to encode. Here that's movie.tagline. We're going to specify what embedding model we want to use. That's OpenAI. And because OpenAI requires a key, we're also going to pass in a little bit of configuration here. It says here's the token for OpenAI. It's going to be this OpenAI API key. Now, this value here is what we call a query parameter.



```python
kg.query("""
    MATCH (movie:Movie) WHERE movie.tagline IS NOT NULL
    WITH movie, genai.vector.encode(
        movie.tagline, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS vector
    CALL db.create.setNodeVectorProperty(movie, "taglineEmbedding", vector)
    """, 
    params={"openAiApiKey":OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )
```



### Running the Embedding Query

Okay. This query may take a few seconds to run because it calls out to the OpenAI API.



```python
result = kg.query("""
    MATCH (m:Movie) 
    WHERE m.tagline IS NOT NULL
    RETURN m.tagline, m.taglineEmbedding
    LIMIT 1
    """
)
```

```python
result[0]['m.tagline']
```


    'Welcome to the Real World'



<!--
3:28 This query may take a few seconds to run because it calls out to the OpenAI API
3:33 to calculate the vector embeddings for each movie in the dataset.
3:45 So let's pull out from that result just the tagline itself.
3:49 You can see what that is.
3:51 Since we only have one movie that we did the tagline,
3:53 it's welcome to the real world.
3:55 Super.
3:56 And let's also take a look at what the embedding looks like.
3:58 I'm not going to show the entire embedding.
4:00 We'll just get the first 10 values out of it.
4:04 Okay, great. That looks like a good embedding to me.
4:07 And for the last step in verifying what we've got for the embeddings,
4:10 we'll make sure that those embeddings are the right size.
4:12 We're expecting them to be 1536 . So,
4:20 Great. The vector size is 1536, just as we expected.
4:31 So now, we can actually query the database and
4:36 We'll start by specifying what the question is we want to ask
4:39 and find similar movies that might match that question.
4:45 Remember, we've done vector indexing on the taglines.
4:51 Here, we're going to start with a call towards calculating
4:54 and embedding using that same function we had before.
4:56 We're going to do that by saying, with this function call,
4:59 JNAI vector and code and a parameter for the question that will pass in.
5:04 We want to calculate an embedding using the OpenAI model and the OpenAI
5:07 of course needs an API key so we're going to pass that in as well.
5:11 The result of that function call we're going to
5:14 assign to something we call question embedding.
5:16 We're then going to call another function for actually
5:18 doing the vector similarity search itself.
5:26 That's the name of the index that we created earlier.
5:29 And this is another parameter that is interesting. We just wanted the top k results.
5:40 And then, of course, we're going to pass in the embedding that we just calculated.
5:50 Do the similarity search and actually give us those results.
5:53 Now from the results we want to be able to yield the nodes that we found,
5:57 and we will rename those as movies, and also what the similarity score was.
6:01 With that we're going to return the movie title, the movie tagline, and the score.
6:05 We're passing in some query parameters for the OpenAI API key itself,
6:09 the question that we asked that's going to be calculated into an embedding,
6:13 and here the top case is 5 , so we only want the 5 closest embeddings.
6:21 Cool. So, we've got movie titles like Joe vs. the Volcano,
6:27 You can see through all of these tag lines,
6:29 that's a pretty good match for movies that are about laws.
6:48 We'll save that question and we'll run this query again.
6:52 Oh yeah, Castaway Ninja Assassin. That sounds like something adventurous.
6:57 Duel vs. the Volcano. Apparently, it's about love and adventure.
7:00 Maybe that's a good one to have on your Netflix list.
-->
### Embedding Calculation and Verification

This query may take a few seconds to run because it calls out to the OpenAI API to calculate the vector embeddings for each movie in the dataset. So let's pull out from that result just the tagline itself. You can see what that is. Since we only have one movie that we did the tagline, it's welcome to the real world. Super. And let's also take a look at what the embedding looks like. I'm not going to show the entire embedding. We'll just get the first 10 values out of it. 

```python
result[0]['m.taglineEmbedding'][:10]
```


    [0.017445066943764687,
     -0.005481892731040716,
     -0.002013522433117032,
     -0.025571243837475777,
     -0.014404304325580597,
     0.016737302765250206,
     -0.017078077420592308,
     0.000485358847072348,
     -0.025217361748218536,
     -0.029516370967030525]


Okay, great. That looks like a good embedding to me. And for the last step in verifying what we've got for the embeddings, we'll make sure that those embeddings are the right size. We're expecting them to be 1536. Great. The vector size is 1536, just as we expected.

### Performing Vector Similarity Search

So now, we can actually query the database. We'll start by specifying what the question is we want to ask and find similar movies that might match that question. Remember, we've done vector indexing on the taglines. Here, we're going to start with a call towards calculating and embedding using that same function we had before. We're going to do that by saying, with this function call, GNAI vector and code and a parameter for the question that will pass in. We want to calculate an embedding using the OpenAI model and the OpenAI of course needs an API key so we're going to pass that in as well. The result of that function call we're going to assign to something we call question embedding. We're then going to call another function for actually doing the vector similarity search itself. That's the name of the index that we created earlier. And this is another parameter that is interesting. We just wanted the top K results. And then, of course, we're going to pass in the embedding that we just calculated.


```python
question = "What movies are about love?"
```


```python
kg.query("""
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'movie_tagline_embeddings', 
        $top_k, 
        question_embedding
        ) YIELD node AS movie, score
    RETURN movie.title, movie.tagline, score
    """, 
    params={"openAiApiKey":OPENAI_API_KEY,
            "openAiEndpoint": OPENAI_ENDPOINT,
            "question": question,
            "top_k": 5
            })
```




    [{'movie.title': 'Joe Versus the Volcano',
      'movie.tagline': 'A story of love, lava and burning desire.',
      'score': 0.9062913656234741},
     {'movie.title': 'As Good as It Gets',
      'movie.tagline': 'A comedy from the heart that goes for the throat.',
      'score': 0.9022631645202637},
     {'movie.title': 'Snow Falling on Cedars',
      'movie.tagline': 'First loves last. Forever.',
      'score': 0.9013131856918335},
     {'movie.title': 'Sleepless in Seattle',
      'movie.tagline': 'What if someone you never met, someone you never saw, someone you never knew was the only someone for you?',
      'score': 0.8945093154907227},
     {'movie.title': 'When Harry Met Sally',
      'movie.tagline': 'Can two friends sleep together and still love each other in the morning?',
      'score': 0.8942364454269409}]


### Analyzing Search Results

Do the similarity search and actually give us those results. Now from the results we want to be able to yield the nodes that we found, and we will rename those as movies, and also what the similarity score was. With that we're going to return the movie title, the movie tagline, and the score. We're passing in some query parameters for the OpenAI API key itself, the question that we asked that's going to be calculated into an embedding, and here the top K is 5, so we only want the 5 closest embeddings. Cool. 


```python
question = "What movies are about adventure?"
```


```python
kg.query("""
    WITH genai.vector.encode(
        $question, 
        "OpenAI", 
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'movie_tagline_embeddings', 
        $top_k, 
        question_embedding
        ) YIELD node AS movie, score
    RETURN movie.title, movie.tagline, score
    """, 
    params={"openAiApiKey":OPENAI_API_KEY,
            "openAiEndpoint": OPENAI_ENDPOINT,
            "question": question,
            "top_k": 5
            })
```




    [{'movie.title': 'RescueDawn',
      'movie.tagline': "Based on the extraordinary true story of one man's fight for freedom",
      'score': 0.8998090028762817},
     {'movie.title': 'Cast Away',
      'movie.tagline': 'At the edge of the world, his journey begins.',
      'score': 0.8982737064361572},
     {'movie.title': 'Ninja Assassin',
      'movie.tagline': 'Prepare to enter a secret world of assassins',
      'score': 0.8880558013916016},
     {'movie.title': 'Joe Versus the Volcano',
      'movie.tagline': 'A story of love, lava and burning desire.',
      'score': 0.8870121240615845},
     {'movie.title': 'As Good as It Gets',
      'movie.tagline': 'A comedy from the heart that goes for the throat.',
      'score': 0.8856385350227356}]


So, we've got movie titles like Joe vs. the Volcano. You can see through all of these tag lines, that's a pretty good match for movies that are about laws. We'll save that question and we'll run this query again. Oh yeah, Castaway Ninja Assassin. That sounds like something adventurous. Duel vs. the Volcano. Apparently, it's about love and adventure. Maybe that's a good one to have on your Netflix list.
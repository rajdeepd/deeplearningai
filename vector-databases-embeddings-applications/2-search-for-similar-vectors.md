---
layout: default
title: 2. Search for Similar Data
nav_order: 3
description: ".."
has_children: false
parent:  Vector Databases and Embeddings - Weaviate
---

In this lesson, you build an intuition of vector or semantic search using the brute-force k-nearest-neighbors algorithm. You code up an implementation of brute-force KNN, and see how it can be used to accurately obtain the nearest vectors in embedding space to a query vector. You then explore the issues around the runtime complexity of the brute-force KNN algorithms and this leads you to the class of approximate nearest-neighbors algorithms that lie at heart of vector database technology. All right, let's dive in. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-30_at_1.28.08 PM.png" width="80%" />
<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-30_at_1.28.12 PM.png" width="80%" />
<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-30_at_1.28.33 PM.png" width="80%" />

Vectors capture the meaning behind our data and thus in order to find data points similar in meaning to our query we can search and retrieve the closest objects in vector space and return them. This is known as semantic or vector search. And by semantic search I mean search that utilizes the meaning of words or images in question. One way to find similar vectors is through brute force which goes in following steps. Step one, given a query find the distances between all the vectors and the query vector. Step two, sort the distances and finally return the top K best matching objects with the best distance. This is known in classical machine learning as the K nearest neighbor algorithm. The thing is that brute-force searches comes with a large computational cost. You can see here that the overall query time grows with the amount of objects that we have in our star. And if over time the amount of data doubles or triples, the query time will also double and triple. Let's demonstrate this algorithm in code and try to scale it up both with number of data points and dimensions.

So, we 
already have a number of libraries loaded into our notebook and the one that is worth looking into is the nearest 
neighbor, which we'll use to demonstrate brute force algorithm and 
how it works. Let's start by generating 20 random points of 
two dimensions. And then, we can plot 
it nicely on a graph, so that you can see how they are distributed across the 
screen and across the vector space. 

``python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time
np.random.seed(42)
```


```python
# Generate 20 data points with 2 dimensions
X = np.random.rand(20,2)
```


```python
# Display Embeddings
n = range(len(X))

fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], label='Embeddings')
ax.legend()

for i, txt in enumerate(n):
    ax.annotate(txt, (X[i,0], X[i,1]))
```

And now, let's add all the data points into nearest neighbors' index. And you can see here that we are using the brute force algorithm. And then, if I run this it returns an instance that is ready to query. 




```python
k = 4

neigh = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
neigh.fit(X)
```


```python
# Display Query with data
n = range(len(X))

fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1])
ax.scatter(0.45,0.2, c='red',label='Query')
ax.legend()

for i, txt in enumerate(n):
    ax.annotate(txt, (X[i,0], X[i,1]))
```


```python
neighbours = neigh.kneighbors([[0.45,0.2]], k, return_distance=True)
print(neighbours)
```
   
Now, this is the query that finds the four nearest vectors is our case set to four and this is the query vector that we are looking for and if we run this, we will see that vectors 10, 4, 19 and 15 are the four nearest ones with the following distances. And even though we only have 20 objects we could already measure how long this query took. 

    (array([[0.09299859, 0.16027853, 0.1727928 , 0.17778682]]), array([[ 9, 15, 10, 18]]))

```python
t0 = time.time()
neighbours = neigh.kneighbors([[0.45,0.2]], k, return_distance=True)
t1 = time.time()

query_time = t1-t0
print(f"Runtime: {query_time: .4f} seconds")
```

So, if I run this and grab the time before and after we could see that like searching across 20 vectors takes very little time in this case. And now, let's see the time complexity for brute force for much larger data set than just 20 objects. And for this we have this handy function called speedtest which takes the count of the number of objects that we are tested on. And it works in three steps. First, we are randomly generating a number of objects based on the count. Then again, we are building the index using the nearest neighbors. And then finally, we measure the time of the actual query and that's basically what we return back as a result.

```python
def speed_test(count):
    # generate random objects
    data = np.random.rand(count,2)
    
    # prepare brute force index
    k=4
    neigh = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    neigh.fit(data)

    # measure time for a brute force query
    t0 = time.time()
    neighbours = neigh.kneighbors([[0.45,0.2]], k, return_distance=True)
    t1 = time.time()

    total_time = t1-t0
    print (f"Runtime: {total_time: .4f}")

    return total_time
```
So, let's test it on 20,000 objects and we can see that it run pretty fast but to really give it a go. Let's test it on way bigger data sets so we're going to run it on 200,000, 2 million, 20 million, 200 million objects and you can already see that like we've asked the number of objects increase the query time takes longer and longer and even between 2 million 20 million that's already a 10x increase and a 200 million objects took 12 seconds. 

```python
time20k = speed_test(20_000)
```

    Runtime:  0.0830    

```python
# Brute force examples
time200k = speed_test(200_000)
time2m = speed_test(2_000_000)
time20m = speed_test(20_000_000)
time200m = speed_test(200_000_000)
```
    Runtime:  0.0026
    Runtime:  0.0180
    Runtime:  0.1631


So, you can imagine what would happen if we actually had a billion objects or way more than that. This will get very quickly out of hands. The complexity that you just saw was only for two dimensions. 

## Brute force kNN implemented by hand on `768` dimensional embeddings

What will happen if we increase the dimensionality of our vector embeddings let's say to 768 dimensions and see what happens. 

Lets start with 1,000 documents with 768 dimensions so in here we are generating those vectors over here and then we are normalizing it and then also we have our query that we'll use to test the performance.



```python
documents = 1000
dimensions = 768

embeddings = np.random.randn(documents, dimensions) # 1000 documents, 768-dimensional embeddings
embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True)) # L2 normalize the rows, as is common

query = np.random.randn(768) # the query vector
query = query / np.sqrt((query**2).sum()) # normalize query
```


```python
# kNN
t0 = time.time()
# Calculate Dot Product between the query and all data items
similarities = embeddings.dot(query)
# Sort results
sorted_ix = np.argsort(-similarities)
t1 = time.time()

total = t1-t0
print(f"Runtime for dim={dimensions}, documents_n={documents}: {np.round(total,3)} seconds")

print("Top 5 results:")
for k in sorted_ix[:5]:
    print(f"Point: {k}, Similarity: {similarities[k]}")
```

And now, let's run a query where at the beginning we start a timer, then we use dot product, and we calculate the distances across all 4,000 vector embeddings.

    Runtime for dim=768, documents_n=1000: 0.002 seconds
    Top 5 results:
    Point: 434, Similarity: 0.11658853328636033
    Point: 41, Similarity: 0.10319523603880205
    Point: 677, Similarity: 0.09041193794869379
    Point: 13, Similarity: 0.0856056395158937
    Point: 438, Similarity: 0.08410763673528118

Finally once we have the results we'll sort all of them and then stop the timer and this will give us the time it takes to find the top five nearest results, and we can see that searching across 1,000 vector embeddings. It took us one half millisecond, and these are the nearest matches. 
 
So now, let's really test out how long it takes. To run a vector query across 1,000, 10,000, 100,000 and half a million objects with the 768 dimensions.

```python
n_runs = [1_000, 10_000, 100_000, 500_000]

for n in n_runs:
    embeddings = np.random.randn(n, dimensions) #768-dimensional embeddings
    query = np.random.randn(768) # the query vector
    
    t0 = time.time()
    similarities = embeddings.dot(query)
    sorted_ix = np.argsort(-similarities)
    t1 = time.time()

    total = t1-t0
    print(f"Runtime for 1 query with dim={dimensions}, documents_n={n}: {np.round(total,3)} seconds")
```

    Runtime for 1 query with dim=768, documents_n=1000: 0.0 seconds
    Runtime for 1 query with dim=768, documents_n=10000: 0.006 seconds
    Runtime for 1 query with dim=768, documents_n=100000: 0.098 seconds
    Runtime for 1 query with dim=768, documents_n=500000: 0.592 seconds

```python
print
 (f"To run 1,000 queries: {total * 1_000/60 : .2f} minutes")
```

    To run 1,000 queries:  9.86 minutes

And we can see that like up to 100,000 that returns pretty fast. But half a million takes a bit longer and you can see a single query of just half a million vectors takes almost two seconds. And if we were to run a thousand queries across half a million objects that would take in total about half an hour. Which is not great.  In conclusion, we saw how the number of vectors affects the query time. So, the more vectors we have, the longer it takes for the query to complete. And then, it got really tricky when we got to the use cases that are closer to real-life scenarios where the dimensionality of our vectors was 768 vectors. And then in this case, as soon as we reached half a million objects, brute force was just not cutting it as it was taking almost two seconds per query. And in real-life scenarios, you'd be dealing with tens or hundreds of millions of objects. In the next lesson, we'll cover different methods on how you can query across many vectors and still return results in a reasonable time. 

---
layout: default
title: 3. Approximate Nearest Neighbors
nav_order: 3
description: ".."
has_children: false
parent:  Vector Databases
---

In this lesson you will get a theoretical and 
practical understanding of how approximate nearest neighbors' algorithms, also known as ANN, trade a little bit of accuracy or recall for a lot of performance. Specifically, you'll explore the hierarchical navigable small words ANN algorithm to understand how this powers the world's most powerful vector databases. We'll also demonstrate the scalability of HNSW and how it solves the runtime complexity issues of brute-force KNN. Let's get coding! 




So, if you look at this example with 20 vectors, searching for the nearest one is not necessarily a big issue.

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_12.11.52 PM.png" width="80%" />

However, as soon as we get to thousands or millions of vectors, this is becoming a way bigger problem and definitely a big no-go. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_12.12.05 PM.png" width="80%" />


There are many algorithms that allow us to actually find the nearest vectors in a more efficient way. One of the algorithms that solve this problem is HNSW, which is based on small world phenomenon in human social networks. And the whole idea is that on average we are all connected from each other by six degrees of separation. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_12.12.19 PM.png" width="80%" />

So, it's very possible that you have a friend, who has a friend, who has a friend, who has a friend, who is connected to Andrew. And the whole idea is that you probably know somebody that knows everyone. And that person probably also knows somebody that's really well connected and through that you could actually through within six steps find the connection to the person you're looking for. And then, automatically we could apply the same concepts to vector embeddings if we build those kinds of connections. 

Lets have a look at navigable small world which is an algorithm that allows us to construct these connections between different nodes. So, for example, in here we have eight randomly generated vectors and let's have a look how we can connect all of them together to their nearest neighbors. 



If we start with vector 0, there's nothing to connect just yet. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.47.48 PM.png" width="80%" />

Then, we add vector 1 and the only possible connection is to a vector 0. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.48.32 PM.png" width="80%" />

And now ,we can add vector 2 and we can build two connections between 1 and 0. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.48.37 PM.png" width="80%" />

Following the same method, we can go to vector 3 and the two nearest connections will be to vectors 2 and 0. 


<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.48.46 PM.png" width="80%" />

Cool! Next, let's go to number four and the two nearest connections probably two and zero. Great! 
<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.48.54 PM.png" width="80%" />


Now, let's go to five is well connected to two and zero. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.49.00 PM.png" width="80%" />

Now, we can go to six connects to two and four. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.49.05 PM.png" width="80%" />

And then, finally, seven connects to vectors five and three and just like that we build a Navigo small world, and just as a note for you in this example 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-31_at_4.49.09 PM.png" width="80%" />

 
Now, let's see how the search of the Navigo small world can be performed. In this example we have a query vector which is on the left and we could already 
kind of guess that number six will be the nearest vector and usually how you work with NSW, and we start with a random entry node and we try to move across towards the nearest neighbors. So, starting from node 7, which is connected to nodes 3 and 5, we 
can see that 5 is closer than node 7. So, we can move there. Great! 
Now, from node 5 we can see that we are also connected to 0 and 2 and 2 is definitely way closer. 
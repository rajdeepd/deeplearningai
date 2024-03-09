---
layout: default
title: 1. Getting Started with textembeddings
nav_order: 2
description: ".."
has_children: false
parent:  Google Cloud Vertex AI Embeddings
---
Exploring text embeddings is incredibly engaging. Let's dive into some practical examples of text embeddings. Here we are, starting with a blank Jupyter notebook. To follow along on your own machine, you'll need the Google Cloud AI platform installed, which can be done via the command pip install Google Cloud AI platform. However, since I already have it installed on this computer, I'll skip this step. Next, I'll authenticate myself to the Google Cloud AI platform using a special authenticate function. This function is a convenient tool in this Jupyter Notebook environment for loading my credentials and project ID. You have the option to display these details if you wish.


#### Project environment setup

- Load credentials and relevant Python Libraries
- If you were running this notebook locally, you would first install Vertex AI.  In this classroom, this is already installed.
```Python
!pip install google-cloud-aiplatform
```

The project ID is essentially a string that identifies the project you're working with. Following this, I'll execute my commands targeting a server in the US Central region. Then, by importing Vertex AI and initializing it with my project ID, specifying the region for my API calls, and providing my authentication credentials, I'm set to interact with the Vertex AI platform.



```python
from utils import authenticate
credentials, PROJECT_ID = authenticate() # Get credentials and project ID
```

#### Enter project details


```python
print(PROJECT_ID)
```


```python
REGION = 'us-central1'
```
For those setting up their Google Cloud accounts, a few steps are necessary: registering an account, identifying your project ID (which you'll insert here as a string), and choosing a server region—US Central is a good default, but you can opt for a server closer to your location. There's also an optional Jupyter notebook available that goes into detail on how to secure your own Google Cloud platform credentials.


```python
# Import and initialize the Vertex AI Python SDK

import vertexai
vertexai.init(project = PROJECT_ID, 
              location = REGION, 
              credentials = credentials)
```

But for the purposes of this course, that's all the setup required. If you're running this on your own device, detailed instructions for obtaining your credentials and project ID are provided in an optional Jupyter notebook later on. Our main focus in this course is on utilizing text embedding models. Hence, I'll import the text embedding model like this. Then, I'll select the Gecko-001 text embedding model for our session today. This command essentially assigns the embedding model to this variable for our use.


#### Use the embeddings model

To compute an embedding, here's the process: Assign the variable embedding by invoking the embedding model to obtain an embedding. We'll begin with the simple word "life" as our input. Following that, we set vector to embedding[0].values, which effectively pulls the numerical values from the embedding. Then, we'll display the dimensionality of vector by printing its length, and also show the first 10 elements of this vector. In this case, vector is a 768-dimensional vector, and we're looking at its first 10 elements. You're encouraged to print more elements if you're curious about the full array of numbers.


##### Import and load the model.


```python
from vertexai.language_models import TextEmbeddingModel
```


```python
embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")
```

- Generate a word embedding


```python
embedding = embedding_model.get_embeddings(
    ["life"])
```

- The returned object is a list with a single `TextEmbedding` object.
- The `TextEmbedding.values` field stores the embeddings in a Python list.

What we've done here is taken the word "life", encapsulated it as a text string, and generated its embedding. Now, let's examine another scenario. We input the question, "What is the meaning of life?" and compute its embedding. Similarly, we obtain a 768-dimensional vector representing various features of this sentence, showcasing the first 10 elements. Given the vast number of values in each embedding, it's challenging to interpret these numbers directly. However, a prime use of embeddings is to evaluate the similarity between texts, whether they be sentences, phrases, or paragraphs.


```python
vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])
```

Print output below lists the output

```
Length = 768
[-0.006005102302879095, 0.015532972291111946, -0.030447669327259064, 0.05322219058871269, 0.014444807544350624, -0.0542873740196228, 0.045140113681554794, 0.02127358317375183, -0.06537645310163498, 0.019103270024061203]
```

##### Generate a sentence embedding.



```python
embedding = embedding_model.get_embeddings(
    ["What is the meaning of life?"])
```


```python
vector = embedding[0].values
print(f"Length = {len(vector)}")
print(vector[:10])
```


```
Length = 768
[0.020522113889455795, 0.02229207195341587, -0.009265718050301075, 0.005001612473279238, 0.016248879954218864, -0.018983161076903343, 0.04320966452360153, 0.02643178217113018, -0.04369377717375755, 0.023666976019740105]
```

#### Similarity

Next, we'll explore how to compare the similarity between different embeddings. For this, we'll employ the cosine similarity measure from the scikit-learn library. This method normalizes two vectors to unit length and then calculates their dot product as a means to gauge their similarity. We'll compute embeddings for three sentences, including "What is the meaning of life? Is it 42, or is it something else?" For those unfamiliar, the reference to "42" is a nod to a famous novel, a fun fact you might enjoy looking up online using the number "42" as your query.

Additionally, we'll embed the phrase "How does one spend their time well on Earth?" which intriguingly mirrors the existential query, "What's the meaning of life?" Our third sentence for comparison is "Would you like a salad?" followed by a humorous reflection, "I hope the meaning of my life is much more than eating salads." This third sentence, while slightly related, hopefully doesn't bear too much resemblance to the first one in terms of meaning. Following the same procedure as before, we'll extract the vectors from these embeddings.

```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
emb_1 = embedding_model.get_embeddings(
    ["What is the meaning of life?"]) # 42!

emb_2 = embedding_model.get_embeddings(
    ["How does one spend their time well on Earth?"])

emb_3 = embedding_model.get_embeddings(
    ["Would you like a salad?"])

vec_1 = [emb_1[0].values]
vec_2 = [emb_2[0].values]
vec_3 = [emb_3[0].values]
```

- Note: the reason we wrap the embeddings (a Python list) in another list is because the `cosine_similarity` function expects either a 2D numpy array or a list of lists.
```Python
vec_1 = [emb_1[0].values]
```

 Next, I will calculate and display the similarity scores for each pair of sentences. Upon recalculating, we observe that the similarity score between the first and second sentences (Vec1 and Vec2) is relatively high, at 0.655, indicating that the question of life's meaning is perceived to be closely related to how one might best utilize their time on Earth. 




```python
print(cosine_similarity(vec_1,vec_2)) 
print(cosine_similarity(vec_2,vec_3))
print(cosine_similarity(vec_1,vec_3))
```
Output of the cosine similarity


```
[[0.65503744]]
[[0.52001556]]
[[0.54139322]]
```


The similarity scores between sentences 2 and 3, and between sentences 1 and 3, are 0.52 and 0.54, respectively, suggesting that the first two sentences share more in meaning than either does with the third.

This demonstrates that even in the absence of common words, the first two sentences are deemed more similar to each other than either is to the third sentence. I encourage you to pause here and experiment by typing different sentences into the Jupyter Notebook. Try sentences related to your favorite programming language, algorithm, animals, or weekend activities, and see how the system evaluates their similarity. It's noteworthy that while cosine similarity theoretically ranges from 0 to 1, due to the high-dimensional nature of these 768-dimensional vectors, the resulting similarity scores tend to cluster within a narrower range. You're unlikely to encounter scores at the extreme ends of the scale, but even within this confined spectrum, the distinctions in similarity can be quite revealing.

#### From word to sentence embeddings


A common approach in natural language processing is to create individual embeddings for each word in a sentence and then combine these to form a single vector representation for the entire sentence. For instance, by averaging the embeddings of each word in the phrase "the kids play in the park" to derive a unified sentence vector. However, simply averaging the embeddings can lead to a loss of valuable context, as it treats all words equally and fails to capture the nuanced relationships between them.

When embeddings for words in a sentence are averaged, the resulting vector often has the same value for identical words across different instances. This is because the operation does not consider the position or function of the word within the sentence, leading to a loss of the unique meaning that may be imparted by the syntax or word order.

In contrast, more advanced embedding techniques consider the full sentence structure, recognizing the importance of word order and the role of function words like "the", "is", "at", which are often disregarded in simpler models. These sophisticated models result in embeddings that reflect a deeper understanding of sentence semantics, distinguishing, for example, between "the kids play in the park" as an activity and "the play was for kids in the park" as a performance event.

To explore the capabilities of these models, you're encouraged to experiment by inputting different sentences and observing how the embeddings reflect the inherent meaning. By examining the first few elements or the entire array of the vector, you can gain insight into how different sentences are encoded by the embedding process. This exercise will help you appreciate the complexity and power of sentence-level embeddings over word-level averaging.

Let's take a deeper look at why sentence embeddings are more powerful, I think, than word embeddings. Let's look at another two different inputs. First input, the kids play in the park. You know, during recess, the kids play in the park. And in the second input is the play was for kids in the park. So, someone puts on a play that is a show for a bunch of kids to watch. If you were to remove what's called stop words, so stop words like the, in, for, and is, those are words that are often perceived to have less semantic meaning in English sometimes. 

But if you were to remove the stop words from both of these sentences, you really end up with an identical set of three words. Kids play park and play kids park. Now, let's compute the embedding of the words in the first inputs. I'm gonna do a little bit of data wrangling in a second. So, I'm gonna import the NumPy library. And then, let me use this code snippet to call the embedding model on the first input, kids play park.
And then, the rest of this code here using an iterator and then NumPy stack, It's just a little bit of data wrangling to reformat the outputs of the embedding model into a 3 by 768 dimensional array. So, that just takes the embeddings and puts it in $a$, in an array like that. If you want, feel free to pause the video and print out the intermediate values to see what this is doing. But now, let me just do this as well for the second input. So, embedding array 2 is another 3 by 768 dimensional array. And there are three rows because there are three embeddings, one for each of these three words.

```python
in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."
```

- Remove stop words like ["the", "in", "for", "an", "is"] and punctuation.


```python
in_pp_1 = ["kids", "play", "park"]
in_pp_2 = ["play", "kids", "park"]
```

- Generate one embedding for each word.  So this is a list of three lists.


```python
embeddings_1 = [emb.values for emb in embedding_model.get_embeddings(in_pp_1)]
```

- Use numpy to convert this list of lists into a 2D array of 3 rows and 768 columns.


```python
import numpy as np
emb_array_1 = np.stack(embeddings_1)
print(emb_array_1.shape)
```
    (3, 768)

```python
embeddings_2 = [emb.values for emb in embedding_model.get_embeddings(in_pp_2)]
emb_array_2 = np.stack(embeddings_2)
print(emb_array_2.shape)
```
    (3, 768)

- Take the average embedding across the 3 word embeddings 
- You'll get a single embedding of length 768.


```python
emb_1_mean = emb_array_1.mean(axis = 0) 
print(emb_1_mean.shape)
```
    (768,)

```python
emb_2_mean = emb_array_2.mean(axis = 0)
```

- Check to see that taking an average of word embeddings results in two sentence embeddings that are identical.


```python
print(emb_1_mean[:4])
print(emb_2_mean[:4])
```
    [-0.00385805 -0.00522636  0.00574341  0.03331106]
    [-0.00385805 -0.00522636  0.00574341  0.03331106]

So, one way that many people used to build sentence level embeddings is to, then take these three embeddings for the different words and to average them together. So, if I were to say the embedding for my first input, the kids play in the park after stop word removal. So, kids play park is, I'm going to take the embedding array one, and take the mean along $\mathrm{x}$ is zero. So that just averages it across the three words we have. And, you know, do the same for my second embedding. If I then print out the two embedding vectors, not surprisingly, you end up with the same value. So, because these two lists have exactly the same words, when you embed the words, and then average the embeddings of the individual words, you end up with very similar sentence embeddings for sentences that have quite different meanings. This illustrates a significant limitation of traditional word embedding approaches when applied to sentence-level understanding: the inability to capture the context and semantic relationships between words effectively.

#### Get sentence embeddings from the model.

Word embeddings represent words in a high-dimensional space, where each dimension captures some aspect of the word's meaning. These embeddings are trained on large corpora of text, learning representations that encapsulate a word's relationships and associations with other words. However, when used to construct sentence embeddings by averaging the embeddings of individual words, the resulting vector loses much of the context and nuance. As your example illustrates, despite the two sentences having distinct meanings, the process of averaging word embeddings fails to differentiate between the contexts in which "play" is used—as a verb in one sentence and as a noun in another.

Sentence embeddings aim to overcome these limitations by encoding entire sentences, capturing not just the presence of words but also the context in which they appear. Models designed for generating sentence embeddings, such as BERT (Bidirectional Encoder Representations from Transformers) and its variants, are trained to understand the meaning of sentences by considering the words in context. This training allows these models to capture the subtleties and nuances that differentiate sentences with similar words but different meanings.





- These sentence embeddings account for word order and context.
- Verify that the sentence embeddings are not the same.


```python
print(in_1)
print(in_2)
```


```python
embedding_1 = embedding_model.get_embeddings([in_1])
embedding_2 = embedding_model.get_embeddings([in_2])
```


```python
vector_1 = embedding_1[0].values
print(vector_1[:4])
vector_2 = embedding_2[0].values
print(vector_2[:4])
```

    [0.0039385221898555756, -0.020830577239394188, -0.002994248876348138, -0.007580515928566456]
    [-0.01565515622496605, -0.012884826399385929, 0.01229254249483347, -0.0005865463172085583]

The power of sentence embeddings lies in their ability to encapsulate the semantic meaning of a sentence as a whole, including the relationships between words and the overall context. This is particularly important for applications requiring a deep understanding of text, such as question answering, text summarization, and natural language inference. By capturing the nuanced differences between sentences that may share similar words, sentence embeddings enable more sophisticated and accurate natural language processing tasks.


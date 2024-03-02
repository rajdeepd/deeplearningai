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

Next, we'll explore how to compare the similarity between different embeddings. For this, we'll employ the cosine similarity measure from the scikit-learn library. This method normalizes two vectors to unit length and then calculates their dot product as a means to gauge their similarity. We'll compute embeddings for three sentences, including "What is the meaning of life? Is it 42, or is it something else?" For those unfamiliar, the reference to "42" is a nod to a famous novel, a fun fact you might enjoy looking up online using the number "42" as your query.

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

Additionally, we'll embed the phrase "How does one spend their time well on Earth?" which intriguingly mirrors the existential query, "What's the meaning of life?" Our third sentence for comparison is "Would you like a salad?" followed by a humorous reflection, "I hope the meaning of my life is much more than eating salads." This third sentence, while slightly related, hopefully doesn't bear too much resemblance to the first one in terms of meaning. Following the same procedure as before, we'll extract the vectors from these embeddings. Next, I will calculate and display the similarity scores for each pair of sentences. Upon recalculating, we observe that the similarity score between the first and second sentences (Vec1 and Vec2) is relatively high, at 0.655, indicating that the question of life's meaning is perceived to be closely related to how one might best utilize their time on Earth. 



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
- One possible way to calculate sentence embeddings from word embeddings is to take the average of the word embeddings.
- This ignores word order and context, so two sentences with different meanings, but the same set of words will end up with the same sentence embedding.


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


```python
embeddings_2 = [emb.values for emb in embedding_model.get_embeddings(in_pp_2)]
emb_array_2 = np.stack(embeddings_2)
print(emb_array_2.shape)
```

- Take the average embedding across the 3 word embeddings 
- You'll get a single embedding of length 768.


```python
emb_1_mean = emb_array_1.mean(axis = 0) 
print(emb_1_mean.shape)
```


```python
emb_2_mean = emb_array_2.mean(axis = 0)
```

- Check to see that taking an average of word embeddings results in two sentence embeddings that are identical.


```python
print(emb_1_mean[:4])
print(emb_2_mean[:4])
```

#### Get sentence embeddings from the model.
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


```python

```
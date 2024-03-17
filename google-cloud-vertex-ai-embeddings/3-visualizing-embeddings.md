---
layout: default
title: 3. Visualizing Embeddings
nav_order: 4
description: ".."
has_children: false
parent:  Google Cloud Vertex AI Embeddings
---
## Lesson 3: Visualizing Embeddings


In this video, we're going to explore visualizations of embeddings. While creating practical applications, visualizing the output isn't always the final step—unless the goal is to analyze a set of documents to understand similarities or differences in their content. However, such visualizations are less common in other applications. Despite this, we'll delve into visualizations in this session to deepen our understanding of how embeddings function, and I anticipate we'll uncover some fascinating insights. So, let's dive in.

#### Project environment setup
Firstly, I'll authenticate with the Veritex AI platform, as we've done previously. 

- Load credentials and relevant Python Libraries


```python
from utils import authenticate
credentials, PROJECT_ID = authenticate() #Get credentials and project ID
```


```python
REGION = 'us-central1'
```

#### Enter project details


```python
# Import and initialize the Vertex AI Python SDK

import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)
```
## Embeddings capture meaning

For our visualization exercise, we'll work with a unique set of seven sentences: "Mustang flamingo," "discover that swimming pool," "see all the spotters who have bought baby panda," "boat ride," "breakfast theme," "food truck, new curry restaurants," along with "Python developers are wonderful people" (a statement I wholeheartedly agree with), and "TypeScript, C++, and Java are all great." While I have personal preferences among these, I'll keep them to myself in this video. Next, just like before, I'll import NumPy and set up our embedding model. Then, I'll demonstrate a code snippet that processes these seven sentences, generating embeddings for each to illustrate the process.

```python
in_1 = "Missing flamingo discovered at swimming pool"

in_2 = "Sea otter spotted on surfboard by beach"

in_3 = "Baby panda enjoys boat ride"


in_4 = "Breakfast themed food truck beloved by all!"

in_5 = "New curry restaurant aims to please!"


in_6 = "Python developers are wonderful people"

in_7 = "TypeScript, C++ or Java? All are great!" 


input_text_lst_news = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]
```


```python
import numpy as np
from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")
```


For the given list of sentences, we start by applying our embedding model to each sentence to generate embeddings. These embeddings are then collected into a list. Following this, we perform some data manipulation to transform this list into a NumPy array. Upon executing this process, it completes swiftly. To give you an idea of what we have now, I'll share the dimensions of the resulting embeddings array: it's structured as 7 by 768. This means we have seven sentences, each represented by an embedding with 768 values.

```python
embeddings = []
for input_text in input_text_lst_news:
    emb = embedding_model.get_embeddings(
        [input_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings) 
```


```python
print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)
```


```
Shape: (7, 768)
[[ 0.04559246 -0.01472285 -0.02949955 ...  0.04057328 -0.03193641
  -0.01936668]
 [-0.01995482  0.00037652  0.0116593  ...  0.02617216 -0.03978169
  -0.02036468]
 [ 0.01030084  0.02219611  0.02433357 ...  0.03538613 -0.0273955
  -0.04193578]
 ...
 [-0.0263201  -0.01767797 -0.01261324 ... -0.01372547  0.00060259
   0.01581882]
 [-0.00561961 -0.02237099 -0.03271009 ... -0.02777804 -0.03388645
  -0.01553735]
 [ 0.00867064 -0.0131854   0.04283332 ... -0.04224638  0.01800203
   0.01088098]]
```

#### Reduce embeddings from 768 to 2 dimensions for visualization

Next, we aim to visualize these seven embeddings. However, directly visualizing a 768-dimensional vector on a two-dimensional computer screen is not feasible. To address this, we'll employ a method known as PCA, or Principal Components Analysis. For those already familiar with PCA, you'll know its utility; for others, no need to worry. Essentially, PCA is a technique used to reduce the dimensionality of high-dimensional data (like our 768-dimensional embeddings) down to two dimensions, making it possible to visualize on our screens.

```python
from sklearn.decomposition import PCA

# Perform PCA for 2D visualization
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings_array)
new_values = PCA_model.transform(embeddings_array)
```
If you're curious about diving deeper into PCA (Principal Components Analysis), you might consider enrolling in an online machine learning course, like the Machine Learning Specialization. However, for the purpose of this video, what you need to understand is that PCA is a method used to compress high-dimensional data—like our 768-dimensional dataset—into just two dimensions. This process simplifies our dataset, making it manageable to plot on a two-dimensional display by using the PCA function from the Scikit-learn library. After applying PCA, our dataset transforms from a 7x768 matrix to a 7x2 matrix, significantly reducing the dimensionality while unavoidably losing some information but enabling us to visualize it on our screens.

```python
print("Shape: " + str(new_values.shape))
print(new_values)
```

```
Shape: (7, 2)
[[-0.40980753 -0.10084478]
 [-0.39561909 -0.18401444]
 [-0.29958523  0.07514691]
 [ 0.16077688  0.32879395]
 [ 0.1893873   0.48556638]
 [ 0.31516547 -0.23624716]
 [ 0.4396822  -0.36840086]]
```

We then employ matplotlib, a widely-used plotting library, to create a 2D visualization of these embeddings, mapping each of the now two-dimensional vectors onto a graph. This visualization clusters similar sentences closer together, demonstrating the model's ability to group related concepts. For example, sentences related to animals cluster together, as do those related to food or programming, illustrating the embedding's effectiveness in capturing semantic similarities.

I encourage viewers to experiment by inputting their own sentences to see how they cluster. This can be a fun exercise, perhaps even sharing the outcomes with friends. However, it's crucial to note that for practical applications requiring similarity measurements, one should rely on the original high-dimensional space rather than the reduced two-dimensional version. While 2D embeddings are useful for visualization, the original, higher-dimensional space provides a more accurate representation for calculating distances and similarities, as the compression process for visualization omits significant amounts of information.


```python
import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl

from utils import plot_2D
plot_2D(new_values[:,0], new_values[:,1], input_text_lst_news)
```

Principal Component Analysis (PCA) simplifies our data for visualization by discarding a considerable amount of information, which affects the accuracy of distance metrics. This reduction is essential for visual clarity but can lead to the loss of nuanced details.

In another example, we analyze four sentences. The first, "He couldn't desert his post at the power plant, the power plant needed him at the time," and a closely related second sentence, suggest thematic similarity, as do the third and fourth sentences regarding the resilience of cacti and desert plants in arid conditions, despite both sets containing the words "desert" and "plant" in different contexts. To explore this, we employ the same method to generate embeddings for these sentences.

Following this, we introduce a heat map to visualize the embeddings' values, using a color gradient from blue to red to indicate the variance in values across the embedding components. Although heat maps are not commonly used for embedding visualization, they offer an insightful look into the high-dimensional data structure. This visualization reveals that the embeddings for sentences about the desert plant and cacti share similar patterns, indicating their thematic closeness compared to the other sentences.

As an engaging activity, viewers are encouraged to pause the video and experiment by calculating the similarity between these embeddings, particularly to see if the first two sentences are more closely related than when compared to the third or fourth. This can be done using the cosine similarity function, revisiting the method introduced in our initial discussion. This exercise highlights the practical application of embeddings in identifying and quantifying thematic similarities between different texts.

#### Embeddings and Similarity
- Plot a heat map to compare the embeddings of sentences that are similar and sentences that are dissimilar.


```python
in_1 = """He couldn’t desert 
          his post at the power plant."""

in_2 = """The power plant needed 
          him at the time."""

in_3 = """Cacti are able to 
          withstand dry environments.""" 

in_4 = """Desert plants can 
          survive droughts.""" 

input_text_lst_sim = [in_1, in_2, in_3, in_4]
```


```python
embeddings = []
for input_text in input_text_lst_sim:
    emb = embedding_model.get_embeddings([input_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings) 
```


```python
from utils import plot_heatmap

y_labels = input_text_lst_sim

# Plot the heatmap
plot_heatmap(embeddings_array, y_labels = y_labels, title = "Embeddings Heatmap")
```

Note: the heat map won't show everything because there are 768 columns to show.  To adjust the heat map with your mouse:

- Hover your mouse over the heat map.  Buttons will appear on the left of the heatmap.  Click on the button that has a vertical and horizontal double arrow (they look like axes).
- Left click and drag to move the heat map left and right.
- Right click and drag up to zoom in.
- Right click and drag down to zoom out.

#### Compute cosine similarity

- The `cosine_similarity` function expects a 2D array, which is why we'll wrap each embedding list inside another list.
- You can verify that sentence 1 and 2 have a higher similarity compared to sentence 1 and 4, even though sentence 1 and 4 both have the words "desert" and "plant".


```python
from sklearn.metrics.pairwise import cosine_similarity
```


```python
def compare(embeddings,idx1,idx2):
    return cosine_similarity([embeddings[idx1]],[embeddings[idx2]])
```


```python
print(in_1)
print(in_2)
print(compare(embeddings,0,1))
```

```text
He couldn’t desert 
          his post at the power plant.
The power plant needed 
          him at the time.
[[0.80995305]]
```


```python
print(in_1)
print(in_4)
print(compare(embeddings,0,3))
```

```
He couldn’t desert 
          his post at the power plant.
Desert plants can 
          survive droughts.
[[0.48815018]]
```


Feel encouraged to explore these pairwise similarities further. Before concluding this section, I must note a critical caveat regarding our visualization method. Though we've used this technique for illustrative purposes, it's important to recognize that it doesn't hold complete mathematical validity. To delve a bit into the technical side for a moment—and it's perfectly fine if the following details seem complex—the orientation of the axes in an embedding space is essentially arbitrary, subject to change due to random rotations. This means that while calculating pairwise similarity between embeddings is a robust and meaningful operation, interpreting individual components of an embedding, such as a single value on a heat map, can be misleading. These components can't easily be tied to specific, interpretable features, making such visualizations more of an informal tool for gaining insights into embeddings rather than a precise analytical method.

If the technical details are a bit much, no worries. The essential point is that these visualizations should be viewed as informal tools for understanding embedding behavior, not definitive guides. Different embedding models might produce varying visualizations, but the underlying concept—that certain sentences share closer similarities in the embedding space—remains valid and insightful.
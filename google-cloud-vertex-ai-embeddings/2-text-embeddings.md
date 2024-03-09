---
layout: default
title: 2.  Understanding Text Embeddings
nav_order: 3
description: ".."
has_children: false
parent:  Google Cloud Vertex AI Embeddings
---

## What is an Embedding?

Embeddings represent an intriguing technology that truly fascinates me. In this tutorial, we're going to explore the mechanics behind embeddings. Let's jump right in. From what we've briefly touched upon in a prior session, embeddings serve as a method for transforming data into points within a space, where these points' positions hold semantic value. Essentially, "semantic" pertains to the meaning, indicating that these positions reflect the underlying meaning of a text segment. For instance, consider the sentence "missing flamingo discovered at swimming pool" being converted into an embedding. When we process various sentences, such as "missing flamingo discovered," "sea otter spotted on surfboard by beach," or "baby panda enjoys boat ride," we anticipate these to occupy proximate locations in this semantic space.


<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-06_at_4.56.09 PM.png" width="90%"/>

Conversely, sentences like "breakfast in food truck beloved by all" and "new curry restaurant is delightful" would be positioned further apart. The semantic distances between sentences involving animals and water activities are expected to be significantly larger than the distance between, say, the flamingo narrative and the breakfast scenario. Now, let's delve a bit into the technical side of how these embeddings are generated. I'll introduce more complex concepts on the next slide, so feel free to take it at your pace. This part isn't mandatory for understanding the rest of our course, so skipping it won't hinder your progress. However, for those interested in creating their embeddings, a basic method involves individually embedding each word of a sentence and then aggregating these through summation or averaging. Traditionally, embeddings were computed by cataloging the most frequent English words and training unique parameters for each to derive their embeddings. Then, by averaging these embeddings, you could represent an entire sentence. However, this older method doesn't account for the order of words within sentences.

Contemporary embedding techniques employ a more refined approach, utilizing transformer neural networks to generate context-sensitive word representations. Don't worry if the diagram or the concept seems complex; the key takeaway is that transformers assess each word in the context of its surrounding words. This method distinguishes the various meanings of the word "play" based on context, for example, distinguishing between children playing and a theatrical play.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-06_at_4.56.23 PM.png" width="90%"/>

This advanced technique allows for the generation of sentence embeddings that more accurately reflect the nuanced meanings of each word within its context. Moreover, this approach introduces an even more potent modification to embedding technology, enhancing its ability to capture and represent semantic intricacies.

## How are sentence Embeddings Computed?

Rather than relying on a fixed list of words, contemporary embedding techniques generate embeddings for each token, which typically represent parts of words. This approach has the advantage of accommodating new or misspelled words effectively. For instance, if "unverse" is inputted as a misspelling of "universe" from a favorite novel title, the system can still produce a reasonably accurate embedding for it, despite the error. This flexibility contrasts sharply with traditional embedding methods, where a misspelled word like "unverse" would not be closely associated with its correct form, "universe."

Modern large language models deconstruct sentences into smaller components or tokens, including subwords, and learn embeddings for these tokens. This means that virtually any string of text, regardless of its novelty or accuracy in spelling, can yield a meaningful embedding. The learning process for these embeddings typically involves contrastive learning within transformer neural networks. This method begins with pre-training the network on extensive text data, often sourced from the internet or other large datasets, without any labels.

Following pre-training, the network undergoes fine-tuning with pairs of sentences deemed similar. This process adjusts the neural network to bring embeddings of similar sentences closer together while distancing those of dissimilar sentences. The criteria for sentence similarity vary according to the specific application, such as a database of sentence pairs or a question-answering system where questions and their answers are considered similar. Conversely, dissimilar sentences are usually just randomly chosen pairs, providing a broad learning spectrum for the network.


The underlying principle behind the continuous improvement of embedding algorithms, which enhances every few months, is a testament to the vibrant area of research that makes these technologies increasingly effective for practical use today.We  will delve into the practical applications of text embeddings, spanning text classification, clustering, outlier detection, and semantic search. While we won't focus extensively on product recommendations within this course, it's worth noting the potential of embeddings in identifying similar products based on their descriptions, thereby facilitating tailored product suggestions.

## Multimodal Embeddings

Another intriguing concept I'd like to introduce, though not covered in detail in this course, is the idea of multimodal embeddings. This cutting-edge approach allows for the embedding of both text and images into a unified dimensional space, such as a 768-dimensional one. 


Multimodal algorithms are designed to handle data of various types, including text and images, and research is extending to audio as well. For example, these algorithms can map both a text description like "oranges are in season" and an image of oranges to proximate points within this shared space.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-07_at_2.56.49 PM.png" width="90%"/>

The development of multimodal embeddings represents a significant leap forward, broadening the scope of possible applications that can comprehend both text and visual information. As we proceed to the next video, we'll explore more about embeddings and visualize some of their unique characteristics. Let's move on to the next segment.
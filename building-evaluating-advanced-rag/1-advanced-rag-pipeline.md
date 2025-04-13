---
layout: default
title: 1. Advanced RAG Pipeline
nav_order: 2
description: "Computational challenges of training LLMs"
has_children: false
parent:  Deeplearning RAG
---

In this lesson, you'll get a full overview of how to set up both the basic and advanced rag pipeline with Lama index. We'll load in an evaluation benchmark and use true lens to find a set of metrics so that we can benchmark advanced rag techniques against the baseline or basic pipeline. In the next few lessons, we'll explore each lesson a little bit more in depth. Let's first walk through how a basic retrieval augmented generation pipeline works or a rag pipeline. It consists of three different components, ingestion, retrieval and synthesis. 

Going through the **injection** phase. We first load in a set of documents. For each document, we split it into a set of tax trunks using a tax splitter. Then for each chunk, we generate an embedding for that trunk using an embedding model. And then for each chunk with embedding, we offload it to an index which is a view of a storage system such as a vector database. Once the data is stored within the index, we then perform retrieval against that index. 

<img src="./images/Screenshot_2023-12-23_at_12.14.45 PM.png" width="80%" />

<p align="Center"><i>Figure 1: Ingestion Process</i></p>

First, we launch a user query against the index and then we fetch the **top k** most similar chunks to the user query. Afterwards, we take these relevant chunks, combine it with the user query and put it into the prompt window of the LLM in the synthesis phase. And this allows us to generate a final response. This notebook will walk you through how to set up a basic and advanced rag pipeline with **LlamaIndex**. We will also use **TruEra** to help set up an evaluation benchmark so that we can measure improvements against the baseline. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-23_at_12.15.32 PM.png" width="80%" />

<p align="Center"><i>Figure 2: Setup</i></p>

For this quick start, you will need an OpenAPI Key note that for this lesson, we'll use a set of helper functions to get you setup and running quickly and we'll do a deep dive into some of these sections and the future lessons. 

```python
import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()
```

Next, we'll create a simple LM application using `llama_index` which internally uses an openai llm in terms of the data source. We'll use the **how to build a career in AI** PDF written by Andrew NG. 


```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()
```
Note that you can also upload your own PDF file if you wish. and for this lesson, we encourage you to do. 

So let's do some basic sanity tracking of what the document consists of as well as the length of the document. We see that we have a list of documents. There are 41 elements in there. 

Each item of that list is a document object and we'll also show a snippet of the text for a given document. 

```python
print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])
```

Output of the type of documents, lenght and sample document is given below:
```
41 

<class 'llama_index.schema.Document'>
Doc ID: eae3a305-599b-45fc-8a4c-38fb6bba948b
Text: PAGE 1Founder, DeepLearning.AICollected Insights from Andrew Ng
How to  Build Your Career in AIA Simple Guide
```
## Basic RAG pipeline

Next, we'll merge these into a single document because it helps with overall accuracy when using more advanced retrieval methods such as a sentence window retrieval as well as an autom meing retrieval. 

```python
from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))
```
The next step here is to index these documents and we can do this with the vector store index within Lama index. Next, we define a service context object which contains both the L LA we're going to use as well as the embedding model. We're gonna use the L we're gonna use is **GPT 3.5 turbo** from OpenAI. And then the embedding model that we're gonna use is the [huggingface bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).

```python
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],
                                        service_context=service_context)
```

These few steps show this injection process in Figure 1. We've loaded in documents and then in one line vectors and knocks off from documents. We're doing the chunking embedding and indexing under the hood with the embedding model that you specified. 

Next, we obtain a query engine from this index that allows us to send user queries that do retrieval and synthesis against this data. 

```python
query_engine = index.as_query_engine()
```

Let's try out our first request and the query is what are subs to take when finding projects to build your experience, that's fine, start small and gradually increase the scope and capacity of your projects. 


```python
response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))
```
Response is show below

```
When finding projects to build your experience, there are several steps you can take. First, you can join existing projects by asking to join someone else's project if they have an idea. Additionally, you can develop a side hustle or personal project, even if you have a full-time job, to stir your creative juices and strengthen bonds with collaborators. It's important to choose projects that will help you grow technically, by being challenging enough to stretch your skills but not too hard that you have little chance of success. Having good teammates or people to discuss things with is also important, as we learn a lot from the people around us. Finally, consider if the project can be a stepping stone to larger projects, both in terms of technical complexity and business impact.
Great. So it's working. So now you've set up the basic rag pipeline. 
```

The next step is to set up some evaluations against this pipeline to understand how well it performs. A

## Evaluation

Evaluation will also provide the basis for defining our advanced retrieval methods of a sentence window retriever as well as an auto merging retriever. In this section, we use **TrueLens** to initialize feedback functions. We initialize a helper function get feedbacks to return a list of feedback functions to evaluate our app. Here we have created a rag evaluation triad which consists of pairwise comparisons between the query response and context. 


```python
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)
```
Printed items will showup as 


```
What are the keys to building a career in AI?
How can teamwork contribute to success in AI?
What is the importance of networking in AI?
What are some good habits to develop for a successful career?
How can altruism be beneficial in building a career?
What is imposter syndrome and how does it relate to AI?
Who are some accomplished individuals who have experienced imposter syndrome?
What is the first step to becoming good at AI?
What are some common challenges in AI?
Is it normal to find parts of AI challenging?
```

This really creates three different evaluation models answer relevance, context relevance and grounded. This answer relevance is, is the response relevant to the query context. Relevance is is the retrieve context relevant to the query and grounded. This is is the response supported by the context, but walk through how to set this up yourself in the next few notebooks. The first thing we need to do is to create a set of questions on which to test our application here. We've prewritten the 1st 10 and we encourage you to add to the list and now we have some evaluation questions. What are the keys to building a career in A I? 

```python
# You can try your own question:
new_question = "What is the right AI job for me?"
eval_questions.append(new_question)
```


```python
print(eval_questions)
```

How can teamwork contribute to success in AI et cetera? The first thing we need to do is to create a set of questions on which to test our application. Here. We've prewritten the 1st 10, but we encourage you to also add to this list. Here we specify a fund new question. What is the right AI job for me? And we add it to the eval questions list. 

Now we can initialize the TruLens module to begin our evaluation process. We've initialized the TruLens module and now we've reset the database. We can now initialize our evaluation models. 


```python
from trulens_eval import Tru
tru = Tru()

tru.reset_database()
```

LLMs are growing as a standard mechanism for evaluating generative AI applications at scale. Rather than relying on expensive human evaluation or set benchmarks, allons allow us to evaluate our applications in a way that is custom to the domain in which we operate and dynamic to the changing demands for our application. Here. We've prebuilt a TruLens recorder to use for this example. 

```python
tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)
```

In the recorder, we've included the standard triad of evaluations for evaluating rags ground in this context, relevance and answer relevance. Uh We'll also specify an ID so that we can track this version of our app. As we experiment, we can track new versions by simply changing the app ID. Now we can run the query engine again with the TruLens context. So what's happening here is that we're sending each query to our query engine and in the background, uh the true lens recorder is evaluating each four queries against these three metrics. 


```python
for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))
```

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-25_at_11.34.52 AM.png" width="80%" />

If you see some warning messages, uh don't worry about it. Some of it is just something about it. Here. We can see a list of queries as well as their associated responses. You can see the input, output the record ID tags and more you can also see the answer relevance context, relevance and crowdedness for each row. 


```python
tru.get_leaderboard(app_ids=[])
```

Starting dashboard ...
Config file already exists. Skipping writing process.
Credentials file already exists. Skipping writing process.

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-25_at_11.35.49 AM.png" width="80%" />


In this dashboard, you can see your evaluation metrics like context, relevance, answer relevance and crowdedness as well as average A and C total cost and more. And, and A U I here, we see that the answer relevance and grounded this are decently high, but clo relevance is pretty low. 

Now, let's see if we can improve these metrics with more advanced retrieval techniques like sentence window retrieval as well as on meg retrieval. 

## Sentence window retrieval
The first advance technique we'll talk about is sentence window retrieval. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-25_at_11.37.24 AM.png"  width="80%" />

This works by embedding and retrieving single sentences. So more granular chunks. But after retrieval, the sentences are replaced with a larger window of sentences around the original retrieve sentence. The intuition is that this allows for the lab to have more context for the information retrieved in order to better answer queries while still retrieving a more granular pieces of information. So ideally improving both retrieval as well as synthesis performance. Now let's take a look at how it started up. First, we'll use OpenAI  GPT-3.5-turbo model. 

```python
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
```

Next, we'll construct our sentence window index over the given document. Just a reminder that we have a helper function for constructing the sentence window index. And we'll do a deep dive in how this works under the hood in the next lesson similar to before we'll get a query engine from the sentence window index. 

```python
from utils import build_sentence_window_index

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)
```

```python
from utils import get_sentence_window_query_engine

sentence_window_engine = get_sentence_window_query_engine(sentence_index)
```

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-25_at_2.27.27 PM.png"  width="80%" />

Now that we've set this up, we can try running an example query here. The question is how do I get started on a personal project in AI? And we get back a response got started on a personal project in AI.

```python
window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))
```

Output of the response

```text
To get started on a personal project in AI, it is important to first identify and scope the project. Consider your career goals and choose a project that complements them. Ensure that the project is responsible, ethical, and beneficial to people. As you progress in your career, aim for projects that grow in scope, complexity, and impact over time. Building a portfolio of projects that shows skill progression can also be helpful. Additionally, there are resources available in the provided chapters that can guide you in starting your AI project and finding the right job in the field.
```

It is first important to identify the project. Great. Similarly to before, let's try doubting the *TrueLens* evaluation context and try benchmarking the results. So here we import the true recorder sentence window which is a prebuilt true lines recorder for the sentence window index. 

```python
tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)
```

And now we'll run the sentence by no retriever on top of these evaluation questions and then compare performance on the rag triad of evaluation modules. 

```python
for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))
```
    What are the keys to building a career in AI?
    The keys to building a career in AI are learning foundational technical skills, working on projects, and finding a job, all of which is supported by being part of a community.
    How can teamwork contribute to success in AI?
    Teamwork can contribute to success in AI by allowing individuals to leverage the expertise and insights of their colleagues. When working on larger AI projects that require collaboration, the ability to lead and work effectively as a team becomes crucial. By working together, team members can share their deep technical insights, make informed decisions about technical architecture or data collection, and ultimately improve the project. Additionally, being surrounded by colleagues who are dedicated, hardworking, and continuously learning can inspire individuals to do the same, leading to greater success in AI endeavors.
    What is the importance of networking in AI?
    Networking is important in AI because it can help individuals build connections with people who are already in the field. These connections can provide valuable insights, advice, and potential job opportunities. By reaching out to individuals in their network or attending industry events, individuals can expand their professional connections and gain access to informational interviews. These interviews allow individuals to learn more about specific positions and companies in the AI field, which can be helpful in preparing for a job search. Additionally, networking allows individuals to pay it forward by helping others who are interested in AI, creating a supportive community within the industry.
    What are some good habits to develop for a successful career?
    Developing good habits in eating, exercise, sleep, personal relationships, work, learning, and self-care can be beneficial for a successful career. These habits can help individuals move forward while staying healthy and contribute to their overall success in their professional lives.
    How can altruism be beneficial in building a career?
    Altruism can be beneficial in building a career by helping others even as one focuses on their own career growth. By aiming to lift others during every step of their own journey, individuals can achieve better outcomes for themselves. This can create a positive reputation and build strong personal relationships, which can lead to networking opportunities and potential referrals for job opportunities. Additionally, practicing altruism can enhance personal satisfaction and fulfillment, which can contribute to overall well-being and success in one's career.
    What is imposter syndrome and how does it relate to AI?
    Imposter syndrome is a psychological phenomenon where individuals doubt their abilities and fear being exposed as frauds, despite evidence of their success. In the context of AI, newcomers to the field sometimes experience imposter syndrome, questioning whether they truly belong in the AI community and if they are capable of contributing. It is important to address imposter syndrome in order to encourage individuals to continue growing in AI and not let self-doubt hinder their progress.
    Who are some accomplished individuals who have experienced imposter syndrome?
    Sheryl Sandberg, Michelle Obama, Tom Hanks, and Mike Cannon-Brookes are some accomplished individuals who have experienced imposter syndrome.
    What is the first step to becoming good at AI?
    The first step to becoming good at AI is to learn foundational technical skills.
    What are some common challenges in AI?
    Some common challenges in AI include keeping up-to-date with evolving technology, finding suitable projects and estimating timelines and return on investment, managing the highly iterative nature of AI projects, collaborating with stakeholders who lack expertise in AI, and struggling with technical challenges while reading research papers or tuning neural network hyperparameters.
    Is it normal to find parts of AI challenging?
    Yes, it is normal to find parts of AI challenging. The author of the text acknowledges that they still find many research papers challenging to read and have made mistakes while working with neural networks. They assure the reader that everyone who has published a seminal AI paper has also faced similar technical challenges at some point. Therefore, it is normal to find certain aspects of AI challenging.
    What is the right AI job for me?
    The right AI job for you will depend on your career goals and the skills you have developed. Building a portfolio of projects that show skill progression can help you identify the areas of AI that you are most interested in and skilled at. Additionally, using a simple framework for starting your AI job search and conducting informational interviews can help you find the right AI job that aligns with your goals and interests.

Here we can see the responses come in as they're being run some examples of questions or responses. Um How can teamwork contribute to success and AI teamwork can contribute to success and AI by allowing individuals to leverage the expertise and insights of their colleagues. What what's the importance of networking in AI networking is important in AI because it allows individuals to connect with others who have experience and knowledge in the field. 

Great. Now that we've run evaluations for two techniques, the basic rag pipeline, as well as the sentence window retrieval pipeline. Let's get a lead award of the results and see what's going on here.


```python
tru.get_leaderboard(app_ids=[])
```

We see that  sentence window based grounded is 8% points better than the baseline rag pipe point answer. Relevance is more or less the same; context relevance is also better for the sentence, latency is more or less the same. The total cost is lower since the grounded and context of elements are higher, but the total cost is lower, we can intuit that the sentence window retriever is actually giving us more relevant context and more efficiently as well.


<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-25_at_2.32.58 PM.png"  width="80%" />


When we go back into the UI, we can see that we now have a comparison between the two techniques - window based and the basic technique.  We can see the metrics that we just saw in the notebook displayed on the dashboard as well. 

## 2. Auto-merging retrieval

The next advanced retrieval technique we'll talk about is the auto merging retriever. Here we construct a hierarchy of larger parent notes with smaller child notes that reference the parent note. So for instance, we might have a parent node of chunk size 512 tokens. And underneath there are four child nodes of chunk size, 128 tokens that link to this parent node. The auto emerging retriever works by merging retrieve nodes into larger parent nodes. Which means that during retrieval, if a parent actually has a majority of its Children nodes retrieved, then we'll replace the Children nodes with the parent node.

So this allows us to hierarchically merge and retrieve nodes. The combination of all the trial nodes is the same text as the parent node. Similarly to the sentence window retriever. And the next few lessons we'll do a bit more of a deep dive on how it works here. It will show you how to set it up with  helper functions here. We've built the auto emerging index um again using GPT 3.5 turbo for the LLM, as well as the BGE small model for the embedding model. 

```python
from utils import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)
```

We got the query engine from the automerging retriever and let's try running an example period. 

```python
from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)
```


To build a portfolio of AI projects, it is important to start with simple undertakings and gradually progress to more complex ones. This progression will demonstrate your growth and development over time. Additionally, effective communication is key in order to explain your thinking and showcase the value of your work. Being able to articulate your ideas will help others see the potential in your projects and trust you with resources for larger endeavors
Here, you actually see the merging process go on or merging nodes into a parent node uh to basically retrieve the parent node as opposed to the child node to build a portfolio of A I projects. It is important to start with simple undertakings and gradually progress to more complex ones. Great. So we see that it's working now lets benchmark results with true ones. 

We got a prebuilt TrueLens recorder. On top of our Auto retriever, we then run the auto verging retriever with true lens. On top of our evaluation questions here. For each question, we actually see the merging process going on such as merging three nodes into the parent node. For the first question, if we scroll down just a little bit, we see that for some of these other questions, we're also performing the merging process, merging three nodes into a parent node, merging one node into a parent node. 
*How do I build a portfolio of AI projects in the logs?* 

```python
auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))
```
    > Merging 1 nodes into parent node.
    > Parent node id: a7158f0e-89e1-43b0-b8e6-11f04e879232.
    > Parent node text: PAGE 21Building a Portfolio of 
    Projects that Shows 
    Skill Progression CHAPTER 6
    PROJECTS

    > Merging 1 nodes into parent node.
    > Parent node id: a6ab7eb6-0861-4984-80f8-56852540b0d7.
    > Parent node text: PAGE 21Building a Portfolio of 
    Projects that Shows 
    Skill Progression CHAPTER 6
    PROJECTS

```python
tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                    app_id="Automerging Query Engine")
```


```python
for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)
```
Is it normal to find parts of AI challenging?
Yes, it is normal to find parts of AI challenging. The context information suggests that even accomplished individuals in the AI community have faced technical challenges and struggled with certain aspects of AI. The author encourages newcomers to not be discouraged by these challenges and assures them that everyone has been in a similar position at some point.
> Merging 1 nodes into parent node.
> Parent node id: 58289b95-de6d-41be-92a8-b8d18e203b93.
> Parent node text: PAGE 31Finding the Right 
AI Job for YouCHAPTER 9
JOBS

    > Merging 1 nodes into parent node.
    > Parent node id: 1a4f4835-4cfa-4da6-ad1a-19c95c1cbbd2.
    > Parent node text: If you’re leaving 
    a job, exit gracefully. Give your employer ample notice, give your full effort...

    > Merging 1 nodes into parent node.
    > Parent node id: 991ffbc3-fe39-4707-88c8-3179252ae2d6.
    > Parent node text: PAGE 28Using Informational 
    Interviews to Find 
    the Right JobCHAPTER 8
    JOBS

    > Merging 1 nodes into parent node.
    > Parent node id: a8e08ee5-a57e-4855-b218-85e18b0194d5.
    > Parent node text: PAGE 31Finding the Right 
    AI Job for YouCHAPTER 9
    JOBS

    > Merging 1 nodes into parent node.
    > Parent node id: e687f33f-3566-44f0-80c4-3ae29dcb12ae.
    > Parent node text: PAGE 28Using Informational 
    Interviews to Find 
    the Right JobCHAPTER 8
    JOBS

```python
tru.get_leaderboard(app_ids=[])
```

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-25_at_5.01.34 PM.png  width="80%" />

An example question response pair is what is the importance of networking in AI networking is important in AI as it helps in building a strong professional networking community. Now that we've run all three retrieval techniques, the basic rag pipeline, as well as the two advanced retrieval methods. We can view a comprehensive leader board to see how all three techniques stack up. We get pretty nice results for the auto version query engine. On top of the evaluation questions, we get 100% in terms of ground in this 94% in terms of a relevance and 43% in terms of context relevance, which is higher than both the sentence window and the baseline rack pipeline. And we get roughly equivalent total costs to a sentence by Prairie Engine implying that the retrieval here is more efficient with equivalent latency. And at the end, you can view this in the dashboard as well. This lesson gives you a comprehensive overview of how to set up a basic and advanced Rag pipeline and also how to set up evaluation modules to measure performance. In the next lesson, Aon will do a deep dive into these evaluation modules, specifically the Rag triad of gravis answer relevance and context relevance. And you'll learn a bit more about how to use these modules and what each module means.
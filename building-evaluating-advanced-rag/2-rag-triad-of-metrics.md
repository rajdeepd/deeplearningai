---
layout: default
title: 2. RAG Triad of metrics
nav_order: 3
description: "Computational challenges of training LLMs"
has_children: false
parent:  Deeplearning RAG
---

In this lesson, we do a deep dive into evaluation. We'll walk you through some core concepts on how to evaluate rag systems. Specifically, we will introduce the rag triad, a triad of metrics for the three main steps of Iran's execution, context, relevance, groundedness and answer relevance. These are examples of an extensible framework of feedback functions, programmatic evaluations of LLM apps. 

We then show you how to synthetically generate an evaluation data set. Given any unstructured corpus. Let's get started. Now I'll use a notebook to walk you through the rag triad answer relevance, context relevance and groundedness to understand how each can be used with Trulens to detect hallucinations. 

At this point, you have already people installed TruLens and LLamaIindex. So I'll not show you that step. The first step for you will be to set up and OpenAI API key. The OpenAI key is used for the completion step of the rag and to implement the evaluations with TruLens. So here's a code snippet that does exactly that and we are now all set up with the OpenAI key. 


```python
import utils
import os
import openai
openai.api_key = utils.get_openai_api_key()
```

The next section, I will quickly recap the query engine construction with LlAMAIndex. In lesson one in some detail, we will largely build on that lesson. The first step now is to set up a `tru`` object from TruLens. We are going to import the `Tru` class, then we'll set up a `tru` object and instance of this class. 

```python
from trulens_eval import Tru

tru = Tru()
tru.reset_database()
```

    🦑 Tru initialized with db url sqlite:///default.sqlite .
    🛑 Secret keys may be written to the database. See the `database_redact_keys` option of `Tru` to prevent this.

And then this object will be used to reset the database. This database will be used later on to record the prompts responses, intermediate results of the Lama index app as well as the results of the various evaluations. We will be setting up with Trulens. Now let's set up the LlamaIndex reader. This snippet reads this PDF document from a directory on *how to build a career in AI written* by Andrew and then loads this data into this document object.

```python
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()
```
The next step is to merge all of this content into a single large document rather than having one document for each page, which is the default set up. 

```python
from llama_index import Document

document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))
```

Next, we set up the sentence index leveraging some of the llama index utilities. So you can see here that we are using OpenAI GPT 3.5 turbo set at a temperature of 0.1 as the LLM that will be used for completion of the rag. The embedding model is set to BG small and version 1.5. 


```python
from utils import build_sentence_window_index
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)
```
You will see the following output as the sentence index is being setup.

    config.json: 100%
    743/743 [00:00<00:00, 75.6kB/s]
    model.safetensors: 100%
    133M/133M [00:01<00:00, 81.2MB/s]
    tokenizer_config.json: 100%
    366/366 [00:00<00:00, 52.5kB/s]
    vocab.txt: 100%
    232k/232k [00:00<00:00, 3.29MB/s]
    tokenizer.json: 100%
    711k/711k [00:00<00:00, 3.66MB/s]
    special_tokens_map.json: 100%
    125/125 [00:00<00:00, 17.1kB/s]

And all of this content is being indexed with the `sentence_index` object. Next we set up the sentence window engine. And this is the query engine that will be used later on to do retrieval effectively from this advanced rag application. 

```python
from utils import get_sentence_window_query_engine

sentence_window_engine = \
get_sentence_window_query_engine(sentence_index)
```

Now that we have set up the query engine for sentence window based rag. 
Let's see it in action by actually asking a specific question. 


```python
output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response
```

*How do you create your AI portfolio?* This will return a full object with the final response from the LLM the intermediate pieces of retrieved context as well as some additional metadata.


```python
output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response
```

Response generated is shown below

    'To create your AI portfolio, you should focus on building a collection of projects that demonstrate your skill progression. This can be achieved by starting with simpler projects and gradually increasing the complexity as you gain more experience. Additionally, it is important to ensure that your portfolio showcases a variety of AI techniques and applications to highlight your versatility and breadth of knowledge. By following this approach, you can effectively showcase your skills and attract potential employers in the field of AI.'

Let's take a look at what the final response looks like. So here you can see the final response that came out of this sentence window based rag. 

It provides a pretty good answer on the surface to this question of how do you create your A I portfolio? Later on, we will see how to evaluate answers of this form against the rag triad to build confidence and identify failure modes for rags of this form.

## Feedback functions



Now that we have an example of a response to this question that looks quite good on the surface, we will see how to make use of feedback functions such as the rag triad to evaluate this kind of response. More deeply identify failure modes as well as build confidence or iterate to improve the LLM application. Now that we have set up the sentence window based drag application. Let's see how we can evaluate it with the rag triad. We'll do a little bit of housekeeping in the beginning. First step is this piece of code snippet that lets us launch a stream lid dashboard from inside the notebook. You'll see later that we'll make use of that dashboard to see the results of the evaluation and to run experiments to look at different choices of apps and to see which one is doing better.

```python
import nest_asyncio

nest_asyncio.apply()
```


```python
from trulens_eval import OpenAI as fOpenAI

provider = fOpenAI()
```

Next up, we initialize OpenAI GPT 3.5 turbo as the default provider for our evaluations. And this provider will be used to implement the different feedback functions or evaluations such as context, relevance, sponsor, relevance and groundedness. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_9.51.37 AM.png"  width="70%" />


Now let's go deeper into each of the evaluations of the rag triad. And we'll go back and forth a bit between slides and the notebook to give you the full context. 

## 1. Answer Relevance

First up, we'll discuss answer relevance. Recall that answer relevance is checking whether the final response is relevant to the query that was asked by the user to give you a concrete example of what the output of answer relevance might look like. 





Here is an example. The user asked the question, how can altruism be beneficial in building a career? This was the response that came out of the rag application. And the answer I relevance evaluation produces two pieces of output. One is a score on a scale of 0 to 1. The answer was assessed to be highly relevant. So it got a score of 0.9. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_9.51.51 AM.png"  width="60%" />
<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_9.52.12 AM.png"  width="70%" />
<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_9.57.28 AM.png"  width="70%" />

The second piece is the supporting evidence or the rationale or the chain of thought reasoning behind why the evaluation produced this score. So here you can see that supporting evidence found in the answer itself which indicates to the LM evaluation that it is a meaningful and relevant answer. I also want to use this opportunity to introduce the abstraction of a feedback function answer. Relevance is a concrete example of a feedback function. More generally a feedback function provides a score on a scale of 0 to 1. After reviewing an LLM apps, inputs, outputs and intermediate results. Let's now look at the structure of feedback functions using the answer relevance feedback function. As a concrete example, the first component is a provider. And in this case, we can see that we are using an LLM from OpenAI to implement these feedback functions. Note that feedback functions don't have to be implemented necessarily using LMS. We can also use birth models and other kinds of mechanisms to implement feedback functions that I'll talk about in some more detail later in the lesson. The second component is that leveraging that provider, we will implement a feedback function. In this case. That's the relevance feedback function. We give it a name, a human readable name that will be shown later in our evaluation dashboard. And for this particular feedback function, we run it on the user input the user query and it also takes as input the final output or response from the app. So given the user question and the final answer from the rag, this feedback function will make use of an LLM provider such as OpenAI GPT 3.5 to come up with a score for how relevant the responses to the question that was asked. And in addition, it'll also provide supporting evidence or chain of thought reasoning for the justification of that score. Let's now switch back to the notebook and look at the code in some more detail. 


Now let's see how to define the question answer relevance feedback function in court from Truls. Well, we will import the feedback loss. Then we set up the different pieces of the question answer relevance function that we were just discussing. First up, we have the provider that is opening OpenAI GPT 3.5. And we set up this particular feedback function where the relevant score will also be augmented with the chain of thought reasoning much like I showed in the slides, we give this feedback function a human understandable name we call it answer relevance. This will be show up later in the dashboard making it easy for users to understand what the feedback function is setting up. Then we also will give the feedback function access to the input, that is the prompt and the output (using `on_input_output()` function), which is the final response coming out of the rag application. 

```python
from trulens_eval import Feedback

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

```

With this set up later on in the notebook, we will see how to apply this feedback function on a set of records, get the evaluation scores for answer relevant as well as the chain of thought reasons for why for that particular answer that was the judged score to be appropriate for as part of the evaluation. 


## 2. Context Relevance

The next feedback function that we will go deep into is context relevance. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_2.02.35 PM.png"  width="80%" />

Recall that context relevance is checking how good the retrieval processes that is given a query. We will look at each piece of retrieved context from the vector database and assess how relevant that piece of context is to the question that was asked, let's look at a simple example. The question here or the prompt from the user is how can altruism be beneficial in building a career? These are the two pieces of retrieve context. 

And after the evaluation with context relevance, each of these pieces of retrieve context gets a score between zero and one. You can see here. The left context got a relevant score of 0.5. The right context got a relevant score of 0.7. So it was assessed to be more relevant to this particular query. And then the mean context relevant score is the average of the relevant scores of each of these retrieved pieces of context that gets also reported out. Let's now look at the structure of the feedback function for context relevance. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_2.03.00 PM.png"  width="80%" />

Various pieces of this structure are similar to the structure for S elements which we reviewed a few minutes ago. There is a provider that's OpenAI. And the feedback function is makes use of that provider to implement the context relevance feedback function. The differences are in the inputs to this particular feedback function. In addition to the user input or prompt, we also share with this feedback function. A pointer to the retrieve contexts that is the intermediate results in the execution of the rag application (referred to as `context_selection`). We get back a score for each of the retrieved pieces of context assessing how relevant or good that context is with respect to the query that was asked and then we aggregate and average those scores across all the retrieve pieces of context to get the final score. Now you will notice that in the answer relevance feedback function, we had only made use of the original input, the prompt and the final response from the rag. In this feedback function, we are making use of the input or prompt from the user as well as intermediate results, the set of retrieve contexts to assess the quality of the retrieval between these two examples. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_2.03.46 PM.png"  width="80%" />

The full power of feedback functions is leveraged by making use of inputs, outputs and intermediate results of a rag application to assess its quality. Now that we have the context selection set up, we are in a position to define the context, relevance feedback function in code. You'll see that it's pretty much the code segment that I walked through on the slide. We are still using OpenAI as the provider GPT 3.5 as the evaluation LLM we are calling the question statement or context relevance feedback function. It gets the input prompt, the set of retrieved pieces of context. It runs the evaluation function on each of those retrieve pieces of context separately, gets a score for each of them and then averages them to report a final aggregate score. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_2.04.05 PM.png"  width="80%" />
<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_2.19.22 PM.png" width="80%" />

```python
from trulens_eval import TruLlama

context_selection = TruLlama.select_source_nodes().node.text
```


```python
import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)
```

Now, one additional variant that you can also use if you like is in addition to reporting a context relevant score for each piece of retrieve context, you can also augment it with chain of thought reasoning so that the evaluation LLM provides not only a score but also a justification or explanation for its assessment score. And that can be done with uss relevance with chain of thought reasoning method. 

```python
import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)
```


```python
import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)
```

## Groundedness

And if I give you a concrete example of this in action, you can see here's the question or the user prompt, how can altruism be beneficial in building a career? This is an example of a retrieved piece of context that takes out a chunk from Andrews article on this topic. You can see the context relevant feedback function gives a score of 0.7 on a scale of 0 to 1 to this piece of retrieved context. And because we have also invoked the chain of thought reasoning on the evaluation LM, it provides this justification for why the score is 0.7. Let me now show you the code snippet to set up the groundedness feedback function, 

we kick it off in much the same way as the previous feedback functions, leveraging LM provider for evaluation, which is if you recall open A I GPD 3.5 then we define the groundedness feedback function. 

This definition is structurally very similar to the definition for context relevance. The groundedness measure comes with chain of thought reasons, justifying the scores. Much like I discussed on the slides, we gave it the name groundedness which is easy to understand. It gets access to the set of retrieved contexts (refer `context_selection`) in the rag application, much like for context relevance as well as the final output or response from the rag. Each sentence in the final response gets a groundedness score and those are aggregated averaged to produce the final grounded test score for the full response using `aggregate(grounded.grounded_statements_aggregator)` function call.

```python
from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=provider)
```


```python
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
```

The context election here is the same context selection that was used for setting up the context relevance feedback function. If you recall that just gets the set of retrieved pieces of context from the retrieval step of the rag and then can access each node within that list, recover the text of the context from that node and proceed to work with that to do the context relevance as well as the groundedness evaluation win that we are now in a position to start executing the evaluation of the rag application. 

We have set up all three feedback functions *answer relevance*, *context relevance* and *groundedness*.  All we need is an evaluation set on which we can run the application and the evaluations and see how they're doing. And if there are opportunities to iterate and improve them further, lets now look at the workflow to evaluate and iterate to improve LM applications. 


We will start with the basic Lama index rag that we introduced in the previous lesson and which we have already evaluated with the true lens drag triad. We'll focus a bit on the failure modes related to the context size. Then we will iterate on that basic rag with an advanced drag technique, the Lama index sentence wind R.



Next, we will re evaluate this new advanced drag with the TruLens rag triad focusing on these kinds of questions. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.09.20 PM.png"  width="80%" />



Do we see improvements specifically in context relevance? What about the other metrics? The reason we focus on context relevance is that often failure modes arise because the context is too small. Once you increase the context up to a certain point, you might see improvements in context relevance. In addition, when context relevance goes up, often we find improvements in groundedness as well because the LM in the completion step has enough relevant context to produce the summary. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.09.37 PM.png"  width="80%" />
<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.09.45 PM.png"  width="80%" />

When it does not have enough relevant context, it tends to leverage its own internal knowledge from the pretraining data set to try to fill those gaps which results in a loss of groundedness.
<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.10.06 PM.png"  width="80%" />


Finally, we will experiment with different window sizes to figure out what window size results in the best evaluation metrics. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.11.23 PM.png"  width="80%" />

Recall that if the window size is too small, there may not be enough relevant context to get a good score on context relevance and groundedness. If the window size becomes too big. On the other hand, irrelevant context can creep into the final response resulting in not such great scores in groundedness or answer elev. We walked through three examples of evaluations or feedback functions, context, relevance answer, relevance and groundedness. 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.12.30 PM.png"  width="80%" />

In our notebook. All three were implemented with LLM evaluations. I do want to point out that feedback functions can be implemented in different ways. Often we see practitioners starting out with ground truth valves which can be expensive to collect but nevertheless, a good starting point. We also see people leverage humans to do evaluations that's also helpful and meaningful but hard to scale in practice ground truth evals just to give you a concrete example, think of a summarization use case where there's a large passage and then the LLM produces a summary, a human expert would then give that summary a score indicating how good it is. This can be used for other kinds of use cases as well such as chatbot, like use cases or even classification use cases. Human valves are similar in some ways to ground through the valves. In that as the LLM produces an output or a rag application produces an output, the human users of that application are going to provide a rating for that output. How good it is. The difference with ground truth develops is that these human users may not be as much of an expert in the topic as the ones who produced the curated ground wells. It's nevertheless a very meaningful evaluation. It'll scale a bit better then the ground dels. But our degree of confidence in it is lower. One very interesting result from the research literature is that if you ask a set of humans to radar question, there's about 80% agreement. And interestingly enough, when you use lens for evaluation, the agreement between the LM evaluation and the human evaluation is also about the 80 to 85% mark. So that suggests that LLM evaluations are quite comparable to human evaluations for the benchmark data data sets to which they have been applied. So feedback functions provide us a way to scale up evaluations in a programmatic manner. In addition to the Lalami valves that you have seen feedback functions also provide can, can implement traditional and LP metrics such as rouge scores and blue scores, they can be helpful in certain scenarios. But one weakness that they have is that they are quite syntactic. They look for overlap between words in two in two pieces of text. So for example, if you have one piece of text that's referring to a riverbank and the other to a financial bank syntactically, they might be viewed as similar and these references might end up being viewed as similar references by a traditional NLP evaluation. Whereas the surrounding context will get used to provide a more meaningful evaluation when you're using either large language models such as GPD four or medium sized language models such as bur models and to perform your evaluation. While in the course, we have given you three examples of feedback functions and evaluations answer relevance, context, relevance and groundedness Trance provides a much broader set of evaluations to ensure that the apps that you're building are honest harmless and helpful. These are all available in the open source library and we encourage you to play with them as you are working through the course and building your LM applications. 

Now that we have set up all the feedback functions, we can set up an object to start recording which will be used to record the execution of the application on various records. So you'll see here that we are importing this Truelama class, creating an object through recorder of this true LMA class. 

```python
from trulens_eval import TruLlama
from trulens_eval import FeedbackMode

tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)
```
This is our integration of TrueLens with LlamaIndex. It takes in the sentence window engine from Lama index that we had created earlier sets the APP I and makes use of the three feedback functions of the rag triad that we created earlier. This true recorder object will be used in a little bit to run the LMA index application as well as the evaluation of these feedback functions and to record it all in a local database. Let us now load some evaluation questions in the set up. The evaluation questions are set up already in this text file and then we just execute this code stet to load them in. 

```python
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)
```

Let's take a quick look at these questions that we will use for evaluation can see what are the keys to building a career in a eye and so on. And this file, you can edit yourself and add your own questions that you might want to get answers from Andrew. Or you can also append directly to the eval questions list in this way.

```python
eval_questions
```
    ['What are the keys to building a career in AI?',
    'How can teamwork contribute to success in AI?',
    'What is the importance of networking in AI?',
    'What are some good habits to develop for a successful career?',
    'How can altruism be beneficial in building a career?',
    'What is imposter syndrome and how does it relate to AI?',
    'Who are some accomplished individuals who have experienced imposter syndrome?',
    'What is the first step to becoming good at AI?',
    'What are some common challenges in AI?',
    'Is it normal to find parts of AI challenging?']

```python
eval_questions.append("How can I be successful in AI?")
```


```python
eval_questions
```
    ['What are the keys to building a career in AI?',
    'How can teamwork contribute to success in AI?',
    'What is the importance of networking in AI?',
    'What are some good habits to develop for a successful career?',
    'How can altruism be beneficial in building a career?',
    'What is imposter syndrome and how does it relate to AI?',
    'Who are some accomplished individuals who have experienced imposter syndrome?',
    'What is the first step to becoming good at AI?',
    'What are some common challenges in AI?',
    'Is it normal to find parts of AI challenging?',
    'How can I be successful in AI?']

```python
for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)
```



 Now let's take a look at the questions list and you can see that this question has been added at the end. Go ahead and add your own questions. And now we have everything set up to get to the most exciting step in this notebook with this code, skip it, we can execute the sentence window engine on each question in the list of Eval questions that we just looked at. And then with `TrueRecorder`, we are going to run each record against the rag triad. We will record the prompts responses, intermediate results and the evaluation results in the true database. And you can see here as each, as the execution of the steps are happening for each record. There is a hash that's an identifier for the record as the record gets added, we have an indicator here that that step has executed effectively. In addition, the feedback results or answer relevance is done and so on for context relevance and so on. 

Now that we have the recording done, we can see the logs in the notebook by executing, by getting the records and feedback and executing this code snippet. And I don't want you to necessarily read through all of the information here. The main point I want to make is that you can see the depth of instrumentation in the application. A lot of information gets logged through the true recorder and this information around prompts responses, evaluation results and so forth can be quite valuable to identify failure modes in the apps and to inform iteration and improvement of the apps. All of this information is available in a flexible adjacent format so they can be exported and consumed by downstream processes.



```python
records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()
```


```python
import pandas as pd

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]
```

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_7.02.15 PM.png"  width="120%" />

Next up. Let's look at some more human readable format for prompts responses and the feedback function evaluations with the cold stamped. You can see that for each input gram or question, we see the output and the respective scores for context, relevance, groundedness and answer relevance. 


And this is run for each and every entry in the list of questions in evaluations underscore questions dot text. And you can see here the last question is how can I be successful in AI was the question that I manually appended to that list at the end sometimes in running the evaluations, you might CNN that likely happens because of hepi all failures. You'll just want to rerun it to ensure that the execution successfully completes. I just showed you a record level view of the evaluations, the prompts, responses and evaluations. 

### Aggregate View in the Leader board

Let's now get an aggregate view in the leader board which aggregates across all of these individual records and produces an average score across the 10 records in that database. So you can see here in the leaderboard, the aggregate view across all the 10 records we had sent the app to app one, the average context relevance is 0.56. Similarly, their average scores for groundedness answer relevance and latency across all the 10 records of questions that were asked of the rag application. And then the cost is the total cost in dollars across these 10 records. 
```python
tru.get_leaderboard(app_ids=[])
```



### Aggregate View of the Scores

It's useful to get this aggregate view to see how well your app is performing and at what level of latency and cost. In addition to the notebook interface 

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.33.18 PM.png"  width="120%" />

**TruLens** also provides a local stream lit app dashboard with which you can examine the applications that you're building. 

Look at the evaluation results drill down into record level views to both get aggregate and detailed evaluation views into the performance of your app. So we can get the dashboard going with the true dot on dashboard method and this sets up a local database at a certain ul. Now once I click on this, this might show up in some window which is not within this range. 

Let's take a few minutes to walk through this dashboard. You can see here the aggregate view of the app's performance 11 records were processed by the app and evaluated. The average latency is 3.55 seconds. We have the total cost, the total number of tokens that were processed by the LMS and then scores for the rag triad for context relevance. It's 0.56 for groundedness 0.86 and answer relevant is 0.92. We can select the app here to get a more detailed record level view of the evaluations for each of the records. You can see that the user input the prompt, the response, there's meta the timestamp and then scores for answer relevance, context relevance and groundedness that have been recorded along with latency, total number of tokens and total cost. Let me pick a role in which the LM indicates evaluation indicates that the LM the rag application has done well speak this row. Once we click on a row, we can scroll down and get a more detailed view of the different components of that rule from the table. So the question here, the prompt was, what is the first step to becoming good at AI? The final response from the rag was has to learn foundational technical skills down here. You can see that the answer relevance was viewed to be one on a scale of 0 to 1 and so relevant. Quite a relevant answer to the question that was asked up here. You can see that context relevance, the average context relevance score is 0.8. For the two pieces of context that were retrieved, both of them individually got scores of 0.8. We can see the chain of thought reason for why the LM evaluation gave a score of 0.8 to this particular a response from the rag and in the retrieval step. And then down here, you can see the groundedness evaluations. So this was one of the clauses and the final answer. Uh It got a score NF one and over here is the reason for that score, you can see this was the statement sentence and the supporting evidence backs it up. And so it got a full score of one on a scale of 0 to 1 or a full score of 10 on a scale of 0 to 10. So previously, the kind of reasoning and information we were talking about through slides and in the notebook. 


```python
tru.run_dashboard()
```

    Starting dashboard ...
    Config file already exists. Skipping writing process.
    Credentials file already exists. Skipping writing process.
    Accordion(children=(VBox(children=(VBox(children=(Label(value='STDOUT'), Output())), VBox(children=(Label(valu…
    Dashboard started at https://s172-31-7-241p21235.lab-aws-production.deeplearning.ai/ .
    <Popen: returncode: None args: ['streamlit', 'run', '--server.headless=True'...>

Now you can see that quite neatly in this streamlit local app that runs on your machine, you can also get a detailed view of the timeline as well as get access to the full Jason object. Now let's look at an example where the rag did not do so well. So as I look through the evaluations, 



<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_7.04.12 PM.png"  width="120%" />

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.48.35 PM.png"  width="120%" />



I see this row with a low groundedness score of 0.5. So let's click on that. That brings up this example. The question is how can altruism be beneficial in building a career? There's a response. If I scroll down to the groundedness evaluation, then both of the sentences and the final response have low groundedness score. Let's pick one of these and look at why the grounded this score is low. So you can see this, the overall response got broken down into four statements and the top two were good, but the bottom two did not have good supporting evidence in the retrieve pieces of context and particular. If you look at this last one, the final output from the lab says additionally, practicing altruism can contribute to personal fulfillment in a sense of purpose which can enhance motivation and overall well being ultimately benefiting one's career success. While that might very well be the case, there was no supporting evidence found in the retrieved pieces of context to ground that statement. And that's why our evaluation gives this a low score. 


<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.48.45 PM.png"  width="120%" />

<img src="/deeplearningai/building-evaluating-advanced-rag/images/Screenshot_2023-12-26_at_6.49.01 PM.png"  width="120%" />

You can play around with the dashboard and explore some of these other examples where the the final rag output uh does not do so well to get a feeling for the kinds of failure modes that are quite common when you're using rag applications. And some of these will get addressed as we go into the sessions on more advanced drag techniques which can do better in terms of addressing these failure modes. Lesson two is a wrap with that. In the next lesson, we will walk through the mechanism for sentence window based retrieval and advanced track technique. And also show you how to evaluate the V technique leveraging the rag triad and true lance.
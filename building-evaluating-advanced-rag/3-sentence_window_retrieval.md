
# Lesson 3: Sentence Window Retrieval

```python
import warnings
warnings.filterwarnings('ignore')
````

```python
import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()
```

```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()
```

```python
print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])
```

```
<class 'list'> 

41 

<class 'llama_index.schema.Document'>
Doc ID: cb806dcd-c367-4273-8976-9ab58cc1755f
Text: PAGE 1Founder, DeepLearning.AICollected Insights from Andrew Ng
How to  Build Your Career in AIA Simple Guide
```

```python
from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))
```

## Window-sentence retrieval setup

### Introducing Sentence Window Retrieval

Our **sentence window retrieval method** is an advanced RAG (Retrieval Augmented Generation) technique. This approach involves retrieving information based on smaller units, specifically individual sentences, to achieve a more precise match with the relevant context. Subsequently, the retrieved sentences are used to synthesize a response, drawing from an expanded context window that surrounds the original sentence. Let's explore how to set this up.

### Understanding the Need for Sentence Window Retrieval

In a standard RAG pipeline, the same text chunk is typically utilized for both embedding (creating numerical representations for search) and synthesis (generating the answer). A challenge arises because **embedding-based retrieval generally performs better with smaller text chunks**, while a Large Language Model (LLM) requires a broader and larger context to generate a comprehensive and high-quality answer.


**Sentence window retrieval** addresses this by decoupling these two processes. We begin by embedding smaller chunks, or individual sentences, and storing them in a vector database. Crucially, we also augment each of these sentence chunks with the context of sentences that appear before and after them. During the retrieval phase, we use a similarity search to identify the sentences most relevant to the user's question. Then, the retrieved sentence is replaced with its full surrounding context. This mechanism enables us to expand the context that is ultimately fed to the LLM, leading to more robust and accurate answers.

### Constructing a Sentence Window Retriever with LlamaIndex

This guide will walk through the essential components required to build a **sentence window retriever using LlamaIndex**. Each component will be explored in detail.

Later, Anupam will demonstrate how to experiment with different parameters and perform evaluations using **TruEra**. This setup is consistent with previous lessons, so ensure you install the necessary packages like LlamaIndex and TruLens Eval. For this quick start, an OpenAI API key is required, similar to prior lessons, as it's used for embeddings, LLMs, and the evaluation process.

### Document Preparation

We'll now prepare and examine the documents for iteration and experimentation. You're encouraged to upload your own PDF file, similar to the first lesson.


As in previous sessions, we'll load the "How to Build a Career in AI" eBook. This is the same document as before. We observe that it's a list of documents, comprising 41 pages with a document object schema. A sample of the text from the first page is also available. Subsequently, these will be merged into a single document, which enhances overall text blending accuracy when more advanced retrievers are employed.

### Setting Up Sentence Window Retrieval

Now, let's configure the **sentence window retrieval method**, delving into its setup in more detail. We'll begin by setting a **window size of 3** and a **top-K value of 6**. The first step involves importing the `SentenceWindowNodeParser`. This object is responsible for splitting a document into discrete sentences and then enriching each sentence chunk with its surrounding contextual information.

```python
from llama_index.node_parser import SentenceWindowNodeParser

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
```

```
[nltk_data] Downloading package punkt to /tmp/llama_index...
[nltk_data]   Unzipping tokenizers/punkt.zip.
```

#### Demonstrating the Node Parser

To illustrate, let's see how the node parser functions with a brief example. If we have a text comprising three sentences, it gets segmented into three distinct nodes. Each node encapsulates a single sentence, with its associated metadata providing a larger contextual window around that sentence. Examining the metadata for the second node reveals that it includes not only the original sentence but also the sentences immediately preceding and succeeding it.

```python
text = "hello. how are you? I am fine!  "

nodes = node_parser.get_nodes_from_documents([Document(text=text)])
```

```python
print([x.text for x in nodes])
```

```
['hello. ', 'how are you? ', 'I am fine!  ']
```

```python
print(nodes[1].metadata["window"])
```

```
hello.  how are you?  I am fine!  
```

We encourage you to experiment with your own text as well. Consider a sample text: for its first node, with a window size of 3, the surrounding metadata would include two additional adjacent nodes occurring before it (and none after, as it's the initial node). Thus, the metadata would contain the original sentence ("hello") along with "foobar" and "cat dog."

```python
text = "hello. foo bar. cat dog. mouse"

nodes = node_parser.get_nodes_from_documents([Document(text=text)])
```

```python
print([x.text for x in nodes])
```

```
['hello. ', 'foo bar. ', 'cat dog. ', 'mouse']
```

```python
print(nodes[0].metadata["window"])
```

```
hello.  foo bar.  cat dog. 
```

### Building the index

The subsequent phase involves constructing the index. We'll begin by configuring the LLM. For this purpose, we'll employ **OpenAI's GPT-3.5 Turbo** with a temperature setting of 0.1. Following this, we set up a **service context object**. This is a wrapper containing all necessary indexing contexts, including the LLM, embedding model, and node parser. It's worth noting that the specified embedding model is the **"bge small model,"** which is downloaded and run locally from HuggingFace. This model is recognized for being compact, fast, and accurate given its size. Alternative embedding models, such as "bge large," are also available and shown in the commented-out code.

```python
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-4", temperature=0.1)
```

```python
from llama_index import ServiceContext

sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    # embed_model="local:BAAI/bge-large-en-v1.5"
    node_parser=node_parser,
)
```


The next step is to set up the **VectorStoreIndex** using the source document. Since the node parser has been defined within the service context, this process will transform the source document into a collection of sentences, augmented with their surrounding contexts. These augmented sentences are then embedded and loaded into the VectorStore. The index can be saved to disk, allowing for later loading without the need for rebuilding. A provided code snippet facilitates loading an existing index if available, or building it otherwise. Once these steps are complete, the index is built.

```python
from llama_index import VectorStoreIndex

sentence_index = VectorStoreIndex.from_documents(
    [document], service_context=sentence_context
)
```

```python
sentence_index.storage_context.persist(persist_dir="./sentence_index")
```

```python
# This block of code is optional to check
# if an index file exist, then it will load it
# if not, it will rebuild it

import os
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import load_index_from_storage

if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(
        [document], service_context=sentence_context
    )

    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./sentence_index"),
        service_context=sentence_context
    )
```

### Building the postprocessor

The next phase involves configuring and executing the **query engine**. Initially, we'll define a **metadata replacement post-processor**. This processor takes a value from the metadata and uses it to replace the text of a node. This operation occurs after the nodes have been retrieved but before they are sent to the LLM.

```python
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)
```

#### Post-Processor Demonstration


To illustrate its function, we'll demonstrate how this post-processor works using nodes previously created with the sentence window node parser (a backup of the original nodes was made). After applying the post-processor to these nodes, an inspection of the second node's text reveals that it has been replaced with its full context, encompassing the sentences that appeared both before and after it.

```python
from llama_index.schema import NodeWithScore
from copy import deepcopy

scored_nodes = [NodeWithScore(node=x, score=1.0) for x in nodes]
nodes_old = [deepcopy(n) for n in nodes]
```

```python
nodes_old[1].text
```

```
'foo bar. '
```

```python
replaced_nodes = postproc.postprocess_nodes(scored_nodes)
```

```python
print(replaced_nodes[1].text)
```

```
hello.  foo bar.  cat dog.  mouse
```

### Adding a reranker


The subsequent step involves integrating a **sentence transformer re-rank model**. This model processes the query and the initially retrieved nodes, then reorders them based on relevance using a specialized model. Typically, the initial similarity search yields a larger "top K" set, which the re-ranker then rescores to return a more focused and smaller "top N" set. An example of such a re-ranker is the "bge-re-ranker," derived from BGE embeddings. The model's name is a string representing its identifier on HuggingFace, where further details can be found.

```python
from llama_index.indices.postprocessor import SentenceTransformerRerank

# BAAI/bge-reranker-base
# link: [https://huggingface.co/BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)
```

```
config.json:   0%|          | 0.00/799 [00:00<?, ?B/s]

model.safetensors:   0%|          | 0.00/1.11G [00:00<?, ?B/s]

tokenizer_config.json:   0%|          | 0.00/443 [00:00<?, ?B/s]

sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]

tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]

special_tokens_map.json:   0%|          | 0.00/279 [00:00<?, ?B/s]
```

#### Re-Ranker Demonstration


To illustrate its functionality, let's examine how this re-ranker operates. We'll provide some sample data to observe how it reorders an initial set of nodes into a newly ranked set. Consider an original query: "I want a dog." Imagine an initial set of scored nodes: "this is a cat" with a score of 0.6, and "this is a dog" with a score of 0.4. Intuitively, the second node ("this is a dog") should have a higher relevance score, aligning more closely with the query. This is precisely where the re-ranker proves valuable. In this scenario, the re-ranker correctly surfaces the node related to "dogs," assigning it a higher score of relevance.

```python
from llama_index import QueryBundle
from llama_index.schema import TextNode, NodeWithScore

query = QueryBundle("I want a dog.")

scored_nodes = [
    NodeWithScore(node=TextNode(text="This is a cat"), score=0.6),
    NodeWithScore(node=TextNode(text="This is a dog"), score=0.4),
]
```

```python
reranked_nodes = rerank.postprocess_nodes(
    scored_nodes, query_bundle=query
)
```

```python
print([(x.text, x.score) for x in reranked_nodes])
```

### Running the query engine

#### Applying to the Query Engine


Now, let's integrate this into our actual query engine. As previously noted, it's beneficial to have a larger **similarity top K** value than the **top N** value selected for the re-ranker. This strategy allows the re-ranker a better opportunity to surface the most pertinent information. We'll configure the **top K to 6** and the **top N to 2**. This implies that we will initially retrieve the six most similar chunks using sentence window retrieval, and then the sentence re-ranker will further filter these down to the two most relevant chunks.

```python
sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=6, node_postprocessors=[postproc, rerank]
)
```

#### Running a Basic Example


With the entire query engine now configured, let's execute a fundamental example. We'll pose a question to our dataset: "What are the keys to building a career in AI?" The system provides a response, which indicates that the keys to building an AI career involve acquiring foundational technical skills, engaging in projects, and securing a job.

```python
window_response = sentence_window_engine.query(
    "What are the keys to building a career in AI?"
)
```

```python
from llama_index.response.notebook_utils import display_response

display_response(window_response)
```

## Putting it all Together

With the sentence window query engine established, let's consolidate all the components. Although this notebook cell will contain a substantial amount of code, it essentially mirrors the functionality of the `utils.BAAI` file.


Functions are available for constructing the sentence window index, as demonstrated earlier. This process involves using the **sentence window node parser** to extract sentences from documents and enrich them with surrounding contexts. It also includes setting up the **service context object**, which contains the necessary LLM embedding model and node parser, and then configuring a **vector store index** with the source documents.

```python
import os
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage


def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine
```

The second part involves obtaining the **sentence window query engine**. This was shown to comprise getting the sentence window retriever, employing the **metadata replacement post-processor** to substitute a node with its surrounding context, and finally, utilizing a **re-ranking module** to filter for the top N results. All these elements are combined using the `as_query_engine` module.


First, we'll invoke the `build_sentence_window_index` function with the source document and the specified save directory. Subsequently, we'll call the second function to retrieve the sentence window query engine. With these steps complete, you are now prepared to experiment with sentence window retrieval.

```python
from llama_index.llms import OpenAI

index = build_sentence_window_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./sentence_index",
)
```

```python
query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
```

## TruLens Evaluation


In the upcoming section, Anupam will demonstrate how to conduct evaluations using the **sentence window retriever**. This will allow you to assess results, adjust parameters, and observe their impact on the engine's performance. After reviewing these examples, we encourage you to incorporate your own questions and even establish custom evaluation benchmarks to further explore and understand the system's behavior.

```python
eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)
```

Thank you, Jerry. Now that the sentence window retriever is configured, let's explore its evaluation using the **RAG triad** and compare its performance against a basic RAG setup, complete with experiment tracking.

```python
from trulens_eval import Tru

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)
```

```python
from utils import get_prebuilt_trulens_recorder

from trulens_eval import Tru

Tru().reset_database()
```

```
🦑 Tru initialized with db url sqlite:///default.sqlite .
🛑 Secret keys may be written to the database. See the `database_redact_keys` option of `Tru` to prevent this.
```

### Evaluating and Iterating on Sentence Window Size


We will now examine how to evaluate and iterate on the **sentence window size parameter**. The goal is to identify the optimal balance between crucial evaluation metrics (application quality) and the operational costs associated with running the application and its evaluations. Our approach will involve progressively increasing the **sentence window size, starting from 1**, evaluating each successive application version using **TruLens and the RAG triad**, and meticulously tracking experiments to pinpoint the most effective sentence window size. Throughout this exercise, we will carefully observe the trade-offs concerning token usage and associated costs.

#### Impact of Window Size on Performance and Cost


As the window size increases, **token usage and cost will predictably rise**. In many scenarios, context relevance will also improve. Initially, increasing the window size is expected to enhance context relevance, which, in turn, indirectly boosts groundedness. This is because if the retrieval step fails to provide sufficiently relevant context, the LLM in the completion phase may resort to its pre-existing knowledge from its training data, rather than exclusively relying on the retrieved information. This can lead to lower groundedness scores, as groundedness requires that all components of the final response be traceable to the retrieved context.

Consequently, we anticipate that as the sentence window size is incrementally increased, **context relevance and groundedness will improve up to a certain threshold**. Beyond this point, context relevance is expected to either stabilize or decline, with groundedness likely following a similar trend.

#### Interplay Between Context Relevance and Groundedness


Furthermore, a notable relationship exists between **context relevance and groundedness**. When context relevance is low, groundedness also tends to be low. This occurs because the LLM often attempts to bridge gaps in retrieved information by drawing upon its pre-trained knowledge, which can reduce groundedness even if the answers appear relevant. As context relevance increases, groundedness typically rises up to a certain point. However, if the context size becomes excessively large, even with high context relevance, groundedness might decline. This is because the LLM can become overwhelmed by overly extensive contexts during the completion step, leading it to revert to its pre-existing knowledge base from its training phase.

### Experimenting with Sentence Window Size


Let's proceed with experimenting with the **sentence window size**. I'll guide you through a notebook that involves loading a selection of questions for evaluation. We will then progressively increase the sentence window size and observe its impact on the **RAG triad evaluation metrics**.

#### Initial Setup and Evaluation (Window Size 1)


First, we load a pre-generated set of evaluation questions, a sample of which is displayed. Next, evaluations are run for each question in this set. Using the **TruRecorder object**, prompts, responses, intermediate application results, and evaluation outcomes are all recorded in the TruLens database.

```python
eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)
```

```python
from trulens_eval import Tru

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)
```

```python
from utils import get_prebuilt_trulens_recorder

from trulens_eval import Tru

Tru().reset_database()
```

```
🦑 Tru initialized with db url sqlite:///default.sqlite .
🛑 Secret keys may be written to the database. See the `database_redact_keys` option of `Tru` to prevent this.
```

Now, we'll adjust the **sentence window size parameter** to observe its effect on the RAG triad evaluation metrics. We begin by resetting the TruLens database. With a provided code snippet, the sentence window size is set to **1**. All other settings remain consistent with previous configurations. We then configure the sentence window engine using `get_sentence_window_query_engine` associated with the index. Subsequently, the TruRecorder is set up with the sentence window size at 1, which defines all feedback functions for the **RAG triad**, including **answer relevance, context relevance, and groundedness**. With everything configured, we are ready to execute the evaluations for this setup. The execution was successful.

```python
sentence_index_1 = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=1,
    save_dir="sentence_index_1",
)
```

```python
sentence_window_engine_1 = get_sentence_window_query_engine(
    sentence_index_1
)
```

```python
tru_recorder_1 = get_prebuilt_trulens_recorder(
    sentence_window_engine_1,
    app_id='sentence window engine 1'
)
```
```python
run_evals(eval_questions, tru_recorder_1, sentence_window_engine_1)
```

#### Dashboard Analysis (Window Size 1)

Let's now examine the results within the **dashboard**. This instruction launches a locally hosted Streamlit application, accessible via a provided link. The application's leaderboard presents aggregate metrics for all 21 records processed and evaluated by TruLens. Key metrics include an average latency of 4.57 seconds, a total cost of approximately two cents, and around 9,000 tokens processed. Reviewing the evaluation metrics, we observe that the application performs reasonably well in terms of **answer relevance and groundedness**, but its **context relevance is notably poor**.

```python
Tru().run_dashboard()
```

```
Starting dashboard ...
Config file already exists. Skipping writing process.
Credentials file already exists. Skipping writing process.

Accordion(children=(VBox(children=(VBox(children=(Label(value='STDOUT'), Output())), VBox(children=(Label(valu…

Dashboard started at https://s172-29-57-46p38560.lab-aws-production.deeplearning.ai/ .

<Popen: returncode: None args: ['streamlit', 'run', '--server.headless=True'...>
```

#### Drilling Down into Individual Records (Window Size 1)

Now, we'll delve into the individual records processed and evaluated by the application. Scrolling right reveals instances where the application underperforms on these metrics. Let's select a specific row to examine its performance more closely. The question posed in this instance is: "In the context of project selection and execution, explain the difference between ready-fire and ready-fire-aim approaches. Provide examples where each approach might be more beneficial."

You can view the comprehensive response from the RAG system. Scrolling further down reveals the overall scores for **groundedness, context relevance, and answer relevance**. In this particular example, two pieces of context were retrieved. For one of these retrieved contexts, the **context relevance is quite low**. Drilling into this specific example, it becomes evident that the piece of context is notably small. Recall that we are operating with a **sentence window size of 1**, meaning only one extra sentence was appended at the beginning and one at the end of the retrieved context. This results in a relatively small piece of context that lacks crucial information necessary for it to be highly relevant to the posed question.


Similarly, regarding **groundedness**, both retrieved pieces for the final summary show quite low scores. Let's examine the one with a slightly higher groundedness score for further justification. In this example, some initial sentences possess strong supporting evidence within the retrieved context, resulting in a high score of 10 out of 10. However, for subsequent sentences, no supporting evidence was found, leading to a groundedness score of 0. Consider a specific sentence: "It's often used in situations where the cost of execution is relatively low and where the ability to iterate and adapt quickly is more important than upfront planning." While this text seems relevant to the question, it was not present in the retrieved context, and thus lacks supporting evidence. It's plausible that the model acquired this information during its training phase, either from Andrew's document on AI career advice or another source discussing similar topics. However, in this specific instance, the sentence is not *grounded* in the retrieved context.


This highlights a common problem when the **sentence window is too small**: **context relevance** tends to be low. As a result, **groundedness** also suffers because the LLM begins to rely on its pre-existing knowledge from its training phase to answer questions, rather than exclusively utilizing the provided context.

### Note about the dataset of questions

  - Since this evaluation process takes a long time to run, the following file `generated_questions.text` contains one question (the one mentioned in the lecture video).
  - If you would like to explore other possible questions, feel free to explore the file directory by clicking on the "Jupyter" logo at the top right of this notebook. You'll see the following `.text` files:

>   - `generated_questions_01_05.text`
>   - `generated_questions_06_10.text`
>   - `generated_questions_11_15.text`
>   - `generated_questions_16_20.text`
>   - `generated_questions_21_24.text`

Note that running an evaluation on more than one question can take some time, so we recommend choosing one of these files (with 5 questions each) to run and explore the results.

  - For evaluating a personal project, an eval set of 20 is reasonable.
  - For evaluating business applications, you may need a set of 100+ in order to cover all the use cases thoroughly.
  - Note that since API calls can sometimes fail, you may occasionally see null responses, and would want to re-run your evaluations.  So running your evaluations in smaller batches can also help you save time and cost by only re-running the evaluation on the batches with issues.

<!-- end list -->

```python
eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)
```

### Sentence window size = 3

#### Improving Metrics with Increased Window Size


Having demonstrated a failure mode with a sentence window size of 1, I'll now guide you through further steps to observe how metrics improve with adjustments to the sentence window size. To expedite the notebook walkthrough, I'll reload the evaluation questions, but this time, specifically focusing on the single problematic question we just reviewed with a sentence window size of 1. Then, I'll run the evaluation again with the **sentence window size set to 3**.

This code snippet configures the RAG system with a **sentence window size of 3** and also sets up the TruRecorder. With the feedback function definitions now in place, alongside the RAG configured for a sentence window of size 3, we're ready to execute the evaluations for that specific question. This is the same evaluation question we examined in detail with the sentence window set to 1, where we observed the failure mode. The execution was successful.

```python
sentence_index_3 = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index_3",
)
sentence_window_engine_3 = get_sentence_window_query_engine(
    sentence_index_3
)

tru_recorder_3 = get_prebuilt_trulens_recorder(
    sentence_window_engine_3,
    app_id='sentence window engine 3'
)
```

```
✅ In Answer Relevance, input prompt will be set to __record__.main_input or `Select.RecordInput` .
✅ In Answer Relevance, input response will be set to __record__.main_output or `Select.RecordOutput` .
✅ In Context Relevance, input prompt will be set to __record__.main_input or `Select.RecordInput` .
✅ In Context Relevance, input response will be set to __record__.app.query.rets.source_nodes[:].node.text .
✅ In Groundedness, input source will be set to __record__.app.query.rets.source_nodes[:].node.text .
✅ In Groundedness, input statement will be set to __record__.main_output or `Select.RecordOutput` .
```

```python
run_evals(eval_questions, tru_recorder_3, sentence_window_engine_3)
```

#### Results with Sentence Window Size 3



Now, let's review the results in the **TruLens dashboard** with the sentence window engine configured to a size of 3. You'll observe the results here; I ran it on the single record that proved problematic when using a sentence window size of 1. There's a substantial increase in **context relevance**, which jumped from 0.57 to 0.9.

```python
Tru().run_dashboard()
```

```
Starting dashboard ...
Config file already exists. Skipping writing process.
Credentials file already exists. Skipping writing process.
Dashboard already running at path: https://s172-29-57-46p38560.lab-aws-production.deeplearning.ai/

<Popen: returncode: None args: ['streamlit', 'run', '--server.headless=True'...>
```


Once the completion step processes these significantly improved pieces of context, the **groundedness score increases considerably**. By identifying supporting evidence across these two highly relevant context pieces, the groundedness score actually escalates to a perfect 1. Therefore, increasing the **sentence window size from 1 to 3** resulted in a substantial enhancement across the RAG triad evaluation metrics, with both groundedness and context relevance showing significant improvement, and answer relevance also rising.

### Sentence window size = 5

#### Results with Sentence Window Size 5 and Trade-offs


Now, let's examine the metrics for a **sentence window size of 5**. Several observations can be made. First, the total number of tokens processed has increased, which could impact the cost, especially with a higher number of records. This illustrates one of the trade-offs previously mentioned: increasing the sentence window size leads to higher expenses due to more tokens being processed by the LLMs during evaluation.


Another observation is that while **context relevance and answer relevance have remained stable, groundedness has actually decreased** with the larger sentence window size. This phenomenon can occur beyond a certain point: as the context size expands, the LLM may become overwhelmed by an excessive amount of information during the completion step. Consequently, during summarization, it might begin to incorporate its own pre-existing knowledge rather than exclusively relying on the information from the retrieved context pieces.

-----

## Conclusion and Next Steps


In summary, our evaluation indicates that incrementally increasing the **sentence window size from 1 to 3, and then to 5**, reveals that a size of **3 is the optimal choice** for this specific evaluation. We observed improvements in context relevance, answer relevance, and groundedness when moving from a size of 1 to 3. However, a further increase to a size of 5 resulted in a reduction or degradation of the groundedness score.


As you engage with the notebook, we highly recommend re-running these two steps with a greater number of records. Take the time to meticulously examine individual records that contribute to issues in specific metrics, such as context relevance or groundedness. This practice will help you develop an intuitive understanding of why certain failure modes occur and how to effectively address them. In the subsequent section, we will delve into another advanced RAG technique: **auto-merging**, which aims to mitigate some of these observed failure modes, particularly instances where irrelevant context infiltrates the final response, leading to suboptimal scores in groundedness or answer relevance.

```
```
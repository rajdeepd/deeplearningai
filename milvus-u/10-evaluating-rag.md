---
layout: default
title: 10. Evaluating RAG Llama Index
nav_order: 10
description: ""
has_children: false
parent:  Milvus (U)
---
Evaluate the Ideal Chunk Size for a RAG System using LlamaIndex and GPT-4o model
Setup
Before proceeding on the experiment, we need to ensure all required modules are imported. Make sure llama-index and pypdf python modules are installed.


!pip install llama-index pypdf
     

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

from llama_index.llms.openai import OpenAI


import openai
import time
openai.api_key = 'sk-..'#'OPENAI-API-KEY' # set your openai api key
     
Download Data
We will be using the Uber 10K SEC Filings for 2021 for this experiment. First use wget to download the pdf into directory data/10k/uber_2021.pdf


!mkdir -p 'data/10k/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
     
--2024-09-30 14:25:49--  https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1880483 (1.8M) [application/octet-stream]
Saving to: ‘data/10k/uber_2021.pdf’

data/10k/uber_2021. 100%[===================>]   1.79M  --.-KB/s    in 0.07s   

2024-09-30 14:25:50 (24.2 MB/s) - ‘data/10k/uber_2021.pdf’ saved [1880483/1880483]

Load Data
Let’s load our document using SimpleDirectoryReader. The SimpleDirectoryReader is the most commonly used data connector Pass in a input directory or a list of files. It selects the best file reader based on the file extensions.


# Load Data

reader = SimpleDirectoryReader("./data/10k/")
documents = reader.load_data()
     
Question Generation
To select the right chunk_size, we willcompute metrics like Average Response time, Faithfulness, and Relevancy for various chunk_sizes. The DatasetGenerator will help generate questions from the documents.


# To evaluate for each chunk size, we will first generate a set of 40 questions from first 20 pages of the document
eval_documents = documents[:20]
data_generator = DatasetGenerator.from_documents(eval_documents)
eval_questions = data_generator.generate_questions_from_nodes(num = 40)
     
/usr/local/lib/python3.10/dist-packages/llama_index/core/evaluation/dataset_generation.py:200: DeprecationWarning: Call to deprecated class DatasetGenerator. (Deprecated in favor of `RagDatasetGenerator` which should be used instead.)
  return cls(
/usr/local/lib/python3.10/dist-packages/llama_index/core/evaluation/dataset_generation.py:296: DeprecationWarning: Call to deprecated class QueryResponseDataset. (Deprecated in favor of `LabelledRagDataset` which should be used instead.)
  return QueryResponseDataset(queries=queries, responses=responses_dict)
Setting Up Evaluators
We are setting up the GPT-4o model to serve as the backbone for evaluating the responses generated during the experiment. Two evaluators, FaithfulnessEvaluator and RelevancyEvaluator, are initialised.Settings is used for setting the llm, embed model, node parser etc .

Faithfulness Evaluator - It helps measure if the response was hallucinated and measures if the response from a query engine matches any source nodes.
Relevancy Evaluator - It helps measure if the query was actually answered by the response and measures if the response + source nodes match the query.

# We will be using GPT-4o for evaluating the responses
gpt4o = OpenAI(temperature=0, model="gpt-4o")


from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Set model, embedding model chunk size etc using Settings
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

# We will define Faithfulness and Relevancy Evaluators, based on GPT-4o

faithfulness_gpt4o = FaithfulnessEvaluator()

relevancy_gpt4o = RelevancyEvaluator()

     
Response Evaluation For A Chunk Size
We will evaluate each chunk_size based on 3 metrics.

Average Response Time.
Average Faithfulness.
Average Relevancy.
Function, evaluate_response_time_and_accuracy, that does that which has:

VectorIndex Creation.
Building the Query Engine.
Metrics Calculation.

# Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size
# We use GPT-3.5-Turbo to generate response and GPT-4 to evaluate it.
def evaluate_response_time_and_accuracy(chunk_size, eval_questions):
    """
    Evaluate the average response time, faithfulness, and relevancy of responses generated by GPT-3.5-turbo for a given chunk size.

    Parameters:
    chunk_size (int): The size of data chunks being processed.

    Returns:
    tuple: A tuple containing the average response time, faithfulness, and relevancy metrics.
    """

    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # create vector index
    llm = OpenAI(model="gpt-4o")
    #service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size)
    #vector_index = VectorStoreIndex.from_documents(
    #    eval_documents, service_context=service_context
    #)
    vector_index = VectorStoreIndex.from_documents(
        eval_documents
    )
    # bu
    # build query engine
    # By default, similarity_top_k is set to 2. To experiment with different values, pass it as an argument to as_query_engine()
    query_engine = vector_index.as_query_engine()
    num_questions = len(eval_questions)

    # Iterate over each question in eval_questions to compute metrics.
    # While BatchEvalRunner can be used for faster evaluations (see: https://docs.llamaindex.ai/en/latest/examples/evaluation/batch_eval.html),
    # we're using a loop here to specifically measure response time for different chunk sizes.
    for question in eval_questions:
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time

        faithfulness_result = faithfulness_gpt4o.evaluate_response(
            response=response_vector
        ).passing

        relevancy_result = relevancy_gpt4o.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy
     
Testing Across Different Chunk Sizes
We will evaluate a range of chunk sizes to identify which offers the most promising metrics, then iterate over difference chunk size to evaluate metrics and get Average faithfullness and Average Relevancy.


# Iterate over different chunk sizes to evaluate the metrics to help fix the chunk size.

for chunk_size in [128, 256, 512, 1024, 2048]:
  avg_response_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size,eval_questions)
  print(f"Chunk size {chunk_size} - Average Response time: {avg_response_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
     
Chunk size 128 - Average Response time: 2.79s, Average Faithfulness: 0.95, Average Relevancy: 0.93
Chunk size 256 - Average Response time: 1.79s, Average Faithfulness: 0.95, Average Relevancy: 0.90
Chunk size 512 - Average Response time: 1.98s, Average Faithfulness: 0.95, Average Relevancy: 0.93
Chunk size 1024 - Average Response time: 2.07s, Average Faithfulness: 0.95, Average Relevancy: 0.97
Chunk size 2048 - Average Response time: 2.03s, Average Faithfulness: 0.97, Average Relevancy: 0.95
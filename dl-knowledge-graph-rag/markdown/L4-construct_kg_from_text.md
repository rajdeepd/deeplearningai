# Lesson 4: Constructing a Knowledge Graph from Text Documents

<p style="background-color:#fd4a6180; padding:15px; margin-left:20px"> <b>Note:</b> This notebook takes about 30 seconds to be ready to use. Please wait until the "Kernel starting, please wait..." message clears from the top of the notebook before running any cells. You may start the video while you wait.</p>

### Import packages and set up Neo4j


```python
from dotenv import load_dotenv
import os

# Common data processing
import json
import textwrap

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI


# Warning control
import warnings
warnings.filterwarnings("ignore")
```


```python
# Load from environment
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Note the code below is unique to this course environment, and not a 
# standard part of Neo4j's integration with OpenAI. Remove if running 
# in your own environment.
OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

# Global constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'
```

### Take a look at a Form 10-K json file

- Publicly traded companies are required to fill a form 10-K each year with the Securities and Exchange Commision (SEC)
- You can search these filings using the SEC's [EDGAR database](https://www.sec.gov/edgar/search/)
- For the next few lessons, you'll work with a single 10-K form for a company called [NetApp](https://www.netapp.com/)


```python
first_file_name = "./data/form10k/0000950170-23-027948.json"

```


```python
first_file_as_object = json.load(open(first_file_name))
```


```python
type(first_file_as_object)
```




    dict




```python
for k,v in first_file_as_object.items():
    print(k, type(v))
```

    item1 <class 'str'>
    item1a <class 'str'>
    item7 <class 'str'>
    item7a <class 'str'>
    cik <class 'str'>
    cusip6 <class 'str'>
    cusip <class 'list'>
    names <class 'list'>
    source <class 'str'>



```python
item1_text = first_file_as_object['item1']
```


```python
item1_text[0:1500]
```




    '>Item 1.  \nBusiness\n\n\nOverview\n\n\nNetApp, Inc. (NetApp, we, us or the Company) is a global cloud-led, data-centric software company. We were incorporated in 1992 and are headquartered in San Jose, California. Building on more than three decades of innovation, we give customers the freedom to manage applications and data across hybrid multicloud environments. Our portfolio of cloud services, and storage infrastructure, powered by intelligent data management software, enables applications to run faster, more reliably, and more securely, all at a lower cost.\n\n\nOur opportunity is defined by the durable megatrends of data-driven digital and cloud transformations. NetApp helps organizations meet the complexities created by rapid data and cloud growth, multi-cloud management, and the adoption of next-generation technologies, such as AI, Kubernetes, and modern databases. Our modern approach to hybrid, multicloud infrastructure and data management, which we term ‘evolved cloud’, provides customers the ability to leverage data across their entire estate with simplicity, security, and sustainability which increases our relevance and value to our customers.\n\n\nIn an evolved cloud state, the cloud is fully integrated into an organization’s architecture and operations. Data centers and clouds are seamlessly united and hybrid multicloud operations are simplified, with consistency and observability across environments. The key benefits NetApp brings to an organization’s hybrid multicloud envir'



### Split Form 10-K sections into chunks
- Set up text splitter using LangChain
- For now, split only the text from the "item 1" section 


```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)
```

```python
item1_text_chunks = text_splitter.split_text(item1_text)
```


```python
type(item1_text_chunks)
```




    list




```python
len(item1_text_chunks)
```




    254




```python
item1_text_chunks[0]
```




    '>Item 1.  \nBusiness\n\n\nOverview\n\n\nNetApp, Inc. (NetApp, we, us or the Company) is a global cloud-led, data-centric software company. We were incorporated in 1992 and are headquartered in San Jose, California. Building on more than three decades of innovation, we give customers the freedom to manage applications and data across hybrid multicloud environments. Our portfolio of cloud services, and storage infrastructure, powered by intelligent data management software, enables applications to run faster, more reliably, and more securely, all at a lower cost.\n\n\nOur opportunity is defined by the durable megatrends of data-driven digital and cloud transformations. NetApp helps organizations meet the complexities created by rapid data and cloud growth, multi-cloud management, and the adoption of next-generation technologies, such as AI, Kubernetes, and modern databases. Our modern approach to hybrid, multicloud infrastructure and data management, which we term ‘evolved cloud’, provides customers the ability to leverage data across their entire estate with simplicity, security, and sustainability which increases our relevance and value to our customers.\n\n\nIn an evolved cloud state, the cloud is fully integrated into an organization’s architecture and operations. Data centers and clouds are seamlessly united and hybrid multicloud operations are simplified, with consistency and observability across environments. The key benefits NetApp brings to an organization’s hybrid multicloud environment are:\n\n\n•\nOperational simplicity: NetApp’s use of open source, open architectures and APIs, microservices, and common capabilities and data services facilitate the creation of applications that can run anywhere.\n\n\n•\nFlexibility and consistency: NetApp makes moving data and applications between environments seamless through a common storage foundation across on-premises and multicloud environments.'



- Set up helper function to chunk all sections of the Form 10-K
- You'll limit the number of chunks in each section to 20 to speed things up


```python
def split_form10k_data_from_file(file):
    chunks_with_metadata = [] # use this to accumlate chunk records
    file_as_object = json.load(open(file)) # open the json file
    for item in ['item1','item1a','item7','item7a']: # pull these keys from the json
        print(f'Processing {item} from {file}') 
        item_text = file_as_object[item] # grab the text of the item
        item_text_chunks = text_splitter.split_text(item_text) # split the text into chunks
        chunk_seq_id = 0
        for chunk in item_text_chunks[:20]: # only take the first 20 chunks
            form_id = file[file.rindex('/') + 1:file.rindex('.')] # extract form id from file name
            # finally, construct a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk, 
                # metadata from looping...
                'f10kItem': item,
                'chunkSeqId': chunk_seq_id,
                # constructed metadata...
                'formId': f'{form_id}', # pulled from the filename
                'chunkId': f'{form_id}-{item}-chunk{chunk_seq_id:04d}',
                # metadata from file...
                'names': file_as_object['names'],
                'cik': file_as_object['cik'],
                'cusip6': file_as_object['cusip6'],
                'source': file_as_object['source'],
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata
```


```python
first_file_chunks = split_form10k_data_from_file(first_file_name)
```

    Processing item1 from ./data/form10k/0000950170-23-027948.json
    	Split into 20 chunks
    Processing item1a from ./data/form10k/0000950170-23-027948.json
    	Split into 1 chunks
    Processing item7 from ./data/form10k/0000950170-23-027948.json
    	Split into 1 chunks
    Processing item7a from ./data/form10k/0000950170-23-027948.json
    	Split into 1 chunks



```python
first_file_chunks[0]
```




    {'text': '>Item 1.  \nBusiness\n\n\nOverview\n\n\nNetApp, Inc. (NetApp, we, us or the Company) is a global cloud-led, data-centric software company. We were incorporated in 1992 and are headquartered in San Jose, California. Building on more than three decades of innovation, we give customers the freedom to manage applications and data across hybrid multicloud environments. Our portfolio of cloud services, and storage infrastructure, powered by intelligent data management software, enables applications to run faster, more reliably, and more securely, all at a lower cost.\n\n\nOur opportunity is defined by the durable megatrends of data-driven digital and cloud transformations. NetApp helps organizations meet the complexities created by rapid data and cloud growth, multi-cloud management, and the adoption of next-generation technologies, such as AI, Kubernetes, and modern databases. Our modern approach to hybrid, multicloud infrastructure and data management, which we term ‘evolved cloud’, provides customers the ability to leverage data across their entire estate with simplicity, security, and sustainability which increases our relevance and value to our customers.\n\n\nIn an evolved cloud state, the cloud is fully integrated into an organization’s architecture and operations. Data centers and clouds are seamlessly united and hybrid multicloud operations are simplified, with consistency and observability across environments. The key benefits NetApp brings to an organization’s hybrid multicloud environment are:\n\n\n•\nOperational simplicity: NetApp’s use of open source, open architectures and APIs, microservices, and common capabilities and data services facilitate the creation of applications that can run anywhere.\n\n\n•\nFlexibility and consistency: NetApp makes moving data and applications between environments seamless through a common storage foundation across on-premises and multicloud environments.',
     'f10kItem': 'item1',
     'chunkSeqId': 0,
     'formId': '0000950170-23-027948',
     'chunkId': '0000950170-23-027948-item1-chunk0000',
     'names': ['Netapp Inc', 'NETAPP INC'],
     'cik': '1002047',
     'cusip6': '64110D',
     'source': 'https://www.sec.gov/Archives/edgar/data/1002047/000095017023027948/0000950170-23-027948-index.htm'}



### Create graph nodes using text chunks


```python
merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET 
        mergedChunk.names = $chunkParam.names,
        mergedChunk.formId = $chunkParam.formId, 
        mergedChunk.cik = $chunkParam.cik, 
        mergedChunk.cusip6 = $chunkParam.cusip6, 
        mergedChunk.source = $chunkParam.source, 
        mergedChunk.f10kItem = $chunkParam.f10kItem, 
        mergedChunk.chunkSeqId = $chunkParam.chunkSeqId, 
        mergedChunk.text = $chunkParam.text
RETURN mergedChunk
"""
```

- Set up connection to graph instance using LangChain


```python
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
```

- Create a single chunk node for now


```python
kg.query(merge_chunk_node_query, 
         params={'chunkParam':first_file_chunks[0]})
```




    [{'mergedChunk': {'formId': '0000950170-23-027948',
       'f10kItem': 'item1',
       'names': ['Netapp Inc', 'NETAPP INC'],
       'cik': '1002047',
       'cusip6': '64110D',
       'source': 'https://www.sec.gov/Archives/edgar/data/1002047/000095017023027948/0000950170-23-027948-index.htm',
       'text': '>Item 1.  \nBusiness\n\n\nOverview\n\n\nNetApp, Inc. (NetApp, we, us or the Company) is a global cloud-led, data-centric software company. We were incorporated in 1992 and are headquartered in San Jose, California. Building on more than three decades of innovation, we give customers the freedom to manage applications and data across hybrid multicloud environments. Our portfolio of cloud services, and storage infrastructure, powered by intelligent data management software, enables applications to run faster, more reliably, and more securely, all at a lower cost.\n\n\nOur opportunity is defined by the durable megatrends of data-driven digital and cloud transformations. NetApp helps organizations meet the complexities created by rapid data and cloud growth, multi-cloud management, and the adoption of next-generation technologies, such as AI, Kubernetes, and modern databases. Our modern approach to hybrid, multicloud infrastructure and data management, which we term ‘evolved cloud’, provides customers the ability to leverage data across their entire estate with simplicity, security, and sustainability which increases our relevance and value to our customers.\n\n\nIn an evolved cloud state, the cloud is fully integrated into an organization’s architecture and operations. Data centers and clouds are seamlessly united and hybrid multicloud operations are simplified, with consistency and observability across environments. The key benefits NetApp brings to an organization’s hybrid multicloud environment are:\n\n\n•\nOperational simplicity: NetApp’s use of open source, open architectures and APIs, microservices, and common capabilities and data services facilitate the creation of applications that can run anywhere.\n\n\n•\nFlexibility and consistency: NetApp makes moving data and applications between environments seamless through a common storage foundation across on-premises and multicloud environments.',
       'chunkId': '0000950170-23-027948-item1-chunk0000',
       'chunkSeqId': 0}}]



- Create a uniqueness constraint to avoid duplicate chunks


```python
kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
""")

```




    []




```python
kg.query("SHOW INDEXES")
```




    [{'id': 1,
      'name': 'index_343aff4e',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'LOOKUP',
      'entityType': 'NODE',
      'labelsOrTypes': None,
      'properties': None,
      'indexProvider': 'token-lookup-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': 0},
     {'id': 2,
      'name': 'index_f7700477',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'LOOKUP',
      'entityType': 'RELATIONSHIP',
      'labelsOrTypes': None,
      'properties': None,
      'indexProvider': 'token-lookup-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': 0},
     {'id': 3,
      'name': 'unique_chunk',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'RANGE',
      'entityType': 'NODE',
      'labelsOrTypes': ['Chunk'],
      'properties': ['chunkId'],
      'indexProvider': 'range-1.0',
      'owningConstraint': 'unique_chunk',
      'lastRead': None,
      'readCount': None}]



- Loop through and create nodes for all chunks
- Should create 23 nodes because you set a limit of 20 chunks in the text splitting function above


```python
node_count = 0
for chunk in first_file_chunks:
    print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
    kg.query(merge_chunk_node_query, 
            params={
                'chunkParam': chunk
            })
    node_count += 1
print(f"Created {node_count} nodes")
```

    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0000
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0001
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0002
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0003
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0004
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0005
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0006
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0007
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0008
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0009
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0010
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0011
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0012
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0013
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0014
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0015
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0016
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0017
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0018
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1-chunk0019
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item1a-chunk0000
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item7-chunk0000
    Creating `:Chunk` node for chunk ID 0000950170-23-027948-item7a-chunk0000
    Created 23 nodes



```python
kg.query("""
         MATCH (n)
         RETURN count(n) as nodeCount
         """)
```




    [{'nodeCount': 23}]



### Create a vector index


```python
kg.query("""
         CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}
""")
```




    []




```python
kg.query("SHOW INDEXES")
```




    [{'id': 5,
      'name': 'form_10k_chunks',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'VECTOR',
      'entityType': 'NODE',
      'labelsOrTypes': ['Chunk'],
      'properties': ['textEmbedding'],
      'indexProvider': 'vector-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': None},
     {'id': 1,
      'name': 'index_343aff4e',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'LOOKUP',
      'entityType': 'NODE',
      'labelsOrTypes': None,
      'properties': None,
      'indexProvider': 'token-lookup-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': 0},
     {'id': 2,
      'name': 'index_f7700477',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'LOOKUP',
      'entityType': 'RELATIONSHIP',
      'labelsOrTypes': None,
      'properties': None,
      'indexProvider': 'token-lookup-1.0',
      'owningConstraint': None,
      'lastRead': None,
      'readCount': 0},
     {'id': 3,
      'name': 'unique_chunk',
      'state': 'ONLINE',
      'populationPercent': 100.0,
      'type': 'RANGE',
      'entityType': 'NODE',
      'labelsOrTypes': ['Chunk'],
      'properties': ['chunkId'],
      'indexProvider': 'range-1.0',
      'owningConstraint': 'unique_chunk',
      'lastRead': None,
      'readCount': 0}]



### Calculate embedding vectors for chunks and populate index
- This query calculates the embedding vector and stores it as a property called `textEmbedding` on each `Chunk` node.


```python
kg.query("""
    MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text, 
      "OpenAI", 
      {
        token: $openAiApiKey, 
        endpoint: $openAiEndpoint
      }) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """, 
    params={"openAiApiKey":OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )
```

    []



```python
kg.refresh_schema()
print(kg.schema)
```

    Node properties are the following:
    Chunk {textEmbedding: LIST, f10kItem: STRING, chunkSeqId: INTEGER, text: STRING, cik: STRING, cusip6: STRING, names: LIST, formId: STRING, source: STRING, chunkId: STRING}
    Relationship properties are the following:
    
    The relationships are the following:
    


### Use similarity search to find relevant chunks

- Setup a help function to perform similarity search using the vector index


```python
def neo4j_vector_search(question):
  """Search for similar nodes using the Neo4j vector index"""
  vector_search_query = """
    WITH genai.vector.encode(
      $question, 
      "OpenAI", 
      {
        token: $openAiApiKey,
        endpoint: $openAiEndpoint
      }) AS question_embedding
    CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
    RETURN score, node.text AS text
  """
  similar = kg.query(vector_search_query, 
                     params={
                      'question': question, 
                      'openAiApiKey':OPENAI_API_KEY,
                      'openAiEndpoint': OPENAI_ENDPOINT,
                      'index_name':VECTOR_INDEX_NAME, 
                      'top_k': 10})
  return similar
```

- Ask a question!


```python
search_results = neo4j_vector_search(
    'In a single sentence, tell me about Netapp.'
)
```


```python
search_results[0]
```




    {'score': 0.9357025623321533,
     'text': '>Item 1.  \nBusiness\n\n\nOverview\n\n\nNetApp, Inc. (NetApp, we, us or the Company) is a global cloud-led, data-centric software company. We were incorporated in 1992 and are headquartered in San Jose, California. Building on more than three decades of innovation, we give customers the freedom to manage applications and data across hybrid multicloud environments. Our portfolio of cloud services, and storage infrastructure, powered by intelligent data management software, enables applications to run faster, more reliably, and more securely, all at a lower cost.\n\n\nOur opportunity is defined by the durable megatrends of data-driven digital and cloud transformations. NetApp helps organizations meet the complexities created by rapid data and cloud growth, multi-cloud management, and the adoption of next-generation technologies, such as AI, Kubernetes, and modern databases. Our modern approach to hybrid, multicloud infrastructure and data management, which we term ‘evolved cloud’, provides customers the ability to leverage data across their entire estate with simplicity, security, and sustainability which increases our relevance and value to our customers.\n\n\nIn an evolved cloud state, the cloud is fully integrated into an organization’s architecture and operations. Data centers and clouds are seamlessly united and hybrid multicloud operations are simplified, with consistency and observability across environments. The key benefits NetApp brings to an organization’s hybrid multicloud environment are:\n\n\n•\nOperational simplicity: NetApp’s use of open source, open architectures and APIs, microservices, and common capabilities and data services facilitate the creation of applications that can run anywhere.\n\n\n•\nFlexibility and consistency: NetApp makes moving data and applications between environments seamless through a common storage foundation across on-premises and multicloud environments.'}



### Set up a LangChain RAG workflow to chat with the form


```python
neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    text_node_properties=[VECTOR_SOURCE_PROPERTY],
    embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
)

```


```python
retriever = neo4j_vector_store.as_retriever()
```

- Set up a RetrievalQAWithSourcesChain to carry out question answering
- You can check out the LangChain documentation for this chain [here](https://api.python.langchain.com/en/latest/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html)


```python
chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever
)
```


```python
def prettychain(question: str) -> str:
    """Pretty print the chain's response to a question"""
    response = chain({"question": question},
        return_only_outputs=True,)
    print(textwrap.fill(response['answer'], 60))
```

- Ask a question!


```python
question = "What is Netapp's primary business?"
```


```python
prettychain(question)
```

    NetApp's primary business is enterprise storage and data
    management, cloud storage, and cloud operations.



```python
prettychain("Where is Netapp headquartered?")
```

    Netapp is headquartered in San Jose, California.



```python
prettychain("""
    Tell me about Netapp. 
    Limit your answer to a single sentence.
""")
```

    NetApp is a global cloud-led, data-centric software company
    that provides customers with the freedom to manage
    applications and data across hybrid multicloud environments.



```python
prettychain("""
    Tell me about Apple. 
    Limit your answer to a single sentence.
""")
```

    Apple is a global cloud-led, data-centric software company
    headquartered in San Jose, California, that provides
    customers with the freedom to manage applications and data
    across hybrid multicloud environments.



```python
prettychain("""
    Tell me about Apple. 
    Limit your answer to a single sentence.
    If you are unsure about the answer, say you don't know.
""")
```

    I don't know.


### Ask you own question!
- Add your own question to the call to prettychain below to find out more about NetApp
- Here is NetApp's website if you want some inspiration: https://www.netapp.com/


```python
prettychain("""ADD YOUR OWN QUESTION HERE
""")
```

    I don't know the answer.



```python

```

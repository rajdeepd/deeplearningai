# Lesson 1: Advanced RAG Pipeline


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

## Basic RAG pipeline


```python
from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))
```


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


```python
query_engine = index.as_query_engine()
```


```python
response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))
```

## Evaluation setup using TruLens


```python
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)
```


```python
# You can try your own question:
new_question = "What is the right AI job for me?"
eval_questions.append(new_question)
```


```python
print(eval_questions)
```


```python
from trulens_eval import Tru
tru = Tru()

tru.reset_database()
```

For the classroom, we've written some of the code in helper functions inside a utils.py file.  
- You can view the utils.py file in the file directory by clicking on the "Jupyter" logo at the top of the notebook.
- In later lessons, you'll get to work directly with the code that's currently wrapped inside these helper functions, to give you more options to customize your RAG pipeline.


```python
from utils import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")
```


```python
with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)
```


```python
records, feedback = tru.get_records_and_feedback(app_ids=[])
```


```python
records.head()
```


```python
# launches on http://localhost:8501/
tru.run_dashboard()
```

## Advanced RAG pipeline

### 1. Sentence Window retrieval


```python
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
```


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


```python
window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))
```


```python
tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)
```


```python
for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))
```


```python
tru.get_leaderboard(app_ids=[])
```


```python
# launches on http://localhost:8501/
tru.run_dashboard()
```

### 2. Auto-merging retrieval


```python
from utils import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)
```


```python
from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)
```


```python
auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))
```


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


```python
tru.get_leaderboard(app_ids=[])
```


```python
# launches on http://localhost:8501/
tru.run_dashboard()
```

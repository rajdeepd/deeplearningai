import requests
import json

# Download the data
resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

# Parse the JSON and preview it
print(type(data), len(data))

def json_print(data):
    print(json.dumps(data, indent=2))

json_print(data[0])

import weaviate, os
from weaviate.embedded import EmbeddedOptions
import openai


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_APIKEY']


client = weaviate.Client('http://localhost:8080',
                         #embedded_options=EmbeddedOptions(),
    additional_headers={
        "X-OpenAI-Api-Key": openai.api_key  # Replace this with your actual key
    })
print(f"Client created? {client.is_ready()}")



# resetting the schema. CAUTION: This will delete your collection

# if client.schema.exists("Question"):
#     client.schema.delete_class("Question")
# class_obj = {
#     "class": "Question",
#     "vectorizer": "text2vec-openai",  # Use OpenAI as the vectorizer
#     "moduleConfig": {
#         "text2vec-openai": {
#             "model": "ada",
#             "modelVersion": "002",
#             "type": "text"
#             #"baseURL": os.environ["OPENAI_API_BASE"]
#         }
#     }
# }
#
# client.schema.create_class(class_obj)

# reminder for the data structure
import_questions = False

if import_questions:
    with client.batch.configure(batch_size=5) as batch:
        for i, d in enumerate(data):  # Batch import data

            print(f"importing question: {i + 1}")

            properties = {
                "answer": d["Answer"],
                "question": d["Question"],
                "category": d["Category"],
            }

            batch.add_data_object(
                data_object=properties,
                class_name="Question"
            )


# write a query to extract the vector for a question
result = (client.query
          .get("Question", ["category", "question", "answer"])
          .with_additional("vector")
          .with_limit(1)
          .do())

json_print(result["data"]['Get']['Question'][0]["_additional"]["vector"][:5])#["Question"][0]["_addtional"]["vector"][0]

response = (
    client.query
    .get("Question",["question","answer","category"])
    .with_near_text({"concepts": "biology"})
    .with_additional('distance')
    .with_limit(2)
    .do()
)

json_print(response)
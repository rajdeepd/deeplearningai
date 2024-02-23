import weaviate
import json
import os

client = weaviate.Client(
    url = 'http://localhost:8080',  # Replace with your cluster url
    #auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Replace with your inference API key
    }
)
print(client.get_meta())

with open("faq.json", "r") as f:
    json_data = json.load(f)

queries = [{"question": item["question"], "answer": item["answer"]} for item in json_data["questions"]]
print(len(queries))
# Ragas wants ['question', 'answer', 'contexts', 'ground_truths'] as
'''
{
    "question": ['What is ref2vec?', ...], <-- question from faq doc
    "answer": [], <-- answer from generated result
    "contexts": [], <-- content
    "ground_truths": [] <-- answer from faq doc
}
'''
questions = []
answers = []
contexts = []
ground_truths = []

for query in queries:
    question = query["question"]
    graphql_query = """
    {
        Get {
            Document(
                hybrid: {
                    query: "%s",
                    alpha: 1
                },
                limit: 5
            ){
                content
                source
                title
                _additional {
                    generate(
                        groupedResult: {
                            task: "Please answer the question %s. Make sure your answer is based on the following search results."
                        }
                    ){
                        groupedResult
                        error
                    }
                }
            }
        }
    }""" % (question, question)

    questions.append(question)
    ground_truths.append([query["answer"]])
    res = client.query.raw(graphql_query)
    responses = client.query.raw(graphql_query)["data"]["Get"]["Document"]
    new_answer = responses[0]["_additional"]["generate"]["groupedResult"]
    answers.append(new_answer)
    new_contexts = [response["content"] for response in responses]
    contexts.append(new_contexts)

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

from datasets import Dataset
dataset = Dataset.from_dict(data)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

result = evaluate(
    dataset = dataset,
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
)

df = result.to_pandas()
print(df.to_csv('./output_ragas.csv'))
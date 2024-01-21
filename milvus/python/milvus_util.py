from openai import OpenAI
client = OpenAI()
model="text-embedding-ada-002"

def embed(text):
    return client.embeddings.create(input=[text], model=model).data[0].embedding
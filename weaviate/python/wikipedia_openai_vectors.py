import weaviate
import os

import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_APIKEY']
client = weaviate.Client('http://localhost:8080',
    additional_headers={
        "X-OpenAI-Api-Key": openai.api_key  # Replace this with your actual key
    })
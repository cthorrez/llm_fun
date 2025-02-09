import os
from mistralai import Mistral
from typing import Literal
from pydantic import BaseModel

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

class NameResponse(BaseModel):
    name: Literal['Assistant', 'Bot', 'AI', 'Bob']

messages = [
    {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]},
    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello, my name is '}], 'prefix': True},
]
response = client.chat.parse(
    model="mistral-small-latest",
    messages=messages,
    response_format=NameResponse,
    max_tokens=32
)

print(response.choices[0].message.content)
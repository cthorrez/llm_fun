import os
from enum import Enum
import json
from typing import Literal
import ell
from openai import OpenAI
from gpqa import FourChoiceAnswer
from pydantic import BaseModel


class Choice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

def main():

    class NameResponse(BaseModel):
        name: str
        age: int
        country: str
        gender: Literal['male', 'female', 'other']
        favorite_movie: str
        python_implementation_of_elo: str

    client = OpenAI(
        base_url = "https://api.scaleway.ai/8b2c7bde-831d-4972-b96f-c03d47763941/v1",
        api_key = os.environ["SCALEWAY_API_KEY"]
    )

    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello, my name is '}], 'prefix': True},
    ]
    response = client.beta.chat.completions.parse(
    # response = client.beta.chat.completions.parse(
        model="llama-3.3-70b-instruct",
        messages=messages,
        response_format=FourChoiceAnswer,
        max_tokens=2048
    )

    print(response.choices[0].message.content)

    # data = json.loads(response.choices[0].message.content)
    # print(data)



if __name__ == '__main__':
    main()
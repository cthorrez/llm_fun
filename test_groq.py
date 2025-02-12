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
        base_url = "https://api.groq.com/openai/v1",
        api_key = os.environ["GROQ_API_KEY"]
    )

    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hello, (respond in json)'}]},
        {'role': 'assistant', 'content': '(json) hello, my name is '},
    ]
    response = client.beta.chat.completions.parse(
    # response = client.chat.completions.create(
        # model="llama-3.3-70b-versatile",
        model="llama-3.3-70b-versatile",
        messages=messages,
        response_format={ "type": "json_object" },
        max_tokens=2048
    )

    print(response.choices[0].message.content)

    # data = json.loads(response.choices[0].message.content)
    # print(data)



if __name__ == '__main__':
    main()
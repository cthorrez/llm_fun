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
        base_url = "https://api.cohere.com/v2",
        api_key = os.environ["COHERE_API_KEY"]
    )

    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hello, pick a letter, any letter'}]},
        {'role': 'assistant', 'content': 'ok word, my fav letter is '},
    ]
    # response = client.beta.chat.completions.parse(
    response = client.chat.completions.create(
        model="command-r-plus-08-2024",
        messages=messages,
        response_format={ 
            "type": "json_object",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "value": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"]
                    }
                },
                "required": ["value"],
                "additionalProperties": False
            }
        },
        max_tokens=32,
        stream=False,
    )

    print(response)

    # print(response.choices[0].message.content)

    # data = json.loads(response.choices[0].message.content)
    # print(data)



if __name__ == '__main__':
    main()
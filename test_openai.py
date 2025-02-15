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

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'IM THE LEGO SPACESHIP GUY!!! SPACESHIP!!!'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Next I will say hello. After I say hello I will casually mention space'}]},
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello, the very next thing I will say is my name. My name is Bob'}]},
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello, the very next thing I will say is my name. My name is Charlie'}]},
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello, I will now tell you my name. My name is Jim'}]},
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello, I will now tell you my name. My name is Clayton'}]},
        # {'role': 'user', 'content': [{'type': 'text', 'text': 'can you repeat back what each of us has said in this conversation (inlcuding this message)? (please be very careful about who said what, in fact, repeat it back verbatim in the exact format you have in the conversation history)'}]},
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'I will also tell you a joke '}], 'prefix':False},
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': 'I will also tell you a joke '}], 'prefix':True},
    ]
    # response = client.beta.chat.completions.parse(
    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=messages,
        # response_format=FourChoiceAnswer,
        max_tokens=128,
        temperature=1.5,
    )

    print(response.choices[0].message.content)

    # data = json.loads(response.choices[0].message.content)
    # print(data)



if __name__ == '__main__':
    main()
import os
from typing import Literal
import ell
from mistralai import Mistral
from clients import register_clients
from gpqa import FourChoiceAnswer
from pydantic import BaseModel

def main():
    register_clients()

    @ell.complex("mistral-small-latest", max_tokens=32, response_format=FourChoiceAnswer)
    def explain(topic):
        """You are an expert in explaining technical topics"""
        return f"please explain {topic}"
    
    print(explain("Which is most accurate: A: Elo\nB: Glicko\nC: Bradley-Terry\nD:Glicko 2"))

def bug():
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    class NameResponse(BaseModel):
        # name: Literal['Assistant', 'Bot', 'AI', 'Bob']
        name: str

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


if __name__ == '__main__':
    # main()
    bug()
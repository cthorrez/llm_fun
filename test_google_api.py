import os
from typing import Literal
from enum import Enum
from pydantic import BaseModel
from google import genai
from google.genai import types

class Response(BaseModel):
    value: Literal['A', 'B', 'C', 'D']

class Choice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

def main():
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    question = 'Which of the following physical theories never requires regularization at high energies?'
    ans = "(A) Quantum Electrodynamics\n(B) Quantum Chromodynamics\n(C) Classical Electrodynamics\n(D) Superstring Theory"

    response = client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05",
        contents=[
            {'role': 'user', 'parts': [{'text': f'What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\".'}]},
            {'role': 'model', 'parts': [{'text': 'Let’s think step by step...'}]},
            {'role': 'model', 'parts': [{'text': 'The correct answer is '}]}
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            response_mime_type= 'application/json',
            response_schema=Choice,
            # safety_settings=[
            #     types.SafetySetting(
            #         category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            #         threshold=types.HarmBlockThreshold.OFF,
            #     )
            # ]
        )
    )
    print(response)
    print(response.text)

if __name__ == '__main__':
    main()
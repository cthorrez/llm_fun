import numpy as np
import polars as pl
from typing import Literal
from pydantic import BaseModel, Field
import ell
from clients import register_clients 
from utils import deterministic_hash

class FourChoiceAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D']
    # answer: str = Field(description="The answer, A, B, C, or D")


LETTERS = ['A', 'B', 'C', 'D']

def main():
    register_clients()
    model = "gemini-2.0-flash-lite-preview-02-05"

    # stop = [')', '.', '\n']
    stop = None
    config = {
        'response_mime_type': 'application/json',
        'response_schema': Literal['A', 'B', 'C', 'D'],
    }
    @ell.complex(model, response_format=FourChoiceAnswer, temperature=0.0, max_tokens=32)
    def zero_shot(question, answers) -> FourChoiceAnswer:
        ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
        return [
            ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
            ell.assistant("The correct answer is ")
        ]
    
    df = pl.read_parquet('data/gpqa.parquet').head(1)
    for row in df.to_dicts():
        question = row['Question']
        answers = [row['Correct Answer']] + [row[f'Incorrect Answer {idx}'] for idx in range(1,4)]
        ids = [deterministic_hash(ans) for ans in answers]
        idxs = np.argsort(ids)
        resp = zero_shot(question, [answers[idx] for idx in idxs])
        print(idxs)

        # print(question)
        print(resp)
        print(resp.parsed)
        print(resp.parsed.answer)
    



if __name__ == '__main__':
    main()
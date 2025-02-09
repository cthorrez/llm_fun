import time
import numpy as np
import polars as pl
from typing import Literal
from pydantic import BaseModel, Field
import ell
from clients import register_clients 
from utils import deterministic_hash
from google import genai
from google.genai import types

class FourChoiceAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D']

LETTERS = ['A', 'B', 'C', 'D']

def main():

    register_clients(timeout=10.0)
    model = "gemini-2.0-flash-lite-preview-02-05"

    @ell.complex(model, response_format=FourChoiceAnswer, temperature=0.0, max_tokens=32)
    def zero_shot(question, answers, force_retry=False) -> FourChoiceAnswer:
        ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
        return [
            ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
            ell.assistant("The correct answer is ")
        ]
    
    def zero_shot_cot(question, answers, force_retry=False):
        ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])

        @ell.simple(model, temperature=0.0, max_tokens=512, force_retry=force_retry)
        def cot():
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.assistant("Let’s think step by step:")
            ]
        thoughts = cot()

        @ell.complex(model, response_format=FourChoiceAnswer, temperature=1.0, max_tokens=32, force_retry=force_retry)
        def final_answer() -> FourChoiceAnswer:
            ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.assistant(f"Let’s think step by step:\n{thoughts}\nThe correct answer is ")
            ]
        return final_answer()
    
    # zs_results = run_eval(zero_shot)
    # print(f'Zero shot: {np.mean(zs_results)}')

    zs_cot_results = run_eval(zero_shot_cot)
    print(f'Zero shot CoT 512: {np.mean(zs_cot_results)}')

    
def run_eval(func):
    df = pl.read_parquet('data/gpqa.parquet')
    print(len(df))
    df  = df.slice(56,1)
    # df = df.with_row_index().filter(~pl.col('index').is_in({57, 185}))
    print(len(df))

    # print(df.slice(181,5)['Question'])
    df = df.head(200)
    max_retries = 3

    results = []
    for idx, row in enumerate(df.to_dicts()):
        question = row['Question']
        print(f'{idx}: {question}')
        answers = [row['Correct Answer']] + [row[f'Incorrect Answer {idx}'] for idx in range(1,4)]
        for ans in answers:
            print(ans)
        ids = [deterministic_hash(ans) for ans in answers]
        idxs = np.argsort(ids)
        shuffled_answers = [answers[idx] for idx in idxs]
        correct_idx = shuffled_answers.index(row['Correct Answer'])
        correct_letter = LETTERS[correct_idx]
        attempt_num = 0
        while attempt_num < max_retries:
            try:
                resp = func(question, shuffled_answers, force_retry=attempt_num>0)
                break
            except:
                print(f'attempt {attempt_num} failed, retrying')
                attempt_num += 1


        print(f'{idx}: {resp.parsed.answer}\n')
        results.append(float(resp.parsed.answer == correct_letter))

    return results
        
    

if __name__ == '__main__':
    main()
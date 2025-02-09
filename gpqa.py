import time
import numpy as np
import polars as pl
from types import SimpleNamespace
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

    register_clients(timeout=5.0)
    models = [
        "gemini-2.0-flash-lite-preview-02-05",
        "mistral-small-latest",
        "open-mistral-nemo",
        "mistral-large-latest",
    ][3:]

    prompts = [
        'zero_shot',
        # 'zero_shot_cot',
    ]

    for model in models:
        lmps = build_lmps(model)
        for prompt in prompts:
            print(f'Running eval with {model}, {prompt}')
            results = run_eval(lmps[prompt])
            print(f'Accuracy: {np.mean(results)}')

def build_lmps(model):
    
    def zero_shot(question, answers, force_retry=False):
        @ell.complex(model, max_tokens=32, response_format=FourChoiceAnswer, force_retry=force_retry)
        def zero_shot_lmp() -> FourChoiceAnswer:
            ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.user("The correct answer is ")
            ]
        return zero_shot_lmp()
    
    def zero_shot_cot(question, answers, force_retry=False):
        ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
        @ell.simple(model, max_tokens=512, force_retry=force_retry)
        def cot():
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.assistant("Let’s think step by step:")
            ]
        thoughts = cot()

        @ell.complex(model, response_format=FourChoiceAnswer, max_tokens=32, force_retry=force_retry)
        def final_answer() -> FourChoiceAnswer:
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.assistant(f"Let’s think step by step:\n{thoughts}"),
                ell.user("The correct answer is ")
            ]
        return final_answer()
    
    return {
        'zero_shot' : zero_shot,
        'zero_shot_cot': zero_shot_cot,
    }
    
def run_eval(func, n=int(1e9)):
    df = pl.read_parquet('data/gpqa.parquet')
    # df  = df.slice(57,1)
    # df = df.with_row_index().filter(~pl.col('index').is_in({57, 185}))

    # print(df.slice(181,5)['Question'])
    df = df.head(n)
    max_retries = 3

    results = []
    for idx, row in enumerate(df.to_dicts()):
        question = row['Question']
        print(f'{idx}: {question}')
        answers = [row['Correct Answer']] + [row[f'Incorrect Answer {idx}'] for idx in range(1,4)]
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
            except Exception as e:
                print(e)
                if attempt_num + 1 < max_retries:
                    print(f'attempt {attempt_num} failed, retrying')
                attempt_num += 1
                resp = None
        
        if resp is None:
            print(f'failed {max_retries} times, guessing A')
            resp = SimpleNamespace(parsed=SimpleNamespace(answer='A'))

        print(f'{idx}: {resp.parsed.answer}\n')
        results.append(float(resp.parsed.answer == correct_letter))

    return results
        
    

if __name__ == '__main__':
    main()
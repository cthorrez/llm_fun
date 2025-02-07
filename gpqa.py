import os
import numpy as np
import polars as pl
import ell.stores
import ell.stores.sql
from ell.configurator import config
import openai
import ell
from cache_utils import CachedOpenAIProvider
import clients
from utils import deterministic_hash


LETTERS = ['A', 'B', 'C', 'D']

def main():
    model = "gemini-2.0-flash-lite-preview-02-05"

    @ell.simple(model, temperature=0.9, max_tokens=1000)
    def zero_shot(question, answers):
        ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
        return [
            ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\".")
        ]
    
    df = pl.read_parquet('data/gpqa.parquet').head(3)
    for row in df.to_dicts():
        question = row['Question']
        answers = [row['Correct Answer']] + [row[f'Incorrect Answer {idx}'] for idx in range(1,4)]
        ids = [deterministic_hash(ans) for ans in answers]
        idxs = np.argsort(ids)
        resp = zero_shot(question, [answers[idx] for idx in idxs])

        print(question)
        print(resp)
    



if __name__ == '__main__':
    main()
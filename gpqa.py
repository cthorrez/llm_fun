import time
from tqdm import tqdm
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Literal
from pydantic import BaseModel
import ell
from registration import register_clients 
from utils import deterministic_hash

class FourChoiceAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D']

LETTERS = ['A', 'B', 'C', 'D']
TIMEOUT = 2.5


def main():
    register_clients(timeout=TIMEOUT)
    models = [
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-flash",
        # "gemini-2.0-pro-exp-02-05", # limit too small
        "mistral-small-latest",
        "open-mixtral-8x7b",
        "open-mixtral-8x22b",
        "open-mistral-nemo",
        "llama-3.3-70b-instruct",
        "command-r-plus-08-2024",
        "deepseek-r1",
        "deepseek-r1-distill-llama-70b",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20"
        "gpt-4o-mini-2024-07-18",
    ]

    prompts = [
        'zero_shot',
        'zero_shot_cot',
    ]

    all_results = []

    for model in models:
        lmps = build_lmps(model)
        for prompt in prompts:
            print(f'Running eval with {model}, {prompt}')
            results_df = run_eval(
                lmps[prompt], 
                model=model,
                prompt=prompt,
                n=200
            )
            all_results.append(results_df)

    # Combine and pivot results
    combined_df = pl.concat(all_results)
    
    # Create model-prompt column for pivoting
    combined_df = combined_df.with_columns(
        (pl.col("model") + "-" + pl.col("prompt")).alias("model_prompt")
    )

    # Pivot to wide format
    pivot_df = combined_df.pivot(
        values="result",
        index=["question", "correct_letter"],
        on="model_prompt",
        aggregate_function="first"
    )

    # Calculate solve rate across all model-prompt pairs
    model_prompt_cols = [col for col in pivot_df.columns 
                       if col not in ["question", "correct_letter"]]
    
    pivot_df = pivot_df.with_columns(
        pl.mean_horizontal(model_prompt_cols).alias("solve_rate")
    )

    for col in model_prompt_cols:
        print(f'{col}: {pivot_df[col].to_numpy().mean():.4f} ')

    # Save and print results
    pivot_df.write_csv("data/collated_results.csv")
    print("\nFinal results dataframe:")
    print(pivot_df.head())

    plt.hist(pivot_df['solve_rate'], bins=11)
    plt.show()
    
    return pivot_df

def build_lmps(model):
    
    def zero_shot(question, answers, force_retry=False, **kwargs):
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 32
        @ell.complex(model, response_format=FourChoiceAnswer, force_retry=force_retry, **kwargs)
        def zero_shot_lmp() -> FourChoiceAnswer:
            ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.user("The correct answer is ")
            ]
        return zero_shot_lmp()
    
    def zero_shot_cot(question, answers, force_retry=False, **kwargs):
        ans = '\n'.join([f'({letter}) {answer}' for letter, answer in zip(LETTERS, answers)])

        cot_kwargs = {k:v for k,v in kwargs.items() if k != 'max_tokens'}
        @ell.simple(model, max_tokens=512, force_retry=force_retry, **cot_kwargs)
        def cot():
            return [
                ell.user(f"What is the correct answer to this question: {question}\nChoices:\n{ans}\nFormat your response as follows: \"The correct answer is (insert answer here)\"."),
                ell.assistant("Let’s think step by step:")
            ]
        thoughts = cot()

        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 32
        @ell.complex(model, response_format=FourChoiceAnswer, force_retry=force_retry, **kwargs)
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
    
def run_eval(func, model: str, prompt: str, n: int = int(1e9)):
    df = pl.read_parquet('data/gpqa.parquet').head(n)
    max_retries = 3
    records = []

    for row in tqdm(df.to_dicts()):
        question = row['Question']
        answers = [row['Correct Answer']] + [row[f'Incorrect Answer {idx}'] 
                 for idx in range(1,4)]
        
        # Consistent answer shuffling
        ids = [deterministic_hash(ans) for ans in answers]
        idxs = np.argsort(ids)
        shuffled_answers = [answers[idx] for idx in idxs]
        correct_idx = shuffled_answers.index(row['Correct Answer'])
        correct_letter = LETTERS[correct_idx]

        # Retry logic
        result = None
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    resp = func(question, shuffled_answers)
                else:
                    resp = func(question, shuffled_answers, force_retry=True)
                result = float(resp.parsed.answer == correct_letter)
                break
            except Exception as e:
                print(e)
                if attempt == max_retries - 1:
                    result = float(correct_letter == 'A')
                    print(f"Failed after {max_retries} attempts for: {question[:50]}...\nGuessing A")

        records.append({
            "question": question,
            "correct_letter": correct_letter,
            "result": result,
            "model": model,
            "prompt": prompt
        })

    return pl.DataFrame(records)
        
    

if __name__ == '__main__':
    main()
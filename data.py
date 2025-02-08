import os
from datasets import load_dataset

def get_data():
    os.makedirs('data', exist_ok=True)
    df = load_dataset("Idavidrein/gpqa", 'gpqa_diamond', split='train').to_polars().select(
        'Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3'
    )
    df.write_parquet('data/gpqa.parquet')

if __name__ == '__main__':
    get_data()
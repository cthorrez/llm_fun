from datasets import load_dataset

def get_data():
    dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond', split='train').to_polars()
    print(dataset.columns)

if __name__ == '__main__':
    get_data()
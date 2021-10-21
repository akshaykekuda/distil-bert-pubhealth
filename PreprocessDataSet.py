import pandas as pd
from datasets import Dataset

class PreprocessDataSet:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def preprocess_data(self):
        # remove samples with lables=-1
        df = pd.DataFrame(self.dataset['train'])
        df = df.drop(index = df[(df.label==-1)].index.tolist())
        dataset_train = Dataset.from_pandas(df)
        df = pd.DataFrame(self.dataset['test'])
        df = df.drop(index = df[(df.label==-1)].index.tolist())
        dataset_test = Dataset.from_pandas(df)
        df = pd.DataFrame(self.dataset['validation'])
        df = df.drop(index = df[(df.label==-1)].index.tolist())
        dataset_val = Dataset.from_pandas(df)
        dataset_train = dataset_train.remove_columns(['claim_id', 'date_published', 'fact_checkers', 'main_text', 'sources', 'subjects'])
        dataset_test = dataset_test.remove_columns(['claim_id', 'date_published', 'fact_checkers', 'main_text', 'sources', 'subjects'])
        dataset_val = dataset_val.remove_columns(['claim_id', 'date_published', 'fact_checkers', 'main_text', 'sources', 'subjects'])
        return dataset_train, dataset_test, dataset_val
    
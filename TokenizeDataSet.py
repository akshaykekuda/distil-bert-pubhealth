class TokenizeDataSet:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def preprocess_function(self, examples):
        return self.tokenizer(examples['explanation'], truncation=True)

    def preprocess_function1(self, examples):
        return self.tokenizer(examples['claim'], examples['explanation'], truncation=True)

    def tokenize_dataset(self, dataset):
        encoded_dataset = dataset.map(self.preprocess_function1, batched=True)
        return encoded_dataset

class TokenizeDataSet:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def preprocess_function(self, examples):
        # use of only the explanation as input to the model
        return self.tokenizer(examples['explanation'], truncation=True)

    def preprocess_function1(self, examples):
        # use of both claim and explanation as inputs to the model
        return self.tokenizer(examples['claim'], examples['explanation'], truncation=True)

    def tokenize_dataset(self, dataset):
        # tokenize the dataset
        encoded_dataset = dataset.map(self.preprocess_function1, batched=True)
        return encoded_dataset

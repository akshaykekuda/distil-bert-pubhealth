from Model import Model
from TokenizeDataSet import TokenizeDataSet
from PreprocessDataSet import PreprocessDataSet
from datasets import load_dataset, Dataset, load_metric
import sys

if __name__=="__main__":
    dataset = load_dataset('health_fact')
    ds =  PreprocessDataSet(dataset)
    dataset_train, dataset_val, dataset_test = ds.preprocess_data()
    
    if sys.argv[1] == 'full_train':
        model_checkpoint = "distilbert-base-uncased"
        model = Model(model_checkpoint)
        tk=TokenizeDataSet(model.tokenizer)
        encoded_train_dataset = tk.tokenize_dataset(dataset_train)
        encoded_val_dataset = tk.tokenize_dataset(dataset_val)
        trainer = model.load_trainer(encoded_train_dataset, encoded_val_dataset)
        trainer.train()
        encoded_test_dataset = tk.tokenize_dataset(dataset_test)
        print(trainer.evaluate(encoded_test_dataset))
        model_path = "best_model"
        model.model.save_pretrained(model_path)
        model.tokenizer.save_pretrained(model_path)
    if sys.argv[1] == "testing":
        model_checkpoint = "best_model"
        model = Model(model_checkpoint)
        finetuned_model = model.model
        tk = TokenizeDataSet(model.tokenizer)
        encoded_test_dataset = tk.tokenize_dataset(dataset_test)
        eval = model.load_evaluator()
        print(eval.evaluate(encoded_test_dataset))







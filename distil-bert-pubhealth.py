from Model import Model
from TokenizeDataSet import TokenizeDataSet
from PreprocessDataSet import PreprocessDataSet
from datasets import load_dataset, Dataset, load_metric
import sys

def evaluate_best_model(model_checkpoint, dataset_test):
    # evaluate the best model on the test set
    # args: location of the fine-tuned model, test dataset
    best_distil_bert = Model(model_checkpoint)
    tk = TokenizeDataSet(best_distil_bert.tokenizer)
    encoded_test_dataset = tk.tokenize_dataset(dataset_test)
    tester = best_distil_bert.load_evaluator()
    print(tester.evaluate(encoded_test_dataset, ignore_keys=['eval_loss']))

def fine_tune_distil_bert(model_checkpoint, dataset_train, dataset_val):
    #finetune distil bert model on the pubhealth dataset and save the best model
    # args: model checkpoint to finetune, training dataset and validation dataset
    distil_bert = Model(model_checkpoint)
    tk=TokenizeDataSet(distil_bert.tokenizer)
    encoded_train_dataset = tk.tokenize_dataset(dataset_train)
    encoded_val_dataset = tk.tokenize_dataset(dataset_val)
    trainer = distil_bert.load_trainer(encoded_train_dataset, encoded_val_dataset)
    trainer.train()
    encoded_test_dataset = tk.tokenize_dataset(dataset_test)
    print(trainer.evaluate(encoded_test_dataset))
    distil_bert.save_best_model(model_path = "distil-bert-finetuned-pubhealth")

if __name__=="__main__":
    # Calls functions to fully train the model and to test the fine-tuned model
    # Recommended to run the training on GPU nodes

    dataset = load_dataset('health_fact')
    ds =  PreprocessDataSet(dataset)
    dataset_train, dataset_val, dataset_test = ds.preprocess_data()
    setting = sys.argv[1]
    if setting == 'fine-tune':
        model_checkpoint = "distilbert-base-uncased"
        fine_tune_distil_bert(model_checkpoint, dataset_train, dataset_val)

    elif setting == "testing":
        model_checkpoint = "distil-bert-finetuned-pubhealth-best"
        evaluate_best_model(model_checkpoint, dataset_test)
    else:
        print("Invalid Argument")








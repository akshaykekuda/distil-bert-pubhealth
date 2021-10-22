from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric


class Model:
    def __init__(self, model_checkpoint):
        #initialize the model to finetune
        # Model class has been customized for this task, the hyperparameters here can be changed as needed
        self.model_checkpoint = model_checkpoint
        self.metric = load_metric('accuracy')
        self.batch_size = 32
        self.num_labels = 4
        self.epoch_len = 10
        self.metric_name = "accuracy"                     
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        self.model_name = self.model_checkpoint.split("/")[-1]       

    def compute_metrics(self, eval_pred):
        #metric to evaluate the performance of the model
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)


    def load_trainer(self, encoded_train_dataset, encoded_val_dataset):
        # create a trainer class for fine-tuning the model
        args = TrainingArguments(
        f"{self.model_name}-finetuned-pubhealth",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=self.batch_size,
        per_device_eval_batch_size=self.batch_size,
        num_train_epochs=self.epoch_len,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=self.metric_name,
        )

        trainer = Trainer(
            self.model,
            args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        return trainer

    def load_evaluator(self, ):
        # create a evaluator class to evaluate the fine-tuned model
        
        evaluator = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            compute_metrics = self.compute_metrics
        )
        return evaluator
    def save_best_model(self, model_path):
        # save the best model in model_path
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

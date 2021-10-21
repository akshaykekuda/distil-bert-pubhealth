from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric


class Model:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.metric = load_metric('accuracy')
        self.batch_size = 32
        self.num_labels = 4
        self.metric_name = "accuracy"                     
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        self.model_name = self.model_checkpoint.split("/")[-1]       

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)


    def load_trainer(self, encoded_train_dataset, encoded_val_dataset):
        args = TrainingArguments(
        f"{self.model_name}-finetuned-pubhealth",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=self.batch_size,
        per_device_eval_batch_size=self.batch_size,
        num_train_epochs=10,
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
        trainer = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            compute_metrics = self.compute_metrics
        )
        return trainer

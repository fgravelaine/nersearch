import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
import datasets
import os
from ankor_tokenizer import AnkorSearchTokenizer
from ankorsearch_dataset import AnkorSearchDataset

metric = load_metric("seqeval")

logger = datasets.logging.get_logger(__name__)


def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == "__main__":
    model_n_version = "ankorsearchv1"
    max_epochs = 50
    learning_rate = 2e-5
    batch_size = 16
    model_root_dir = "~/.ankorsearchv1/models/hf/"
    #model = "bert-base-multilingual-uncased"
    model = "bert-base-uncased"

    ankor_dataset = AnkorSearchDataset()
    ankor_preprocessor = AnkorSearchTokenizer(model)

    hf_model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(ankor_dataset.labels))

    hf_model.config.id2label = ankor_dataset.id2label
    hf_model.config.label2id = ankor_dataset.label2id

    tokenized_datasets = ankor_dataset.dataset.map(ankor_preprocessor.tokenize_and_align_labels, batched=True)

    # ---------------------------------------------------------------------------------------------------

    args = TrainingArguments(
        f"test-ner",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(ankor_preprocessor.tokenizer)

    trainer = Trainer(
        hf_model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=ankor_preprocessor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p=p, label_list=ankor_dataset.labels)
    )

    trainer.train()
    trainer.evaluate()

    # Predictions on test dataset and evaluation
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ankor_dataset.labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [ankor_dataset.labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)

    out_dir = os.path.expanduser(model_root_dir) + "/" + model_n_version
    print(out_dir)
    trainer.save_model(out_dir)


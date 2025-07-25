import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import os

DATA_PATH = '../data/greenwashing_dataset.csv'
MODEL_DIR = '../models/greenwashing_detector'


def main():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    dataset = Dataset.from_pandas(df[['text', 'label']])

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_steps=10,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()
    trainer.evaluate()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

if __name__ == '__main__':
    main()

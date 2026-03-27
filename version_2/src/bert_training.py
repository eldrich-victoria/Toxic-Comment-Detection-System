import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split


# -----------------------------
# DEVICE SETUP
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training.")


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    print("Loading data...")

    df = pd.read_csv("data/processed/cleaned_data.csv")

    df = df[["clean_text", "target"]]

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["target"] = df["target"].fillna(0).astype(int)

    df = df[df["clean_text"].str.strip() != ""]

    # 🔥 USE 100k DATA
    df = df.sample(n=100000, random_state=42)

    print("Final dataset:", df.shape)

    return df


# -----------------------------
# PREPARE DATA
# -----------------------------

def prepare_data(df):

    print("Splitting data (80/20)...")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    return train_df, test_df


# -----------------------------
# TOKENIZATION
# -----------------------------

def tokenize_data(train_df, test_df):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("target", "labels")
    test_dataset = test_dataset.rename_column("target", "labels")

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset, test_dataset, tokenizer


# -----------------------------
# TRAIN MODEL WITH CHECKPOINT
# -----------------------------

def train_model(train_dataset, test_dataset):

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    ).to(device)

    # -----------------------------
    # STEP CALCULATION FOR 100 CHECKPOINTS
    # -----------------------------

    train_size = len(train_dataset)
    batch_size = 8
    steps_per_epoch = train_size // batch_size
    total_steps = steps_per_epoch * 3

    save_steps = max(1, total_steps // 100)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Checkpoint every: {save_steps} steps")

    # -----------------------------
    # TRAINING CONFIG
    # -----------------------------

    training_args = TrainingArguments(
        output_dir="models/bert",

        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,

        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=100,

        logging_steps=50,

        load_best_model_at_end=False,
        fp16=(device.type == "cuda"),  # auto enable if GPU

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Training BERT with checkpointing...")

    checkpoint_dir = "models/bert"

    # -----------------------------
    # RESUME LOGIC (FIXED)
    # -----------------------------

    if os.path.exists(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint")
        ]

        if len(checkpoints) > 0:
            latest_checkpoint = max(
                checkpoints,
                key=lambda x: int(x.split("-")[-1])
            )
            print(f"Resuming from: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
            return trainer.model

    print("No checkpoint found. Starting fresh training...")
    trainer.train()

    return trainer.model


# -----------------------------
# SAVE MODEL
# -----------------------------

def save_model(model, tokenizer):

    os.makedirs("models/bert", exist_ok=True)

    model.save_pretrained("models/bert")
    tokenizer.save_pretrained("models/bert")

    print("BERT model saved.")


# -----------------------------
# MAIN
# -----------------------------

def main():

    df = load_data()

    train_df, test_df = prepare_data(df)

    train_dataset, test_dataset, tokenizer = tokenize_data(train_df, test_df)

    model = train_model(train_dataset, test_dataset)

    save_model(model, tokenizer)

    print("BERT training complete.")


if __name__ == "__main__":
    main()
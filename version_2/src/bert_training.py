import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments
)

from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# PROJECT PATHS
# ============================================================

# Expected location:
# D:\Toxic-Comment-Detection-System\version_2\src\bert_training.py

VERSION_2_DIR = Path(__file__).resolve().parents[1]

PROJECT_ROOT = VERSION_2_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

BERT_DIR = (
    VERSION_2_DIR
    / "models"
    / "bert"
)

CHECKPOINT_DIR = (
    BERT_DIR
    / "checkpoints"
)

FINAL_DIR = (
    BERT_DIR
    / "final"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

RANDOM_STATE = 42

DATASET_SIZE = 100000

TEST_SIZE = 0.20

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5

WARMUP_STEPS = 500

WEIGHT_DECAY = 0.01

MAX_LENGTH = 128

CHECKPOINT_COUNT = 100

LOGGING_STEPS = 50


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is not available. "
        "BERT training has been stopped to prevent CPU training."
    )

device = torch.device("cuda")

print(
    "Using GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA version:",
    torch.version.cuda
)

print(
    "GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3),
        2
    ),
    "GB"
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def prepare_directories():

    BERT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND LAST CHECKPOINT
# ============================================================

def find_last_checkpoint():

    if not CHECKPOINT_DIR.exists():

        return None

    last_checkpoint = get_last_checkpoint(
        str(CHECKPOINT_DIR)
    )

    if last_checkpoint is None:

        print(
            "\nNo previous checkpoint found."
        )

        return None

    print(
        "\nPrevious checkpoint found:"
    )

    print(
        last_checkpoint
    )

    return last_checkpoint


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("BERT TRAINING")
    print("=" * 70)

    print(
        "\nDevice:",
        device
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "\nLoading dataset..."
    )

    print(
        "Dataset path:"
    )

    print(
        DATA_PATH
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "Dataset not found:\n"
            + str(DATA_PATH)
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "\nOriginal dataset shape:"
    )

    print(
        df.shape
    )

    required_columns = [
        "clean_text",
        "target"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if len(missing_columns) > 0:

        raise KeyError(
            "Missing required columns: "
            + str(missing_columns)
        )

    df = df[
        [
            "clean_text",
            "target"
        ]
    ].copy()

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    df["clean_text"] = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # CLEAN TARGET
    # --------------------------------------------------------

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["target"]
    )

    df["target"] = (
        df["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # REMOVE EMPTY TEXT
    # --------------------------------------------------------

    df = df[
        df["clean_text"]
        .str.strip()
        .ne("")
    ]

    # --------------------------------------------------------
    # KEEP BINARY TARGETS
    # --------------------------------------------------------

    df = df[
        df["target"].isin([0, 1])
    ]

    if len(df) == 0:

        raise ValueError(
            "No valid training data remains."
        )

    if df["target"].nunique() < 2:

        raise ValueError(
            "Dataset must contain both target classes."
        )

    # --------------------------------------------------------
    # SAMPLE 100K
    # --------------------------------------------------------

    sample_size = min(
        DATASET_SIZE,
        len(df)
    )

    print(
        "\nUsing",
        sample_size,
        "samples."
    )

    df = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    print(
        "\nFinal dataset shape:"
    )

    print(
        df.shape
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def prepare_data(
    df
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "=" * 70
    )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"]
    )

    train_df = train_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    print(
        "\nTrain:",
        train_df.shape
    )

    print(
        "Test:",
        test_df.shape
    )

    print(
        "\nTrain class distribution:"
    )

    print(
        train_df["target"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTest class distribution:"
    )

    print(
        test_df["target"]
        .value_counts()
        .sort_index()
    )

    return (
        train_df,
        test_df
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_data(
    train_df,
    test_df
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOKENIZATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading tokenizer:",
        MODEL_NAME
    )

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize(
        batch
    ):

        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    print(
        "\nCreating Hugging Face datasets..."
    )

    train_dataset = Dataset.from_pandas(
        train_df
    )

    test_dataset = Dataset.from_pandas(
        test_df
    )

    print(
        "Tokenizing training dataset..."
    )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True
    )

    print(
        "Tokenizing test dataset..."
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True
    )

    train_dataset = train_dataset.rename_column(
        "target",
        "labels"
    )

    test_dataset = test_dataset.rename_column(
        "target",
        "labels"
    )

    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    print(
        "\nTokenization completed."
    )

    return (
        train_dataset,
        test_dataset,
        tokenizer
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_pred
):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_dataset,
    test_dataset,
    last_checkpoint
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BERT MODEL TRAINING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # MODEL INITIALIZATION
    # --------------------------------------------------------

    if last_checkpoint is not None:

        print(
            "\nResuming BERT from checkpoint:"
        )

        print(
            last_checkpoint
        )

        model = BertForSequenceClassification.from_pretrained(
            last_checkpoint,
            num_labels=2
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting fresh from:",
            MODEL_NAME
        )

        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

    model.to(
        device
    )

    # --------------------------------------------------------
    # STEP CALCULATION
    # --------------------------------------------------------

    train_size = len(
        train_dataset
    )

    steps_per_epoch = max(
        1,
        train_size // TRAIN_BATCH_SIZE
    )

    total_steps = (
        steps_per_epoch
        * NUM_EPOCHS
    )

    save_steps = max(
        1,
        total_steps // CHECKPOINT_COUNT
    )

    print(
        "\nTraining samples:",
        train_size
    )

    print(
        "Batch size:",
        TRAIN_BATCH_SIZE
    )

    print(
        "Steps per epoch:",
        steps_per_epoch
    )

    print(
        "Total training steps:",
        total_steps
    )

    print(
        "Checkpoint interval:",
        save_steps
    )

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            CHECKPOINT_DIR
        ),

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        weight_decay=WEIGHT_DECAY,

        save_strategy="steps",

        save_steps=save_steps,

        eval_strategy="steps",

        eval_steps=save_steps,

        save_total_limit=CHECKPOINT_COUNT,

        logging_steps=LOGGING_STEPS,

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        fp16=True,

        report_to="none",

        seed=RANDOM_STATE
    )

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        compute_metrics=compute_metrics
    )

    # --------------------------------------------------------
    # TRAIN / RESUME
    # --------------------------------------------------------

    if last_checkpoint is not None:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RESUMING TRAINING"
        )

        print(
            "=" * 70
        )

        print(
            "\nResuming from:"
        )

        print(
            last_checkpoint
        )

        trainer.train(
            resume_from_checkpoint=last_checkpoint
        )

    else:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "STARTING FRESH TRAINING"
        )

        print(
            "=" * 70
        )

        trainer.train()

    print(
        "\nTraining process completed."
    )

    # --------------------------------------------------------
    # GET TRAINED MODEL
    # --------------------------------------------------------

    trained_model = trainer.model

    if trained_model is None:

        raise RuntimeError(
            "Trainer did not return a trained model."
        )

    return trained_model


# ============================================================
# SAVE TOKENIZER
# ============================================================

def save_tokenizer(
    tokenizer
):

    print(
        "\nSaving tokenizer..."
    )

    tokenizer.save_pretrained(
        str(FINAL_DIR)
    )

    print(
        "Tokenizer saved."
    )


# ============================================================
# SAVE FINAL BERT MODEL
# ============================================================

def save_final_model(
    model
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVING FINAL BERT MODEL"
    )

    print(
        "=" * 70
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nFinal model directory:"
    )

    print(
        FINAL_DIR
    )

    # --------------------------------------------------------
    # SAVE MODEL WEIGHTS
    # --------------------------------------------------------

    print(
        "\nWriting model weights..."
    )

    model.save_pretrained(
        str(FINAL_DIR),
        safe_serialization=True
    )

    # --------------------------------------------------------
    # VERIFY WEIGHTS
    # --------------------------------------------------------

    safetensors_path = (
        FINAL_DIR
        / "model.safetensors"
    )

    pytorch_bin_path = (
        FINAL_DIR
        / "pytorch_model.bin"
    )

    if safetensors_path.exists():

        print(
            "\nFOUND:",
            safetensors_path
        )

        weight_file = "model.safetensors"

    elif pytorch_bin_path.exists():

        print(
            "\nFOUND:",
            pytorch_bin_path
        )

        weight_file = "pytorch_model.bin"

    else:

        raise RuntimeError(
            "\nCRITICAL ERROR: "
            "BERT model weights were not saved."
        )

    # --------------------------------------------------------
    # VERIFY CONFIG
    # --------------------------------------------------------

    config_path = (
        FINAL_DIR
        / "config.json"
    )

    if not config_path.exists():

        raise RuntimeError(
            "BERT config.json was not saved."
        )

    print(
        "FOUND:",
        config_path
    )

    # --------------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------------

    manifest = {
        "model_name": MODEL_NAME,
        "num_labels": 2,
        "max_length": MAX_LENGTH,
        "training_samples": DATASET_SIZE,
        "epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_state": RANDOM_STATE,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "weight_file": weight_file
    }

    manifest_path = (
        FINAL_DIR
        / "training_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print(
        "FOUND:",
        manifest_path
    )


# ============================================================
# VERIFY TOKENIZER
# ============================================================

def verify_tokenizer():

    required_tokenizer_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]

    print(
        "\nVerifying tokenizer files..."
    )

    missing = []

    for filename in required_tokenizer_files:

        path = FINAL_DIR / filename

        if path.exists():

            print(
                "FOUND:",
                filename
            )

        else:

            missing.append(
                filename
            )

    if len(missing) > 0:

        raise RuntimeError(
            "Missing tokenizer files: "
            + str(missing)
        )


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

def verify_saved_model():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MODEL RELOAD VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nReloading BERT from disk..."
    )

    try:

        tokenizer = BertTokenizer.from_pretrained(
            str(FINAL_DIR)
        )

        model = BertForSequenceClassification.from_pretrained(
            str(FINAL_DIR)
        )

    except Exception as error:

        raise RuntimeError(
            "FINAL BERT MODEL COULD NOT BE RELOADED.\n"
            "The saved model is not considered valid.\n"
            + str(error)
        )

    model.to(
        device
    )

    model.eval()

    print(
        "\nBERT model successfully reloaded."
    )

    # --------------------------------------------------------
    # TEST INFERENCE
    # --------------------------------------------------------

    test_text = [
        "This is a simple test comment."
    ]

    inputs = tokenizer(
        test_text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    predictions = torch.argmax(
        outputs.logits,
        dim=1
    )

    print(
        "Test inference successful."
    )

    print(
        "Test prediction:",
        predictions.cpu().tolist()
    )

    return True


# ============================================================
# FINAL ARTIFACT REPORT
# ============================================================

def print_final_artifacts():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL BERT ARTIFACTS"
    )

    print(
        "=" * 70
    )

    for path in sorted(
        FINAL_DIR.iterdir()
    ):

        if path.is_file():

            size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            print(
                path.name,
                "->",
                round(
                    size_mb,
                    2
                ),
                "MB"
            )


# ============================================================
# CHECKPOINT REPORT
# ============================================================

def print_checkpoint_status():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHECKPOINT STATUS"
    )

    print(
        "=" * 70
    )

    if not CHECKPOINT_DIR.exists():

        print(
            "Checkpoint directory does not exist."
        )

        return

    checkpoints = sorted(
        [
            path
            for path in CHECKPOINT_DIR.iterdir()
            if path.is_dir()
            and path.name.startswith("checkpoint-")
        ],
        key=lambda path: int(
            path.name.split("-")[-1]
        )
    )

    if len(checkpoints) == 0:

        print(
            "No checkpoints found."
        )

        return

    print(
        "Available checkpoints:"
    )

    for checkpoint in checkpoints:

        print(
            " -",
            checkpoint.name
        )

    print(
        "\nLatest checkpoint:"
    )

    print(
        checkpoints[-1]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nBERT TRAINING STARTED."
    )

    print(
        "GPU resume-enabled training."
    )

    prepare_directories()

    # --------------------------------------------------------
    # CHECK FOR EXISTING CHECKPOINT
    # --------------------------------------------------------

    last_checkpoint = find_last_checkpoint()

    if last_checkpoint is not None:

        print(
            "\nExisting checkpoint detected."
        )

        print(
            "Training will RESUME from:"
        )

        print(
            last_checkpoint
        )

    else:

        print(
            "\nNo existing checkpoint detected."
        )

        print(
            "Training will START FRESH."
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_df,
        test_df
    ) = prepare_data(
        df
    )

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    (
        train_dataset,
        test_dataset,
        tokenizer
    ) = tokenize_data(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # TRAIN / RESUME
    # --------------------------------------------------------

    model = train_model(
        train_dataset,
        test_dataset,
        last_checkpoint
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    save_final_model(
        model
    )

    save_tokenizer(
        tokenizer
    )

    # --------------------------------------------------------
    # VERIFY EVERYTHING
    # --------------------------------------------------------

    verify_tokenizer()

    verify_saved_model()

    print_final_artifacts()

    print_checkpoint_status()

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BERT TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nFINAL MODEL:"
    )

    print(
        FINAL_DIR
    )

    print(
        "\nThe final BERT model was:"
    )

    print(
        "1. Trained or resumed from checkpoint"
    )

    print(
        "2. Saved to the final model directory"
    )

    print(
        "3. Verified for model weights"
    )

    print(
        "4. Reloaded successfully"
    )

    print(
        "5. Tested with inference"
    )

    print(
        "\nNext stage: evaluation.py"
    )


if __name__ == "__main__":
    main()
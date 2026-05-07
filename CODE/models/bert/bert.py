#!/usr/bin/env python3

"""
make sure to pip install
!pip install transformers datasets scikit-learn -q
!pip install gradio -q
!pip install bertviz -q
"""
import os
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns

import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix
)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR.parent
BERT_EXP_PATH = PROJECT_DIR / "models/bert/experiments/restaurants"
CHECKPOINTS = BERT_EXP_PATH / "checkpoints"
TRAIN_CSV = DATA_DIR / "MISC/data/combined/train.csv"
VAL_CSV = DATA_DIR / "MISC/data/combined/val.csv"
for d in [CHECKPOINTS, BERT_EXP_PATH]:
    os.makedirs(d, exist_ok=True)


# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Training on T4 GPU in Colab.")


# Load data
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
train_df["label"] = train_df["polarity"].map(LABEL_MAP)
val_df["label"]   = val_df["polarity"].map(LABEL_MAP)

print(f"\nTrain rows : {len(train_df)}")
print(f"Val rows   : {len(val_df)}")



# Tokeniser
MODEL_NAME = "bert-base-uncased"
tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)

# Check max token length on training set
lengths = []
for _, row in train_df.iterrows():
    enc = tokenizer(row["text"], row["target"], truncation=False)
    lengths.append(len(enc["input_ids"]))
lengths = np.array(lengths)
MAX_LENGTH = 128 if (lengths > 128).mean() <= 0.05 else 256


# Dataset class
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["text"], 
            row["target"], 
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }
train_dataset = ABSADataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = ABSADataset(val_df, tokenizer, MAX_LENGTH)

# Class weights
train_labels = train_df["label"].values
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_labels,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
    }

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(weight=class_weights_tensor)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training arguments
EPOCHS = 6
BATCH_SIZE = 16
total_steps = math.ceil(len(train_dataset) / BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)
training_args = TrainingArguments(
    output_dir = str(CHECKPOINTS),
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 32,
    learning_rate = 2e-5,
    weight_decay = 0.01,
    warmup_steps = warmup_steps,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "macro_f1",
    greater_is_better = True,
    logging_steps = 50,
    report_to = "none",
    fp16 = torch.cuda.is_available(),
    seed = SEED,
)


# Load model and train
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)

trainer = WeightedTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics = compute_metrics,
)


# Train

train_result = trainer.train()

# Save best model
trainer.save_model(str(BERT_EXP_PATH))
tokenizer.save_pretrained(str(BERT_EXP_PATH))
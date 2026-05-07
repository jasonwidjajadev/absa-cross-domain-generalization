# code for fine-tuning the bert restaurant model on the laptop dataset
# this work is an extension of the code written by Jason in bert.py and evaluation.py

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    AutoTokenizer,
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
RESTAURANT_MODEL = PROJECT_DIR / "models/bert/experiments/restaurants"
LAPTOP_EXP_PATH = PROJECT_DIR / "models/bert/experiments/laptop"
CHECKPOINTS = LAPTOP_EXP_PATH / "checkpoints"
TRAIN_CSV = DATA_DIR / "MISC/data/semeval2014/laptop/processed/train.csv"
VAL_CSV = DATA_DIR / "MISC/data/semeval2014/laptop/processed/val.csv"
TEST_CSV = DATA_DIR / "MISC/data/semeval2014/laptop/processed/test.csv"

os.makedirs(LAPTOP_EXP_PATH, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NAME = "laptop"
MODEL_NAME = "bert finetuned on laptop"

# Load the tokeniser and the model
print(f"\nLoading restaurant model from: {RESTAURANT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(str(RESTAURANT_MODEL))
model = BertForSequenceClassification.from_pretrained(str(RESTAURANT_MODEL))
model.to(device)

# Dataset class
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
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

# Load the data
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df = train_df[train_df["polarity"] != "conflict"].reset_index(drop=True)
val_df = val_df[val_df["polarity"] != "conflict"].reset_index(drop=True)
test_df = test_df[test_df["polarity"] != "conflict"].reset_index(drop=True)

print(f"Train rows: {len(train_df)}")
print(f"Val rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}")
print(f"\nTest polarity distribution:")

train_dataset = ABSADataset(train_df, tokenizer)
val_dataset = ABSADataset(val_df,   tokenizer)
test_dataset = ABSADataset(test_df,  tokenizer)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_df["label"].values,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# custom trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=class_weights_tensor)(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }

# Training args
training_args = TrainingArguments(
    output_dir = str(CHECKPOINTS),
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate = 1e-5,
    weight_decay = 0.01,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "macro_f1",
    greater_is_better = True,
    logging_steps = 50,
    seed = SEED,
    report_to = "none",
)

trainer = WeightedTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics = compute_metrics,
)
trainer.train()

model.save_pretrained(str(LAPTOP_EXP_PATH))
tokenizer.save_pretrained(str(LAPTOP_EXP_PATH))

# Prediction function
def get_predictions(df, tokenizer, model, device, batch_size=32, max_length=128):
    dataset = ABSADataset(df, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# get the predicitons on the test set
true_labels, pred_labels, probs = get_predictions(test_df, tokenizer, model, device)

# classification report
print(f"TEST SET — CLASSIFICATION REPORT on {NAME} test set")
report = classification_report(
    true_labels, pred_labels,
    target_names=LABEL_NAMES,
    digits=4,
)
print(report)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
print(f"Macro F1 : {macro_f1:.4f}")

# save the report
report_path = LAPTOP_EXP_PATH / "evaluation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"BERT ABSA — Evaluation on {NAME} Test Set\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write(f"\nMacro F1 : {macro_f1:.4f}\n")

# plot training curves
state_file = None
for root, dirs, files in os.walk(str(CHECKPOINTS)):
    for f in files:
        if f == "trainer_state.json":
            state_file = Path(root) / f
            break

if state_file is None:
    print("\nWARNING: trainer_state.json not found — skipping training curves.")
else:
    print(f"\nLoading training history from: {state_file}")
    with open(state_file) as f:
        state = json.load(f)

    history = state["log_history"]
    train_loss = [x["loss"] for x in history if "loss" in x and "eval_loss" not in x]
    eval_macro = [x["eval_macro_f1"] for x in history if "eval_macro_f1" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    epochs = list(range(1, len(eval_macro) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("BERT fine-tuning (laptop) — training curves", fontsize=13)

    axes[0].plot(train_loss, color="blue", linewidth=1.5)
    axes[0].set_title("Train loss (per 50 steps)")
    axes[0].set_xlabel("Step (×50)")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, eval_macro, marker="o", color="green",
                 linewidth=2, label="Val macro-F1")
    axes[1].plot(epochs, eval_loss,  marker="s", color="red",
                 linewidth=2, label="Val loss", linestyle="--")
    axes[1].set_title("Val metrics per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xticks(epochs)
    axes[1].legend()

    plt.tight_layout()
    curve_path = LAPTOP_EXP_PATH / "training_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.show()

# Plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    ax=ax,
)
ax.set_title(f"Confusion matrix — {NAME} test set")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
cm_path = LAPTOP_EXP_PATH / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()

# Per-class metrics
report_dict = classification_report(
    true_labels, pred_labels,
    target_names=LABEL_NAMES,
    output_dict=True,
)

plot_order = ["positive", "neutral", "negative"]
per_class_precision = [report_dict[l]["precision"] for l in plot_order]
per_class_recall = [report_dict[l]["recall"] for l in plot_order]
per_class_f1 = [report_dict[l]["f1-score"] for l in plot_order]

x, w = np.arange(len(plot_order)), 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w, per_class_precision, w, label="Precision", color="blue")
ax.bar(x, per_class_recall, w, label="Recall", color="green")
ax.bar(x + w, per_class_f1, w, label="F1", color="orange")
ax.set_title(f"Per-class metrics — {NAME} test set")
ax.set_xticks(x)
ax.set_xticklabels(plot_order)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.axhline(macro_f1, color="red", linestyle="--", linewidth=1,
           label=f"Macro F1 = {macro_f1:.3f}")
ax.legend()
plt.tight_layout()
f1_path = LAPTOP_EXP_PATH / "per_class_f1.png"
plt.savefig(f1_path, dpi=150, bbox_inches="tight")
plt.show()

# missclassified examples per error type.
test_df["predicted"] = [ID2LABEL[p] for p in pred_labels]
test_df["predicted_label"] = pred_labels
test_df["correct"] = (test_df["label"] == test_df["predicted_label"])

misclassified = test_df[~test_df["correct"]].copy()

for true_pol in LABEL_NAMES:
    for pred_pol in LABEL_NAMES:
        if true_pol == pred_pol:
            continue
        subset = misclassified[
            (misclassified["polarity"]  == true_pol) &
            (misclassified["predicted"] == pred_pol)
        ]

# Save predictions into csv file
results_df = test_df[["text", "target", "polarity", "predicted", "correct"]].copy()
results_df.columns = ["sentence", "aspect", "true_sentiment", "predicted_sentiment", "correct"]
results_path = LAPTOP_EXP_PATH / "predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"Predictions saved")

misc_path = LAPTOP_EXP_PATH / "misclassified.csv"
misclassified.to_csv(misc_path, index=False)

import os

import argparse
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

REQUIRED_COLUMNS = ("text", "target", "polarity")

SYSTEM_PROMPT = (
    "You classify sentiment toward one target aspect in English restaurant reviews. "
    "Reply with exactly one word: positive, negative, or neutral."
)


def build_user_message(review: str, target: str) -> str:
    # Build the user prompt from review text and target aspect.
    review = (review or "").strip()
    target = (target or "").strip()
    return (
        f"Review: {review}\n"
        f"Target aspect: {target}\n"
        "What is the sentiment toward this target aspect? Answer with one word only."
    )


def row_to_messages(review: str, target: str, polarity: str) -> list[dict[str, str]]:
    # Convert one CSV row into chat messages for SFT.
    pol = (polarity or "").strip().lower()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(review, target)},
        {"role": "assistant", "content": pol},
    ]


def add_text_column(batch, tokenizer: AutoTokenizer) -> dict[str, list]:
    # Render chat template text for each row in a batch.
    texts: list[str] = []
    n = len(batch["text"])
    for i in range(n):
        messages = row_to_messages(
            batch["text"][i],
            batch["target"][i],
            batch["polarity"][i],
        )
        t = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(t)
    return {"text": texts}


def parse_args() -> argparse.Namespace:
    # Parse stage-2 training hyperparameters and paths.
    p = argparse.ArgumentParser(
        description="Stage-2 LoRA: continue training from stage-1 adapter on laptop CSVs."
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "Qwen2.5-1.5B-Instruct",
        help="Base model (same as stage 1)",
    )
    p.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Stage-1 adapter directory (e.g. outputs/qwen-sft-lora/final_adapter)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "laptop",
        help="Directory with train.csv and val.csv for laptop fine-tuning",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "laptop-lora-stage2",
        help="Where to save the new adapter and tokenizer",
    )
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    return p.parse_args()


def make_label_first_token_ids(tokenizer: AutoTokenizer) -> dict[str, set[int]]:
    # Build candidate first-token ids for each class label.
    classes = ("positive", "neutral", "negative")
    variants = [
        "{label}",
        " {label}",
        "\n{label}",
        "\n {label}",
        "<|im_start|>assistant\n{label}",
    ]
    ids: dict[str, set[int]] = {c: set() for c in classes}
    for c in classes:
        for pattern in variants:
            text = pattern.format(label=c)
            toks = tokenizer(text, add_special_tokens=False).input_ids
            if toks:
                ids[c].add(int(toks[0]))
    return ids


def compute_class_weights_from_train(train_ds) -> dict[str, float]:
    # Compute inverse-frequency class weights from laptop train labels.
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for row in train_ds:
        p = (row["polarity"] or "").strip().lower()
        if p in counts:
            counts[p] += 1
    total = sum(counts.values())
    num_classes = 3
    weights = {}
    for label, n in counts.items():
        if n == 0:
            weights[label] = 1.0
        else:
            weights[label] = total / (num_classes * n)
    return weights


def make_weighted_loss_func(
    tokenizer: AutoTokenizer,
    class_weights: dict[str, float],
):
    # Build a token-weighted loss that upweights minority labels.
    # Label token ids are weighted while other tokens keep weight 1.
    token_sets = make_label_first_token_ids(tokenizer)
    id_to_weight: dict[int, float] = {}
    for label, ids in token_sets.items():
        w = float(class_weights[label])
        for tid in ids:
            id_to_weight[tid] = w

    def weighted_loss_func(outputs, labels, num_items_in_batch=None, **kwargs):
        # Apply per-token weights on shifted LM loss.
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        token_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)

        valid_mask = shift_labels.ne(-100)
        base_loss = token_loss[valid_mask].mean()
        if valid_mask.sum().item() == 0:
            return base_loss

        weights = torch.ones_like(token_loss, dtype=token_loss.dtype)
        for tid, w in id_to_weight.items():
            weights = torch.where(
                shift_labels.eq(tid),
                torch.tensor(w, device=weights.device, dtype=weights.dtype),
                weights,
            )

        weighted = token_loss * weights
        loss = weighted[valid_mask].sum() / weights[valid_mask].sum().clamp_min(1.0)
        if torch.isnan(loss) or torch.isinf(loss):
            return base_loss
        return loss

    return weighted_loss_func


def main() -> None:
    # Continue LoRA training on laptop data and save stage-2 adapter.
    # Loads stage-1 adapter in trainable mode before SFT.
    args = parse_args()
    model_id = str(args.model_path)
    adapter_path = args.adapter_path.resolve()
    data_dir = args.data_dir.resolve()
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Stage-1 adapter not found: {adapter_path}")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Missing {train_csv}")
    if not val_csv.is_file():
        raise FileNotFoundError(f"Missing {val_csv}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    print(f"Loaded stage-1 LoRA from {adapter_path} (trainable continuation)")

    raw = load_dataset(
        "csv",
        data_files={"train": str(train_csv), "validation": str(val_csv)},
    )

    def take_columns(split_name: str):
        # Validate required columns and drop extras.
        ds = raw[split_name]
        missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
        if missing:
            raise ValueError(
                f"{split_name}: missing columns {missing}. Required: {list(REQUIRED_COLUMNS)}"
            )
        return ds.select_columns(list(REQUIRED_COLUMNS))

    train_ds = take_columns("train")
    eval_ds = take_columns("validation")

    class_weights = compute_class_weights_from_train(train_ds)
    print(
        "Class weights (inverse-frequency from laptop train): "
        f"positive={class_weights['positive']:.4f}, "
        f"neutral={class_weights['neutral']:.4f}, "
        f"negative={class_weights['negative']:.4f}"
    )

    train_ds = train_ds.map(
        lambda batch: add_text_column(batch, tokenizer),
        batched=True,
    ).select_columns(["text"])
    eval_ds = eval_ds.map(
        lambda batch: add_text_column(batch, tokenizer),
        batched=True,
    ).select_columns(["text"])

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_params = inspect.signature(SFTConfig).parameters
    length_kw: dict = {}
    if "max_length" in sft_params:
        length_kw["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_params:
        length_kw["max_seq_length"] = args.max_seq_length
    else:
        length_kw["max_length"] = args.max_seq_length

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        dataset_text_field="text",
        packing=False,
        seed=42,
        **length_kw,
    )

    st_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kw: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "compute_loss_func": make_weighted_loss_func(tokenizer, class_weights),
    }
    if "processing_class" in st_params:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kw)

    trainer.train()

    adapter_dir = out_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved stage-2 LoRA adapter and tokenizer to {adapter_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

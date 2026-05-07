import argparse
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 32

SYSTEM_PROMPT = (
    "You classify sentiment toward one target aspect in English restaurant reviews. "
    "Reply with exactly one word: positive, negative, or neutral."
)

LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def build_user_message(review: str, target: str) -> str:
    review = (review or "").strip()
    target = (target or "").strip()
    return (
        f"Review: {review}\n"
        f"Target aspect: {target}\n"
        "What is the sentiment toward this target aspect? Answer with one word only."
    )


def parse_label(text: str) -> str | None:
    m = LABEL_RE.search(text or "")
    return m.group(1).lower() if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo inference with LoRA for aspect sentiment")
    p.add_argument("--model-path", type=str, default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="LoRA adapter directory, e.g. outputs/qwen-sft-lora/final_adapter",
    )
    return p.parse_args()


def load_model_and_tokenizer(model_id: str, adapter_path: str):
    adapter_path = os.path.abspath(adapter_path)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict(model, tokenizer, review: str, target: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(review, target)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0, prompt_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    label = parse_label(raw) or "neutral"
    return label, raw


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)
    print("Model loaded. LoRA demo ready.")

    print("Interactive mode. Press Enter on empty review to exit.\n")
    while True:
        review = input("Review: ").strip()
        if not review:
            print("Bye.")
            break
        target = input("Target aspect: ").strip()
        if not target:
            print("Target cannot be empty.\n")
            continue
        label, raw = predict(model, tokenizer, review, target)
        print(f"Prediction: {label}")
        print(f"Raw output: {raw}\n")


if __name__ == "__main__":
    main()

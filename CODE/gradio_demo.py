# COMP6713 26T1 NLP Project
# Group: Cold Tuna
# Gradio demo for Aspect-Based Sentiment Analysis (ABSA).
# Given a sentence and an aspect term, predicts the sentiment polarity with
# options to use models explored in our project.

## example usage:
# python -m gradio_demo --qwen-model-path "MODELPATH" --qwen-adapter-path "ADAPTERPATH"

import gradio as gr
import torch
import re
import argparse
import numpy as np
from pathlib import Path
from joblib import load
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from models.gpt.cot import predict_sentiment

# Bert
PROJECT_DIR = Path(__file__).resolve().parent
TFIDF_MODEL_PATH = PROJECT_DIR / "models" / "ML" / "artifacts" / "tfidf_lr" / "tfidf_logreg_model.joblib"
BERT_EXP_PATH = PROJECT_DIR / "models/bert/experiments/restaurants"
BERT_MODEL_PATH = BERT_EXP_PATH 
LABEL_NAMES = ["negative", "neutral", "positive"]
MAX_LENGTH = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tfidf_model = None
tfidf_load_error = None
bert_tokenizer = BertTokenizer.from_pretrained(str(BERT_MODEL_PATH), local_files_only=True)
bert_model = BertForSequenceClassification.from_pretrained(str(BERT_MODEL_PATH), local_files_only=True)
bert_model.to(device)
bert_model.eval()

# Qwen LoRA
DEFAULT_QWEN_MODEL_PATH = PROJECT_DIR / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_QWEN_ADAPTER_PATH = PROJECT_DIR / "outputs" / "qwen-sft-lora" / "final_adapter"
QWEN_MODEL_PATH = DEFAULT_QWEN_MODEL_PATH
QWEN_ADAPTER_PATH = DEFAULT_QWEN_ADAPTER_PATH
QWEN_MAX_NEW_TOKENS = 32
QWEN_SYSTEM_PROMPT = (
    "You classify sentiment toward one target aspect in English restaurant reviews. "
    "Reply with exactly one word: positive, negative, or neutral."
)
QWEN_LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def build_qwen_user_message(sentence: str, aspect: str) -> str:
    sentence = (sentence or "").strip()
    aspect = (aspect or "").strip()
    return (
        f"Review: {sentence}\n"
        f"Target aspect: {aspect}\n"
        "What is the sentiment toward this target aspect? Answer with one word only."
    )


def parse_qwen_label(text: str) -> str | None:
    m = QWEN_LABEL_RE.search(text or "")
    return m.group(1).lower() if m else None


def load_qwen_model_and_tokenizer(model_path: Path, adapter_path: Path):
    qwen_model_id = str(model_path)
    adapter_path_str = str(adapter_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Qwen model not found: {qwen_model_id}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Qwen adapter not found: {adapter_path_str}")

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
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
        qwen_model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model = PeftModel.from_pretrained(base, adapter_path_str)
    model.eval()
    return model, tokenizer


qwen_model, qwen_tokenizer = None, None
qwen_load_error = None


def configure_qwen_paths(model_path: str, adapter_path: str) -> None:
    global QWEN_MODEL_PATH, QWEN_ADAPTER_PATH, qwen_model, qwen_tokenizer, qwen_load_error
    QWEN_MODEL_PATH = Path(model_path).expanduser().resolve()
    QWEN_ADAPTER_PATH = Path(adapter_path).expanduser().resolve()
    qwen_model, qwen_tokenizer = None, None
    qwen_load_error = None


def ensure_qwen_loaded() -> None:
    global qwen_model, qwen_tokenizer, qwen_load_error
    if qwen_model is not None and qwen_tokenizer is not None:
        return
    if qwen_load_error is not None:
        return
    try:
        qwen_model, qwen_tokenizer = load_qwen_model_and_tokenizer(
            QWEN_MODEL_PATH, QWEN_ADAPTER_PATH
        )
    except Exception as e:
        qwen_load_error = str(e)

# Method functions
def mark_aspect_text(sentence: str, aspect: str) -> str:
    sentence = sentence.strip()
    aspect = aspect.strip()

    if aspect and aspect.lower() in sentence.lower():
        pattern = re.compile(re.escape(aspect), flags=re.IGNORECASE)
        return pattern.sub(lambda m: f"[ASP] {m.group(0)} [/ASP]", sentence, count=1)

    return f"{sentence} [ASP] {aspect} [/ASP]"


def ensure_tfidf_loaded() -> None:
    global tfidf_model, tfidf_load_error
    if tfidf_model is not None or tfidf_load_error is not None:
        return
    try:
        tfidf_model = load(str(TFIDF_MODEL_PATH))
    except Exception as e:
        tfidf_load_error = str(e)


def predict_tf_idf(sentence: str, aspect: str):
    sentence = sentence.strip()
    aspect = aspect.strip()
    if not sentence or not aspect:
        return "must enter both sentence and aspect term!!!"

    ensure_tfidf_loaded()
    if tfidf_model is None:
        return f"TF-IDF model failed to load: {tfidf_load_error}"

    model_input = mark_aspect_text(sentence, aspect)
    pred_raw = tfidf_model.predict([model_input])[0]

    if isinstance(pred_raw, (int, np.integer)):
        pred_label = LABEL_NAMES[int(pred_raw)] if 0 <= int(pred_raw) < len(LABEL_NAMES) else str(pred_raw)
    else:
        pred_label = str(pred_raw).lower()

    try:
        probs = tfidf_model.predict_proba([model_input])[0]
        confidence = float(np.max(probs))
        return f"{pred_label} ({confidence:.2%})"
    except Exception:
        return pred_label

def predict_bert(sentence: str, aspect: str):
    sentence = sentence.strip()
    aspect = aspect.strip()
    if not sentence or not aspect:
        return "must enter both sentence and aspect term!!!"

    encoding = bert_tokenizer(
        sentence,
        aspect,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return f"{pred_label} ({confidence:.2%})"

def predict_qwen(sentence: str, aspect: str):
    sentence = sentence.strip()
    aspect = aspect.strip()
    if not sentence or not aspect:
        return "must enter both sentence and aspect term!!!"

    ensure_qwen_loaded()
    if qwen_model is None or qwen_tokenizer is None:
        return f"Qwen model failed to load: {qwen_load_error}"

    with torch.inference_mode():
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": build_qwen_user_message(sentence, aspect)},
        ]
        prompt = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        qwen_device = next(qwen_model.parameters()).device
        inputs = {k: v.to(qwen_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        out = qwen_model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=qwen_tokenizer.pad_token_id,
        )
        gen_ids = out[0, prompt_len:]
        raw = qwen_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        label = parse_qwen_label(raw) or "neutral"

    return label

def predict_cot(sentence: str, aspect: str):
    return predict_sentiment(sentence, aspect)

# Create the Gradio interface.
METHODS = {
    "TF-IDF + Logistic Regression": predict_tf_idf,
    "BERT Base Uncased": predict_bert,
    "Qwen2.5": predict_qwen,
    "Chain of Thought (GPT-5.4-mini)": predict_cot,
}

# Run all methods and return a comparison table.
def run_all_models(sentence: str, aspect: str):
    if not sentence.strip() or not aspect.strip():
        return [["Input validation", "must enter both sentence and aspect term!!!"]]

    rows = []
    for method_name, predict_fn in METHODS.items():
        try:
            pred = predict_fn(sentence, aspect)
        except Exception as e:
            pred = f"error: {e}"
        rows.append([method_name, pred])
    return rows


demo = gr.Interface(
    fn=run_all_models,
    inputs=[
        gr.Textbox(label="Sentence", placeholder='e.g. "The steak was to die for!"'),
        gr.Textbox(label="Aspect Term", placeholder='e.g. "steak"'),
    ],
    outputs=gr.Dataframe(
        headers=["Model", "Predicted Sentiment"],
        datatype=["str", "str"],
        label="Predictions from all models",
    ),
    title="Aspect-Based Sentiment Analysis (By Cold Tuna 🐟)",
    description="Enter a sentence and an aspect term to compare predicted sentiment polarity across all models.",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio ABSA demo")
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default=str(DEFAULT_QWEN_MODEL_PATH),
        help="Path to local Qwen base model directory",
    )
    parser.add_argument(
        "--qwen-adapter-path",
        type=str,
        default=str(DEFAULT_QWEN_ADAPTER_PATH),
        help="Path to LoRA adapter directory",
    )
    args = parser.parse_args()
    configure_qwen_paths(args.qwen_model_path, args.qwen_adapter_path)
    demo.launch()

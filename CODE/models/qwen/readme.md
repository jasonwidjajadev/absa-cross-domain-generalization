# Usage

## Preparation

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Download the base model:

```bash
python download_qwen_model.py
python download_qwen_model.py --output-dir models/Qwen2.5-1.5B-Instruct
```

## Training

Training is **two-stage**: first fit LoRA on your primary CSV split (`train_sft.py`), then continue training those LoRA weights on the **laptop** split (`finetune_laptop.py`). Both stages use **class-weighted loss** (inverse frequency from the stage’s train split).

### Stage 1 — `train_sft.py`

LoRA supervised fine-tuning from the base model. **`--data-dir`** must contain **`train.csv`** and **`val.csv`** with columns `text`, `target`, `polarity` (extra columns are ignored).

```bash
pip install -r requirements.txt
python train_sft.py --model-path models/Qwen2.5-1.5B-Instruct --data-dir data --output-dir outputs/qwen-sft-lora
```

The adapter and tokenizer are saved under **`outputs/qwen-sft-lora/final_adapter`** (adjust if you change `--output-dir`).

| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--max-seq-length` | 256 | Max sequence length for SFT (set to 256 based on our max-length testing) |
| `--lora-r` / `--lora-alpha` / `--lora-dropout` | 16 / 32 / 0.05 | LoRA hyperparameters |

### Stage 2 — `finetune_laptop.py`

Continues training from **stage 1’s** adapter on the laptop data. **`--adapter-path`** must point to stage 1’s **`final_adapter`** directory. **`--data-dir`** defaults to **`data/laptop`** (must contain `train.csv` and `val.csv`). Output defaults to **`outputs/laptop-lora-stage2`**; the new adapter is saved as **`final_adapter`** inside that directory.

```bash
python finetune_laptop.py --model-path models/Qwen2.5-1.5B-Instruct --adapter-path outputs/qwen-sft-lora/final_adapter --data-dir data/laptop --output-dir outputs/laptop-lora-stage2
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--adapter-path` | (required) | Stage-1 adapter directory, e.g. `outputs/qwen-sft-lora/final_adapter` |
| `--data-dir` | `data/laptop` | Laptop split with `train.csv` / `val.csv` |
| `--output-dir` | `outputs/laptop-lora-stage2` | Where checkpoints and `final_adapter` are written |
| `--epochs` / `--batch-size` / `--grad-accum` / `--lr` / `--max-seq-length` | same idea as stage 1 | No new LoRA rank/alpha/dropout — weights are updated on top of stage 1. |

For **`eval_lora.py`** / **`demo.py`**, point **`--adapter-path`** at **`outputs/laptop-lora-stage2/final_adapter`** (or your chosen `--output-dir`) when you want the laptop stage-2 checkpoint.

---

## Evaluation

Two standalone scripts score a CSV test set (columns `text`, `target`, `polarity`). Metrics print to the terminal. To also save per-row predictions, pass `--predictions-csv`.

### `eval_baseline.py`

Runs the **base model only** (no LoRA).

```bash
python eval_baseline.py --model-path models/Qwen2.5-1.5B-Instruct --test-csv data/combined/test.csv
```

Example with predictions written under `result/`:

```bash
python eval_baseline.py --model-path models/Qwen2.5-1.5B-Instruct --test-csv data/laptop/test.csv --predictions-csv result/baseline_laptop/laptop_predictions.csv
```

### `eval_lora.py`

Runs **base model + LoRA adapter** (`--adapter-path` is required).

```bash
python eval_lora.py --model-path models/Qwen2.5-1.5B-Instruct --adapter-path outputs/qwen-sft-lora/final_adapter --test-csv data/combined/test.csv
```

Example with metrics on stdout and a predictions CSV saved:

```bash
python eval_lora.py --model-path models/Qwen2.5-1.5B-Instruct --adapter-path outputs/qwen-sft-lora/final_adapter --test-csv data/combined/test.csv --predictions-csv result/lora_both/combined_predictions.csv
```

## `demo.py`

Interactive **LoRA** inference for a single review and target aspect (base model + adapter). `--adapter-path` is required. Run it and enter a review and target when prompted; leave the review empty to exit.

```bash
python demo.py --model-path models/Qwen2.5-1.5B-Instruct --adapter-path outputs/qwen-sft-lora/final_adapter
```

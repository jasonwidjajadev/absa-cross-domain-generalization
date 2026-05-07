"""Download Qwen/Qwen2.5-1.5B-Instruct from Hugging Face to a local folder.

Requires: pip install huggingface_hub

Optional: set HF_TOKEN in the environment if the hub asks for authentication.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    # Download the configured model snapshot to a local directory.
    parser = argparse.ArgumentParser(
        description=f"Download {REPO_ID} snapshot from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "Qwen2.5-1.5B-Instruct",
        help="Local directory for model files (default: ./models/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN)",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        token=args.token,
    )
    print(f"Done. Files are under: {path}")


if __name__ == "__main__":
    main()

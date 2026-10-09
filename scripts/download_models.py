"""
Utility script to reliably download GGUF models from Hugging Face
using HTTP streaming with resume support.
"""
import os
import sys
import argparse
from pathlib import Path
import requests

MODELS = {
    "qwen": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "desc": "Qwen 2.5 3B Instruct (Q4_K_M, ~1.96 GB)"
    },
    "llama": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "desc": "Llama 3.2 3B Instruct (Q4_K_M, ~1.88 GB)"
    },
    "maple": {
        "url": "https://huggingface.co/deepgrove/maple-preview-GGUF/resolve/main/maple-preview-TQ1_0-head-Q4_K.gguf",
        "filename": "maple-preview-TQ1_0-head-Q4_K.gguf",
        "desc": "Maple-Preview (TQ1_0 + Q4_K head, ~4.98 GB)"
    }
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_file_with_resume(url: str, dest_path: Path, desc: str, force: bool = False):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if dest_path.exists() and not force:
        print(f"[OK] {desc} already exists at: {dest_path}")
        return dest_path

    # Check headers for content length
    head = requests.head(url, allow_redirects=True, timeout=15)
    total_size = int(head.headers.get("content-length", 0))

    initial_pos = 0
    headers = {}
    if temp_path.exists() and not force:
        initial_pos = temp_path.stat().st_size
        if total_size and initial_pos < total_size:
            headers["Range"] = f"bytes={initial_pos}-"
            print(f"[RESUME] Resuming {desc} from {initial_pos / 1024 / 1024:.1f} MB...")
        elif total_size and initial_pos == total_size:
            temp_path.rename(dest_path)
            print(f"[DONE] File already complete! Renamed to {dest_path}")
            return dest_path
        else:
            initial_pos = 0

    mode = "ab" if initial_pos > 0 else "wb"
    print(f"[+] Downloading {desc} ({total_size / 1024 / 1024:.1f} MB)...", flush=True)

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
    response.raise_for_status()

    downloaded = initial_pos
    last_reported_mb = downloaded // (20 * 1024 * 1024)

    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                current_mb_block = downloaded // (20 * 1024 * 1024)
                if current_mb_block != last_reported_mb:
                    pct = (downloaded / total_size * 100) if total_size else 0
                    print(f"    Progress: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({pct:.1f}%)", flush=True)
                    last_reported_mb = current_mb_block

    temp_path.rename(dest_path)
    print(f"[SUCCESS] Download completed: {dest_path}", flush=True)
    return dest_path



def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for Lab 1")
    parser.add_argument(
        "--model",
        choices=["all", "qwen", "llama", "maple"],
        default="all",
        help="Model to download (default: all)"
    )
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for key in targets:
        info = MODELS[key]
        dest = MODELS_DIR / info["filename"]
        download_file_with_resume(info["url"], dest, info["desc"], force=args.force)


if __name__ == "__main__":
    main()

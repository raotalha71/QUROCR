
import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import io
import time
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).parent / ".env")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Please install openai: pip install openai python-dotenv")

# ==========================================
# 1. API Endpoints Configuration for Research
# ==========================================
ENDPOINTS = {
    "finetuned": {
        "name": "finetuned 35B A3B parameter",
        "model_id": "qwen3.5-35b-a3b",
        "base_url": "https://ws-zjhkohk1mzvyxnvu.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": False},
    },
    "openrouter": {
        "name": "OpenRouter Global Vision Gateway",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "model_aliases": {
            # Zero-shot base model (same arch as fine-tuned, no LoRA)
            "base-qwen":        "qwen/qwen3.5-35b-a3b",
            "qwen-35b-a3b":     "qwen/qwen3.5-35b-a3b",
            # Google Gemini models
            "gemini":           "google/gemini-3.8-flash",      # Default Gemini -> Gemini 3.8 Flash
            "gemini-flash":     "google/gemini-3.8-flash",      # Gemini 3.8 Flash
            "gemini-3.8-flash": "google/gemini-3.8-flash",      # Explicit Gemini 3.8 Flash
            "gemini-pro":       "google/gemini-2.5-pro",        # Gemini 2.5 Pro
            "gemini-4-argon":   "google/gemini-4-argon",        # Frontier flagship
            # Other commercial VLMs
            "grok":             "x-ai/grok-4.5",
            "gpt-4o-mini":      "openai/gpt-4o-mini",
            "qwen-72b":         "qwen/qwen2.5-vl-72b-instruct",
        }
    }
}

# All available model choices for CLI
ALL_MODEL_CHOICES = [
    "finetuned",
    "base-qwen",
    "gemini",
    "gemini-flash",
    "gemini-3.8-flash",
    "gemini-pro",
    "gemini-4-argon",
    "grok",
    "gpt-4o-mini",
    "qwen-72b",
]

# ==========================================
# 2. Unified OCR Prompt (Plain text, no JSON wrapper)
# ==========================================
# IMPORTANT: This prompt returns PLAIN Urdu text directly.
# No JSON wrapping = no wasted tokens on {"urdu_text": ...} formatting.
# This maximizes the text the model can output within the token limit.

DEFAULT_PROMPT = """You are a high-accuracy OCR engine for printed or handwritten Urdu text in Nastaliq/Naskh. Your ONLY function is to extract and output the raw Urdu text.

Visual Processing Directives (Apply these internally before outputting):
- Analyze the continuous ink flow carefully. Separate words based strictly on Urdu grammar, even if physical spacing is missing.
- Pay strict attention to Urdu-specific orthography (ٹ، ڈ، ڑ، ے، ں، ھ، ہ) and prevent conversion to Arabic styles.

Strict Output Rules (DO NOT VIOLATE):
1) Output in Urdu ONLY when the source text is Urdu. Do not convert wording into Arabic language style.
2) Preserve Urdu orthography and punctuation where visible. 
3) Keep numbers exactly as seen (using Urdu/Eastern Arabic numerals like ۱, ۲, ۳) and preserve line breaks from the source.
4) ABSOLUTELY NO conversational filler. Do not write "Here is the text:", "The image contains", or "Transcribed version".
5) ABSOLUTELY NO Markdown formatting. Do not use headers (###), bolding (**), or bullet points.
6) ABSOLUTELY NO English text, translations, or interpretive notes.
7) If a segment is completely unreadable, skip that segment instead of hallucinating.

Return ONLY the final Urdu transcription text. Do not output a single word of English."""


# ==========================================
# 3. Text Extraction & Utilities
# ==========================================

def extract_urdu_text(raw_text: str) -> str:
    """Clean model output: strip JSON wrappers, markdown fences, and English reasoning."""
    text = (raw_text or "").strip()
    if not text:
        return ""

    # Try to parse as JSON if the model wrapped it
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "urdu_text" in obj:
            return str(obj["urdu_text"]).strip()
    except Exception:
        pass

    # Try to extract from markdown code fence
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "urdu_text" in obj:
                return str(obj["urdu_text"]).strip()
        except Exception:
            pass

    # Try to find embedded JSON
    l_idx = text.find("{")
    r_idx = text.rfind("}")
    if l_idx != -1 and r_idx != -1 and r_idx > l_idx:
        try:
            obj = json.loads(text[l_idx : r_idx + 1])
            if isinstance(obj, dict) and "urdu_text" in obj:
                return str(obj["urdu_text"]).strip()
        except Exception:
            pass

    # Strip markdown fences if present
    m_fence = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, flags=re.S)
    if m_fence:
        text = m_fence.group(1)

    # Clean any accidental English thinking/chain-of-thought lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # If line contains English conversational phrases, skip it
        if re.search(r"\b(let's|word\s+\d+|look at|it has|starts with|deep round bowl|phrase|that is)\b", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)

    if cleaned_lines:
        text = "\n".join(cleaned_lines)

    return text.strip()


def image_data_url(path: Path) -> str:
    """Convert image file to base64 data URL for API."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def get_client(target: str = "finetuned"):
    """Get OpenAI-compatible client for the specified model target."""
    if target == "finetuned":
        cfg = ENDPOINTS["finetuned"]
        key = os.getenv(cfg["env_key"])
        if not key:
            key = os.getenv("OPENROUTER_API_KEY")
            return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"), "qwen/qwen3.5-35b-a3b", cfg["name"]
        return OpenAI(api_key=key, base_url=cfg["base_url"]), cfg["model_id"], cfg["name"]
    else:
        cfg = ENDPOINTS["openrouter"]
        key = os.getenv(cfg["env_key"])
        model_id = cfg["model_aliases"].get(target, target)
        return OpenAI(api_key=key, base_url=cfg["base_url"]), model_id, target


def ocr_image(client: OpenAI, model_id: str, img_path: Path,
              prompt: str = DEFAULT_PROMPT, max_tokens: int = 4096):
    """
    Run OCR on a single image. Returns a dict with:
      - output: extracted Urdu text
      - raw_output: unprocessed model response
      - error: error message if any
      - latency: seconds taken
      - finish_reason: 'stop', 'length' (truncated!), or 'error'
      - usage: token usage dict from the API
      - image: image filename
    """
    data_url = image_data_url(img_path)

    body_kwargs = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    # Suppress thinking tokens for Qwen models
    if "qwen3.5" in model_id or "maas" in str(client.base_url):
        body_kwargs["extra_body"] = {"enable_thinking": False}

    result = {
        "output": "",
        "raw_output": "",
        "error": "",
        "latency": 0.0,
        "finish_reason": "",
        "usage": {},
        "image": img_path.name,
    }

    start = time.time()
    try:
        resp = client.chat.completions.create(**body_kwargs)
        result["latency"] = round(time.time() - start, 6)

        choice = resp.choices[0]
        result["finish_reason"] = choice.finish_reason or "unknown"

        msg = choice.message
        raw = msg.content or getattr(msg, "reasoning_content", "") or ""
        result["raw_output"] = raw
        result["output"] = extract_urdu_text(raw)

        if resp.usage:
            result["usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            if hasattr(resp.usage, "cost"):
                result["usage"]["cost"] = resp.usage.cost

        if result["finish_reason"] == "length":
            print(f" [TRUNCATED!]", end="", flush=True)

    except Exception as e:
        result["latency"] = round(time.time() - start, 6)
        result["error"] = str(e)
        result["finish_reason"] = "error"

    return result


# ==========================================
# 4. Batch Processing with JSONL + CSV Output
# ==========================================

def run_benchmark(client, model_id, display_name, img_dir: Path,
                  output_path: Path, labels_path: Path = None,
                  prompt: str = DEFAULT_PROMPT, max_tokens: int = 4096,
                  resume: bool = True):
    """
    Run OCR on all images in a directory. Saves results as JSONL (raw logs)
    and optionally as a CSV with ground truth for evaluation.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    print(f"Found {len(imgs)} images in {img_dir}\n")

    gt_map = {}
    if labels_path and labels_path.exists():
        with open(labels_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_map[row["filename"]] = row["text"]
        print(f"Loaded {len(gt_map)} ground truth labels from {labels_path.name}\n")

    done = set()
    if resume and output_path.exists():
        valid_lines = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if not d.get("error") and d.get("image"):
                        done.add(d.get("image"))
                        valid_lines.append(line if line.endswith("\n") else line + "\n")
                except Exception:
                    pass
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)
        if done:
            print(f"Resuming: {len(done)} successful images already processed, skipping them.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(imgs)
    truncated_count = 0
    error_count = 0

    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, img in enumerate(imgs, 1):
            if img.name in done:
                continue

            print(f"[{idx:3d}/{total}] {img.name:25s} ... ", end="", flush=True)

            result = ocr_image(client, model_id, img, prompt, max_tokens)

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            if result["error"]:
                error_count += 1
                print(f"FAIL: {result['error'][:60]}")
            elif result["finish_reason"] == "length":
                truncated_count += 1
                print(f"OK ({result['latency']:.1f}s, {result['usage'].get('completion_tokens', '?')} tok) [TRUNCATED]")
            else:
                print(f"OK ({result['latency']:.1f}s, {result['usage'].get('completion_tokens', '?')} tok)")

    print(f"\n{'='*50}")
    print(f"  Benchmark Complete: {display_name}")
    print(f"  Total Images : {total}")
    print(f"  Errors       : {error_count}")
    print(f"  Truncated    : {truncated_count}")
    print(f"  Output       : {output_path}")
    print(f"{'='*50}\n")

    if truncated_count > 0:
        print(f"  WARNING: {truncated_count} outputs were TRUNCATED (finish_reason='length').")
        print(f"    Consider increasing --max-tokens (current: {max_tokens}).\n")

    csv_path = output_path.with_suffix(".csv") if output_path.suffix == ".jsonl" else output_path.parent / (output_path.stem + "_predictions.csv")
    generate_csv(output_path, csv_path, gt_map)
    print(f"  Predictions CSV: {csv_path}")


def generate_csv(jsonl_path: Path, csv_path: Path, gt_map: dict = None):
    """Convert JSONL results to a clean CSV for evaluation."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                row = {
                    "image": d.get("image", ""),
                    "prediction": d.get("output", ""),
                    "latency_s": d.get("latency", 0),
                    "finish_reason": d.get("finish_reason", ""),
                    "prompt_tokens": d.get("usage", {}).get("prompt_tokens", ""),
                    "completion_tokens": d.get("usage", {}).get("completion_tokens", ""),
                }
                if gt_map:
                    row["reference"] = gt_map.get(d.get("image", ""), "")
                rows.append(row)
            except Exception:
                pass

    if not rows:
        return

    fieldnames = ["image"]
    if gt_map:
        fieldnames.append("reference")
    fieldnames.extend(["prediction", "latency_s", "finish_reason", "prompt_tokens", "completion_tokens"])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 5. CLI Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Endpoint OCR API Client for Research Paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fine-tuned model on single image
  py call_ocr.py --model finetuned --image benchmark200/001.jpg

  # Run Gemini 3.8 Flash on full benchmark with labels
  py call_ocr.py --model gemini-flash --dir benchmark200 --labels labels200.csv --out results_final/raw/gemini-3.8-flash.jsonl

  # Run zero-shot Qwen on full benchmark
  py call_ocr.py --model base-qwen --dir benchmark200 --labels labels200.csv --out results_final/raw/qwen-35b-a3b.jsonl

  # Run GPT-4o-mini on full benchmark
  py call_ocr.py --model gpt-4o-mini --dir benchmark200 --labels labels200.csv --out results_final/raw/gpt-4o-mini.jsonl
        """
    )
    parser.add_argument("--image", type=Path, help="Path to single image")
    parser.add_argument("--dir", type=Path, help="Path to directory of images (benchmark mode)")
    parser.add_argument("--labels", type=Path, help="Path to labels CSV (columns: filename, text)")
    parser.add_argument("--model", default="finetuned", choices=ALL_MODEL_CHOICES,
                        help="Model target (default: finetuned)")
    parser.add_argument("--out", type=Path, help="Output file path (.jsonl for raw logs)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Custom prompt")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens for ALL models (default: 4096). Must be uniform for fair comparison!")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from existing output, start fresh")
    args = parser.parse_args()

    client, model_id, display_name = get_client(args.model)

    print(f"\n{'='*60}")
    print(f"  Target Model : {display_name}")
    print(f"  Model ID     : {model_id}")
    print(f"  Base URL     : {client.base_url}")
    print(f"  Max Tokens   : {args.max_tokens}")
    print(f"{'='*60}\n")

    if args.image:
        if not args.image.exists():
            sys.exit(f"[error] Image not found: {args.image}")
        print(f"Processing: {args.image.name} ...")
        result = ocr_image(client, model_id, args.image, args.prompt, args.max_tokens)

        print(f"\n--- EXTRACTED URDU TEXT ---")
        print(result["output"])
        print(f"--- Finish Reason: {result['finish_reason']} | Latency: {result['latency']:.2f}s ---\n")

        if result["error"]:
            print(f"[API ERROR] {result['error']}\n")

        if result["finish_reason"] == "length":
            print("WARNING: Output was TRUNCATED. Increase --max-tokens.")

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"Saved to: {args.out}")

    elif args.dir:
        if not args.dir.exists():
            sys.exit(f"[error] Directory not found: {args.dir}")
        if not args.out:
            sys.exit("[error] --out is required in benchmark mode (use .jsonl extension)")

        if args.no_resume and args.out.exists():
            args.out.unlink()
            print("Deleted existing output file for fresh start.\n")

        run_benchmark(
            client, model_id, display_name,
            img_dir=args.dir,
            output_path=args.out,
            labels_path=args.labels,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            resume=not args.no_resume
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# 🏆 QUROCR: Parameter-Efficient Vision-Language Models for Low-Resource Urdu Nastaliq OCR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Benchmark CER: 23.92%](https://img.shields.io/badge/Benchmark%20CER-23.92%25-brightgreen.svg)](#benchmark-leaderboard)

This repository contains the official evaluation suite, datasets, reproduction scripts, and model configurations for **QUROCR** (*Parameter Efficient Hybrid Vision-Language Models for Low-Resource Urdu Text Recognition*).

QUROCR addresses the severe catastrophic hallucination failure (85–95% error) of foundation Vision-Language Models on multi-line Nastaliq Urdu documents by establishing:
1. **The Single-Line Cropped Paradigm**: Decomposing dense, diagonally stacked Nastaliq pages into discrete text-line crops, completely eliminating multi-line hallucination loops.
2. **QUROCR-5K Dataset**: A curated, high-variance dataset of **5,401 single-line cropped images** (30% Real historical documents, 70% Synthetic Nastaliq fonts) with normalized character-level ground truth.
3. **Parameter-Efficient LoRA Adaptation**: A fine-tuned Mixture-of-Experts (MoE) vision-language adapter on **Qwen3.5-35B-A3B** achieving **23.92% CER** overall and **18.93% CER on real scanned Nastaliq** (substantially outperforming the zero-shot base model and commercial frontier models).
4. **QUROCR Benchmark-200**: A balanced, held-out evaluation suite (60 Real historical book scans, 140 Synthetic lines) rigorously testing zero-shot, commercial frontier, and classical OCR engines.

---

## 📊 Benchmark Leaderboard (`benchmark200`)

Evaluated on 200 held-out cropped line images (60 Real historical book scans, 140 Synthetic Nastaliq lines):

| Rank | Model / OCR System | Model Architecture | Overall CER (%) | Overall WER (%) | Overall SER (%) | Real CER (%) | Synth CER (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 | **QUROCR (Fine-Tuned 35B-A3B)** | **LoRA Fine-Tuned MoE (Ours)** | **23.92%** | **70.39%** | **88.50%** | **18.93%** | **26.80%** |
| 🥈 | **Qwen3.5-35B-A3B (Base)** | Zero-Shot Base MoE | **30.64%** | **83.54%** | **93.50%** | **27.96%** | **32.19%** |
| 🥉 | **Google Gemini 2.5 Flash** | Commercial VLM | **34.68%** | **88.83%** | **91.00%** | **21.80%** | **42.10%** |
| 4 | **Qwen 2.5 VL 72B Instruct** | Open-Weights VLM | **53.59%** | **98.09%** | **97.00%** | **42.84%** | **59.79%** |
| 5 | **Tesseract 5 (`urd`)** | Classical OCR Engine | **57.60%** | **94.42%** | **98.50%** | **57.72%** | **57.54%** |
| 6 | **EasyOCR (`ur`)** | Deep OCR Baseline | **60.70%** | **99.56%** | **96.00%** | **55.50%** | **63.70%** |
| 7 | **OpenAI GPT-4o-mini** | Commercial VLM | **61.93%** | **113.30%** | **94.50%** | **60.35%** | **62.84%** |
| 8 | **Anthropic Claude Haiku 4.5** | Commercial VLM | **66.12%** | **113.81%** | **95.00%** | **71.07%** | **63.26%** |
| 9 | **PaddleOCR (Arabic)** | Generic Arabic Script | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

* **CER** = Character Error Rate (Levenshtein distance / total reference characters).
* **WER** = Word Error Rate (Word-level Levenshtein distance / total words).
* **SER** = Sentence / Sequence Error Rate (fraction of lines with any discrepancy).

---

## 📁 Repository Structure

```text
├── benchmark200/                        # Held-out benchmark suite
│   ├── images/                          # 200 cropped text-line images (img001.jpg - img200.jpg)
│   ├── labels200.csv                    # Ground-truth annotations with source and char lengths
│   └── labels200.json                   # Structured JSON benchmark ground truth
├── QUROCR_dataset/                      # Master 5,401-image dataset splits
│   ├── train/                           # 3,780 single-line crops (70%)
│   ├── val/                             # 810 single-line crops (15%)
│   ├── test/                            # 811 single-line crops (15%)
│   ├── dataset_splits_summary.json      # Dataset statistics and metadata
│   └── *.csv / *.jsonl                  # Training pairs and conversational formats
├── results/                             # Complete raw outputs & prediction CSVs
│   ├── QUROCR_finetuned_35B_A3B.csv     # 23.92% CER QUROCR fine-tuned model predictions
│   ├── QUROCR_finetuned_35B_A3B.jsonl   # Full reasoning & inference traces
│   ├── predictions_qwen_qwen3_5-35b-a3b.csv # 30.64% CER zero-shot base predictions
│   ├── raw_qwen_qwen3_5-35b-a3b.jsonl   # Thinking traces for zero-shot base model
│   ├── predictions_google_gemini-2_5-flash.csv # 34.68% CER Gemini predictions
│   └── ...
├── lora_adapter_config.json             # Hyperparameters for PEFT/LoRA adapter
├── reproduce_lora_training.py           # LoRA fine-tuning training script
├── call_ocr.py                          # CLI tool for single-image and batch inference
├── run_benchmark.py                     # Benchmark runner for VLM APIs
├── evaluate_results.py                  # Evaluation script (CER, WER, SER)
├── QUROCR_Colab_OpenSource_OCRs.ipynb   # Colab notebook for Tesseract, EasyOCR, PaddleOCR
├── QUROCR_Colab_Benchmark.ipynb         # Colab notebook for VLM API benchmarking
├── requirements.txt                     # Python package dependencies
└── README.md                            # Documentation
```

---

## ⚡ Quickstart & Installation

### 1. Clone Repository
```bash
git clone https://github.com/raotalha71/QUROCR.git
cd QUROCR
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the repository root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DASHSCOPE_API_KEY=your_alibaba_dashscope_key_here
```

### 3. Run Inference on a Single Image
```bash
python call_ocr.py --image benchmark200/images/img001.jpg
```

### 4. Reproduce the Held-Out Benchmark Evaluation
To run inference across the entire 200-sample benchmark:
```bash
# Evaluate QUROCR
python run_benchmark.py --provider openrouter --model qwen/qwen3.5-35b-a3b

# Evaluate Google Gemini 2.5 Flash
python run_benchmark.py --provider openrouter --model google/gemini-2.5-flash
```

### 5. Compute Error Rates (CER, WER, SER)
To compute the exact metrics:
```bash
python evaluate_results.py
```

---

## 🛠️ LoRA Fine-Tuning Configuration

The LoRA adapter parameters used to specialize `Qwen3.5-35B-A3B` on Urdu Nastaliq are documented in [`lora_adapter_config.json`](./lora_adapter_config.json):

```json
{
  "base_model": "qwen3.5-35b-a3b",
  "lora_parameters": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  },
  "training_hyperparameters": {
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "batch_size_per_device": 4,
    "gradient_accumulation_steps": 4,
    "epochs": 3,
    "bf16": true,
    "max_seq_length": 2048
  }
}
```

To run the fine-tuning workflow via Alibaba Cloud Model Studio:
```bash
python reproduce_lora_training.py --config lora_adapter_config.json
```

---

## 📓 Google Colab Notebooks

For reviewers and researchers without local GPU setups:
* [`QUROCR_Colab_OpenSource_OCRs.ipynb`](./QUROCR_Colab_OpenSource_OCRs.ipynb): One-click execution of classical engines (Tesseract 5 `urd`, EasyOCR `ur`, PaddleOCR) on free Colab CPUs/T4 GPUs.
* [`QUROCR_Colab_Benchmark.ipynb`](./QUROCR_Colab_Benchmark.ipynb): Interactive evaluation pipeline for Vision-Language Models.

---

## 📖 Citation

If you use QUROCR, its dataset splits, or benchmark suites in your research, please cite:

```bibtex
@article{qurocr2026,
  title   = {Parameter Efficient Hybrid Vision-Language Models for Low-Resource Urdu Text Recognition},
  journal = {Neural Computing and Applications},
  year    = {2026},
  note    = {Under Review}
}
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE). Datasets and images are made available for non-commercial academic research under CC-BY-NC-SA 4.0.

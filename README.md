<div align="center">

# SATT: Slice-Aware Temporal Tokenization
### Radiology Report Generation from 3D Abdominal CT

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HF%20Transformers-5.3-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-6.20-F97316)
![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)

B.Tech Major Project, Department of Computer Engineering<br>
Sardar Patel Institute of Technology, Mumbai, 2026

</div>

> [!WARNING]
> **This is a research prototype and is not for clinical use.** Generated reports leave out most findings, sometimes miss serious disease, and have not been reviewed by a radiologist.

---

## Overview

Computed tomography (CT) is used heavily: more than 85 million scans a year in the US, about a quarter of them abdominal. Reporting one abdominal study takes a radiologist around 20 minutes. Report generation from 2D chest X-rays has moved fast, but 3D CT is harder for two reasons. A volume holds far more visual data than a language model can take in, and where a finding sits along the body axis carries clinical meaning.

**SATT** is a lightweight adapter that connects a **frozen 2D vision encoder** to a **large language model** and keeps the order of the CT slices:

1. A frozen **SigLIP ViT-B/16** encodes every axial slice.
2. Neighbouring slices are grouped into **chunks** and mean-pooled.
3. A small **Transformer** models the ordered sequence of chunks.
4. An **MLP** projects the result into **Llama 3.2-3B**, which is quantised to 4 bits and fine-tuned with **LoRA** (QLoRA).

Only **43.2M of 3.35B parameters (1.29%)** are trained, and everything fits on a **single 48 GB GPU**.

## Key Results

Trained on 15,306 scans of the [Merlin](https://arxiv.org/abs/2406.06512) abdominal CT dataset and evaluated on 1,000 test scans with bootstrap confidence intervals:

| Configuration (visual tokens) | RadGraph-F1 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | ClinicalBERT-F1 |
|---|---|---|---|---|---|---|---|
| Mean-Pool (196) | 0.312 | 0.104 | 0.380 | 0.204 | 0.283 | 0.258 | 0.809 |
| ChunkPool (3,136) | 0.308 | 0.118 | 0.395 | 0.194 | 0.277 | 0.275 | 0.808 |
| SATT c = 8 (1,568) | 0.316 | 0.097 | 0.388 | **0.206** | 0.288 | 0.267 | 0.810 |
| **SATT c = 4 (3,136)** | **0.331** | **0.142** | **0.411** | **0.206** | **0.291** | **0.286** | 0.813 |
| SATT c = 2 (6,272) | 0.328 | 0.119 | 0.403 | 0.201 | 0.286 | 0.265 | **0.814** |

**Findings**

- **Slice-aware tokenization helps, modestly.** SATT c = 4 beats mean pooling by **+0.019 RadGraph-F1** (95% CI [0.015, 0.023]) and improves five of the other six metrics.
- **The gain comes from the Transformer, not from having more tokens.** ChunkPool, with the same 3,136 tokens and no Transformer, does no better than mean pooling.
- **c = 4 gives the best trade-off.** It has the best quality at about a third of the training time of c = 2.
- **Decoding matters more than architecture.** Relaxing the repetition constraints raised RadGraph-F1 from **0.244 to 0.331**, more than four times the architectural gain.
- **The main remaining error is under-reporting.** Generated reports mention about one third as many positive findings as radiologists do (around 13 against 39).
- **It runs offline on a laptop.** A report takes 13 to 26 s per scan on an RTX 4060 (8 GB) with 3.65 GB peak memory.

## Architecture

```
CT volume (.nii.gz)
   │  reorient → resample to 64×224×224 → clip [-200, +300] HU → normalise to [-1, 1]
   ▼
SigLIP ViT-B/16 (frozen, per slice)           E: 64 × 196 × 768
   ▼
┌──────────────── SATT adapter (trainable, 18.9M) ────────────────┐
│ 1. Group 64 slices into T = 64/c chunks                         │
│ 2. Mean-pool the c slices in each chunk     G: T × 196 × 768    │
│ 3. Flatten in slice order                   S: (T·196) × 768    │
│ 4. Transformer encoder (2 layers, 8 heads)                      │
│ 5. MLP 768 → 2048 → 3072 (GELU)             V: (T·196) × 3072   │
└─────────────────────────────────────────────────────────────────┘
   ▼
[V ; prompt] → Llama 3.2-3B-Instruct (4-bit NF4, frozen) + LoRA r=16 (24.3M)
   ▼
Findings section of the radiology report
```

| Configuration | `--model_type` | Chunks | Visual tokens | Trainable (Phase 2) |
|---|---|---|---|---|
| Mean-Pool baseline | `baseline` | 1 (global) | 196 | 32.2M |
| ChunkPool control (no Transformer) | `chunkpool` | 16 | 3,136 | 32.2M |
| SATT c = 8 | `satt` | 8 | 1,568 | 43.2M |
| **SATT c = 4** | `satt` | 16 | 3,136 | 43.2M |
| SATT c = 2 | `satt` | 32 | 6,272 | 43.2M |

## Training

Two-phase training in the style of LLaVA, with the same settings for every configuration:

| Setting | Phase 1 (alignment) | Phase 2 (fine-tuning) |
|---|---|---|
| Trained | adapter only | adapter + LoRA |
| Language model | bfloat16, frozen | 4-bit NF4 + LoRA |
| Optimiser | AdamW, wd 0.01 | AdamW, wd 0.01 |
| Learning rate | 1e-4 | 2e-5 |
| Batch × grad. accumulation | 2 × 4 | 1 × 8 |
| Epochs | 3 | 5 (best by validation loss) |

**Decoding:** greedy, repetition penalty 1.1, no n-gram blocking, at most 600 new tokens.

**Hardware:** NVIDIA RTX 6000 Ada (48 GB) for training. Phase 2 took 26.5 h for SATT c = 4.

## Repository Structure

```
├── Phase_1/                    # early prototype (single-script pipeline, debug runs)
├── Phase_2/                    # final system
│   ├── model.py                # SATT, ChunkPool and Mean-Pool adapters + model builders
│   ├── dataset.py              # NIfTI loading and preprocessing
│   ├── preprocess_dataset.py   # optional offline preprocessing to .npy
│   ├── train.py                # two-phase training loop
│   ├── main.py                 # entry point: training and evaluation
│   ├── evaluate.py             # report generation and the seven metrics
│   ├── eval_paths.py           # matches adapter and LoRA checkpoints of one run
│   ├── bootstrap_ci.py         # bootstrap confidence intervals and paired tests
│   ├── text_analysis.py        # error analysis and test-subset check
│   ├── inference.py            # inference engine (CLI + used by the app)
│   ├── app.py                  # Gradio demonstration application
│   ├── start_*.sh              # SLURM job scripts for every training / eval run
│   └── laptop_requirements.txt # pinned environment for the demo laptop
├── paper/                      # research paper (LaTeX source + PDF)
├── docs/                       # project context notes
└── requirements.txt            # training / evaluation dependencies
```

> Scans, reports, generated reports and model checkpoints are **not** included in this repository, in line with the dataset's research terms.

## Getting Started

### 1. Install

```bash
pip install -r requirements.txt
```

To reproduce the exact demo environment (PyTorch 2.10, CUDA 12.8, Transformers 5.3, Gradio 6.20), use `Phase_2/laptop_requirements.txt` instead.

### 2. Data

Request access to the **Merlin** dataset from its authors. You need the NIfTI scans and the reports spreadsheet. The official train / validation / test splits are used.

### 3. Train

Run from `Phase_2/`. Phase 1 trains the adapter, and Phase 2 adds LoRA:

```bash
python main.py --phase 1 --mode train --model_type satt --chunk_size 4 \
  --data_dir /path/to/merlin_data --reports_xlsx /path/to/reports_final.xlsx \
  --checkpoint_dir ./checkpoints --num_slices 64 \
  --batch_size 2 --grad_accum_steps 4 --num_epochs 3 --lr 1e-4
```

```bash
python main.py --phase 2 --mode train --model_type satt --chunk_size 4 \
  --data_dir /path/to/merlin_data --reports_xlsx /path/to/reports_final.xlsx \
  --checkpoint_dir ./checkpoints --num_slices 64 \
  --batch_size 1 --grad_accum_steps 8 --num_epochs 5 --lr_phase2 2e-5
```

For the other configurations, change `--model_type` (`satt`, `baseline`, `chunkpool`) and `--chunk_size` (2, 4, 8). The `start_*.sh` files contain the SLURM jobs used for every run.

### 4. Evaluate

```bash
python main.py --phase 2 --mode eval --model_type satt --chunk_size 4 \
  --data_dir /path/to/merlin_data --reports_xlsx /path/to/reports_final.xlsx \
  --checkpoint_dir ./checkpoints --num_slices 64 \
  --eval_split test --eval_max_samples 1000 \
  --repetition_penalty 1.1 --no_repeat_ngram_size 0
```

> [!NOTE]
> `main.py` still defaults to the original decoding (repetition penalty 1.3, 4-gram blocking). Pass the two flags above to reproduce the reported results.

### 5. Run the demo (offline)

From `Phase_2/`, with trained checkpoints in `./checkpoints`:

```bash
HF_HUB_OFFLINE=1 python app.py
```

The app opens at `http://127.0.0.1:7860` after the model loads (about a minute). For each scan you can:

1. **Upload** a `.nii.gz` scan and see its size, voxel spacing and coverage.
2. **Browse** axial, coronal and sagittal views.
3. **See what the model sees**: 64 preprocessed slices in 16 colour-coded chunks.
4. **Generate a report** with timing, memory use, a download button and the reference report.
5. **Read how SATT works**: pipeline, results and limitations.

To generate reports from the command line instead:

```bash
python inference.py scan1.nii.gz scan2.nii.gz --variant satt_chunk4 --checkpoint_dir ./checkpoints
```

## Limitations

- One dataset and one institution, with no external validation. Evaluated on 1,000 of the 5,123 test scans.
- One training run per configuration, so the confidence intervals reflect variation across test scans, not across training runs.
- No radiologist evaluation. All metrics and error categories are automatic.
- Fixed preprocessing (64 slices, 224×224, a single soft-tissue window) loses detail on small lesions, the lung bases and bone.
- The vision encoder was pretrained on natural images, and the adapter has no axial positional encoding.
- The model sees only one scan, but it imitates the radiologists' comparisons with earlier studies.

## Future Work

- Training objectives that reward correct clinical findings
- Radiologist evaluation and clinical metrics such as GREEN and RaTEScore
- A medical image encoder, axial positional encoding, more slices, higher resolution and several HU windows
- Using prior examinations, validating at other institutions, and comparing with Merlin under one protocol

## Team

| Name | Roll No. |
|---|---|
| Yash Jadhav | 2023300086 |
| Vandit Kenia | 2023300109 |
| Rishit Kothari | 2023300124 |

**Guide:** Prof. Sunil Ghane
Department of Computer Engineering, Bharatiya Vidya Bhavan's Sardar Patel Institute of Technology, Mumbai

## Acknowledgments

We thank Prof. Sunil Ghane for his guidance, and the Department of Computer Engineering at SPIT for access to its GPU cluster. We also thank the authors of the Merlin dataset, SigLIP, Llama 3.2, LLaVA and QLoRA, whose work this project builds on.

## Citation

```bibtex
@thesis{satt2026,
  title  = {SATT: Slice-Aware Temporal Tokenization for Radiology Report Generation from 3D Abdominal CT},
  author = {Jadhav, Yash and Kenia, Vandit and Kothari, Rishit},
  school = {Sardar Patel Institute of Technology, University of Mumbai},
  year   = {2026},
  type   = {B.Tech Major Project}
}
```

# SATT — Complete Project Context

> **How to use this file with Claude (or any AI assistant).**
> Start a new chat, attach this file, and say for example:
> *"This is the full context of my research project SATT. Read it first. Then help me with: <task>."*
> For paper edits, also attach `paper/main.tex` (and `paper/refs.bib` if references change).
> Everything below is the single source of truth as of **7 November 2026**. If something here
> disagrees with an older note, chat or memory, this file wins.

---

## 0. Status at a glance

| Item | Status |
|---|---|
| Training and evaluation (5 configurations, 1,000 test scans) | **Done** (no further GPU work planned) |
| Research paper (IEEE conference format, 11 pages) | **Final draft done**; Turnitin 4% (30 Oct 2026) |
| Phase 4 demo + final presentation (college) | **Done**; external examiner liked the project (Nov 2026) |
| Black Book / ESE report (54 pages) | **Done**, submitted 4 Nov 2026 |
| Conference submission | **Pending** — the only remaining task |

**People.** Team: Yash (Fundy) Jadhav (2023300086, yash.jadhav23@spit.ac.in), Vandit Kenia
(2023300109, vandit.kenia23@spit.ac.in), Rishit Kothari (2023300124, rishit.kothari23@spit.ac.in).
Guide: Prof. Sunil Ghane (sunil_ghane@spit.ac.in), Department of Computer Engineering,
Sardar Patel Institute of Technology (SPIT), Mumbai. (The project was started under Dr. Sanjuktarani
Jena, who left the college; Prof. Ghane took over in Sept 2026.)
Paper author order: Yash Jadhav, Rishit Kothari, Vandit Kenia, Sunil Ghane.

**Repository.** https://github.com/FundyLord/MAJOR_PROJECT (private until the paper is accepted).
Never commit Merlin scans (incl. preprocessed `.npy`), reports, generated reports (`eval_raw_*.csv`), checkpoints, logs, or images/screenshots of scans.

---

## 1. Problem

- >85 million CT exams per year in the US, ~1/4 abdominal; one abdominal study takes a radiologist ~20 min to report; radiologist shortage.
- Report generation works reasonably for **2D chest X-rays**; **3D CT** is harder because of three challenges:
  1. **Volumetric scale** — encoding every slice gives far more visual features than an LLM can take as input.
  2. **Anatomical continuity** — organs/lesions span many slices; position along the body axis matters.
  3. **Vision-language alignment** — visual features must be mapped into the LLM's embedding space.
- Existing 3D methods either use a heavy 3D encoder (Merlin's I3D ResNet152, RadFM) that needs costly 3D pretraining, or **pool globally** (M3D, 3D-CT-GPT), losing where along the body axis a feature is.
- **Research gap:** no published abdominal-CT method combines a frozen 2D slice encoder + local pooling along the body axis with a controllable chunk size + a sequence model over the chunks, and nobody had reported how chunk size affects quality and cost in a controlled comparison. The effect of decoding settings on templated radiology reports was also under-studied.

**Title (exact):** *SATT: Slice-Aware Temporal Tokenization for Radiology Report Generation from 3D Abdominal CT*

**Research questions**
- **RQ1.** Does slice-aware tokenization beat global mean pooling of the same slice features (everything else fixed), and is any gain from more visual tokens or from the Transformer over the chunk sequence?
- **RQ2.** How does chunk size *c* trade report quality against training/inference cost?
- **RQ3.** What errors do generated reports contain, and how much do they depend on decoding settings?

---

## 2. Dataset — Merlin abdominal CT (Stanford)

- De-identified abdominal CT + radiology reports (Blankemeier et al., *Nature* 2026, vol. 652, pp. 1318–1328; arXiv:2406.06512).
- Spreadsheet lists 25,494 scans: 15,314 train / 5,055 val / 5,125 test (official splits, patient-disjoint).
- 11 scans unusable after download/preprocessing → **15,306 / 5,054 / 5,123**.
- Target text: the **findings** section only (organised by organ system: lower thorax, liver & biliary, gallbladder, spleen, pancreas, adrenals, kidneys, GI tract, …).
- Data stays local; used under the dataset's research terms; never redistributed.

## 3. Preprocessing (fixed in advance, not tuned)

NIfTI → reorient to (depth, H, W) → **cubic (order-3) spline resample to 64×224×224** → **clip HU to [−200, +300]** (soft-tissue window) → scale to [0,1] → replicate to 3 channels → normalise to [−1, 1] (SigLIP input). Same for train/val/test, **no augmentation**. Done once by `preprocess_dataset.py` into `.npy` files.

Why: 224×224 = SigLIP native size (14×14 = 196 patches); 64 slices = fixed depth that fits 48 GB (64×196 = 12,544 patch embeddings before pooling; Merlin uses 160 slices); soft-tissue window covers most abdominal findings but saturates air/bone; cubic = fewer artefacts.
Costs: depth resampling blurs small lesions; slice spacing varies across scans; 512→224 in-plane loses detail; lung bases and bone poorly shown.

## 4. Method — SATT architecture

Notation: Z = 64 slices, N = 196 patches/slice, D_v = 768 (SigLIP), D_l = 3072 (Llama), chunk size c, T = Z/c chunks.

```
CT volume (Z0×512×512)
 → preprocessing (64×3×224×224)
 → SigLIP ViT-B/16, frozen, per slice           E: 64×196×768      (google/siglip-base-patch16-224)
 → SATT adapter (trainable, 18.9M):
     1. group into T = 64/c chunks of c adjacent slices
     2. mean-pool within each chunk              G: T×196×768
     3. flatten in slice order                   S: TN×768
     4. Transformer encoder: 2 layers, 8 heads, FFN 2048, dropout 0.1, NO positional encoding   S': TN×768
     5. MLP 768→2048→3072 with GELU              V: TN×3072
 → [visual tokens V] + instruction prompt → Llama 3.2-3B-Instruct (+LoRA) → findings text
```

- Visual tokens: **c=2: 6,272 · c=4: 3,136 · c=8: 1,568 · ChunkPool: 3,136 · Mean-Pool: 196**.
- Adapter params: 18.9M total = 11.0M Transformer + 7.9M MLP; independent of c.
- **No positional encoding in the adapter (deliberate, simplest design).** Each token still knows its in-plane position (SigLIP's learned patch position embedding), but not which chunk it came from; self-attention without PE is permutation-equivariant, so the adapter can mix chunks but not use their order. Order is **not lost**: tokens leave in slice order and Llama applies RoPE to the whole sequence, so the LLM/LoRA can use axial order. An axial PE would cost only T×768 params (12,288 at c=4) — **not tested** (limitation/future work).
- "Temporal" is an analogy with video frames: slices are an ordered sequence along the body axis; it is depth, not time.
- Prompt: *"Analyze this abdominal CT scan and generate a clinical radiology report."* in Llama chat format, placed **after** the visual tokens; loss on report tokens only.
- c=1 (12,544 tokens) not tested: c=2 already needed gradient checkpointing and 73 h of Phase 2. c≥16 and overlapping windows not tested.

### Baselines / controls
- **Mean-Pool baseline:** average all 64 slices → 196 tokens → same MLP (7.9M). Differs from SATT in two ways at once (16× fewer tokens AND no Transformer).
- **ChunkPool control:** SATT's c=4 grouping+pooling, **no Transformer** → 3,136 tokens, MLP only (7.9M). Separates "more tokens" from "Transformer".
  - c=4 − ChunkPool = effect of the Transformer at fixed token count.
  - ChunkPool − Mean-Pool = effect of chunking alone.

## 5. Language model and training

- **Llama 3.2-3B-Instruct**. LoRA r=16, α=32, dropout 0.05, on all 7 projections (q,k,v,o,gate,up,down) of 28 layers.
- **Phase 1 (alignment, LLaVA-style):** train adapter only; SigLIP and Llama frozen; **Llama loaded in bfloat16** (`build_llm_phase1()` in `model.py`). 3 epochs, lr 1e-4, batch 2×4 accumulation (1×8 for c=2).
- **Phase 2 (fine-tuning):** Llama reloaded in **4-bit NF4 (QLoRA, double quantisation, bf16 compute)** + fresh LoRA; train adapter + LoRA jointly. 5 epochs (best by val loss), lr 2e-5, batch 1×8.
- Both phases: AdamW (wd 0.01), grad clip 1.0, max text 512 tokens; hyperparameters are standard QLoRA values, **not tuned**, identical across configurations (this is what makes the comparison controlled).
- Single-phase control: not run.
- Best Phase-2 val loss: 1.024–1.027 for the four main configs (same epoch); ChunkPool 1.024 (5th epoch).
- Gradient checkpointing used in Phase 2 for c=2 and ChunkPool (ChunkPool trained later with a code version that enables it by default) — changes speed/memory, not gradients.
- Hardware: one **NVIDIA RTX 6000 Ada (48 GB)** on SPIT's shared GPU cluster.
- **Paper fix made on 7 Nov 2026:** the paper earlier said Phase 1 used 4-bit; corrected to bfloat16 to match the code (training table row is now "Language model precision: bfloat16 | 4-bit NF4").

### Parameter counts
| Component | Status | Params |
|---|---|---|
| SigLIP ViT-B/16 (incl. unused 7.1M pooling head; patch features come from the 85.8M encoder) | frozen | 92.9M |
| Llama 3.2-3B-Instruct | frozen | 3,212.7M |
| SATT adapter (any c) = Transformer 11.0M + MLP 7.9M | trainable | 18.9M |
| Mean-Pool / ChunkPool adapter (MLP only) | trainable | 7.9M |
| LoRA (r=16, 7 projections, 28 layers) | trainable (Phase 2) | 24.3M |
| **Phase-2 trainable, SATT** | | **43.2M (1.29% of ~3.35B)** |
| Phase-2 trainable, Mean-Pool / ChunkPool | | 32.2M (0.96%) |
| Whole system, SATT | | 3,348.8M |

LoRA = 0.76% of the LLM (0.75% of 3,237.1M incl. LoRA — the PEFT-printed figure).

## 6. Decoding

- Greedy, max 600 new tokens.
- **Original setting** (initial experiments): repetition penalty **1.3 + blocking repeated 4-grams**.
- Problem: structured reports legitimately repeat short statements ("Normal.") across organ sections; strong constraints forbid that.
- **Selection on 100 validation scans** (c=4 model, criterion fixed in advance = highest RadGraph-F1):

| RP | no-repeat n-gram | RadGraph | ROUGE-2 | BLEU-4 | Loops | "Normal." sections | s/scan |
|---|---|---|---|---|---|---|---|
| 1.3 (original) | 4 | 0.237 | 0.098 | 0.107 | 0% | 8% | 27.6 |
| 1.0 | 0 | 0.283 | 0.178 | 0.126 | 21% | 95% | 14.3 |
| **1.1 (selected)** | **0** | **0.338** | **0.211** | **0.153** | 2% | 86% | 14.3 |
| 1.0 | 12 | 0.322 | 0.195 | 0.143 | 3% | 87% | 14.1 |

  Validation references: 62% of liver/gallbladder/spleen/pancreas/adrenal sections are just "Normal.". With the original setting: 8% "Normal.", **89% of reports said gallbladder surgically absent (refs 22%)**, **98% had prior-study wording** although the model never sees a prior. Selected vs original on validation: +0.10 RadGraph (95% CI [0.078, 0.126]); selected halves generation time.
- **Test confirmation (1,000 scans, c=4):** RadGraph-F1 **0.244 → 0.331 (+0.087)**, ROUGE-2 0.100 → 0.206, BLEU-4 0.112 → 0.142. Largest single gain in the whole study — bigger than any architectural change.
- Selected setting (RP 1.1, no blocking) used once for all five configurations on test.

## 7. Evaluation protocol

- **1,000 test scans** drawn at random from 5,123 with **seed 42**, fixed before any evaluation (generation takes 10–15 s/scan; full test split × 5 configs infeasible on the shared cluster).
- Subset is representative vs the remaining 4,123: reference length KS D=0.022, p=0.81; ten common findings/terms differ ≤1.5 pp (all χ² p>0.3).
- Metrics (all [0,1], higher better): BLEU-4 (sacreBLEU), ROUGE-1/2/L, METEOR, **ClinicalBERT-F1** (BERTScore with Bio_ClinicalBERT; 12 layers, 512-token limit set explicitly), **RadGraph-F1** (RadGraph-XL, entity-and-relation level) — the most clinically meaningful.
- Statistics: 95% **percentile bootstrap** CIs (10,000 resamples; 2,000 for corpus BLEU-4); **paired bootstrap** between configurations. CIs reflect choice of test scans, **not** training-run variance (each config trained once, one seed).

## 8. Results

### Main results (1,000 test scans; 95% CI)
| Config (tokens) | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | ClinicalBERT-F1 | RadGraph-F1 |
|---|---|---|---|---|---|---|---|
| Mean-Pool (196) | 0.104 [.097,.111] | 0.380 [.373,.386] | 0.204 [.197,.211] | 0.283 [.276,.290] | 0.258 [.251,.266] | 0.809 [.807,.811] | 0.312 [.304,.320] |
| ChunkPool (3,136) | 0.118 [.111,.126] | 0.395 [.388,.401] | 0.194 [.188,.200] | 0.277 [.271,.283] | 0.275 [.268,.283] | 0.808 [.806,.810] | 0.308 [.301,.316] |
| SATT c=8 (1,568) | 0.097 [.090,.104] | 0.388 [.380,.395] | 0.206 [.199,.213] | 0.288 [.281,.295] | 0.267 [.260,.275] | 0.810 [.808,.812] | 0.316 [.308,.324] |
| **SATT c=4 (3,136)** | **0.142** [.135,.150] | **0.411** [.405,.418] | **0.206** [.199,.214] | **0.291** [.284,.298] | **0.286** [.279,.293] | 0.813 [.811,.815] | **0.331** [.323,.339] |
| SATT c=2 (6,272) | 0.119 [.112,.126] | 0.403 [.396,.409] | 0.201 [.195,.208] | 0.286 [.279,.292] | 0.265 [.258,.272] | **0.814** [.812,.816] | 0.328 [.320,.336] |

### Paired differences (A − B; * = 95% CI excludes zero)
| Metric | c4 − MeanPool | c4 − ChunkPool | ChunkPool − MeanPool | c4 − c8 | c2 − c4 |
|---|---|---|---|---|---|
| RadGraph-F1 | **+0.019*** [.015,.023] | **+0.023*** [.019,.027] | −0.004* [−.007,−.0003] | +0.015* [.011,.019] | −0.004 [−.008,.0002] |
| BLEU-4 | +0.038* | +0.024* | +0.015* | +0.045* | −0.023* |
| ROUGE-1 | +0.032* | +0.017* | +0.015* | +0.024* | −0.009* |
| ROUGE-2 | +0.003 [−.001,.006] | +0.013* | −0.010* | +0.001 | −0.005* |
| ROUGE-L | +0.008* | +0.014* | −0.006* | +0.003 | −0.005* |
| METEOR | +0.027* | +0.010* | +0.017* | +0.018* | −0.021* |
| ClinicalBERT-F1 | +0.004* | +0.005* | −0.001 | +0.003* | +0.001 |

**Answers**
- **RQ1:** c=4 beats Mean-Pool on 6/7 metrics (all but ROUGE-2): RadGraph 0.312 → 0.331 (+0.019, ~6% relative). Positive but **modest**. ChunkPool (same tokens, no Transformer) is no better than Mean-Pool on RadGraph, and c=4 beats ChunkPool on all 7 metrics → **the gain comes from the Transformer over the chunk sequence, not from more tokens.** (Untested explanation: without the Transformer each token is one pooled patch and relating thousands of them is left to the LLM.) Transformer tested only at c=4, not on the 196 mean-pooled tokens.
- **RQ2:** quality is **not monotonic** in c: c=4 best on 6/7; c=8 ≈ baseline; c=2 not better than c=4 (RadGraph difference not significant). c=4 = best trade-off (best quality at ~1/3 of c=2's Phase-2 time).
- **RQ3:** decoding matters as much as architecture (+0.087); main remaining error is **under-reporting**.

### Cost (one RTX 6000 Ada)
| Config | Tokens | Trainable | Phase-2 time | Train samples/s | Inference scans/min | Peak mem |
|---|---|---|---|---|---|---|
| Mean-Pool | 196 | 32.2M | 5.5 h | 3.9 | 5.3 | 2.7 GB |
| ChunkPool | 3,136 | 32.2M | 28.8 h † | 0.7 | 4.8 | 3.4 GB |
| c=8 | 1,568 | 43.2M | 12.3 h | 1.7 | 5.7 | 3.0 GB |
| c=4 | 3,136 | 43.2M | 26.5 h | 0.8 | 4.2 | 3.4 GB |
| c=2 | 6,272 | 43.2M | 73.4 h † | 0.3 | 4.7 | 4.2 GB |

† gradient checkpointing (slower; not directly comparable). Phase-2 time grows faster than tokens (attention is O(L²·D)). Inference: 10.5–14.3 s/scan, mostly driven by report length (c=4 writes the longest reports → slowest). Phase-1 times omitted (jobs interrupted/resumed).

### Context: published results on Merlin data (NOT a controlled comparison)
| Model | BLEU | ROUGE-2 | RadGraph-F1 |
|---|---|---|---|
| RadFM (as reported in Merlin paper) | 0.000 | 0.011 | 0.008 |
| Merlin (as reported) | 0.102 | 0.262 | 0.293 |
| SATT c=4 (ours) | 0.142 (BLEU-4) | 0.206 | 0.331 (entity+relation level) |

Why not controlled: Merlin generates organ by organ with organ-specific prompts; full 5,137-scan test split vs our 1,000; BLEU n-gram order, metric implementations and RadGraph level not stated; Merlin's BERTScore uses a general encoder (so omitted). Merlin = RadLlama-7B + I3D ResNet152 pretrained on CT; its 160 h covers foundation pretraining. **We do not claim SATT beats Merlin.**

### Error analysis (1,000 test scans; automatic, NOT radiologist-validated)
| | Ref. | Mean-Pool | ChunkPool | c=8 | c=4 | c=2 |
|---|---|---|---|---|---|---|
| Words / report | 303 | 121 | 136 | 114 | 160 | 136 |
| Positive findings / report | 39.1 | 12.9 | 15.8 | 11.6 | 12.6 | 13.2 |
| Negated findings / report | 10.1 | 4.6 | 4.9 | 4.4 | 7.2 | 6.8 |
| Positive-finding precision | — | 0.33 | 0.31 | 0.38 | 0.36 | 0.36 |
| Positive-finding recall | — | 0.11 | 0.13 | 0.11 | 0.12 | 0.12 |
| Gallbladder "absent" | 23% | 29% | 42% | 47% | 10% | 21% |
| … precision of that claim | — | 0.35 | 0.36 | 0.33 | 0.35 | 0.37 |
| Prior-study wording | 66% | 16% | 15% | 9% | 27% | 15% |
| Original image-number citation | 45% | 26% | 17% | 14% | 33% | 25% |

- Error types: **factual error** (contradicts reference), **unsupported information** (cannot come from the input, e.g. prior comparison, image numbers), **omission**, **potentially clinically significant**. "Hallucination" = first two.
- **Omission dominates**: ~1/3 as many positive findings as radiologists; ~12% recall (exact-text match, so true recall somewhat higher, but the 3× count gap doesn't depend on matching). Merlin's authors report the same tendency.
- Prior-study wording is learned from references (2/3 of test refs compare with a prior); the count is an upper bound (includes "prior colectomy" etc.).
- When c=4 says gallbladder absent, it is right only 35% of the time.
- Representative c=4 cases (paper Table): (1) gallbladder normal → "surgically absent" (factual, low impact); (2) stable hepatic hemangiomas + mild steatosis → "Normal." (omission, moderate); (3) focal fat + tiny likely cysts → "Normal." (omission, low); (4) post-ileocolectomy inflammation at neo-terminal ileum → "prior partial colectomy … rectal wall thickening" (factual + omission, potentially significant). Clinical-impact column is the authors' judgement.

## 9. Demo application (Phase 4)

- Gradio app (`Phase_2/app.py`) using the same inference engine as evaluation (`inference.py`); fully offline: `HF_HUB_OFFLINE=1 python app.py`. Loads SigLIP + adapter + 4-bit Llama once (~1 min).
- Laptop: Fedora, **RTX 4060 Laptop GPU 8 GB**; PyTorch 2.10 (CUDA 12.8), Transformers 5.3.0, Gradio 6.20.0 (model libs pinned).
- Page sections: (1) upload `.nii.gz` or pick one of 7 prepared test scans, header info shown; (2) axial/coronal/sagittal viewer; (3) "what the model sees": 64 preprocessed slices in 16 colour-coded chunks; (4) generate report (formatted card, time, memory, download, reference in collapsed panel); (5) how SATT works. Research-prototype warning shown at top and with every report.
- Results: **13–26 s per scan** incl. preprocessing, **3.65 GB peak GPU memory**. Of 7 scans, **3 reproduced the saved evaluation report word for word**, 4 differed in small details (plausibly floating-point differences between GPU types flipping a near-tied greedy token — **not verified**).
- Demo-scan observations (team's own comparison, not a radiologist's): 2 got main finding right but missed incidentals; 2 got a similar "appendicitis with abscess"; 1 invented a 1.2 cm right-middle-lobe nodule; 2 had critical omissions (missed liver metastases; missed gastric/PEG-tube leak with abscesses). Shows why it must not be used clinically.
- The demo report card can end with a short numbered "impression" although the model is trained on the findings section (seen in the Black Book screenshot) — worth knowing if asked.

## 10. Engineering challenges (good viva / reviewer-response material)

- **OOM at c=2** (44.5 of 47.4 GiB) → gradient checkpointing (+ `enable_input_require_grads`, `use_cache=False`).
- **NaN/Inf losses** → skip the batch; after one job resumed from a corrupted checkpoint and skipped every batch for ~20 h, added a **dead-model guard** that aborts if no finite loss.
- **Metric crashes**: ClinicalBERT scorer (missing layer count, huge placeholder max length) → set 12 layers, 512 tokens; RadGraph/ClinicalBERT retry scripts re-score saved predictions without regenerating.
- **Checkpoint mix-ups** → `eval_paths.py` resolves adapter/LoRA paths from the configuration name; the inference engine checks the recorded model type and chunk size.
- **Shared cluster interruptions** → regular checkpoints (adapter, LoRA, optimiser, step) and resume.

## 11. Limitations (as stated in the paper)

Single dataset/institution, no external validation · 1,000-scan test subset · **single training run per config** (differences of 0.004–0.019 may be partly run-to-run variation) · ChunkPool control only at c=4, one run; no Transformer-over-196-tokens test · **no radiologist evaluation**; not for clinical use, not even as an unsupervised draft · no axial positional encoding · SigLIP is natural-image pretrained · fixed preprocessing (64 slices, 224², soft-tissue window) · LLM/report prior bias (stock phrases, prior-study wording) · single-scan input (no priors).

## 12. Future work

Training objectives that reward correct findings (e.g. RadGraph/clinical reward) · radiologist evaluation + GREEN / RaTEScore · full test split and multiple seeds; extend controls (Transformer on 196 tokens, ChunkPool at other c) · medical image encoder, axial PE, more slices, higher resolution, multiple HU windows · prior examinations, external validation, and comparing with Merlin under one protocol.

## 13. History (how we got here)

1. Early prototype (`Phase_1/`) → full pipeline (`Phase_2/`).
2. First end-to-end run: preprocessed ~25K scans; two-phase training; eval on 300 test samples with the original decoding (old numbers like BLEU 8.22 / ROUGE-L 0.28 / BERTScore 0.84 are **obsolete** — do not use).
3. Phase 3 experiments: Mean-Pool baseline, c=2, c=8, re-evaluation of c=4 on 1,000 scans with RadGraph-F1 and ClinicalBERT-F1, bootstrap CIs; decoding pilot on 100 validation scans.
4. Paper drafted, reviewed by Prof. Ghane (Oct 2026); he asked to clarify trainable parameters etc. (`param_count.py`); on 23 Oct decided to add only the ChunkPool ablation (no new positional-encoding experiment), then finalise.
5. Phase 4 demo + PPT (30 Oct), Turnitin 4% (30 Oct), Black Book submitted (4 Nov), final presentation done (Nov 2026).

## 14. Turnitin

Checked by Prof. Ghane on 30 Oct 2026, submission ID trn:oid:::3618:156520696; 11 pages, 9,882 words. **4% overall similarity**, no integrity flags, no source >1% (bibliography, quotes and matches <8 words excluded). The only edits after this check: the Phase-1 precision correction and the author e-mail addresses (7 Nov 2026).

## 15. Files and where they are

Repository layout after this bundle is added:
```
MAJOR_PROJECT/
├── Phase_1/                 early prototype
├── Phase_2/                 final code
│   ├── model.py             SATTAdapter, MeanPoolAdapter, ChunkPool; LLM builders (phase1 bf16, phase2 4-bit+LoRA, eval)
│   ├── dataset.py           Merlin loader
│   ├── preprocess_dataset.py  one-time NIfTI → .npy
│   ├── main.py, train.py    two-phase training, checkpoint/resume, dead-model guard
│   ├── evaluate.py          generation + 7 metrics
│   ├── retry_radgraph.py, radgraph_per_sample.py, retry_clinicalbert.py   re-scoring saved outputs
│   ├── bootstrap_ci.py      CIs + paired bootstrap
│   ├── text_analysis.py     error analysis + subset representativeness check
│   ├── case_examples.py     picks representative error cases
│   ├── pilot_summary.py     decoding-pilot table
│   ├── param_count.py       exact parameter counts
│   ├── eval_paths.py        checkpoint path resolution
│   ├── inference.py         inference engine (scan → report)
│   ├── app.py               Gradio demo
│   ├── test_local.py        local tests without the dataset
│   └── start_*.sh           cluster job scripts
├── paper/                   IEEE paper: main.tex, refs.bib, SATT_paper.pdf (figures are TikZ inside main.tex)
└── docs/
    └── SATT_CONTEXT.md      this file
```
**Kept only in the personal backup (`SATT_Backup_Nov2026.zip`, on the laptop / Google Drive), NOT on GitHub**, because they contain images of Merlin scans or screenshots of reports:
- `blackbook/` — Black Book (ESE report): `main.tex` + `figures/` (pdfLaTeX, no BibTeX), `SATT_BlackBook.pdf` (54 pages)
- `docs/SATT_Project_Study_Guide.pdf` — 31-page study guide + 52 viva Q&A
- `docs/SATT_Phase4_Review.pptx` — final presentation deck
- plus copies of `paper/` and this file.
Compile the paper: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (or Overleaf with pdfLaTeX).
Compile the Black Book: `pdflatex main` twice.

## 16. Conference submission checklist (the remaining task)

1. **Choose the venue** and read its call for papers: page limit, template (paper is IEEE conference format, 11 pages — many venues allow 4–8 pages, so it may need cutting), review type (single/double-blind), deadline, fees, presence requirement.
2. **Double-blind?** Make an anonymised copy: remove author block, e-mails, SPIT, the GitHub link and any acknowledgments; cite own prior work in third person. Keep the repo private until acceptance.
3. **One venue at a time** unless both venues' policies allow parallel submission.
4. **Shortening plan** (if needed): move the literature table and cost table to supplementary material; shorten Related Work and Preprocessing rationale; keep Tables main/paired/decoding/errors (they carry the claims).
5. **Dataset terms:** cite Merlin correctly (Nature 2026) and do not include any report text beyond short excerpts already in the case table.
6. **Claims to keep honest:** modest gain (+0.019); single training run; automatic metrics only; no radiologist; no claim of beating Merlin.
7. Likely reviewer questions: Why no axial PE? Why only one seed? Why 1,000 scans? Why SigLIP (natural images)? How does it compare with Merlin under one protocol? Why does BLEU/ROUGE improve more than RadGraph? — answers are in sections 4, 7, 8 and 11 above.
8. Candidate venue types (check current deadlines yourself): Indian IEEE conferences (e.g. INDICON, INDIACom, IEEE CONECCT) for a faster decision; medical-imaging venues (ISBI, MIDL, MICCAI workshops) for higher visibility; or a journal if deadlines don't fit.

# Skeleton-JEPA for Real-Time LSF Recognition (Experimental R&D)

An experimental research pipeline exploring the application of Joint-Embedding Predictive Architecture (JEPA) to French Sign Language (LSF) skeleton sequences for self-supervised representation learning and keypoint-to-text translation.

## Context & Background

This exploratory repository is a personal follow-up to the **[HearMyHands (HMH)](https://hearmyhands.asia/)** project ([GitHub repository](https://github.com/nmqx/hearmyhands)), where I served as team lead. 

Following the final project deliverables, I wanted to benchmark whether a self-supervised JEPA framework operating on spatial-temporal skeleton coordinates could capture latent sign transitions better than conventional supervised architectures.

### Outcomes & Findings
- **Mixed Results / Data Scarcity**: While the model successfully learns structured representations on curated/synthetic sets, downstream translation and zero-shot sign generalization suffered significantly from the lack of large-scale, diverse, continuous LSF video datasets.
- **Key Takeaway**: Self-supervised skeleton-JEPA requires substantially larger corpus diversity (thousands of continuous signers) to avoid overfitting and generalize robustly across unseen signing styles.

## Pipeline Overview

```text
Raw Video -> Keypoint Extraction (MediaPipe) -> Multi-reference Normalization -> Skeleton-JEPA Pre-training -> Downstream Fine-Tuning -> Real-Time Inference
```

The system processes two data modalities:
- **Type A (Unlabelled)**: Continuous signing videos for self-supervised Skeleton-JEPA pre-training.
- **Type B (Labelled)**: Segmented clips aligned with French text/glosses for downstream classification and generative translation.

## Features

- **Skeleton-JEPA Pre-training**: Spatial and temporal masking with an EMA target encoder predicting latent trajectory representations.
- **Graph Transformer Topology**: Canonical 89-point keypoint layout (8 body, 42 hands, 39 face landmarks) with normalized spatial, velocity, and acceleration features.
- **Real-Time Webcam Inference**: Low-latency sign and alphabet recognition with CTC decoding.
- **JEPA-to-LLM Soft-Prompting**: Experimental bridge projecting JEPA latent embeddings into causal language models (SmolLM2).

## Installation

```bash
git clone https://github.com/NarenkuII/Jepa_for_real_time_lsf.git
cd Jepa_for_real_time_lsf
pip install -e ".[vision,text,viz]"
```

## Quickstart

### 1. Synthetic Dataset Generation & Verification

```bash
python tools/generate_synthetic_dataset.py --output_dir data/synthetic
pytest tests/
```

### 2. Dataset Processing (Type B Alphabet)

```bash
python tools/build_alphabet_dataset.py \
  --source_root data/raw_datasets \
  --output_dir data/alphabet_type_b \
  --delegate gpu
```

### 3. Skeleton-JEPA Pre-training

```bash
python tools/train_jepa.py \
  --train-manifest data/canonical/manifests/type_a_train.jsonl \
  --val-manifest data/canonical/manifests/type_a_val.jsonl \
  --output-dir runs/jepa_pretrain
```

### 4. Downstream Alphabet Fine-Tuning

```bash
python tools/train_alphabet_classifier.py \
  --checkpoint runs/jepa_pretrain/final.pt \
  --output-dir runs/alphabet_jepa
```

## Real-Time Inference

### Single Letter Recognition

```bash
python tools/realtime_alphabet.py --camera 0
```

### Continuous Fingerspelling with CTC

```bash
# Build sequence corpus
python tools/build_continuous_alphabet_dataset.py --train-count 4000 --val-count 400 --test-count 400

# Train CTC decoder head
python tools/train_continuous_ctc.py \
  --checkpoint runs/jepa_pretrain/best.pt \
  --output-dir runs/alphabet_continuous_ctc

# Run live inference
python tools/realtime_continuous_alphabet.py --camera 0 --checkpoint runs/alphabet_continuous_ctc/best.pt
```

## JEPA-to-LLM Soft-Prompting (Experimental)

Projects learned JEPA representations as soft-prompt embeddings into a lightweight causal language model:

```bash
pip install -e ".[llm]"

# Train adapter projector
python tools/train_jepa_llm.py \
  --jepa-checkpoint runs/jepa_pretrain/best.pt \
  --train-manifest data/mediapi_rgb_canonical/manifests/mediapi_rgb_text_train.jsonl \
  --val-manifest data/mediapi_rgb_canonical/manifests/mediapi_rgb_text_val.jsonl \
  --output-dir runs/jepa_llm_final \
  --llm-name HuggingFaceTB/SmolLM2-360M-Instruct \
  --freeze-encoder

# Evaluate generated sentences
python tools/evaluate_jepa_llm.py \
  --checkpoint runs/jepa_llm_final/best_adapter.pt \
  --manifest data/mediapi_rgb_canonical/manifests/mediapi_rgb_text_test.jsonl \
  --output-dir runs/jepa_llm_final/evaluation
```

## Debugging and Analysis Tools

- `tools/realtime_skeleton_viewer.py`: Real-time skeleton overlay with FPS and confidence tracking.
- `tools/normalized_skeleton_viewer.py`: Visualizer for raw vs. normalized multi-reference body coordinates.
- `tools/jepa_mask_debugger.py`: Interactive inspection of spatial-temporal JEPA masks.
- `tools/analyze_text_errors.py`: Evaluation metrics reporting (BLEU, chrF, WER, CER).

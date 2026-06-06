# Jepa_for_real_time_lsf

Pipeline de recherche Skeleton-JEPA pour apprendre des représentations de squelettes LSF et fine-tuner une traduction directe `keypoints -> phrase française`.

Le modèle n'apprend pas depuis les pixels. Les vidéos servent à extraire les keypoints MediaPipe et au debug visuel. L'entrée réelle des modèles est une séquence de keypoints visage / corps / mains.

## Pipeline

```text
vidéos brutes
-> extraction keypoints visage / corps / mains
-> normalisation multi-repères
-> pré-entraînement Skeleton-JEPA sur données non annotées
-> fine-tuning skeleton-to-text sur segments alignés avec phrase française
-> inférence offline ou temps réel
-> phrase française prédite
```

Deux types de données sont supportés :

- Type A : vidéos non annotées, utilisées pour pré-entraîner Skeleton-JEPA.
- Type B : vidéos ou segments alignés avec une phrase française, utilisés pour l'alignement skeleton-text optionnel et le fine-tuning génératif.

## Objectif génératif

La tâche principale est une traduction directe `keypoints -> phrase française` avec une architecture encoder-decoder et une cross entropy token-level avec teacher forcing.

## Skeleton-JEPA

Skeleton-JEPA adapte le principe JEPA aux séquences de keypoints : un encodeur contexte observe des frames ou articulateurs visibles, un predictor prédit les latents cibles, et un encodeur cible EMA produit les représentations à prédire. Le système prédit des représentations latentes de squelette, jamais des pixels.

Références conceptuelles à ajouter en submodules :

```bash
git submodule add https://github.com/facebookresearch/ijepa external/ijepa
git submodule add https://github.com/facebookresearch/jepa external/vjepa
git submodule add https://github.com/facebookresearch/jepa-intuitive-physics external/jepa-intuitive-physics
git submodule add https://github.com/facebookresearch/td_jepa external/td_jepa
git submodule update --init --recursive
```

## Installation

```bash
git clone https://github.com/NarenkuII/Jepa_for_real_time_lsf.git
cd Jepa_for_real_time_lsf
pip install -e ".[dev]"
```

Pour l'extraction réelle de keypoints :

```bash
pip install -e ".[vision,text,viz]"
```

## Dataset synthétique

Le dépôt contient un générateur de mini dataset synthétique pour tester le pipeline sans vraies vidéos.

```bash
python tools/generate_synthetic_dataset.py --output_dir data/synthetic
pytest tests/
```

## Dataset alphabet Type B

Le builder alphabet reconstruit les keypoints uniquement depuis les vidéos des dossiers `clips`. Il n'utilise jamais les anciens dossiers `keypoints` ou `preview`.

```bash
python tools/build_alphabet_dataset.py \
  --source_root "C:\Users\Narenku\Documents\000000000000000_test_projet_2a\segemntation-last-04-05-26\workspace\datasets" \
  --output_dir data/alphabet_type_b \
  --delegate gpu
```

Le traitement est reprenable : les fichiers `.npz` existants sont conservés. Les sorties principales sont :

```text
data/alphabet_type_b/keypoints/{train,val,test}/
data/alphabet_type_b/manifests/alphabet_all.jsonl
data/alphabet_type_b/manifests/alphabet_clean.jsonl
data/alphabet_type_b/manifests/alphabet_review.jsonl
data/alphabet_type_b/manifests/alphabet_train.jsonl
data/alphabet_type_b/manifests/alphabet_val.jsonl
data/alphabet_type_b/manifests/alphabet_test.jsonl
data/alphabet_type_b/reports/dataset_report.json
data/alphabet_type_b/reports/dataset_report.html
```

Chaque NPZ contient les keypoints bruts, les 18 features normalisées, la confiance, le masque de validité et les métadonnées du clip. Les splits sont séparés par signeur.

## Vérifier le dataset

```bash
python tools/check_dataset.py --config configs/default.yaml
```

Sorties attendues :

```text
reports/dataset_report.html
reports/dataset_stats.json
reports/figures/dataset/
```

## Extraction keypoints

```bash
python tools/extract_keypoints.py \
  --manifest data/manifests/unlabelled_videos.jsonl \
  --output_manifest data/manifests/unlabelled_with_keypoints.jsonl \
  --output_dir data/keypoints/unlabelled \
  --config configs/default.yaml
```

```bash
python tools/extract_keypoints.py \
  --manifest data/manifests/labelled_segments.jsonl \
  --output_manifest data/manifests/labelled_segments_with_keypoints.jsonl \
  --output_dir data/keypoints/labelled \
  --config configs/default.yaml
```

## Pré-entraînement Skeleton-JEPA

```bash
python -m src.training.pretrain_jepa --config configs/pretrain_jepa.yaml
```

## Alignement skeleton-text optionnel

```bash
python -m src.training.train_skeleton_text_alignment --config configs/train_skeleton_text_alignment.yaml
```

## Fine-tuning skeleton-to-text

```bash
python -m src.training.finetune_skeleton_to_text --config configs/finetune_skeleton_to_text.yaml
```

## Inférence

```bash
python tools/predict_segment.py \
  --video data/raw/test/video_001.mp4 \
  --start 12.5 \
  --end 16.8 \
  --checkpoint checkpoints/best_skeleton_to_text.ckpt \
  --config configs/inference.yaml
```

```bash
python tools/realtime_skeleton_to_text.py \
  --checkpoint checkpoints/best_skeleton_to_text.ckpt \
  --config configs/realtime.yaml
```

## Outils de debug

- `tools/realtime_skeleton_viewer.py` : webcam, overlay squelette, FPS et confiance.
- `tools/normalized_skeleton_viewer.py` : vues brut / torse / visage / mains / vitesses.
- `tools/jepa_mask_debugger.py` : visualisation des masques JEPA.
- `tools/overfit_tiny_batch.py` : surapprentissage rapide pour détecter bugs de shapes, loss, tokenizer.
- `tools/analyze_text_errors.py` : erreurs texte, chrF/BLEU, longueurs, répétitions.

## Graphiques produits

Les entraînements écrivent dans `runs/{experiment_name}/` :

- `metrics.csv`
- `figures/`
- `report.html`

Les métriques incluent JEPA latent loss, variance embeddings, normes, collapse score, métriques contrastives Recall@k, et métriques texte BLEU, chrF, ROUGE-L, WER, CER, exact match, phrases vides et répétitions.

## Limites

- Les vidéos sont nécessaires pour extraction et debug, mais le modèle apprend depuis les keypoints.
- La supervision phrase française est faible : elle ne donne ni glosses, ni timestamps de signes.
- JEPA améliore les représentations mais ne remplace pas les annotations alignées.
- BLEU/chrF ne garantissent pas la qualité linguistique.
- Un petit sous-ensemble vérifié manuellement est recommandé.
- La qualité dépend fortement de MediaPipe, de la visibilité des mains, du visage, du regard et du cadrage.
- Le temps réel est expérimental si le modèle est entraîné seulement sur segments phrase-level.

## Extensions futures

Si des glosses ou timestamps deviennent disponibles, ajouter des branches dédiées `keypoints -> glosses`, classification frame-level ou segment-level. Les pseudo-unités `SU_0001` ne sont pas des glosses LSF et ne doivent pas être présentées comme vérité linguistique.

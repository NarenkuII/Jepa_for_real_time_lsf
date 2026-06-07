# Expérience JEPA vers modèle de langue

## Architecture

Le modèle ne demande pas au LLM de comprendre directement des coordonnées.

1. Le `context_encoder` JEPA encode les keypoints corps, mains et visage.
2. Un resampler par attention réduit la séquence à 16 tokens temporels.
3. Un projecteur transforme ces tokens vers la dimension d'embedding du LLM.
4. Ces soft tokens précèdent les tokens français dans un modèle causal.
5. Par défaut, le LLM est gelé et seuls l'encodeur, le resampler et le
   projecteur sont optimisés.

Cette voie est complémentaire au CTC. Le CTC est le meilleur contrôle pour
l'alphabet et l'alignement continu. Le LLM devient pertinent pour produire du
français lorsque des phrases LSF annotées sont disponibles.

## Installation

```bash
.\.venv-jepa\Scripts\python.exe -m pip install -e ".[llm]"
```

Le modèle par défaut est petit pour tenir sur une carte 8 Go. Il peut être
remplacé avec `--llm-name`.

## Format des manifests

Chaque ligne doit pointer vers des keypoints canoniques et une transcription :

```json
{"id":"s01_phrase_0042","keypoints":"data/phrases/s01_0042.npz","text_fr":"je vais a la gare"}
```

Les splits train/validation/test doivent être séparés par signeur.

## Entraînement

```bash
.\.venv-jepa\Scripts\python.exe tools\train_jepa_llm.py \
  --train-manifest data\phrases\train.jsonl \
  --val-manifest data\phrases\val.jsonl \
  --jepa-checkpoint runs\graph_jepa_context_fix\best.pt \
  --output-dir runs\jepa_llm \
  --max-minutes 60
```

Ne pas utiliser `--unfreeze-llm` pour le premier test. Cela augmente fortement
la mémoire et le risque que le langage mémorise les textes sans exploiter le
signal visuel.

Évaluation générative :

```bash
.\.venv-jepa\Scripts\python.exe tools\evaluate_jepa_llm.py \
  --checkpoint runs\jepa_llm\best_adapter.pt \
  --manifest data\phrases\test.jsonl
```

## Webcam

```bash
.\.venv-jepa\Scripts\python.exe tools\realtime_jepa_llm.py --list-cameras
.\.venv-jepa\Scripts\python.exe tools\realtime_jepa_llm.py \
  --camera 0 \
  --checkpoint runs\jepa_llm\best_adapter.pt
```

La génération se déclenche après une absence prolongée des mains. Pour une
prévisualisation pendant le geste, ajouter `--live-every 45`; cela réduit le
débit car la génération est synchrone.

## Limite essentielle

Un LLM ne supprime pas le besoin de paires LSF/français. Sans ces annotations,
le projecteur ne sait pas quelle représentation JEPA correspond à quelle
phrase. Les vidéos non annotées restent utiles au pré-entraînement JEPA, mais
elles ne suffisent pas à apprendre la traduction.

Cette architecture suit l'idée générale des adaptateurs visuel-langage de
[Sign2GPT](https://arxiv.org/abs/2405.04164). Le choix de garder un contrôle CTC
et de pré-entraîner d'abord la représentation visuelle tient compte du résultat
de [FLa-LLM](https://aclanthology.org/2024.lrec-main.620/) : un LLM introduit
trop tôt peut dominer l'apprentissage au détriment de l'encodeur visuel.

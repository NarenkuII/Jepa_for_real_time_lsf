# Pipeline direct mixte

## Objectif

Apprendre directement :

```text
clip A -> A
clip B -> B
clip A+B+C -> ABC
clip de mot épelé -> MOT
phrase LSF continue -> phrase française
```

Il n'y a ni glosses ni CTC. Un Graph Transformer pré-entraîné par JEPA encode
les keypoints. Un Transformer autorégressif produit les caractères.

## Étapes

### 1. JEPA sur Type A et Matignon

```powershell
.\.venv-jepa\Scripts\python.exe tools\train_jepa.py `
  --train-manifest data\type_a_canonical\manifests\type_a_train.jsonl `
  --train-manifest data\matignon_canonical\manifests\matignon_train.jsonl `
  --val-manifest data\type_a_canonical\manifests\type_a_val.jsonl `
  --val-manifest data\matignon_canonical\manifests\matignon_val.jsonl `
  --output-dir runs\jepa_type_a_matignon `
  --max-minutes 60
```

### 2. Alphabet isolé

Les manifests existants sont utilisés directement. Une ligne avec `label: A`
devient la cible texte `"A"`.

### 3. Alphabet synthétique enchaîné

```powershell
.\.venv-jepa\Scripts\python.exe tools\build_direct_alphabet_sequences.py `
  --source-root data\alphabet_canonical `
  --output-dir data\direct_alphabet `
  --train-count 4000 --val-count 400 --test-count 400
```

Le générateur produit aussi `aabb.jsonl`. Il insère des mouvements vers une
position neutre, mais ces transitions restent approximatives.

Un corpus de mots peut contrôler les chaînes générées :

```powershell
.\.venv-jepa\Scripts\python.exe tools\build_direct_alphabet_sequences.py `
  --text-corpus data\words.txt --max-letters 24
```

### 4. Alphabet continu réel

Format attendu :

```json
{"id":"s03_aabb_01","split":"train","source_type":"alphabet_real","signer_id":"s03","keypoints":"data/real_spelling/keypoints/s03_aabb_01.npz","text":"AABB"}
```

Priorités :

- `ABC`, `AABB`, `LL`, `BONJOUR` ;
- 3 à 12 lettres ;
- mains qui descendent ou restent en position entre deux lettres ;
- au moins trois signeurs réservés au test ;
- plusieurs vitesses et distances caméra.

### 5. Matignon et Mediapi-RGB vers phrase française

Voir `docs/matignon_lsf.md`.

Mediapi-RGB est préférable pour la supervision directe car les vidéos sont en
LSF native et les sous-titres français sont alignés aux clips. Préparation :

```powershell
.\.venv-jepa\Scripts\python.exe tools\bootstrap_lsf_datasets.py `
  --open-ortolang --watch-downloads
```

Voir `docs/mediapi_rgb.md`.

### 6. Fine-tuning mixte

```powershell
.\.venv-jepa\Scripts\python.exe tools\train_mixed_direct_text.py `
  --checkpoint runs\jepa_type_a_matignon\best.pt `
  --train-manifest data\alphabet_canonical\manifests\alphabet_train.jsonl `
  --train-manifest data\direct_alphabet\manifests\direct_alphabet_train.jsonl `
  --train-manifest data\real_spelling\manifests\train.jsonl `
  --train-manifest data\matignon_canonical\manifests\matignon_train.jsonl `
  --train-manifest data\mediapi_rgb_canonical\manifests\mediapi_rgb_text_train.jsonl `
  --val-manifest data\alphabet_canonical\manifests\alphabet_val.jsonl `
  --val-manifest data\direct_alphabet\manifests\direct_alphabet_val.jsonl `
  --val-manifest data\real_spelling\manifests\val.jsonl `
  --val-manifest data\matignon_canonical\manifests\matignon_val.jsonl `
  --val-manifest data\mediapi_rgb_canonical\manifests\mediapi_rgb_text_val.jsonl `
  --test-manifest data\alphabet_canonical\manifests\alphabet_test.jsonl `
  --test-manifest data\direct_alphabet\manifests\direct_alphabet_test.jsonl `
  --test-manifest data\real_spelling\manifests\test.jsonl `
  --test-manifest data\matignon_canonical\manifests\matignon_test.jsonl `
  --test-manifest data\mediapi_rgb_canonical\manifests\mediapi_rgb_text_test.jsonl `
  --output-dir runs\mixed_direct_text `
  --max-minutes 60
```

Les manifests `real_spelling` sont omis tant que les clips n'existent pas.

Par défaut, chaque source reçoit le même poids total. Pour privilégier les
vraies transitions :

```powershell
--source-weight alphabet_isolated=1.0 `
--source-weight alphabet_synthetic=1.0 `
--source-weight alphabet_real=2.0 `
--source-weight matignon=1.0
```

## Évaluation

```powershell
.\.venv-jepa\Scripts\python.exe tools\evaluate_mixed_direct_text.py `
  --checkpoint runs\mixed_direct_text\best.pt `
  --manifest data\direct_alphabet\manifests\aabb.jsonl `
  --manifest data\matignon_canonical\manifests\matignon_test.jsonl
```

Le rapport sépare CER, WER, chrF et exact-match par source.

## Interprétation

L'apprentissage synthétique vérifie d'abord que le modèle peut produire
plusieurs caractères. La généralisation est mesurée uniquement sur les clips
continus réels. Matignon apporte des transitions et phrases naturelles, mais
ne remplace pas les exemples explicites d'épellation.

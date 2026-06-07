# Campagne autonome de 5 heures

## Objectif

Comparer sous un budget global de cinq heures :

- pré-entraînement JEPA et initialisation aléatoire ;
- alphabet isolé et alphabet continu ;
- CTC continu, Transformer direct et adaptateur JEPA vers LLM ;
- données canoniques avec les 39 points visage ou avec ces points neutralisés.

Les mêmes splits, graines, clips et budgets sont conservés entre les variantes
avec et sans visage.

## Lancement

```powershell
.\.venv-jepa\Scripts\python.exe tools\run_benchmark_campaign.py `
  --config configs\benchmark_5h.json `
  --max-hours 5 `
  --status-minutes 12
```

Le runner :

- reprend les expériences déjà terminées ;
- saute les solutions dont les données ou dépendances sont absentes ;
- interrompt une commande qui dépasse son budget ;
- réserve douze minutes au reporting ;
- génère systématiquement `reports/benchmark_5h/report.md`.

## Données requises

Disponibles :

- Type A canonique pour JEPA ;
- alphabet Type B isolé, séparé par signeur ;
- alphabet continu synthétique ;
- variante sans visage de l'alphabet continu.

Optionnelles mais nécessaires pour mesurer la traduction LSF continue :

- `data/mediapi_rgb_canonical/manifests/mediapi_rgb_text_{train,val,test}.jsonl` ;
- Matignon peut compléter le pré-entraînement mais ne remplace pas la LSF
  native annotée.

Sans Mediapi-RGB, les expériences Transformer direct et JEPA vers LLM sont
marquées `skipped_missing`. Le rapport distingue explicitement ce cas d'un
échec de modèle.

## Sorties

```text
runs/benchmark_5h/campaign_state.json
runs/benchmark_5h/<experience>/summary.json
runs/benchmark_5h/<experience>/evaluation/
reports/benchmark_5h/report.json
reports/benchmark_5h/experiment_metrics.csv
reports/benchmark_5h/report.md
```

Les classifications produisent accuracy, précision, recall et F1 macro,
micro, pondérés et par classe, ainsi qu'une matrice de confusion. Les modèles
de séquence produisent CER, WER, chrF, exact-match et une matrice de confusion
caractère par caractère incluant insertions et suppressions.

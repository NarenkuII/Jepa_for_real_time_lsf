# Matignon-LSF

Sources : [dépôt Matignon-LSF](https://github.com/JulieLascar/Matignon-LSF) et
[article LREC 2024](https://aclanthology.org/2024.signlang-1.10/).

## Contenu utile

Le corpus annonce :

- 39 heures de LSF interprétée ;
- 67 vidéos de conférences gouvernementales ;
- 15 interprètes ;
- audio et sous-titres français ;
- environ 18 490 segments phrase-level ;
- environ 447 000 tokens français.

Il sert au pré-entraînement JEPA non supervisé et au fine-tuning direct
faiblement supervisé `segment LSF -> phrase française`.

## Limite d'alignement

Les sous-titres suivent la parole française. L'interprétation LSF possède un
retard variable et une formulation différente. Les exemples générés portent
donc `alignment: weak_subtitle_shifted`.

Valeurs par défaut :

- décalage vers le futur : 1,5 seconde ;
- marge avant : 0,5 seconde ;
- marge après : 1 seconde ;
- rejet des fenêtres de plus de 30 secondes.

Un sous-ensemble de 200 à 500 segments réalignés manuellement est nécessaire
pour une évaluation fiable.

## Téléchargement et recadrage

Le dépôt fournit les identifiants et outils, pas les vidéos elles-mêmes :

```powershell
git clone https://github.com/JulieLascar/Matignon-LSF.git external\Matignon-LSF
.\.venv-jepa\Scripts\python.exe -m pip install -e ".[vision,matignon]"

.\.venv-jepa\Scripts\python.exe tools\download_matignon_videos.py `
  --subtitle-zip "external\Matignon-LSF\preprocess_subtitles\data\données_V2.0.0.0\gold_audio_aligned_cr_sentences_based.zip" `
  --output-dir data\matignon_raw
```

Le crop par défaut reproduit le notebook du corpus :

```text
largeur=494, hauteur=494, x=1334, y=417
```

Certaines vidéos peuvent nécessiter un crop différent ou être indisponibles.

## Extraction et manifests

```powershell
.\.venv-jepa\Scripts\python.exe tools\prepare_matignon_lsf.py `
  --subtitle-zip "external\Matignon-LSF\preprocess_subtitles\data\données_V2.0.0.0\gold_audio_aligned_cr_sentences_based.zip" `
  --video-dir data\matignon_raw\cropped `
  --output-dir data\matignon_canonical `
  --delegate gpu
```

Pour un smoke test, ajouter `--max-videos 1`.

## Splits par interprète

Le fallback par vidéo évite une fuite entre segments d'une même vidéo, mais ne
garantit pas des interprètes distincts. Pour l'évaluation finale :

```csv
video_id,signer_id
8ZUIw7jcaZE,interpreter_01
hegyfM0YipI,interpreter_02
```

Puis ajouter `--signer-map data\matignon_signers.csv`.

## Licence

Le dépôt GitHub ne déclare pas actuellement de licence explicite. Les vidéos
proviennent de chaînes gouvernementales/YouTube. Vérifier les droits d'usage et
ne pas redistribuer les vidéos ou visages sans autorisation.

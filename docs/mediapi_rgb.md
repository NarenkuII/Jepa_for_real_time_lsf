# Mediapi-RGB

Mediapi-RGB apporte des vidéos LSF continues, des sous-titres français et des
keypoints MediaPipe/OpenPose. Dans ce projet, les keypoints MediaPipe sont
convertis vers le format canonique commun `89 x 10`.

## Bootstrap

ORTOLANG impose une connexion et l'acceptation explicite de la politique et de
la licence. Le script ne valide pas ces cases à la place de l'utilisateur. Il
peut ouvrir la page, attendre le téléchargement, puis exécuter automatiquement
l'extraction, la conversion et la création des manifests :

```powershell
.\.venv-jepa\Scripts\python.exe tools\bootstrap_lsf_datasets.py `
  --open-ortolang `
  --watch-downloads
```

Après avoir lancé la commande :

1. accepter les conditions sur la page ORTOLANG ;
2. demander l'export contenant les keypoints MediaPipe et les métadonnées ;
3. laisser le téléchargement se terminer dans `Downloads`.

Le script reprend automatiquement lorsque la taille de l'archive est stable.
Il peut être relancé sans recommencer l'extraction ou les conversions déjà
terminées.

Avec une archive déjà téléchargée :

```powershell
.\.venv-jepa\Scripts\python.exe tools\bootstrap_lsf_datasets.py `
  --mediapi-archive "$HOME\Downloads\mediapi-rgb.zip"
```

Avec une URL d'export autorisée :

```powershell
.\.venv-jepa\Scripts\python.exe tools\bootstrap_lsf_datasets.py `
  --mediapi-url "URL_SIGNEE"
```

Un fichier local de cookies peut être fourni avec `--cookie-file`. Il doit
contenir une entrée `nom=valeur` par ligne et ne doit jamais être ajouté à Git.
Les fichiers `*.cookies.txt` et le dossier `.secrets/` sont ignorés.

## Sorties

```text
data/mediapi_rgb_raw/
data/mediapi_rgb_canonical/keypoints/{train,val,test}/
data/mediapi_rgb_canonical/manifests/mediapi_rgb_{split}.jsonl
data/mediapi_rgb_canonical/manifests/mediapi_rgb_text_{split}.jsonl
data/mediapi_rgb_canonical/report.json
```

Les manifests sans suffixe `_text` contiennent tous les clips convertibles et
servent au pré-entraînement JEPA. Les manifests `_text` ne gardent que les clips
avec phrase française et servent au décodeur direct.

Le détecteur de métadonnées accepte CSV, JSONL et JSON. Les colonnes usuelles
`keypoints`, `path`, `clip_id`, `text_fr`, `subtitle` et `split` sont reconnues.
Si l'archive ORTOLANG utilise un autre schéma, lancer d'abord :

```powershell
.\.venv-jepa\Scripts\python.exe tools\prepare_mediapi_rgb.py `
  --source-root data\mediapi_rgb_raw `
  --index CHEMIN_VERS_INDEX
```

## Entraînement

JEPA :

```powershell
--train-manifest data\mediapi_rgb_canonical\manifests\mediapi_rgb_train.jsonl
```

Traduction directe sans glosses ni CTC :

```powershell
--train-manifest data\mediapi_rgb_canonical\manifests\mediapi_rgb_text_train.jsonl
```

Les splits officiels doivent être conservés. Ne pas mélanger MEDIAPI-SKEL avec
Mediapi-RGB si les identités de test se recouvrent.

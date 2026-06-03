# Data Format

## Manifest non annoté

```json
{"id":"unlabelled_000001","video":"data/raw/unlabelled/session_001.mp4","signer_id":"unknown_or_signer_001","source":"unlabelled_corpus","split":"train"}
```

## Manifest segment annoté français

```json
{"id":"clip_000001","video":"data/raw/labelled/video_001.mp4","start":175.72,"end":181.36,"text_fr":"Demain, je vais à l'école.","signer_id":"signer_001","source":"corpus_name","split":"train","quality":{"hands_visible":true,"face_visible":true,"framing":"good","lighting":"good"}}
```

## Manifest enrichi avec keypoints

```json
{"id":"clip_000001","video":"data/raw/labelled/video_001.mp4","start":175.72,"end":181.36,"text_fr":"Demain, je vais à l'école.","keypoints":"data/keypoints/labelled/clip_000001.npz","split":"train","quality_stats":{"missing_ratio":0.04,"left_hand_presence":0.91,"right_hand_presence":0.88,"face_presence":0.96}}
```

## Format `.npz`

Champs attendus :

```text
keypoints: float32 [T, J, F]
confidence: float32 [T, J]
valid_mask: bool [T, J]
fps: float
topology_name: str
source_video: str
start: float optional
end: float optional
```

Features minimales : `x, y, z, confidence, visibility, valid_mask`.

Features dérivées recommandées : coordonnées globales normalisées, coordonnées relatives au torse, au nez et aux poignets, vitesse, accélération, bone vectors et bone lengths.

## Règles de split

- Préférer `split_by_signer: true` pour éviter la fuite d'identité entre train et test.
- Conserver `source` et `signer_id` quand ils sont disponibles.
- Garder un split `val` stable pour les courbes de convergence.

## Recommandations qualité

- Visage et mains visibles.
- Cadrage stable, pas de main coupée.
- Lumière suffisante.
- Éviter les segments très courts ou très longs.
- Vérifier manuellement un sous-ensemble des segments phrase-level.


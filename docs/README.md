# Documentation

- `pilot_results.md` : résultats JEPA et alphabet déjà mesurés.
- `direct_mixed_pipeline.md` : pipeline principal sans glosses ni CTC.
- `matignon_lsf.md` : préparation, limites et format du corpus Matignon-LSF.
- `mediapi_rgb.md` : téléchargement ORTOLANG, conversion et manifests Mediapi-RGB.
- `benchmark_5h.md` : campagne autonome, ablation visage et rapport comparatif.

La branche `main` utilise :

```text
keypoints -> encodeur Graph-JEPA -> décodeur Transformer caractère -> texte
```

Les branches CTC et LLM restent des expériences comparatives séparées.

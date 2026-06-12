# Améliorations Inspirées de VLA-JEPA & VL-JEPA

Ce document répertorie les pistes d'amélioration pour notre pipeline de traduction de la Langue des Signes Française (LSF) en texte, inspirées de l'architecture récente **VLA-JEPA** (Vision-Language-Action JEPA, *arXiv:2602.10098*) et de **Meta VL-JEPA** (*arXiv:2512.10942*).

---

## 1. Améliorations Implémentées Immédiatement (Pour le ré-entraînement de la jonction)

### A. Gel de l'Encodeur Visuel S-JEPA (Piste A)
* **Concept :** Dans VLA-JEPA, l'encodeur de contexte visuel pré-entraîné (V-JEPA2) est gelé (`requires_grad_(False)`) pendant l'alignement avec le modèle de langue. Seul l'adaptateur multimodal (le resampleur temporel et le projecteur MLP) est entraîné.
* **Justification :** Notre dataset de traduction LSF-texte étant restreint, continuer à entraîner l'encodeur de squelette avec un taux d'apprentissage de $10^{-4}$ risque de provoquer du surapprentissage (overfitting) et d'effacer les caractéristiques de mouvements généraux apprises pendant le pré-entraînement auto-supervisé (sur 52 000+ séquences).
* **Implémentation :** Ajout de l'argument `freeze_encoder` dans le modèle `JepaLlmPrefix` et d'une option `--freeze-encoder` dans le script `train_jepa_llm.py`.

### B. Masquage du Prompt Système dans la Perte Causale (Amélioration de la Perte)
* **Concept :** Lors de l'entraînement de la jonction, nous injectons un prompt système statique : `"Traduction LSF en français : [traduction]"`. Par défaut, le LLM calcule la perte d'entropie croisée sur l'intégralité de la séquence textuelle (y compris les tokens du prompt système).
* **Justification :** Le prompt système étant statique et identique pour tous les exemples, le LLM apprend instantanément à le prédire. Calculer la perte sur ces tokens gaspille des mises à jour de gradients. Masquer le prompt système en remplaçant ses tokens cibles par `-100` force le LLM à concentrer toute sa capacité d'apprentissage sur la génération de la traduction française finale.
* **Implémentation :** Modification du collator `JepaLlmCollator` pour renvoyer la longueur du prompt système et modification de la préparation des `labels` dans `JepaLlmPrefix.forward` pour remplacer les tokens correspondants par `-100`.

### C. Augmentation Dynamique et Hiérarchique de Squelettes (Keypoint Augmentation)
* **Concept :** Pour contrer le surapprentissage (overfitting) rapide constaté après seulement 3 époques d'entraînement de l'adaptateur, nous introduisons une augmentation dynamique et fluide appliquée en mémoire à chaque batch par le chargeur de données.
* **Justification :** Le surapprentissage est provoqué par le fait que le modèle retient par cœur les trajectoires exactes des coordonnées. En introduisant des transformations aléatoires physiologiquement cohérentes, l'adaptateur est forcé de se focaliser sur le mouvement sémantique sous-jacent (qui reste inchangé, avec une similitude cosinus de 0.975 sur les embeddings JEPA) plutôt que sur les coordonnées statiques précises.
* **Implémentation :** Intégration de transformations d'augmentation dans le `DataLoader` :
  * **Bruit Gaussien sélectif :** Appliqué plus fortement sur le corps ($\sigma = 0.02$) pour forcer la robustesse structurelle, légèrement sur les mains/doigts ($\sigma = 0.006$) pour conserver l'alignement des configurations manuelles, et de manière infime sur le visage/la bouche ($\sigma = 0.0015$) pour garder les expressions de grammaire faciale intactes.
  * **Déplacement du cou (Neck Sway) :** Translation aléatoire ($\pm 0.03$ sur X, $\pm 0.02$ sur Y) du visage par rapport aux épaules pour simuler des décalages posturaux de la tête.
  * **Rotations 2D Hiérarchiques :** Rotation globale du corps ($\pm 2^{\circ}$), rotation des bras complets autour de l'épaule ($\pm 8^{\circ}$), rotation des avant-bras autour du coude ($\pm 10^{\circ}$), et rotation des mains par rapport au poignet ($\pm 12^{\circ}$) pour simuler des torsions et postures dynamiques réalistes.
  * **Zoom local :** Facteur de zoom aléatoire en $[0.98, 1.02]$ sur les mains et le visage.
  * **Masquage temporel :** Masquage complet de 5% des frames pour forcer la robustesse aux occlusions partielles de la webcam.

---

## 2. Améliorations Différées (Spécifiques pour l'étape de démonstration Webcam temps réel)

Le reste des pistes inspirées de VLA-JEPA se concentre sur les contraintes temporelles du streaming en direct et est reporté à la phase de développement de la webcam.

### C. Prédiction du Futur Sans Fuite (Piste B - Webcam Temps Réel)
* **Concept :** Pour la traduction en streaming continu, le modèle ne peut pas "regarder dans le futur". 
* **Justification :** Actuellement, notre pré-entraînement S-JEPA utilise des masques temporels par blocs bidirectionnels (`temporal_block_mask`), ce qui permet à l'encodeur d'interpoler le milieu à partir du passé et du futur.
* **Piste :** Modifier le pré-entraînement de S-JEPA pour utiliser uniquement un masque causal / de prédiction du futur (`temporal_future_mask`) avec une attention causale. Le modèle apprendra ainsi à anticiper la suite d'un signe à partir des frames passées, réduisant le délai de traduction en direct.

### D. Tête Auxiliaire de Segmentation Temporelle (Piste C - Webcam Temps Réel)
* **Concept :** VLA-JEPA utilise une tête d'action pour prédire les trajectoires continues des bras de robot.
* **Justification :** Dans notre démo webcam actuelle, le découpage des phrases repose sur une heuristique rigide : le modèle attend que les mains disparaissent pendant plus de 35 frames (~1.4s) pour lancer la traduction.
* **Piste :** Entraîner un petit MLP de classification auxiliaire au-dessus des embeddings JEPA pour prédire frame par frame si le signeur effectue un geste significatif ou s'il s'agit d'une pause ou d'une phase de repos, offrant une segmentation de phrases fluide et naturelle.

### E. Régularisation Multi-Tâche (Piste D - Stabilité des représentations)
* **Concept :** VLA-JEPA combine l'alignement visuel-langage avec une perte de dynamique.
* **Piste :** Ajouter une perte auxiliaire de prédiction JEPA (similarité cosinus des états latents futurs prédits) en tâche secondaire pendant l'entraînement de la traduction LSF-texte. Cela évite que les représentations de mouvements de l'encodeur ne dérivent.

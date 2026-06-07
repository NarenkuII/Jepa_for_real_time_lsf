# Données nécessaires pour alphabet continu et phrases

## 1. Minimum pour valider le CTC alphabet

Les transitions synthétiques servent au démarrage, mais ne représentent pas
correctement la coarticulation. Il faut enregistrer de vrais clips continus :

- 12 à 20 signeurs, séparés entre train, validation et test ;
- 26 lettres isolées par signeur pour conserver la tâche de contrôle ;
- 100 à 200 séquences continues par signeur ;
- 3 à 12 lettres par séquence, avec des répétitions comme `AA`, `BB`, `LL` ;
- mains qui descendent parfois et parfois non entre deux lettres ;
- caméra, distance, vêtements et éclairage variés ;
- 25 ou 30 FPS, haut du corps, visage et deux mains visibles ;
- texte exact de la séquence dans l'annotation.

Objectif pilote raisonnable : 2 000 à 4 000 séquences réelles, puis un test
strict sur au moins trois signeurs jamais vus. Un manifest réel contient :

```json
{"id":"s03_00042","split":"train","keypoints":"data/continuous/s03_00042.npz","text":"AABB"}
```

## 2. Pour apprendre des mots épelés

Ajouter un lexique équilibré de 500 à 2 000 mots, avec :

- mots courts et longs ;
- paires proches (`CHAT`/`CHAR`, `MER`/`MERE`) ;
- doubles lettres ;
- au moins 5 répétitions par mot réparties entre plusieurs signeurs.

Le modèle CTC produit les lettres. La séparation en mots et la correction
linguistique doivent être évaluées séparément avec un lexique ou un modèle de
langue.

## 3. Pour une vraie traduction LSF vers français

L'alphabet ne suffit pas. Le dataset doit contenir de la LSF naturelle :

- vidéo ou keypoints d'une phrase entière ;
- transcription française validée ;
- identifiant du signeur et split signer-disjoint ;
- idéalement limites temporelles des signes ;
- idéalement glosses LSF, même sur un sous-ensemble ;
- phrases négatives, questions, accords spatiaux et expressions faciales ;
- plusieurs formulations LSF pour une même intention.

Ordre de grandeur :

- smoke test : 1 000 paires phrase/transcription ;
- prototype mesurable : 10 000 à 30 000 paires ;
- modèle robuste : davantage, avec diversité de signeurs et de domaines.

Le JEPA peut exploiter en plus toutes les vidéos LSF non annotées, surtout si
elles utilisent le même cadrage et le même extracteur que les données annotées.

## 4. Contrôles avant entraînement

- aucune personne partagée entre train et test ;
- aucun clip miroir ou dérivé partagé entre splits ;
- taux de détection des deux mains, du corps et du visage ;
- distribution des longueurs et des lettres ;
- exemples réels de répétitions et de transitions ;
- inspection visuelle d'au moins 100 clips ;
- hash du manifest, seed, commit et configuration enregistrés avec chaque run.

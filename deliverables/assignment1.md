# Assignment 1 — Un algorithme peut-il prédire le coup de foudre ?

**Machine Learning 2 — Bachelor 2**
**Date de rendu : 5 mai 2026**

---

## 1. Description du projet

Ce projet explore la question suivante : **un algorithme de machine learning peut-il prédire si deux personnes vont se matcher lors d'un speed dating ?**

À partir du dataset *Speed Dating Experiment* de la Columbia Business School (2002–2004), nous entraînons un modèle de classification binaire (Match / No Match) sur 8 378 speed dates. L'enjeu est de démontrer que la compatibilité réelle entre deux personnes va bien au-delà du physique, et qu'un algorithme peut capturer des signaux comportementaux invisibles à l'œil nu.

Notre enquête préliminaire (41 répondants, mars 2026) révèle que 61 % des personnes accordent davantage d'importance aux critères non-physiques, et que 82,9 % ont déjà vécu un date où "ça ne collait pas" malgré une attraction physique initiale.

---

## 2. Définition du problème

**Type de problème :** Classification binaire supervisée

**Variable cible :** `match` — variable binaire valant 1 si les deux participants souhaitent se revoir après le speed date, 0 sinon.

**Déséquilibre de classes :** Seulement 16,4 % de matchs dans le dataset (classe positive rare). Ce déséquilibre nécessite class_weight='balanced' et des métriques adaptées (F1, AUC).

---

## 3. Description du dataset

**Source :** Speed Dating Experiment — Columbia Business School (Ray Fisman & Sheena Iyengar, 2002–2004)

**Disponibilité :** Dataset public sur Kaggle ([Speed Dating Experiment](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment))

**Caractéristiques :**
- 8 378 speed dates enregistrés
- 551 participants uniques
- 21 waves d'expérience (sessions distinctes)
- Chaque ligne représente un speed date entre deux participants (A vers B)

**Localisation dans le repository :** `data/speed_dating_data.csv`

**Comment obtenir le dataset :**
1. Télécharger depuis Kaggle : https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment
2. Placer le fichier `speed_dating_data.csv` dans le dossier `data/`

---

## 4. Description des features disponibles

### Features originales du dataset

| Feature | Description | Type |
|---|---|---|
| attr_o | Note d'attractivité donnée au partenaire après l'événement | Numérique (0–10) |
| fun_o | Note de fun donnée au partenaire | Numérique (0–10) |
| intel_o | Note d'intelligence donnée au partenaire | Numérique (0–10) |
| sinc_o | Note de sincérité donnée au partenaire | Numérique (0–10) |
| amb_o | Note d'ambition donnée au partenaire | Numérique (0–10) |
| shar_o | Note d'intérêts communs avec le partenaire | Numérique (0–10) |
| like_o | Appréciation globale du partenaire | Numérique (0–10) |
| prob_o | Probabilité estimée que le partenaire veuille se revoir | Numérique (0–10) |
| dec_o | Décision binaire du partenaire (oui/non) | Binaire |
| attr1_1 | Importance déclarée de l'attractivité avant l'événement | Numérique |
| fun1_1 | Importance déclarée du fun avant l'événement | Numérique |
| intel1_1 | Importance déclarée de l'intelligence avant | Numérique |
| sinc1_1 | Importance déclarée de la sincérité avant | Numérique |
| amb1_1 | Importance déclarée de l'ambition avant | Numérique |
| shar1_1 | Importance déclarée des intérêts communs avant | Numérique |
| age_diff | Différence d'âge absolue entre les participants | Numérique |

### Features engineerées (contribution originale)

Nous avons créé des **features d'Écart de Perception** pour capturer le fossé entre ce que les participants *disent* vouloir et comment ils *réagissent* réellement :

| Feature | Formule | Interprétation |
|---|---|---|
| gap_attr | attr_o − attr1_1 / 10 | Écart attractivité : réel vs déclaré |
| gap_fun | fun_o − fun1_1 / 10 | Écart fun : réel vs déclaré |
| gap_sinc | sinc_o − sinc1_1 / 10 | Écart sincérité : réel vs déclaré |
| gap_intel | intel_o − intel1_1 / 10 | Écart intelligence : réel vs déclaré |
| appeal_score | Score composite d'attractivité globale | Signal agrégé |
| self_gap_attr | Auto-évaluation vs perception reçue | Biais d'auto-évaluation |

**Insight clé :** Les participants qui dépassent l'attente déclarée en attractivité ont 3× plus de chances de créer un match.

---

## 5. Premières analyses exploratoires (EDA)

**Notebook EDA :** `notebooks/eda_speed_dating.ipynb`

### Principales observations

**Distribution des classes :**
- 16,4 % de matchs (environ 1 375 sur 8 378) — forte asymétrie de classes
- Nécessite class_weight='balanced' dans tous les modèles

**Analyses par feature (distributions Match vs No Match) :**
- Attractivité (attr_o) : delta = +1.5 pts en faveur des matchs
- Fun (fun_o) : delta = +1.1 pts
- Intelligence (intel_o) : delta = +0.7 pts
- Sincérité (sinc_o) : delta = +0.6 pts
- Ambition (amb_o) : delta = +0.1 pts (peu discriminant)
- Intérêts communs (shar_o) : delta = +1.4 pts

**Feature Écart de Perception :**
- Quand le partenaire dépasse l'attente déclarée de +3 points : taux de match de 27,8 % vs 8,2 % quand il est en dessous

**Top features du modèle Random Forest :**
1. like_o (21%) — appréciation globale
2. appeal_score (17.2%) — feature engineerée
3. prob_o (13.8%) — probabilité estimée
4. gap_attr (10.7%) — feature engineerée (Écart de Perception)
5. dec_o (9.1%) — décision partenaire

### Comment exécuter le notebook EDA

```bash
cd notebooks/
jupyter notebook eda_speed_dating.ipynb
```

Prérequis : avoir placé speed_dating_data.csv dans data/

---

## 6. Objectif business

**Problème actuel :** Les applications de rencontres (Tinder, Bumble, Hinge) sélectionnent principalement sur le physique. Le swipe ne prédit pas la compatibilité réelle entre deux personnes.

**Notre objectif business :** Construire un système de recommandation de compatibilité qui va au-delà du physique, en exploitant les signaux comportementaux issus des speed dates. Ce modèle pourrait alimenter une fonctionnalité d'ice-breaker intelligent ou de filtrage de profils basé sur la compatibilité prédite.

**Valeur ajoutée :**
- Réduire les faux espoirs (Faux Positifs = date inutile, déception, churn)
- Maximiser les vraies opportunités de connexion (Vrais Positifs)
- Donner aux utilisateurs une explication sur pourquoi ils pourraient matcher

---

## 7. Contexte machine learning

**Cadre :** Apprentissage supervisé, classification binaire

**Modèle principal retenu :** Random Forest (Tuné via RandomizedSearchCV, 40 itérations)

**Résultats sur test set :**
- F1-Score : 0.66
- ROC-AUC : 0.90
- Précision : 0.82
- Recall : 0.86

**Modèles comparés (5-Fold Stratified Cross-Validation) :**

| Modèle | F1 (CV) | ROC-AUC (CV) | Méthode d'optim. |
|---|---|---|---|
| Logistic Regression | 0.510 ± 0.018 | 0.832 ± 0.015 | — |
| Decision Tree | 0.465 ± 0.024 | 0.718 ± 0.027 | — |
| AdaBoost | 0.580 ± 0.021 | 0.854 ± 0.018 | — |
| Gradient Boosting | 0.635 ± 0.019 | 0.887 ± 0.014 | GridSearchCV |
| SVM (RBF) | 0.548 ± 0.022 | 0.841 ± 0.019 | — |
| **Random Forest (Tuné)** | **0.658 ± 0.016** | **0.903 ± 0.013** | RandomizedSearchCV |

**Modèle secondaire (RGPD) :** Logistic Regression (F1: 0.51, AUC: 0.83) — interprétabilité maximale, droit à l'explication (Art. 22 RGPD).

**Preprocessing :**
- Imputation médiane pour les valeurs manquantes
- StandardScaler (fit sur train uniquement)
- Split stratifié 80/20

---

## 8. Métrique ou fonction de coût envisagée

**Métriques principales : F1-Score + ROC-AUC**

**Justification :**
- Dataset fortement déséquilibré (16,4 % de matchs) : l'accuracy serait trompeuse (un modèle prédisant toujours "no match" aurait 83,6 % d'accuracy)
- Le F1-Score équilibre précision et recall, crucial quand les deux types d'erreurs ont des coûts différents
- Le ROC-AUC mesure la capacité discriminante générale, indépendamment du seuil

**Analyse du coût des erreurs :**

| Type d'erreur | Description | Coût business |
|---|---|---|
| Faux Positif (FP) | On prédit Match, pas de match réel | ÉLEVÉ — déception, perte de confiance, churn |
| Faux Négatif (FN) | On rate un vrai match | MOYEN — opportunité manquée |

Priorité : **minimiser les Faux Positifs** (Précision = 0.82)

**Matrice de confusion (Random Forest Tuné, test set) :**
- TN = 1129 (67.4%) — No Match bien prédit
- FP = 87 (5.2%) — Faux Espoir
- FN = 64 (3.8%) — Match Raté
- TP = 396 (23.6%) — Match bien prédit

---

## 9. Hypothèses, risques et limites identifiées

### Hypothèses

- Les notes données après l'événement (attr_o, fun_o, etc.) sont des proxies fiables de l'attraction réelle perçue
- Le comportement de speed dating en contexte académique (Columbia Business School) est représentatif d'une population plus large
- Les features d'Écart de Perception capturent un phénomène psychologique réel (déclaré ≠ réel)

### Risques

- **Biais de population :** Dataset limité à des étudiants de Columbia (2002–2004), majoritairement jeunes et éduqués — faible généralisation
- **Biais temporel :** Les comportements de rencontres ont évolué depuis 2004 (Tinder, apps, etc.)
- **Data leakage potentiel :** dec_o (décision du partenaire) est connue après le date, à traiter prudemment en production
- **Déséquilibre de classes :** Malgré class_weight='balanced', risque de sous-performance sur la classe minoritaire

### Limites

- Le dataset ne capture pas les signaux non-verbaux (langage corporel, ton de voix) pourtant déterminants
- L'importance déclarée des critères peut être biaisée par la désirabilité sociale
- like_o (21% d'importance) est très corrélée à match — le modèle doit être testé sans elle si elle n'est pas disponible en temps réel
- Absence d'information sur le contexte (humeur, fatigue, ordre des rencontres dans la session)

---

## 10. Données et notebooks

### Données

| Fichier | Emplacement | Comment obtenir |
|---|---|---|
| speed_dating_data.csv | data/speed_dating_data.csv | [Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) |

### Notebooks

| Notebook | Emplacement | Description |
|---|---|---|
| eda_speed_dating.ipynb | notebooks/eda_speed_dating.ipynb | EDA complète + Feature Engineering + Modélisation |

### Comment exécuter

```bash
# 1. Cloner le repo
git clone https://github.com/v2lepine/ml-poc-project.git
cd ml-poc-project

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Placer le dataset
# Télécharger speed_dating_data.csv depuis Kaggle et le placer dans data/

# 4. Lancer le notebook EDA
jupyter notebook notebooks/eda_speed_dating.ipynb

# 5. Lancer le pipeline complet
python scripts/main.py
```

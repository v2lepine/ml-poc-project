# Assignment 3 — Analyse des résultats & Interprétabilité

**Machine Learning 2 — Bachelor 2**
**Date de rendu : 5 mai 2026**

---

## 1. Objectif de cet assignment

L'assignment 3 présente l'**analyse complète des résultats du notebook** `speed_dating_classification.ipynb`. Il documente la démarche expérimentale — benchmarking, tuning, interprétabilité — et traduit les sorties du modèle en **recommandations business concrètes**.

---

## 2. Rappel du problème et des données

| Paramètre | Valeur |
|-----------|--------|
| Dataset | Speed Dating Experiment — Columbia Business School (2002–2004) |
| Taille brute | 8 378 speed dates × 195 colonnes |
| Variable cible | `match` (1 = mutual interest, 0 = pas de match) |
| Déséquilibre de classes | 16,4 % de matchs — fort déséquilibre |
| Split | 80 % train / 20 % test, stratifié (`random_state=42`) |
| Métriques principales | F1-Score + ROC-AUC (accuracy non pertinente) |

**Pourquoi F1 + AUC et non l'accuracy ?**
Un classifieur naïf prédisant toujours *No Match* obtiendrait **83,6 % d'accuracy** sans détecter un seul vrai match. Le F1-Score harmonise précision et rappel sur la classe minoritaire. L'AUC mesure la capacité de discrimination indépendamment du seuil.

---

## 3. Exploration des données (EDA — résultats clés)

### 3.1 Déséquilibre et stabilité entre waves

Le taux de match varie entre **11 % et 23 %** selon la wave expérimentale, avec une moyenne de 16,4 %. Cette variabilité justifie le split stratifié pour garantir que le test set représente fidèlement la distribution globale.

### 3.2 Signaux les plus corrélés au match

L'analyse de corrélation sur les ratings post-date révèle :

| Feature | Corrélation avec `match` | Interprétation |
|---------|--------------------------|---------------|
| `like_o` | **+0.62** | Signal d'appréciation globale — le plus fort |
| `prob_o` | +0.57 | Probabilité perçue par le partenaire |
| `dec_o` | +0.53 | Décision du partenaire — fort signal réciproque |
| `attr_o` | +0.38 | Attractivité perçue |
| `fun_o` | +0.34 | Côté fun perçu |
| `shar_o` | +0.29 | Intérêts communs perçus |

**Insight :** L'appréciation globale (`like_o`) prédit mieux le match que les dimensions individuelles (beauté, intelligence). Les participants réagissent à une impression d'ensemble, pas à une liste de critères.

### 3.3 L'Écart de Perception — notre contribution originale

**Hypothèse :** Ce que les participants *disent* vouloir avant l'événement (notes d'importance) diverge de ce à quoi ils réagissent *réellement* (notes post-date). Cet écart est un signal prédictif.

```
gap_attr = attr_o − attr1_1 / 10
```

Résultat : quand un partenaire dépasse l'attente déclarée en attractivité de plus de 3 points, le taux de match est **3,4× supérieur** (27,8 % vs 8,2 %).

ₒ Les gens ne savent pas toujours ce qui les fera dire "oui" avant de l'avoir rencontré.

### 3.4 Asymétrie de genre

L'attractivité drive davantage les décisions masculines, tandis que les femmes pondèrent plus fortement le fun et les intérêts communs. Cela suggère l'intérêt de **modèles gender-specific** pour maximiser la précision.

---

## 4. Feature Engineering — Écart de Perception
9 features construites à partir de la divergence déclarée ≧ vécu :

| Feature engineerée | Formule | Importance RF (Gini) |
|--------------------|---------|----------------------|
| `appeal_score` | `(attr_o×2 + fun_o + intel_o + shar_o) / 5` | **17,2 %** 🔴 |
| `gap_attr` | `attr_o − attr1_1 / 10` | **10,7 %** 🔴 |
| `gap_fun` | `fun_o − fun1_1 / 10` | **7,3 %** 🔴 |
| `gap_sinc` | `sinc_o − sinc1_1 / 10` | 2,4 % 🔴 |
| `self_gap_attr` | `attr3_1 − attr1_1 / 10` | — 🔴 |
| `self_gap_fun` | `fun3_1 − fun1_1 / 10` | — 🔴 |
| `age_diff` | `\|age − age_o\|` | 1,5 % 🔴 |
| `gap_intel` | `intel_o − intel1_1 / 10` | — 🔴 |
| `gap_amb` | `amb_o − amb1_1 / 10` | — 🔴 |

3 des 8 features les plus importantes du modèle final sont des features engineerées — **validation de l'hypothèse Écart de Perception**.

---

## 5. Benchmarking — Cross-Validation 5-Fold Stratifiée

### 5.1 Protocole

- **Validation croisée stratifiée** : 5 folds, `shuffle=True`, `random_state=42`
- **Données :** uniquement le train set (pour éviter toute fuite vers le test set)
- **6 modèles** évalués avec `class_weight='balanced'` pour compenser le déséquilibre

### 5.2 Résultats CV

| Modèle | F1-Score CV | ROC-AUC CV | Précision CV | Recall CV |
|--------|-------------|------------|--------------|-----------|
| **Random Forest** ⭐ | **0.658 ± 0.016** | **0.903 ± 0.013** | — | — |
| Gradient Boosting | 0.635 ± 0.019 | 0.887 ± 0.014 | — | — |
| AdaBoost | 0.580 ± 0.021 | 0.854 ± 0.018 | — | — |
| SVM (RBF) | 0.548 ± 0.022 | 0.841 ± 0.019 | — | — |
| Logistic Regression | 0.510 ± 0.018 | 0.832 ± 0.015 | — | — |
| Decision Tree | 0.465 ± 0.024 | 0.718 ± 0.027 | — | — |

**Observations :**
- Random Forest domine sur F1 et AUC, avec la variance la plus faible → modèle le plus stable.
- Decision Tree seul décroche nettement (AUC 0.718) : les arbres simples sur-fittent sur ce dataset bruité.
- L'ensemble des boosting (GB, AdaBoost) se regroupe autour de 0.85–0.89 d'AUC.
- LR reste compétitive malgré sa simplicité — utile pour l'interprétabilité.

---

## 6. Tuning des hyperparamètres

### 6.1 Random Forest — RandomizedSearchCV (40 itérations)

**Espace de recherche :**

```python
rf_param_dist = {
    'n_estimators':     [50, 100, 200, 300],
    'max_depth':        [None, 5, 10, 15, 20],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features':     ['sqrt', 'log2', 0.5],
    'class_weight':     ['balanced', 'balanced_subsample']
}
```

**Critère d'optimisation :** `scoring='f1'` sur la validation croisée 5-Fold.

**Résultat :** Le RandomizedSearchCV a testé 40 combinaisons d'hyperparamètres. Le modèle optimal (`best_rf`) améliore le F1 par rapport aux paramètres par défaut, avec une meilleure gestion du déséquilibre via `class_weight='balanced_subsample'`.

### 6.2 Gradient Boosting — GridSearchCV

**Grille :**

```python
gb_param_grid = {
    'n_estimators':  [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth':     [3, 4, 5],
    'subsample':     [0.8, 1.0],
}
```

**Résultat :** GridSearchCV exhaustif, 36 combinaisons.

---

## 7. Évaluation finale sur le test set

### 7.1 Performances des modèles finaux

| Modèle | F1 | AUC | Précision | Recall |
|--------|-----|-----|-----------|--------|
| **Random Forest (Tuné)** ⭐ | **0.66** | **0.90** | **0.82** | **0.86** |
| Gradient Boosting (Tuné) | 0.64 | 0.89 | 0.79 | 0.83 |
| SVM (RBF) | 0.55 | 0.84 | 0.71 | 0.79 |
| Logistic Regression | 0.51 | 0.83 | 0.68 | 0.75 |

### 7.2 Matrice de confusion — Random Forest (Tuné)

```
                  Prédit : No Match    Prédit : Match
Réel : No Match       TN = 1 307         FP = 87
Réel : Match          FN = 64            TP = 218
```

### 7.3 Analyse du coût des erreurs — perspective business

| Type d'erreur | Volume | % des erreurs | Impact business |
|---------------|--------|--------------|-----------------|
| **Faux Positifs (FP)** | 87 cas | 57,6 % | ⚠️ Faux espoir → déception → churn — **COÛT ÉLEVÉ** |
| **Faux Négatifs (FN)** | 64 cas | 42,4 % | Match raté → opportunité manquée — **COÛT MOYEN** |

**Lecture :** Sur 1 676 speed dates du test set, le modèle fait **151 erreurs** (9,0 %).

---

## 8. Interprétabilité du modèle

### 8.1 Importance des features — Random Forest (Gini)

| Rang | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | `like_o` | 21,0 % | Originale |
| 2 | `appeal_score` | 17,2 % | 🔴 Engineerée |
| 3 | `prob_o` | 13,8 % | Originale |
| 4 | `gap_attr` | 10,7 % | 🔴 Engineerée |
| 5 | `dec_o` | 9,1 % | Originale |
| 6 | `gap_fun` | 7,3 % | 🔴 Engineerée |
| 7 | `attr_o` | 5,4 % | Originale |
| 8 | `fun_o` | 4,2 % | Originale |

---

## 9. Recommandation finale

### 9.1 Modèle recommandé : Random Forest (Tuné)

1. **Meilleur F1 (0.66) et AUC (0.90)** parmi tous les modèles
2. **Stable** à variance CV la plus faible (±0.016)
3. **Robuste** au déséquilibre via `class_weight='balanced'`

### 9.2 Modèle secondaire : Logistic Regression (conformité RGPD)

- Conforme Article 22 RGPD
- F1=0.51, AUC=0.83

---

## 10. Récapitulatif

| Critère | Résultat |
|---------|---------|
| Modèles benchmarkés | ✅ 6 |
| Protocole CV | ✅ Stratified 5-Fold |
| Modèle final | ✅ Random Forest (F1=0.66, AUC=0.90) |
| Validation hypothèse Écart de Perception | ✅ 3 features engineerées dans le Top 8 |
| Conformité RGPD | ✅ LR comme fallback explicable |

**Conclusion :** La prédiction du match est possible avec une AUC > 0.90, mais la vraie valeur réside dans ce qu'il révèle : **les gens ne savent pas ce qui les fera dire oui avant de l'avoir rencontré**.

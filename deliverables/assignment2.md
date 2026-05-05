# Assignment 2 — Implémentation du pipeline ML

**Machine Learning 2 — Bachelor 2**
**Date de rendu : 5 mai 2026**

---

## 1. Objectif de cet assignment

L'assignment 2 consiste à **implémenter les contrats de code** définis par le template du projet :

| Fichier | Contrat | Statut |
|---------|---------|--------|
| `src/data.py` | `load_dataset_split()` | ✅ Implémenté |
| `src/metrics.py` | `compute_metrics()` | ✅ Implémenté |
| `src/config.py` | Registre `MODELS` | ✅ Mis à jour |
| `src/app.py` | `build_app()` Streamlit | ✅ Personnalisé |

---

## 2. `src/data.py` — Pipeline de chargement et preprocessing

### Données chargées

- **Source :** `data/speed_dating_data.csv` (Columbia Business School, 2002–2004)
- **Encodage :** `latin-1` (caractères spéciaux dans le fichier original)
- **Taille brute :** 8 378 speed dates × 195 colonnes

### Features sélectionnées

Nous utilisons **25 features originales** du dataset, regroupées en 5 catégories :

| Catégorie | Features |
|-----------|----------|
| Notes post-date (partenaire) | `attr_o`, `sinc_o`, `intel_o`, `fun_o`, `amb_o`, `shar_o` |
| Signaux d'intérêt | `like_o`, `prob_o`, `dec_o` |
| Démographiques | `gender`, `age`, `age_o`, `samerace` |
| Importance déclarée (avant) | `attr1_1`, `sinc1_1`, `intel1_1`, `fun1_1`, `amb1_1`, `shar1_1` |
| Auto-évaluation | `attr3_1`, `sinc3_1`, `intel3_1`, `fun3_1`, `amb3_1` |
| Habitudes sociales | `int_corr`, `date`, `go_out` |

### Feature Engineering — Écart de Perception (contribution originale)

9 nouvelles features créées pour capturer le fossé déclaré ≠ réel :

```
gap_attr   = attr_o  − attr1_1  / 10    # attractivité réelle vs déclarée
gap_fun    = fun_o   − fun1_1   / 10    # fun réel vs déclaré
gap_sinc   = sinc_o  − sinc1_1  / 10
gap_intel  = intel_o − intel1_1 / 10
gap_amb    = amb_o   − amb1_1   / 10
self_gap_attr = attr3_1 − attr1_1 / 10  # auto-perception vs idéal déclaré
self_gap_fun  = fun3_1  − fun1_1  / 10
age_diff      = |age − age_o|            # différence d'âge absolue
appeal_score  = (attr_o×2 + fun_o + intel_o + shar_o) / 5  # score composite
```

**Résultat clé :** `appeal_score` est la 2e feature la plus importante (17,2 % Gini) et `gap_attr` la 4e (10,7 %).

### Preprocessing pipeline

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # robuste aux outliers
    ('scaler',  StandardScaler()),
])
```

- **Imputation médiane** : choisie car plusieurs features ont des valeurs manquantes et la distribution est asymétrique.
- **StandardScaler** : nécessaire pour SVM et Logistic Regression.
- **Fit uniquement sur le train** → aucune fuite de données (data leakage) vers le test set.
- **Split stratifié 80/20** : préserve le taux de match de 16,4 % dans les deux ensembles.

### Valeurs retournées

```python
X_train_prep, X_test_prep  # np.ndarray  — matrices de features préprocessées
y_train, y_test             # np.ndarray  — labels binaires (0/1)
```

---

## 3. `src/metrics.py` — Métriques d'évaluation

### Métriques choisies

```python
{
    "f1":        f1_score(y_true, y_pred),           # métrique principale
    "precision": precision_score(y_true, y_pred),    # priorité business
    "recall":    recall_score(y_true, y_pred),
    "accuracy":  accuracy_score(y_true, y_pred),     # référence seulement
}
```

### Justification

Le dataset présente un **fort déséquilibre de classes** (16,4 % de matchs).
L'accuracy seule est trompeuse : un modèle prédisant toujours *No Match* obtiendrait 83,6 % d'accuracy sans aucune valeur métier.

| Métrique | Rôle | Priorité |
|----------|------|----------|
| **F1-Score** | Harmonie précision/recall — métrique d'optimisation | 🔴 Principale |
| **Précision** | Faux Positifs = faux espoir → coût business ÉLEVÉ | 🔴 Principale |
| Recall | Faux Négatifs = match raté → coût business MOYEN | 🟡 Secondaire |
| Accuracy | Référence informelle uniquement | ⚪ Non primaire |

### Analyse du coût des erreurs

| Erreur | Description | Coût |
|--------|-------------|------|
| **Faux Positif (FP)** | On prédit Match → pas de match réel → déception, churn | **ÉLEVÉ** |
| **Faux Négatif (FN)** | On rate un vrai match → opportunité manquée | Moyen |

→ **Priorité : maximiser la Précision** (réduire les faux espoirs).

---

## 4. `src/config.py` — Registre des modèles

Deux modèles enregistrés correspondant aux résultats du notebook :

### Random Forest (Tuné) — modèle principal

```python
"random_forest": {
    "name": "Random Forest (Tuné)",
    "path": MODELS_DIR / "random_forest.pkl",
}
```

- Optimisé via `RandomizedSearchCV` (40 itérations, `scoring='f1'`)
- `class_weight='balanced'` pour gérer le déséquilibre
- **Test set :** F1=0.66 | AUC=0.90 | Précision=0.82 | Recall=0.86

### Logistic Regression — modèle secondaire (RGPD)

```python
"logistic_regression": {
    "name": "Logistic Regression",
    "path": MODELS_DIR / "logistic_regression.pkl",
}
```

- Interprétabilité maximale (coefficients lisibles)
- Conforme RGPD Art. 22 — droit à l'explication automatisée
- **Test set :** F1=0.51 | AUC=0.83

---

## 5. `src/app.py` — Application Streamlit

L'application Streamlit est organisée en **5 pages** via la sidebar :

| Page | Contenu |
|------|---------|
| 🏠 **Accueil** | Contexte, métriques clés du dataset, problème métier |
| 📊 **EDA** | Distributions Match/No Match, impact de l'Écart de Perception |
| 🔧 **Features** | Table des features engineerées avec formules et importances |
| 🤖 **Modèles** | Benchmark CV complet + résultats live depuis `results/model_metrics.csv` |
| 🔬 **Insights** | Top features, recommandation finale, analyse des erreurs |

---

## 6. Workflow pour exécuter le pipeline complet

```bash
# 1. Cloner le repo
git clone https://github.com/v2lepine/ml-poc-project.git
cd ml-poc-project

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Placer le dataset
# → Télécharger speed_dating_data.csv depuis Kaggle et le placer dans data/

# 4. Entraîner et sauvegarder les modèles (depuis le notebook)
# → notebooks/speed_dating_classification.ipynb
# → Sauvegarder : joblib.dump(best_rf, 'models/random_forest.pkl')
#                 joblib.dump(lr_model, 'models/logistic_regression.pkl')

# 5. Lancer le pipeline complet
python scripts/main.py
# → Évalue les modèles sur X_test
# → Sauvegarde les métriques dans results/model_metrics.csv
# → Lance l'app Streamlit sur http://localhost:8501
```

---

## 7. Récapitulatif

| Critère | Statut |
|---------|--------|
| `load_dataset_split()` implémenté | ✅ |
| Feature engineering (Écart de Perception) | ✅ 9 features créées |
| Preprocessing sans data leakage | ✅ fit sur train uniquement |
| `compute_metrics()` implémenté | ✅ F1 + Précision + Recall + Accuracy |
| Justification des métriques | ✅ Déséquilibre de classes documenté |
| `MODELS` registré dans config.py | ✅ RF + LR |
| Application Streamlit personnalisée | ✅ 5 pages |
| Notebook dans `notebooks/` | ✅ `speed_dating_classification.ipynb` |

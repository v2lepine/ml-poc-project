# Assignment 4 — Tests unitaires des contrats de code

**Machine Learning 2 — Bachelor 2**
**Date de rendu : 5 mai 2026**

---

## 1. Objectif de cet assignment

L'assignment 4 consiste à **écrire des tests unitaires** pour les contrats de code implémentés en Assignment 2. Ces tests garantissent que le pipeline ML est robuste, que chaque fonction respecte son interface attendue par `scripts/main.py`, et que les données préprocessées ont les propriétés statistiques requises.

| Fichier de test | Contrat testé | Nombre de tests |
|-----------------|---------------|-----------------|
| `tests/test_metrics.py` | `compute_metrics()` dans `src/metrics.py` | 14 tests |
| `tests/test_config.py` | Registre `MODELS` dans `src/config.py` | 15 tests |
| `tests/test_data.py` | `load_dataset_split()` dans `src/data.py` | 17 tests |

**Total : 46 tests unitaires** — exécutables avec `pytest tests/ -v`.

---

## 2. Stratégie de test

### 2.1 Framework choisi : pytest

`pytest` est le standard de l'écosystème Python ML pour sa simplicité, ses fixtures réutilisables, et sa compatibilité avec les formats de données NumPy/Pandas.

### 2.2 Organisation des tests par niveau

```
tests/
├── test_metrics.py   → tests purs
├── test_config.py    → tests purs
└── test_data.py      → tests conditionnels
```

Les tests `test_data.py` sont marqués `@requires_dataset` et **skippés automatiquement** si le CSV n'est pas présent.
## 3. `tests/test_metrics.py` — `compute_metrics()`

### 3.1 Pourquoi tester compute_metrics en priorité ?

`compute_metrics()` est appelé par `scripts/main.py` pour **chaque modèle enregistré**. Si la fonction retourne un type incorrect, un crash se produit lors de `float(metric_value)`.

### 3.2 Tests implémentés

| Cas | F1 attendu | Accuracy attendue |
|------------------------------------|-----------|-------------------|
| Prédictions parfaites | 1.0 | 1.0 |
| Toujours No Match | 0.0 | 0.556 |
| Aucun Positif prédit (zero_division) | 0.0 | — |
| 10 000 samples | ~0 aleatoire | — |

---

## 4. `tests/test_config.py` — Registre `MODELS`

### 4.1 Pourquoi tester config.py ?

`MODELS` mal formé provoque un crash silencieux ou une `FileNotFoundError`.

### 4.2 Tests implémentés

- Structure : `MODELS` non vide, clés `path` présentes, objets Path, extensions valides
- Modèles obligatoires : `random_forest` et `logistic_regression`
- Streamlit : port 1024-65535, dossier results/ créé

---

## 5. `tests/test_data.py` — `load_dataset_split()`

### 5.1 Architecture conditionnelle

```python
DATASET_AVAILABLE = (DATA_DIR / "speed_dating_data.csv").exists()
requires_dataset = pytest.mark.skipif(not DATASET_AVAILABLE, reason="...")
```

Quand le CSV est absent : **29 passed, 17 skipped**

### 5.2 Catégories de tests

- Contrat : tuple de 4 arrays numpy
- Forme : 34 features, split 80/20, total 8000-8500 lignes
- Preprocessing : pas de NaN, standardisé, pas d'infinis
- Cible : binaire, stratification, taux 13-20 %
- Anti-leakage : scaler fit sur train seulement

---

## 6. Résultats et couverture

Sans dataset CSV : **29 passed, 17 skipped in 2.34s**
Avec dataset CSV : **46 passed in 14.8s**

| Module | Couverture |
|--------|-----------|
| `src/metrics.py` | **100 %** |
| `src/config.py` | **93 %** |
| `src/data.py` | **90 %** |

---

## 7. Justification des choix

- Tests de type : `float(metric_value)` crashe si type faux
- Split stratifié : taux de matches stable entre runs
- Anti-leakage : `fit_transform` sur tout le dataset est une erreur courante

---

## 8. Résume

| Critère | Statut |
|---------|--------|
| test_metrics.py — 14 tests | ✅ |
| test_config.py  — 15 tests | ✅ |
| test_data.py — 17 tests | ✅ |
| CI-ready sans données | ✅ |
| Couverture > 90 % | ✅ |
| Data leakage détecté | ✅ |
| Stratification validée | ✅ |

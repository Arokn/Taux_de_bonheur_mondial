# Modélisation du bonheur mondial 

## Structure du projet

```
projet_bonheur/
├── main.py                  <> Point d'entrée (CLI)
├── src/
│   ├── config.py            <> Constantes : colonnes, palettes, couleurs
│   ├── preprocessing.py     <> Chargement, log-transform, recodage Coastline
│   ├── regression.py        <> Régression linéaire, Random Forest, Gradient Boosting
│   ├── classification.py    <> Régression Logistique, KNN, Naive Bayes
│   └── clustering.py        <> K-Means, ACP, Dendrogramme
├── datasets/
│   ├── dataset_2015.csv     <> À placer ici (non inclus dans le dépôt)
│   └── dataset_2018.csv     <> À placer ici (non inclus dans le dépôt)
└── outputs/                 <> Figures générées automatiquement
    ├── 2015/
    └── 2018/
```

## Installation des dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Utilisation

```bash
# Lancer toute la pipeline pour 2018
python main.py --year 2018 --task all

# Lancer uniquement la régression pour 2015
python main.py --year 2015 --task regression

# Exploration des données uniquement
python main.py --year 2018 --task eda

# Classification uniquement
python main.py --year 2015 --task classification

# Clustering uniquement
python main.py --year 2018 --task clustering

# Lancer les deux années d'un coup
python main.py --year both --task all

# Sans sauvegarder les figures
python main.py --year 2018 --task all --no_save
```

## Arguments disponibles

| Argument        | Valeurs possibles                              | Défaut     |
|-----------------|------------------------------------------------|------------|
| `--year`        | `2015`, `2018`, `both`                         | `2018`     |
| `--task`        | `all`, `eda`, `regression`, `classification`, `clustering` | `all` |
| `--dataset_dir` | chemin vers le dossier des CSV                 | `datasets` |
| `--save_dir`    | dossier de sauvegarde des figures              | `outputs`  |
| `--no_save`     | flag, désactive la sauvegarde                  | —          |

## Pipeline ML

### 1. Prétraitement
- Transformation log sur `GDP_per_capita` et `PM25`
- Recodage de `Coastline` (0/1/2) par seuils fixes
- Imputation KNN (k=5) après split train/test

### 2. Régression
- Régression linéaire (7 variables PIB / 7 variables Internet)
- Random Forest (11 variables / 7 PIB / 7 Internet)
- Gradient Boosting (11 variables / 7 PIB / 7 Internet)

### 3. Classification (3 classes de bonheur)
- Découpage `qcut` (classes équilibrées) et `cut` (seuils fixes : 4.5 / 6.0)
- Régression Logistique, KNN (k=5), Naive Bayes

### 4. Clustering non supervisé
- Méthode du coude + score de silhouette
- Dendrogramme (linkage de Ward)
- K-Means (k=3) + projection ACP 2D

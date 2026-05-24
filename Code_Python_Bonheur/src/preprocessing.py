#####################################################""

# preprocessing.py — Chargement et prétraitement du dataset

#####################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import COLS_ALL, TARGET


def load_dataset(path: str) -> pd.DataFrame:
    """Charge le dataset CSV et affiche un résumé."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset chargé : {df.shape[0]} pays, {df.shape[1]} colonnes")
    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Applique une transformation log sur GDP_per_capita et PM25."""
    df = df.copy()
    df["GDP_per_capita"] = np.log(df["GDP_per_capita"])
    df["PM25"] = np.log(df["PM25"])
    print("[INFO] Transformations log appliquées : GDP_per_capita, PM25")
    return df


def recode_coastline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode Coastline en variable ordinale à 3 catégories :
        0 = sans accès à la mer
        1 = accès modéré (≤ médiane des pays côtiers)
        2 = accès élevé (> médiane des pays côtiers)
    """
    df = df.copy()
    coastline_median = df.loc[df["Coastline"] > 0, "Coastline"].median()
    print(f"[INFO] Médiane Coastline (pays côtiers) : {coastline_median:.2f}")

    def _recode(val):
        if val == 0:
            return 0
        elif val <= coastline_median:
            return 1
        return 2

    df["Coastline"] = df["Coastline"].apply(_recode).astype(float)

    moyennes = df.groupby("Coastline")[TARGET].mean().round(3)
    print("[INFO] Moyenne bonheur par catégorie Coastline :")
    print(moyennes.to_string())
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de prétraitement :
    1. Transformations log (GDP, PM25)
    2. Recodage Coastline
    """
    df = apply_log_transforms(df)
    df = recode_coastline(df)

    print(f"\n[INFO] Pays avec Life_evaluation renseignée : "
        f"{df[TARGET].notna().sum()} / {len(df)}")
    return df



# Visualisations exploratoires 


def plot_distributions(df: pd.DataFrame, palette: list, year: str, save_dir: str = "") -> None:
    """Histogrammes des 11 variables explicatives."""
    features = [c for c in COLS_ALL if c != TARGET]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=20, color=palette[0], edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribution des variables explicatives — {year}",
                fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/distributions_{year}.png", bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, year: str, save_dir: str = "") -> None:
    """Heatmap de la matrice de corrélation."""
    corr = df[COLS_ALL].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                center=0, linewidths=0.8, square=True, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Corrélation de Pearson"})
    ax.set_title(f"Matrice de corrélation — {year}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/correlation_{year}.png", bbox_inches="tight")
    plt.show()

#!/usr/bin/env python3

# main.py — Modélisation du bonheur mondial (M1 IDD — Projet ML)
#
# Usage :
#   python main.py --year 2018 --task all
#   python main.py --year 2015 --task regression
#   python main.py --year 2018 --task classification
#   python main.py --year 2015 --task clustering
#   python main.py --year both  --task all    ( lance 2015 + 2018 )

################################################

import os
import argparse
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")


# Arguments CLI

parser = argparse.ArgumentParser(
    prog="ml_bonheur",
    description="Projet ML — Modélisation du bonheur mondial (2015 / 2018)",
)
parser.add_argument(
    "--year",
    type=str,
    default="2018",
    choices=["2015", "2018", "both"],
    help="Année à analyser : '2015', '2018' ou 'both'",
)
parser.add_argument(
    "--task",
    type=str,
    default="all",
    choices=["all", "eda", "regression", "classification", "clustering"],
    help=(
        "Tâche à exécuter : "
        "'all' | 'eda' (exploration) | 'regression' | 'classification' | 'clustering'"
    ),
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="datasets",
    help="Dossier contenant les fichiers CSV des datasets",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="outputs",
    help="Dossier où sauvegarder les figures ('' = pas de sauvegarde)",
)
parser.add_argument(
    "--no_save",
    action="store_true",
    help="Désactive la sauvegarde des figures",
)

args = parser.parse_args()


# Imports internes (après argparse pour un démarrage rapide en cas de --help)

from src.config import PALETTES
from src.preprocessing import (
    load_dataset, preprocess,
    plot_distributions, plot_correlation_matrix,
)
from src.regression import run_all_regression
from src.classification import run_all_classification
from src.clustering import run_all_clustering


# Helpers


def _dataset_path(year: str, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, f"dataset_{year}.csv")


def _get_save_dir(year: str) -> str:
    if args.no_save:
        return ""
    save_dir = os.path.join(args.save_dir, year)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def run_year(year: str) -> None:
    """Exécute la pipeline complète (ou partielle) pour une année donnée."""
    print(f"\n{'='*60}")
    print(f"  Année : {year}")
    print(f"{'='*60}")

    # Chargement et prétraitement
    path = _dataset_path(year, args.dataset_dir)
    if not os.path.isfile(path):
        print(f"[ERREUR] Dataset introuvable : {path}")
        print(f"Placez vos fichiers CSV dans le dossier '{args.dataset_dir}/'")
        return

    df = load_dataset(path)
    df = preprocess(df)

    # Thème graphique
    cfg     = PALETTES[year]
    palette = cfg["palette"]
    accent  = cfg["accent"]
    bg      = cfg["bg"]

    sns.set_theme(
        style="whitegrid", font_scale=1.05,
        rc={
            "figure.facecolor": bg, "axes.facecolor": bg,
            "axes.edgecolor": "#CCCCCC", "grid.color": "#E8E8E8",
            "font.family": "sans-serif",
        },
    )

    save_dir = _get_save_dir(year)

    # Dispatch selon la tâche demandée
    task = args.task

    if task in ("all", "eda"):
        plot_distributions(df, palette, year, save_dir)
        plot_correlation_matrix(df, year, save_dir)

    if task in ("all", "regression"):
        run_all_regression(df, palette, accent, year, save_dir)

    if task in ("all", "classification"):
        run_all_classification(df, year, save_dir)

    if task in ("all", "clustering"):
        run_all_clustering(df, palette, accent, year, save_dir)


# Point d'entrée

if __name__ == "__main__":
    years = ["2015", "2018"] if args.year == "both" else [args.year]
    for y in years:
        run_year(y)

    print("\n !!! Exécution terminée.")

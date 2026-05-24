#######################

# classification.py — Modèles de classification supervisée

#############################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

from src.config import TARGET, COLS_CLF

CLASS_LABELS = ["Malheureux", "Moyen", "Heureux"]



# Helpers


def _build_classes(df: pd.DataFrame, method: str = "qcut") -> pd.DataFrame:
    """
    Crée la colonne Classe_Bonheur :
    - method='qcut'  : 3 classes équilibrées (quantiles)
    - method='cut'   : seuils fixes [0, 4.5, 6.0, 10]
    """
    df = df.copy()
    if method == "qcut":
        df["Classe_Bonheur"] = pd.qcut(df[TARGET], q=3, labels=[0, 1, 2])
    else:
        df["Classe_Bonheur"] = pd.cut(
            df[TARGET], bins=[0, 4.5, 6.0, 10], labels=[0, 1, 2]
        )
    counts = df["Classe_Bonheur"].value_counts().sort_index()
    print(f"[INFO] Classes ({method}) — répartition : {dict(counts)}")
    return df


def _prepare_clf_data(df: pd.DataFrame):
    """Préparation train/test + imputation + standardisation pour la classification"""
    X = df[COLS_CLF]
    y = df["Classe_Bonheur"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test


def _plot_confusion(cm, title: str, cmap: str, year: str, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, linewidths=0.8, linecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")
    plt.tight_layout()
    if save_dir:
        safe_title = title.replace(" ", "_").replace("—", "").replace(".", "")
        fig.savefig(f"{save_dir}/{safe_title}_{year}.png", bbox_inches="tight")
    plt.show()


# Classifieurs individuels


def run_logistic_regression(X_train, X_test, y_train, y_test,
                            year: str, method_label: str, save_dir: str = "") -> None:
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"\n  [Rég. Logistique — {method_label}]")
    print(f"  Train : {model.score(X_train, y_train):.3f}  "
        f"| Test : {model.score(X_test, y_test):.3f}  "
        f"| CV : {cv:.3f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0))

    _plot_confusion(
        confusion_matrix(y_test, y_pred),
        f"Matrice de confusion — Rég. Logistique — {method_label}",
        "Blues", year, save_dir
    )


def run_knn(X_train, X_test, y_train, y_test,
            year: str, method_label: str, save_dir: str = "") -> None:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"\n  [KNN — {method_label}]")
    print(f"  Train : {model.score(X_train, y_train):.3f}  "
        f"| Test : {model.score(X_test, y_test):.3f}  "
        f"| CV : {cv:.3f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0))

    _plot_confusion(
        confusion_matrix(y_test, y_pred),
        f"Matrice de confusion — KNN — {method_label}",
        "Greens", year, save_dir
    )


def run_naive_bayes(X_train, X_test, y_train, y_test,
                    year: str, method_label: str, save_dir: str = "") -> None:
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"\n  [Naive Bayes — {method_label}]")
    print(f"  Train : {model.score(X_train, y_train):.3f}  "
        f"| Test : {model.score(X_test, y_test):.3f}  "
        f"| CV : {cv:.3f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, zero_division=0))

    _plot_confusion(
        confusion_matrix(y_test, y_pred),
        f"Matrice de confusion — Naive Bayes — {method_label}",
        "Purples", year, save_dir
    )


# Point d'entrée principal pour la classification


def run_all_classification(df: pd.DataFrame, year: str, save_dir: str = "") -> None:
    """Lance les 3 classifieurs avec qcut et cut pour une année donnée."""
    print("\n" + "=" * 55)
    print(f"  CLASSIFICATION — {year}")
    print("=" * 55)

    for method in ["qcut", "cut"]:
        print(f"\n{'─'*55}")
        print(f"  Découpage : {method.upper()}")
        print(f"{'─'*55}")

        df_clf = _build_classes(df, method=method)
        # On supprime les lignes dont la cible est NaN après découpage
        df_clf = df_clf.dropna(subset=["Classe_Bonheur"])

        X_train, X_test, y_train, y_test = _prepare_clf_data(df_clf)
        label = f"{method} — {year}"

        run_logistic_regression(X_train, X_test, y_train, y_test, year, label, save_dir)
        run_knn(X_train, X_test, y_train, y_test, year, label, save_dir)
        run_naive_bayes(X_train, X_test, y_train, y_test, year, label, save_dir)

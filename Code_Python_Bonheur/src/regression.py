##############################################

# regression.py — Modèles de régression supervisée

###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from src.config import TARGET, COLS_GDP, COLS_INTERNET, COLS_11


# Helpers

def _prepare_data(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2, seed: int = 42):
    """
    Sépare train/test, impute les NaN (KNN) et standardise.
    Retourne X_train_sc, X_test_sc, y_train, y_test.
    """
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test


def _prepare_data_no_scale(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2, seed: int = 42):
    """Comme _prepare_data mais sans standardisation (pour RF / GB)."""
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=seed)

    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    return X_train, X_test, y_train, y_test


def _print_scores(model_name: str, r2_train: float, r2_cv: float,
                r2_test: float, mse_test: float) -> None:
    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    print(f"  R² Train              : {r2_train:.3f}")
    if r2_cv is not None:
        print(f"  R² Cross-val (5-fold) : {r2_cv:.3f}")
    print(f"  R² Test               : {r2_test:.3f}")
    if mse_test is not None:
        print(f"  MSE Test              : {mse_test:.3f}")

# Régression Linéaire


def run_linear_regression(df: pd.DataFrame, feature_cols: list, label: str,
                        accent: str, palette: list, year: str,
                        save_dir: str = "") -> LinearRegression:
    """
    Entraîne une régression linéaire et affiche les coefficients.
    Retourne le modèle entraîné.
    """
    X_train, X_test, y_train, y_test = _prepare_data(df, feature_cols)

    model = LinearRegression()
    scores_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _print_scores(
        f"Régression Linéaire — {label} — {year}",
        model.score(X_train, y_train),
        scores_cv.mean(),
        r2_score(y_test, y_pred),
        mean_squared_error(y_test, y_pred),
    )

    # Graphique des coefficients
    coef_df = pd.DataFrame({"Variable": feature_cols, "Coefficient": model.coef_})
    coef_df = coef_df.sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [accent if c > 0 else palette[0] for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Variable"], coef_df["Coefficient"],
            color=colors, edgecolor="white", height=0.6)
    ax.axvline(x=0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"Coefficients — Rég. Linéaire ({label}) — {year}",
                fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient standardisé")
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/lr_coefs_{label}_{year}.png", bbox_inches="tight")
    plt.show()

    return model


# Random Forest


def run_random_forest(df: pd.DataFrame, feature_cols: list, label: str,
                    palette: list, year: str, save_dir: str = "") -> RandomForestRegressor:
    """Entraîne un Random Forest et affiche les importances"""
    X_train, X_test, y_train, y_test = _prepare_data_no_scale(df, feature_cols)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        max_features="sqrt", random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    cv_score = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2").mean()

    _print_scores(
        f"Random Forest — {label} — {year}",
        rf.score(X_train, y_train),
        cv_score,
        r2_score(y_test, y_pred),
        mean_squared_error(y_test, y_pred),
    )

    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, len(feature_cols) * 0.6)))
    feat_imp.plot(kind="barh", color=palette[1], edgecolor="white", ax=ax)
    ax.set_title(f"Importance — Random Forest ({label}) — {year}",
                fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Gini)")
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/rf_importance_{label}_{year}.png", bbox_inches="tight")
    plt.show()

    return rf


# Gradient Boosting


def run_gradient_boosting(df: pd.DataFrame, feature_cols: list, label: str,
                        palette: list, year: str, lr: float = 0.05,
                        save_dir: str = "") -> HistGradientBoostingRegressor:
    """Entraîne un Gradient Boosting et affiche la permutation importance"""
    X_train, X_test, y_train, y_test = _prepare_data_no_scale(df, feature_cols)

    gb = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=lr, max_depth=3,
        l2_regularization=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    cv_score = cross_val_score(gb, X_train, y_train, cv=5, scoring="r2").mean()

    _print_scores(
        f"Gradient Boosting — {label} — {year}",
        gb.score(X_train, y_train),
        cv_score,
        r2_score(y_test, y_pred),
        None,
    )

    result    = permutation_importance(gb, X_test, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean
    indices   = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_cols) * 0.6)))
    ax.barh(range(len(indices)), importances[indices], color=palette[2], edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_title(f"Permutation Importance — Gradient Boosting ({label}) — {year}",
                fontsize=13, fontweight="bold")
    ax.set_xlabel("Diminution du R²")
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/gb_importance_{label}_{year}.png", bbox_inches="tight")
    plt.show()

    return gb


# Point d'entrée principal pour la régression


def run_all_regression(df: pd.DataFrame, palette: list, accent: str,
                    year: str, save_dir: str = "") -> None:
    """Lance tous les modèles de régression pour une année donnée."""
    print("\n" + "=" * 55)
    print(f"  RÉGRESSION — {year}")
    print("=" * 55)

    # Régression Linéaire
    run_linear_regression(df, COLS_GDP,      "PIB",      accent, palette, year, save_dir)
    run_linear_regression(df, COLS_INTERNET, "Internet", accent, palette, year, save_dir)

    # Random Forest
    run_random_forest(df, COLS_11,      "11 variables", palette, year, save_dir)
    run_random_forest(df, COLS_GDP,     "PIB",          palette, year, save_dir)
    run_random_forest(df, COLS_INTERNET,"Internet",     palette, year, save_dir)

    # Gradient Boosting
    run_gradient_boosting(df, COLS_11,      "11 variables", palette, year, lr=0.01, save_dir=save_dir)
    run_gradient_boosting(df, COLS_GDP,     "PIB",          palette, year, lr=0.05, save_dir=save_dir)
    run_gradient_boosting(df, COLS_INTERNET,"Internet",     palette, year, lr=0.05, save_dir=save_dir)

###############################################################

# clustering.py — Apprentissage non supervisé (K-Means + ACP + Dendrogramme)

###############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Patch

from src.config import TARGET, COLS_CLUST, CLUSTER_COLORS



# Préparation des données pour le clustering


def _prepare_clustering_data(df: pd.DataFrame):
    """Imputation KNN + standardisation pour le clustering. Retourne X_scaled et df_clust."""
    df_clust = df[["Country", TARGET] + COLS_CLUST].copy()

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(df_clust[COLS_CLUST])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, df_clust


# Choix du nombre de clusters

def plot_elbow_silhouette(X_scaled: np.ndarray, palette: list, accent: str,
                        year: str, save_dir: str = "") -> None:
    """Méthode du coude + score de silhouette."""
    inertias, sil_scores = [], []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, inertias, marker="o", linestyle="--", color=palette[0], linewidth=2)
    ax1.set_title(f"Méthode du Coude — {year}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Nombre de clusters (k)")
    ax1.set_ylabel("Inertie")
    ax1.set_xticks(list(K_range))
    ax1.grid(True, alpha=0.3)

    ax2.plot(K_range, sil_scores, marker="s", linestyle="--", color=accent, linewidth=2)
    ax2.set_title(f"Score de Silhouette — {year}", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Nombre de clusters (k)")
    ax2.set_ylabel("Silhouette")
    ax2.set_xticks(list(K_range))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/elbow_silhouette_{year}.png", bbox_inches="tight")
    plt.show()


# Dendrogramme

def plot_dendrogram(X_scaled: np.ndarray, country_names: np.ndarray,
                    year: str, save_dir: str = "") -> None:
    """Classification hiérarchique de Ward."""
    Z = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(Z, labels=country_names, leaf_rotation=90.,
            leaf_font_size=7., color_threshold=7, ax=ax)
    ax.axhline(y=7, color="#E63946", linestyle="--", linewidth=1.5,
            label="Coupure à 3 clusters")
    ax.set_title(f"Dendrogramme — Classification Hiérarchique (Ward) — {year}",
                fontsize=14, fontweight="bold")
    ax.set_ylabel("Distance de Ward")
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/dendrogram_{year}.png", bbox_inches="tight")
    plt.show()


# K-Means + ACP

def run_kmeans_pca(X_scaled: np.ndarray, df_clust: pd.DataFrame,
                year: str, n_clusters: int = 3, save_dir: str = "") -> pd.DataFrame:
    """Applique K-Means, projette en ACP 2D et affiche le résultat"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clust = df_clust.copy()
    df_clust["Cluster"] = kmeans.fit_predict(X_scaled).astype(int)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clust["PC1"] = X_pca[:, 0]
    df_clust["PC2"] = X_pca[:, 1]

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(13, 9))
    for cid, color in CLUSTER_COLORS.items():
        mask = df_clust["Cluster"] == cid
        ax.scatter(df_clust.loc[mask, "PC1"], df_clust.loc[mask, "PC2"],
                c=color, s=80, alpha=0.75, edgecolors="white", linewidth=0.5,
                label=f"Cluster {cid}")

    for i in range(0, len(df_clust), 8):
        row = df_clust.iloc[i]
        ax.annotate(row["Country"], (row["PC1"], row["PC2"]),
                    fontsize=8, alpha=0.7, ha="left", xytext=(5, 5),
                    textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var1:.1f}% de variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% de variance)", fontsize=12)
    ax.set_title(f"Clusters K-Means (k={n_clusters}) — Projection ACP — {year}",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/kmeans_pca_{year}.png", bbox_inches="tight")
    plt.show()

    return df_clust


# Profils et visualisations complémentaires

def plot_cluster_boxplot(df_clust: pd.DataFrame, year: str, save_dir: str = "") -> None:
    """Boxplot du bonheur par cluster"""
    palette_list = [CLUSTER_COLORS[i] for i in range(len(df_clust["Cluster"].unique()))]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(x="Cluster", y=TARGET, data=df_clust,
                palette=palette_list, width=0.5, ax=ax)
    ax.set_title(f"Répartition du bonheur par cluster — {year}",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Life Evaluation")
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/cluster_boxplot_{year}.png", bbox_inches="tight")
    plt.show()


def plot_ranked_bar(df_clust: pd.DataFrame, year: str, save_dir: str = "") -> None:
    """Barres de bonheur triées et colorées par cluster."""
    df_ranked = df_clust.sort_values(TARGET).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    for idx, row in df_ranked.iterrows():
        ax.bar(idx, row[TARGET], color=CLUSTER_COLORS[row["Cluster"]],
            edgecolor="none", width=1.0)

    legend_elements = [Patch(facecolor=CLUSTER_COLORS[i], label=f"Cluster {i}")
                    for i in range(len(CLUSTER_COLORS))]
    ax.legend(handles=legend_elements, fontsize=11, loc="upper left", framealpha=0.9)
    ax.set_title(f"Score de bonheur par pays (triés) — Clusters K-Means — {year}",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Pays (du moins heureux au plus heureux)")
    ax.set_ylabel("Score de bonheur")
    ax.set_xticks([])
    plt.tight_layout()
    if save_dir:
        fig.savefig(f"{save_dir}/ranked_bar_{year}.png", bbox_inches="tight")
    plt.show()


def print_cluster_members(df_clust: pd.DataFrame, n_clusters: int = 3) -> None:
    """Affiche les pays par cluster."""
    for c in range(n_clusters):
        pays = df_clust[df_clust["Cluster"] == c]["Country"].values
        tail = "..." if len(pays) > 10 else ""
        print(f"\n  Cluster {c} ({len(pays)} pays) : {', '.join(pays[:10])}{tail}")


def print_cluster_profiles(df_clust: pd.DataFrame) -> None:
    """Affiche les profils moyens par cluster."""
    df_clust = df_clust.copy()
    df_clust["Coastline"] = df_clust["Coastline"].astype(float)
    profils = df_clust.groupby("Cluster")[[TARGET] + COLS_CLUST].mean().round(2)
    print("\n[INFO] Profils moyens par cluster :")
    print(profils.to_string())



# Point d'entrée principal pour le clustering


def run_all_clustering(df: pd.DataFrame, palette: list, accent: str,
                    year: str, save_dir: str = "") -> None:
    """Lance la pipeline complète de clustering pour une année donnée"""
    print("\n" + "=" * 55)
    print(f"  CLUSTERING — {year}")
    print("=" * 55)

    X_scaled, df_clust = _prepare_clustering_data(df)

    # 1) Choix du nombre de clusters
    plot_elbow_silhouette(X_scaled, palette, accent, year, save_dir)

    # 2) Dendrogramme
    plot_dendrogram(X_scaled, df_clust["Country"].values, year, save_dir)

    # 3) K-Means k=3 + ACP
    df_clust = run_kmeans_pca(X_scaled, df_clust, year, n_clusters=3, save_dir=save_dir)

    # 4) Profils
    print_cluster_profiles(df_clust)
    print_cluster_members(df_clust, n_clusters=3)

    # 5) Visualisations complémentaires
    plot_cluster_boxplot(df_clust, year, save_dir)
    plot_ranked_bar(df_clust, year, save_dir)

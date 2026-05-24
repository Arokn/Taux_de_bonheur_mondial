##################################

# config.py — Constantes globales du projet

#########################################


# Palettes graphiques (une par année)

PALETTES = {
    "2015": {
        "palette": ["#6B2737", "#C44536", "#E8856E"],  # bordeaux > rouge > saumon
        "accent":  "#457B9D",                           # bleu acier
        "bg":      "#FAF5F5",                           # crème rosé
    },
    "2018": {
        "palette": ["#2D6A4F", "#52B788", "#95D5B2"],  # vert foncé > sauge > menthe
        "accent":  "#D4A373",                           # ocre doré
        "bg":      "#FAFAF5",                           # blanc cassé vert
    },
}


# Colonnes du dataset

TARGET = "Life_evaluation"

COLS_ALL = [
    "Life_evaluation", "Conflit", "Water_Access",
    "population using the Internet", "Coastline", "Net_Migration",
    "GDP_per_capita", "HDI", "GII", "Freedom_Score", "PM25", "Suicide_Rate",
]

# 7 variables — proxy PIB (pas de multicolinéarité)
COLS_GDP = [
    "Conflit", "Coastline", "Net_Migration", "GDP_per_capita",
    "Freedom_Score", "PM25", "Suicide_Rate",
]

# 7 variables — proxy Internet (alternative au PIB)
COLS_INTERNET = [
    "Conflit", "Coastline", "Net_Migration",
    "population using the Internet", "Freedom_Score", "PM25", "Suicide_Rate",
]

# 11 variables — toutes (pour RF / GB qui tolèrent la multicolinéarité)
COLS_11 = [
    "Conflit", "Water_Access", "population using the Internet", "Coastline",
    "Net_Migration", "GDP_per_capita", "HDI", "GII", "Freedom_Score",
    "PM25", "Suicide_Rate",
]

# Variables retenues pour la classification
COLS_CLF = COLS_GDP  # identique aux 7 variables PIB

# Variables retenues pour le clustering
COLS_CLUST = [
    "Conflit", "Water_Access", "population using the Internet", "Coastline",
    "Net_Migration", "Freedom_Score", "PM25", "Suicide_Rate",
    "GDP_per_capita", "HDI", "GII",
]

# Couleurs fixes pour les 3 clusters (communes aux deux années)
CLUSTER_COLORS = {0: "#E63946", 1: "#2D6A4F", 2: "#457B9D"}

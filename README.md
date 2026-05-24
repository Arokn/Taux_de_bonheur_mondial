# Taux de bonheur mondial

Projet réalisé en M1 IDD — l'idée de départ est simple : peut-on prédire le bonheur d'un pays à partir de données chiffrées ?

Pour ça on s'appuie sur le World Happiness Report (2015 et 2018) qu'on enrichit avec 11 variables externes : PIB, accès à internet, conflits armés, pollution, migration, inégalités homme-femme, accès à l'eau potable, IDH, liberté, taux de suicide et accès à la mer.

## Ce qu'il y a dans ce repo

**NoteBook_Bonheur/** — toute la partie Jupyter  
Le preprocessing fusionne 12 fichiers de sources différentes pour construire les datasets finaux. Les notebooks d'analyse couvrent la régression, la classification et le clustering pour 2015 et 2018.

**Code_Python_Bonheur/** — la même pipeline mais en code modulaire  
Exécutable directement en ligne de commande, avec des arguments pour choisir l'année et le type de modèle à lancer. Voir le README dédié dans ce dossier.

**Rapport_Bonheur.pdf** — le rapport complet du projet

## Modèles utilisés

- Régression : linéaire, Random Forest, Gradient Boosting
- Classification : Logistique, KNN, Naive Bayes
- Clustering : K-Means, ACP, dendrogramme

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter

# 🚀 Projet : Building Energy Consumption Prediction
✨ Auteur : Maëva Beauvillain
📅 Date de début : Janvier 2025
📅 Dernière MAJ :  29 mars 2025

## 📌 Contexte du projet

Dans le cadre de son engagement pour atteindre la neutralité carbone d'ici 2050, la **ville de Seattle** cherche à mieux comprendre la **consommation énergétique** de ses bâtiments **non résidentiels**. Ce projet a pour objectif de prédire la consommation d'énergie des bâtiments à partir de leurs caractéristiques structurelles.

## 🎯 Objectif

Construire un **modèle de Machine Learning supervisé** capable de prédire la **consommation énergétique spécifique (SiteEUI - kBtu/sf)** des bâtiments, puis le **déployer sous forme d’API** pour permettre à tout citoyen ou agent municipal de simuler la consommation d’un bâtiment à partir de ses caractéristiques.

---

## Contenu du projet

- Analyse exploratoire des données
- Nettoyage, sélection de variables et **feature engineering**
- Entraînement et évaluation de plusieurs modèles supervisés
- Suivi des expérimentations avec **MLflow**
- Export du meilleur modèle avec **BentoML**
- Création d’un service API prêt à être déployé

---

## Données utilisées

- **Source** : [City of Seattle - 2016 Building Energy Benchmarking](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv)
- **Format** : CSV
- **Taille** : ~ 300 colonnes initialement, dont :
  - Caractéristiques structurelles
  - Consommations d’électricité, gaz, vapeur
  - Emissions GES
  - Scores ENERGY STAR
  - Données géographiques et administratives

---

## ⚙️ Étapes du projet

### 1. Préparation des données

- Suppression des colonnes peu remplies ou non exploitables
- Suppression des **bâtiments résidentiels**
- Suppression des **valeurs aberrantes (outliers)** sur la cible
- Nettoyage de la colonne `PrimaryPropertyType`, regroupement de catégories
- Catégorisation et réduction de variance sur `NumberofFloors`, `Neighborhood`, `NumberofBuildings`

### 2. Feature Engineering

Création de nouvelles variables à forte valeur prédictive :

- `EnergyUsed` : nature de l'énergie consommée (électricité, gaz, les deux, aucune)
- `BuildingAge` : âge du bâtiment
- `HasSecondUse` : bâtiment à usage mixte ou non
- `HasParking` : présence d’un parking

### 3. Modélisation

Trois modèles testés :

| Modèle              | R² Test | MAE Test | RMSE Test |
|---------------------|---------|----------|-----------|
| Régression linéaire | ~0.50   | ~32      | ~45       |
| SVM                 | ~0.37   | ~30.8    | ~50.5     |
| Random Forest    | **0.515** | **29.8** | **44.4** |

➡️ Le **Random Forest** est sélectionné comme **modèle final**, puis optimisé via **GridSearchCV** pour corriger l’overfitting.

### 4. Suivi avec MLflow

- Comparaison des performances des modèles
- Log des métriques, hyperparamètres et artefacts
- Versioning du modèle final

### 5. Déploiement API avec BentoML

Un service BentoML a été développé pour exposer le modèle via une API REST
Validation des données avec Pydantic (types, plages, dépendances logiques).

## Utilisation de l’API
### 1. Lancer le service BentoML

```bash
bentoml serve service:EnergyPrediction
```

### 2. Exemple d’appel API (via curl ou Postman)

#### Requête :

```json
{
  "PrimaryPropertyType": "Office",
  "BuildingAge": 25,
  "PropertyGFATotal": 100000,
  "PropertyGFABuilding": 95000,
  "NumberofFloors": 5,
  "EnergyUsed": "electricity",
  "HasSecondUse": true,
  "Neighborhood": "DOWNTOWN",
  "HasParking": true,
  "NumberofBuildings": 2
}
```
#### Réponse :
```json
{
  "SiteEUI(kBtu/sf)": [68.35]
}
```
## Organisation des fichiers

```bash
├── pyproject.toml                             # Configuration du projet avec Poetry
├── poetry.lock                                # Fichier de verrouillage des dépendances
├── requirements.txt                           # Dépendances (exportées pour BentoML)
├── bentofile.yaml                             # Configuration de BentoML
├── service.py                                 # Service BentoML (exposition de l'API)
├── onehotencoder.pkl                          # Encoder OneHot sauvegardé
├── presentation.pptx                          # Support PowerPoint de présentation
├── Notebook/                                  # Dossier de travail (scripts, données, notebooks)
│   ├── exploration_donees.ipynb               # Analyse exploratoire
│   ├── modelisation_supervisee.ipynb          # Feature engineering & modélisation ML
│   ├── my_functions.py                        # Fonctions utilitaires personnalisées
│   ├── 2016_Building_Energy_Benchmarking.csv  # Jeu de données brut
│   ├── building_consumption_analized.csv      # Données nettoyées et filtrées
│   └── building_consumption_columns_tracking.csv  # Suivi des colonnes sélectionnées
```

## 🛠️ Stack technique
- **Python 3.10+**
- Bibliothèques principales :
  - `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
- **MLflow** : suivi des expériences (metrics, modèles, hyperparamètres)
- **BentoML** : déploiement du modèle via une API REST
- **Pydantic** : validation stricte des données en entrée (types, contraintes, règles métiers)

## Résultat final

- Un modèle performant pour prédire la consommation énergétique des bâtiments non résidentiels
- Des expérimentations suivies et comparées dans **MLflow**
- Une API prête à être déployée pour une utilisation en conditions réelles




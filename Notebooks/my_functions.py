import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.stats import shapiro, chi2_contingency, f_oneway, pearsonr, spearmanr, kruskal
from itertools import combinations

# my_functions.calculate_nan(df) : Calcule le nombre et % de NaN dans un df par colonne
# my_functions.identify_column_types(df) : Identifie le type de chaque colonne d'un dataframe : Date - Binaire - Catégorielle - Continue - Inconue
# my_functions.plot_column_analysis(df, column_types) : Pour chaque colonne d'un dataframe créé une représentation visuelle
# my_functions.analyze_correlations(df) : retourne un tableau contenant les colonnes comparées, les p-values et les coefficients de corrélation.


def calculate_nan(df):
    '''
    Calcule le nombre et % de NaN dans un df par colonne
    '''
    # Compte les valeurs None
    nan_counts = df.isnull().sum()

    # Compte les valeurs string "NaN", "nan", "none"
    str_nan_counts = df.isin(["NaN", "nan", "none"]).sum()

    # Compte les valeurs vides
    empty_counts = (df == "").sum()

    # Compte les valeurs contenant uniquement des espaces
    space_counts = {}
    for col in (df.select_dtypes(include='object')).columns : 
        space_counts[col] = df[col].str.isspace().sum()
    space_counts = pd.Series(space_counts)

    # Création du DataFrame de toutes les données NaN
    merged_nan_counts = pd.DataFrame({
        'NaN counts': nan_counts.reindex(df.columns, fill_value=0),
        'Str nan counts': str_nan_counts.reindex(df.columns, fill_value=0),
        'Empty counts': empty_counts.reindex(df.columns, fill_value=0),
        'Space counts': space_counts.reindex(df.columns, fill_value=0),
    })

    # Ajout d'une colonne total de NaN
    merged_nan_counts['Total NaN'] = merged_nan_counts.sum(axis=1)

    # Ajout % de NaN
    merged_nan_counts['% NaN'] = round(merged_nan_counts['Total NaN'] / len(df) * 100, 2)

    # Retourner le DataFrame
    return merged_nan_counts


def identify_column_types(df):
    '''
    Identifie le type de chaque colonne d'un dataframe : Date - Binaire - Catégorielle - Continue - Inconue
    Une colonne numérique ayant moins de 10 valeurs uniques est considérée comme catégorielle
    '''
    
    column_types = pd.DataFrame({
    'column_name': df.columns,
    'c_type': ['' for _ in df.columns]
    })

    for col in df.columns:
        # Vérifier si la colonne est de type datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types.loc[column_types["column_name"] == col, "c_type"] = 'Date'
        # Vérifier si la colonne est binaire
        elif df[col].nunique() == 2:
            column_types.loc[column_types["column_name"] == col, "c_type"] = 'Binaire'
        # Vérifier si la colonne est catégorielle (objet ou catégorie)
        elif df[col].dtype == 'object' or isinstance(df[col].dtype, CategoricalDtype):
            column_types.loc[column_types["column_name"] == col, "c_type"] = 'Catégorielle'
        # Vérifier si la colonne est numérique mais avec moins de 10 valeurs distinctes
        elif np.issubdtype(df[col].dtype, np.number) and df[col].nunique() < 10:
            column_types.loc[column_types["column_name"] == col, "c_type"] = 'Catégorielle'
        # Vérifier si la colonne est continue (numérique)
        elif np.issubdtype(df[col].dtype, np.number):
            column_types.loc[column_types["column_name"] == col, "c_type"] = 'Continue'
        else:
            column_types.loc[column_types["column_name"] == col, "c_type"] = 'Inconnu'
    
    return column_types


def plot_column_analysis(df, column_types):
    '''
    Pour chaque colonne d'un dataframe créé une représentation visuelle : 
    - colonne continue : boxplot
    - colonne catégorielle : barplot
    '''

    # Créer des graphiques pour les colonnes continues et catégorielles
    for col in df.columns:
        # Si la colonne est continue (numérique)
        if column_types.loc[column_types["column_name"] == col, "c_type"].values[0] == "Continue":
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot pour la colonne {col}')
            
            # Calcul des statistiques
            mean = df[col].mean()
            median = df[col].median()
            std_dev = df[col].std()
            
            # Affichage des statistiques sur le graphique
            plt.figtext(0.15, 0.85, f'Moyenne: {mean:.2f}', fontsize=12)
            plt.figtext(0.15, 0.80, f'Médiane: {median:.2f}', fontsize=12)
            plt.figtext(0.15, 0.75, f'Écart-type: {std_dev:.2f}', fontsize=12)
            
            plt.show()

        # Si la colonne est catégorielle
        elif column_types.loc[column_types["column_name"] == col, "c_type"].values[0] in ["Catégorielle", "Binaire"]:
            plt.figure(figsize=(8, 6))
            value_counts = df[col].value_counts()
            value_percent = df[col].value_counts(normalize=True) * 100
            nan_percent = df[col].isna().sum() / len(df) * 100
            
            # Création du bar plot
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Bar Plot pour la colonne {col}')
            plt.ylabel('Fréquence')
            
            # Affichage des pourcentages sur le graphique
            for patch, (category, percent) in zip(plt.gca().patches, value_percent.items()):
                plt.text(patch.get_x() + patch.get_width() / 2., patch.get_height() + 1, f'{percent:.2f}%', ha='center', fontsize=10)

            # Afficher les NaN
            plt.figtext(0.15, 0.85, f'% NaN: {nan_percent:.2f}%', fontsize=12)
            plt.show()



def analyze_correlations(df):
    '''
    Analyse les corrélations entre les colonnes d'un DataFrame en fonction de leur type (continue ou catégorielle).
    Applique des tests statistiques appropriés (Khi-deux, ANOVA, Pearson ou Spearman, Kruskal-Wallis) et retourne un tableau
    contenant les colonnes comparées, les p-values et les coefficients de corrélation.
    Le df ne doit pas contenir de colonne vide.
    '''
    # Supprimer les lignes contenant des NaN dans le DataFrame
    df_cleaned = df.dropna()  # Tu peux aussi utiliser df.fillna() pour imputer les NaN
    
    # Détection des types de colonnes
    column_types = []
    for col in df_cleaned.columns:
        if np.issubdtype(df_cleaned[col].dtype, np.number):
            # Vérifier la cardinalité pour déterminer si elle est catégorielle ou continue
            if df_cleaned[col].nunique() > 10:
                column_types.append((col, 'continue'))
            else:
                column_types.append((col, 'categorielle'))
        else:
            column_types.append((col, 'categorielle'))

    # Préparation du tableau final
    results = []

    # Boucle sur les combinaisons de colonnes
    for col1, col2 in combinations(df_cleaned.columns, 2):
        type1 = next(ct[1] for ct in column_types if ct[0] == col1)
        type2 = next(ct[1] for ct in column_types if ct[0] == col2)

        if type1 == 'categorielle' and type2 == 'categorielle':
            # Test du khi-deux
            contingency_table = pd.crosstab(df_cleaned[col1], df_cleaned[col2])
            if contingency_table.size == 0:
                results.append((col1, col2, None, "Empty contingency table"))
                continue
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append((col1, col2, p, None))

        elif (type1 == 'categorielle' and type2 == 'continue') or (type1 == 'continue' and type2 == 'categorielle'):
            # Test ANOVA ou Kruskal-Wallis
            if type1 == 'categorielle' and type2 == 'continue':
                groups = [df_cleaned[col2][df_cleaned[col1] == category] for category in df_cleaned[col1].unique()]
            elif type1 == 'continue' and type2 == 'categorielle':
                groups = [df_cleaned[col1][df_cleaned[col2] == category] for category in df_cleaned[col2].unique()]

            # Vérifier qu'il y a au moins deux groupes avec des données valides
            valid_groups = [group for group in groups if len(group) > 0]
            if len(valid_groups) >= 2:
                # Vérification de la normalité des groupes
                normality_check = [shapiro(group)[1] > 0.05 for group in valid_groups]
                if all(normality_check):
                    # Si tous les groupes sont normalement distribués, utiliser ANOVA
                    f_stat, p = f_oneway(*valid_groups)
                    results.append((col1, col2, p, None))
                else:
                    # Si un ou plusieurs groupes ne sont pas normalement distribués, utiliser Kruskal-Wallis
                    h_stat, p = kruskal(*valid_groups)
                    results.append((col1, col2, p, None))
            else:
                results.append((col1, col2, None, "Not enough valid groups"))

        elif type1 == 'continue' and type2 == 'continue':
            # Vérifier la normalité des deux colonnes
            normal1 = shapiro(df_cleaned[col1])[1] > 0.05
            normal2 = shapiro(df_cleaned[col2])[1] > 0.05

            if normal1 and normal2:
                # Corrélation Pearson
                corr, p = pearsonr(df_cleaned[col1], df_cleaned[col2])
            else:
                # Corrélation Spearman
                corr, p = spearmanr(df_cleaned[col1], df_cleaned[col2])

            if np.isnan(corr):  # Gérer les cas où le calcul de la corrélation échoue
                corr = None

            results.append((col1, col2, p, corr))

    # Conversion en DataFrame
    results_df = pd.DataFrame(results, columns=['Column 1', 'Column 2', 'p-value', 'Correlation Coefficient'])
    return results_df

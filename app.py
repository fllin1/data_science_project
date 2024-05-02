import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('src/data')
import get_data as gd
import make_dataset as md
sys.path.append('src/visualization')
import plot as pl
import visualize as vz

st.set_option('deprecation.showPyplotGlobalUse', False)

# Titre de l'application
st.title("Prédiction du prix immobilier")

# Texte
st.write("Ce dashboard s'appuie le projet type kaggle contest\
          (https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/notebook).\
          Le but est de prédire le prix de maisons d'entraîner un modèle de base de forêt aléatoire\
          en utilisant TensorFlow Decision Forests sur un ensemble de données de prix de maisons.")

dataset_df = gd.get_train_data().drop('Id', axis=1)

st.sidebar.title("Données à afficher")

# Sidebar for financial data
st.sidebar.header("Données des maisons")
house_data = st.sidebar.selectbox('Sélectionnez la feature',
                                  ['LotFrontage', 'LotArea', 'YearBuilt',
                                   'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                                   'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea',
                                   'MoSold', 'YrSold', 'SalePrice'])
acronymes = {
    'LotFrontage': "La longueur de la façade du lot.",
    'LotArea': "La superficie du terrain en pieds carrés.",
    'YearBuilt': "L'année de construction de la maison.",
    'TotalBsmtSF': "La superficie totale du sous-sol en pieds carrés.",
    '1stFlrSF': "La superficie du premier étage en pieds carrés.",
    '2ndFlrSF': "La superficie du deuxième étage en pieds carrés.",
    'TotRmsAbvGrd': "Le nombre total de pièces à l'étage (à l'exclusion des salles de bains).",
    'GarageYrBlt': "L'année de construction du garage.",
    'GarageArea': "La superficie du garage en pieds carrés.",
    'MoSold': "Le mois de vente de la maison (en chiffres).",
    'YrSold': "L'année de vente de la maison.",
    'SalePrice': "Le prix de vente de la maison."
}


# Sidebar for death data
st.sidebar.header("Random Forest Info")
select_model = st.sidebar.selectbox('Sélectionnez le modèle',
                                    ['RandomForestModel', 'GradientBoostedTreesModel',
                                     'CartModel', 'DistributedGradientBoostedTreesModel'])

# Sidebar for Twitter data
st.sidebar.header("Évaluation du modèle")
select_info = st.sidebar.selectbox('Sélectionnez le donnée recherchée',
                                   ['RMSE / Nombre d\'arbres',
                                    'Poids des variables'])


# Page for Data Visualization
def data_visualization_page():
    # Données sur les maisons
    if house_data is not None:
        st.subheader("Informations sur la base de données")
        st.write("Vous trouverez les distribution indiquant le nombre de foyers disposant de \
                    certaines caractéristiques.")
        st.write(acronymes[house_data])
        plt.figure(figsize=(9, 8))
        sns.histplot(dataset_df[house_data], color='g', bins=100, kde=True, alpha=0.4)
        fig_data = plt.gcf()
        st.pyplot(fig_data)

    # Select the model
    if select_model is not None:
        st.subheader("En apprendre plus sur les modèles")
        st.write("Les algorithmes disponibles sont tous basés sur des forêts aléatoires, \
            et sont disponibles dans la librairie TensorFlow Decision Forests. Pour rappel, une \
            forêt aléatoire est une collection d'arbres de décision, chacun entraîné \
            indépendamment sur un sous-ensemble aléatoire de l'ensemble de données \
            d'entraînement (échantillonné avec remplacement). L'algorithme est unique en ce qu'il \
            est robuste au surajustement et facile à utiliser.")
        st.write(f"Vous avez choisi le modèle {select_model} :")
        if select_model == 'RandomForestModel':
            st.markdown("""
- C'est un modèle de forêt aléatoire qui combine plusieurs arbres de décision pour la \
    classification ou la régression.
- Chaque arbre est formé sur un sous-ensemble aléatoire des données d'entraînement \
    (échantillonnage avec remplacement).
- Le modèle est robuste au surajustement et est facile à utiliser.
- Il peut gérer des données avec de nombreuses fonctionnalités et est capable de gérer des \
    problèmes de régression et de classification.
""")
        elif select_model == 'GradientBoostedTreesModel':
            st.markdown("""
- C'est un modèle de gradient boosting qui construit des arbres de décision de manière \
    séquentielle pour minimiser une fonction de perte.
- Chaque arbre est formé pour corriger les erreurs faites par les arbres précédents.
- Il est souvent utilisé pour des tâches de régression et de classification.
- Il a tendance à être plus précis que les forêts aléatoires mais peut être plus sensible \
    au surajustement.
""")
        elif select_model == 'CartModel':
            st.markdown("""
- C'est un modèle d'arbre de décision simple basé sur l'algorithme CART (Classification and \
    Regression Trees).
- Il divise récursivement les données en fonction des caractéristiques pour minimiser la \
    variance des nœuds.
- Il est souvent utilisé pour la classification et la régression sur des ensembles de données \
    de petite à moyenne taille.
""")
        elif select_model == 'DistributedGradientBoostedTreesModel':
            st.markdown("""
- C'est une version distribuée du modèle de gradient boosting, conçue pour fonctionner sur de \
    grands ensembles de données.
- Il utilise le parallélisme pour accélérer l'apprentissage et est capable de gérer des \
    ensembles de données massifs.
- Il est souvent utilisé dans des scénarios où les données sont trop grandes pour tenir en \
    mémoire sur une seule machine.
""")

    # Plot results
    if select_info is not None:
        st.subheader("Résultats")
        st.write("Le modèle sur lequel nous avons travaillé est le RandomForestModel")
        st.write(select_info)
        if select_info == 'RMSE / Nombre d\'arbres':
            st.pyplot(vz.evaluate_logs(dataset_df))
        elif select_info == 'Poids des variables':
            st.pyplot(vz.plot_inspector(dataset_df))


def main():
    data_visualization_page()


if __name__ == "__main__":
    main()

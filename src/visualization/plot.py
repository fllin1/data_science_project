"""
Imports:
    seaborn (sns): Une bibliothèque de visualisation Python basée sur matplotlib, fournissant une
    interface de haut niveau pour dessiner des graphiques statistiques attrayants.
    matplotlib.pyplot (plt): Une interface basée sur l'état de matplotlib qui fournit une manière
    de tracer similaire à MATLAB.
"""

import seaborn as sns
import matplotlib.pyplot as plt


def house_price(dataset):
    """
    Trace un histogramme de la colonne 'SalePrice' à partir d'un ensemble de données donné.

    Paramètres:
        dataset (DataFrame): Un DataFrame pandas contenant la colonne 'SalePrice'.

    La fonction utilise la fonction histplot de seaborn pour tracer l'histogramme.
    """
    plt.figure(figsize=(9, 8))
    sns.histplot(dataset['SalePrice'], color='g', bins=100, kde=True, alpha=0.4)
    fig = plt.gcf()
    return fig


def evaluate_model(logs):
    """
    Trace la performance du modèle en fonction du nombre d'arbres utilisés.

    Paramètres:
        logs (list): Une liste d'objets d'inscription contenant les journaux de formation du modèle.

    La fonction utilise la fonction plot de matplotlib.pyplot pour tracer la performance du modèle.
    """
    fig, ax = plt.subplots()
    ax.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
    ax.set_xlabel("Nombre d'arbres")
    ax.set_ylabel("RMSE (hors échantillon)")
    ax.set_title("Performance du modèle en fonction du nombre d'arbres")
    return fig


def variable_weight(inspector):
    """
    Visualise l'importance des différentes variables dans un modèle de forêt de décision.

    Paramètres:
        inspector: Un inspecteur de modèle fourni par TensorFlow Decision Forests.

    La fonction utilise la fonction barh de matplotlib.pyplot pour tracer l'importance des 
    variables.
    """
    plt.figure(figsize=(12, 4))

    # Mean decrease in AUC of the class 1 vs the others.
    variable_importance_metric = "NUM_AS_ROOT"
    variable_importances = inspector.variable_importances()[variable_importance_metric]

    # Extract the feature name and importance values.
    # `variable_importances` is a list of <feature, importance> tuples.
    feature_names = [vi[0].name for vi in variable_importances]
    feature_importances = [vi[1] for vi in variable_importances]
    # The feature are ordered in decreasing importance value.
    feature_ranks = range(len(feature_names))

    bars = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
    plt.yticks(feature_ranks, feature_names)
    plt.gca().invert_yaxis()

    # Label each bar with values
    for importance, patch in zip(feature_importances, bars.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

    plt.xlabel(variable_importance_metric)
    plt.title("NUM AS ROOT of the class 1 vs the others")
    plt.tight_layout()
    fig = plt.gcf() 
    return fig

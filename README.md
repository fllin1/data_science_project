# Mise en production d'un projet en data-science

Nous avons choisi le sujet application interactive.
Ce projet s'appuie le notebook type contest kaggle (https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/comments).

## Fichier config.yaml

Il faudra modifier le fichier `config/config.yaml` avec vos clés permettant d'accéder au stockage MinIO. Assurez vous que le chemin vers votre fichier `config/config.yaml` dans le fichier `src/data/get_data.py` est le bon.

Dans ce fichier, vous pourrez indiquerer vos clés d'authentifications SSPCloud : https://datalab.sspcloud.fr/account/storage

Le fichier devrait avoir le format suivant :

- AWS_ACCESS_KEY_ID : "Votre AWS_ACCESS_KEY_ID"
- AWS_SECRET_ACCESS_KEY : "Votre AWS_SECRET_ACCESS_KEY"
- AWS_SESSION_TOKEN : "Votre AWS_SESSION_TOKEN"
- AWS_DEFAULT_REGION : "Votre AWS_DEFAULT_REGION"

## Notebooks

Les notebooks permettent de voir ce que les différents fichiers .py renvoient. Il y a actuellement 3 notebooks:
- data processing : notebook qui traite les fichiers .csv d'origine;
- visualize plots : les graphiques clés y sont représentés;
- house-price-prediction-using-tfdf : notebook explicatif des applications de la librairie tensorflow decision forest, sous la forme d'un tutoriel.

## Fichiers python (src)

Tous les fichiers python utilisés se trouvent dans le dossier src/ et sont divisés en trois fichiers:
- data/ : contient les fonctions pour récupérer les données depuis MinIO (get_data.py) et celles utiles au data processing (make_data.py)
- models/ : utilise les fonctions de tensorflow decision forests pour entrainer les modèles (train_model.py) et réaliser les prédictions (predict_model.py)
- visualization/ : contient les fichiers permettant de tracer les graphiques retrouvés dans les notebooks différents notebooks

## Fichier app.py

Fichier permettant de lancer le streamlit. Vous pourrez le tester via la commande dans le terminal : `streamlit run app.py` (à condition d'avoir bien paramétré le fichier `config.yaml` au préalable).

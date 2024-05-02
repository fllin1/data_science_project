# Mise en production d'un projet en data-science

https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/comments

## Fichier config.yaml

Il faudra modifier le fichier `config.yaml` avec vos clés permettant d'accéder au stockage MinIO. Assurez vous que le chemin vers votre fichier `config.yaml` dans le fichier `src/data/get_data.py` est le bon.

Dans ce fichier, vous pourrez indiquerer vos clés d'authentifications SSPCloud : https://datalab.sspcloud.fr/account/storage

Le fichier aura le format suivant :

- AWS_ACCESS_KEY_ID : "Votre AWS_ACCESS_KEY_ID"
- AWS_SECRET_ACCESS_KEY : "Votre AWS_SECRET_ACCESS_KEY"
- AWS_SESSION_TOKEN : "Votre AWS_SESSION_TOKEN"
- AWS_DEFAULT_REGION : "Votre AWS_DEFAULT_REGION"

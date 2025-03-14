# Twitch Moderation Backend

API FastAPI pour l'analyse et la modération en temps réel des chats Twitch. Cette API fournit des fonctionnalités d'analyse de contenu pour le projet [twitch-moderation-front](https://github.com/ManoelDaPonte/twitch-moderation-front).

## Fonctionnalités

-   **Modération de contenu**: Détection automatique de contenu toxique, haineux, violent ou inapproprié
-   **Analyse thématique**: Regroupement des messages par sujets principaux
-   **Détection de questions**: Identification et classification des questions posées dans le chat
-   **Clustering par intelligence artificielle**: Utilisation des embeddings OpenAI pour grouper des contenus similaires

## Endpoints API

-   `GET /moderation/{input}`: Évalue un message pour détecter le contenu inapproprié
-   `POST /topics/`: Analyse et regroupe un ensemble de messages par thèmes
-   `POST /questions/`: Extrait et classifie les questions des messages de chat
-   `GET /completion/{input}`: Alternative de modération basée sur le modèle conversationnel

## Prérequis

-   Python 3.8+
-   pip ou conda
-   Un compte [OpenAI](https://platform.openai.com/) avec une clé API

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ManoelDaPonte/twitch-moderation-back.git
cd twitch-moderation-back

# Installer les dépendances
pip install -r requirements.txt

# Créer un fichier .env avec votre clé API OpenAI
echo "OPENAI_API_KEY=votre_clé_api_openai" > .env
```

## Démarrage du serveur

```bash
# Démarrer le serveur FastAPI avec auto-rechargement
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Le serveur sera accessible à l'adresse http://localhost:8000

## API Documentation

Une fois le serveur démarré, la documentation interactive Swagger est disponible à:
http://localhost:8000/docs

## Structure du projet

```
twitch-moderation-back/
├── main.py                   # Point d'entrée de l'application FastAPI
├── openai_moderation.py      # Module de modération via l'API OpenAI
├── openai_completion.py      # Module de complétion via ChatGPT
├── top_topics.py             # Module d'analyse des sujets principaux
├── top_questions.py          # Module d'extraction et analyse des questions
├── utils.py                  # Fonctions utilitaires partagées
├── requirements.txt          # Dépendances du projet
└── .env                      # Variables d'environnement (à créer)
```

## Exemples d'utilisation

### Modération d'un message

```bash
curl -X GET "http://localhost:8000/moderation/Votre%20message%20à%20modérer"
```

### Analyse de sujets

```bash
curl -X POST "http://localhost:8000/topics/" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Message 1", "Message 2", "..."]}'
```

## Configuration avancée

### Paramètres de clustering

Lors de l'appel des endpoints `/topics/` et `/questions/`, vous pouvez personnaliser:

-   `n_clusters`: Nombre de clusters à générer (par défaut: 5)
-   `n_samples`: Nombre maximum de messages à analyser (par défaut: 100)
-   `model_generation`: Modèle OpenAI pour la génération de noms de clusters (par défaut: "gpt-3.5-turbo")
-   `model_embedding`: Modèle OpenAI pour les embeddings (par défaut: "text-embedding-ada-002")

Exemple:

```json
{
	"comments": ["Message 1", "Message 2", "..."],
	"n_clusters": 3,
	"n_samples": 50,
	"model_generation": "gpt-4",
	"model_embedding": "text-embedding-3-small"
}
```

## Dépendances principales

-   [FastAPI](https://fastapi.tiangolo.com/) - Framework API web moderne
-   [OpenAI Python](https://github.com/openai/openai-python) - Client Python officiel pour l'API OpenAI
-   [scikit-learn](https://scikit-learn.org/) - Bibliothèque d'apprentissage automatique pour le clustering
-   [pandas](https://pandas.pydata.org/) - Manipulation et analyse de données

## Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

[MIT](LICENSE)


from sklearn.cluster import KMeans
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_questions(chat_samples):
    questions = [sentence for sentence in chat_samples if '?' in sentence]
    return questions

def initialize_df(list_input, column_name):
    df = pd.DataFrame(list_input, columns=[column_name])
    return df


def embed_sentence(sentence, model):
    response = client.embeddings.create(
        input=sentence,
        model=model
    )
    return response.data[0].embedding

def k_means_clustering(matrix, n_clusters = 5):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    return kmeans.labels_.tolist()
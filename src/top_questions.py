from utils import get_questions, embed_sentence, k_means_clustering, initialize_df
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import json
import os

# Load environment variables from .env file
load_dotenv()

# Enable tqdm pandas
tqdm.pandas()

# Load the openai api key
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def create_cluster_name(df, n_clusters, rev_per_cluster=5, model='gpt-3.5-turbo', column_name = "question"):
    cluster_themes = []

    for i in range(n_clusters):

        # Get the actual number of reviews per cluster
        cluster_size = df[df.Cluster == i].shape[0]
        if cluster_size < rev_per_cluster:
            rev_per_cluster_acutal = cluster_size
        else :
            rev_per_cluster_acutal = rev_per_cluster

        # Get the questions
        questions = "\n".join(
            df[df.Cluster == i].sample(rev_per_cluster_acutal, random_state=42)[column_name].values
        )

        # Prompt the model
        messages = [
            {"role": "user", "content": f'What do the following questions in a stream channel have in common? Try to find a short Theme to describe it. \n\nUser question:\n"""\n{questions}\n"""\n\nTheme:'},
        ]

        # Get the response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)

        # Parse the response
        theme = response.choices[0].message.content.strip()
        questions_list = questions.split("\n")

        # Append the result to the cluster_themes list
        cluster_themes.append({
            "cluster": i,
            "theme": theme,
            "questions": questions_list
        })

    return json.dumps(cluster_themes, ensure_ascii=False, indent=2)



def main_questions(input, n_clusters = 5, n_samples = 100, model_generation = "gpt-3.5-turbo", model_embedding = "text-embedding-ada-002"): #text-embedding-3-small

    # get the last n_samples from the input
    input = input[-n_samples:]

    # get the questions from the input
    questions = get_questions(input)

    # initialize the dataframe
    df = initialize_df(questions, "question")

    # embed the questions
    df["embedding"] = df["question"].progress_apply(lambda x: embed_sentence(x, model_embedding))

    # cluster the questions
    matrix = np.vstack(df["embedding"].values)
    labels = k_means_clustering(matrix, n_clusters)
    df["Cluster"] = labels

    # create the cluster names and return the result
    result = create_cluster_name(df, n_clusters, rev_per_cluster = 5, model=model_generation)

    return json.loads(result)

if __name__ == "__main__":

    input = [
    "What's your favorite game to stream and why?",
    "Any tips for new streamers just starting out?",
    "What setup do you use for streaming?",
    "How do you deal with trolls in chat?",
    "Can you explain your stream schedule?",
    "What was your most memorable streaming moment?",
    "How do you choose what games to play on stream?",
    "What's the best way to support your channel?",
    "How long have you been streaming?",
    "What do you enjoy most about streaming?",
    "Do you have any streaming rituals or routines?",
    "How do you stay motivated to stream regularly?",
    "What's the hardest part about being a streamer?",
    "Any advice on building a community on Twitch?",
    "What are your thoughts on the latest game update?",
    "How do you balance streaming with personal life?",
    "What's your favorite streaming moment with viewers?",
    "How did you come up with your streamer name?",
    "What games are you looking forward to streaming next?",
    "Do you collaborate with other streamers?"
    ]

    main_questions(input=input)
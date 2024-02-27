from utils import embed_sentence, k_means_clustering, initialize_df
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

def create_cluster_name(df, n_clusters, rev_per_cluster=5, model='gpt-3.5-turbo', column_name = "comment"):
    cluster_themes = []

    for i in range(n_clusters):

        # Get the actual number of reviews per cluster
        cluster_size = df[df.Cluster == i].shape[0]
        if cluster_size < rev_per_cluster:
            rev_per_cluster_acutal = cluster_size
        else :
            rev_per_cluster_acutal = rev_per_cluster

        # Get the topics
        topics = "\n".join(
            df[df.Cluster == i].sample(rev_per_cluster_acutal, random_state=42)[column_name].values
        )

        # Prompt the model
        messages = [
            {"role": "user", "content": f'What do the following comments in a stream channel have in common? Please try to find a short theme.\n\nUser comment:\n"""\n{topics}\n"""\n\nTheme:'},
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
        topics_list = topics.split("\n")
        cluster_themes.append({
            "cluster": i,
            "theme": theme,
            "topics": topics_list
        })

    return json.dumps(cluster_themes, ensure_ascii=False, indent=2)



def main_topics(input, n_clusters = 5, n_samples = 100, model_generation = "gpt-3.5-turbo", model_embedding = "text-embedding-ada-002"): #text-embedding-3-small
    
    # get the last n_samples from the input
    input = input[-n_samples:]

    # initialize the dataframe
    df = initialize_df(input, "comment")

    # embed the topics
    df["embedding"] = df["comment"].progress_apply(lambda x: embed_sentence(x, model_embedding))

    # cluster the topics
    matrix = np.vstack(df["embedding"].values)
    labels = k_means_clustering(matrix, n_clusters)
    df["Cluster"] = labels

    # create the cluster names and return the result
    result = create_cluster_name(df, n_clusters, rev_per_cluster = 5, model=model_generation)

    return json.loads(result)

if __name__ == "__main__":
    import random

    # Updated comments with more variety, including casual comments, emojis, and realistic chat behavior
    comments_updated = [
        "What's your favorite game of all time? 😊",
        "LOL, did anyone else see that glitch just now? 🤣",
        "Dude, that was an epic win!!!",
        "What got you into streaming? Always curious.",
        "Any tips for a newbie here? 😅",
        "What's your setup like? I'm looking to upgrade mine.",
        "Been gaming for years or what? You're really good!",
        "Can you play [insert game name] next? Would love to see it!",
        "Streaming seems fun, what do you love about it? ❤️",
        "Trolls suck, but you handle them well! 👏",
        "That last game was so boring, play something else!",
        "OMG, your gameplay is on fire today! 🔥",
        "Why do you always play this game? 🙄",
        "Not this map again... zzz",
        "Hey chat, anyone wanna team up after?",
        "This streamer's the best, change my mind. 😉",
        "Yawn, seen better plays. 😴",
        "Who else is here just for the chat? 😜",
        "Turn up the game volume, can barely hear it!",
        "Your mic is muted... again. 🤦‍♂️"
    ]

    # Generate 100 random chat samples with updated comments, allowing for duplication
    input = []
    for _ in range(40):
        comment = random.choice(comments_updated)
        input.append(comment)

    main_topics(input=input)
from dotenv import load_dotenv
from openai import OpenAI
import json
import os

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_completion(input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content":
                '''
                You are a moderator, your goal is to find if a message is offensing or not. 
                You have to categorize them in 3 categories.
                red : Insulting message, hate, sexual, harrassment, or any other kind of offensing message. It had to be very outraging.
                orange : Message givin an opinion about potential sensitive subject but not offensing anyone or insulting.
                green : a message without any disrepectfull word and not talking about potentital sensitive subject.
                you have to return the result as a json object with a score of confidence for each of the catogory.
                '''
            },
            {
                "role": "user",
                "content": input
            }
        ]
    )
    return completion.choices[0].message.content

def parse_completion(completion):
    return json.loads(completion)

def main_completion(input):
    completion = generate_completion(input)
    return parse_completion(completion)

if __name__ == "__main__":
    input = "I think that the government is doing a terrible job"
    completion = generate_completion(input)
    print(parse_completion(completion))
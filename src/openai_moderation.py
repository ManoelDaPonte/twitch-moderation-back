from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_moderation(input):
    response = client.moderations.create(input=input)
    return response

def parse_moderation(moderation, input):
    dictionnary_moderation = \
        {
            'input' : input,
            'flagged' : moderation.results[0].flagged,
            'categories' : {
                'harassment' : {
                    'flagged' : moderation.results[0].categories.harassment,
                    'score' : moderation.results[0].category_scores.harassment,
                    },
                'harassment threatening' : {
                        'flagged' : moderation.results[0].categories.harassment_threatening,
                        'score' : moderation.results[0].category_scores.harassment_threatening,
                    },
                'hate' : {
                        'flagged' : moderation.results[0].categories.hate,
                        'score' : moderation.results[0].category_scores.hate,
                    },
                'hate threatening' : {
                        'flagged' : moderation.results[0].categories.hate_threatening,
                        'score' : moderation.results[0].category_scores.hate_threatening,
                    },
                'self harm' :   {
                        'flagged' : moderation.results[0].categories.self_harm,
                        'score' : moderation.results[0].category_scores.self_harm,
                    },
                'self harm instructions' : {
                        'flagged' : moderation.results[0].categories.self_harm_instructions,
                        'score' : moderation.results[0].category_scores.self_harm_instructions,
                    },
                'self harm intent' : {
                        'flagged' : moderation.results[0].categories.self_harm_intent,
                        'score' : moderation.results[0].category_scores.self_harm_intent,
                    },
                'sexual' : {
                        'flagged' : moderation.results[0].categories.sexual,
                        'score' : moderation.results[0].category_scores.sexual,
                    },
                'sexual minors' : {
                        'flagged' : moderation.results[0].categories.sexual_minors,
                        'score' : moderation.results[0].category_scores.sexual_minors,
                    },
                'violence' : {
                        'flagged' : moderation.results[0].categories.violence,
                        'score' : moderation.results[0].category_scores.violence,
                    },
                'violence graphic' : {
                        'flagged' : moderation.results[0].categories.violence_graphic,
                        'score' : moderation.results[0].category_scores.violence_graphic,
                    },

                }

        }
    return dictionnary_moderation

def main_moderation(input):
    moderation = generate_moderation(input)
    return parse_moderation(moderation, input)

if __name__ == "__main__":
    input = "I think that the government is doing a terrible job"
    moderation = generate_moderation(input)
    print(parse_moderation(moderation, input))
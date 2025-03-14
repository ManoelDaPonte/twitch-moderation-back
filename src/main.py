from fastapi.middleware.cors import CORSMiddleware
from openai_completion import main_completion
from openai_moderation import main_moderation
from top_questions import main_questions
from top_topics import main_topics
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
import os

app = FastAPI()

class TopicsInput(BaseModel):
    comments: List[str]
    n_clusters: int = 5
    n_samples: int = 100
    model_generation: str = "gpt-3.5-turbo"
    model_embedding: str = "text-embedding-ada-002"

class QuestionsInput(BaseModel):
    comments: List[str]
    n_clusters: int = 5
    n_samples: int = 100
    model_generation: str = "gpt-3.5-turbo"
    model_embedding: str = "text-embedding-ada-002"

# CORS settings - Ajoutez votre URL Vercel ici
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://twitch-moderation-front.vercel.app",  # Sans la barre oblique finale
    "*",  # Option pour autoriser toutes les origines pendant les tests
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "Nina"}

@app.get("/completion/{input}")
def completion(input: str):
    return main_completion(input)

@app.get("/moderation/{input}")
def moderation(input: str):
    return main_moderation(input)

@app.post("/questions/")
def get_topics(input: QuestionsInput):
    """
    Receives a JSON body with a list of topics and returns a list of JSON objects.
    """
    return main_questions(input.comments, input.n_clusters, input.n_samples, input.model_generation, input.model_embedding)

@app.post("/topics/")
def get_topics(input: TopicsInput):
    """
    Receives a JSON body with a list of topics and returns a list of JSON objects.
    """
    return main_topics(input.comments, input.n_clusters, input.n_samples, input.model_generation, input.model_embedding)

@app.get("/test")
def test():
    """
    Simple test endpoint to check if the API is running correctly.
    """
    return {
        "message": "API is running correctly",
        "openai_key_exists": bool(os.environ.get("OPENAI_API_KEY")),
        "origins": origins
    }

# Gardez ce bloc, mais avec une condition qui vérifie s'il s'exécute directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from openai_completion import main_completion
from openai_moderation import main_moderation
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/completion/{input}")
def completion(input: str):
    return main_completion(input)

@app.get("/moderation/{input}")
def moderation(input: str):
    return main_moderation(input)
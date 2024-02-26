from fastapi.middleware.cors import CORSMiddleware
from openai_completion import main_completion
from openai_moderation import main_moderation
from fastapi import FastAPI

app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",  # Assuming your frontend runs on this port
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
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
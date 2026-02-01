from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an AI voice bot that represents Sushyamal Maji.
Answer questions exactly as Sushyamal would in an interview.

Background:
- M.Tech Data Science student
- Interested in multi-omics, deep learning, NLP, SQL
- Strong in problem-solving and research
- Actively preparing for data science interviews
- Clear, honest, confident communication style

Rules:
- Speak in first person ("I")
- Keep answers concise (30–60 seconds spoken)
- Sound human, confident, and thoughtful
- No buzzwords or generic AI phrases
"""

class UserInput(BaseModel):
    message: str

@app.post("/chat")
def chat(user_input: UserInput):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input.message}
        ]
    )
    return {"reply": response.choices[0].message.content}

@app.get("/")
def root():
    return {"status": "AI Voice Bot backend is running"}

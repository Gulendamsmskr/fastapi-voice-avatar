# services/ai.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

def ollama_chat(system: str, user: str) -> str:
    prompt = f"{system}\n\nKullanıcı: {user}\nAsistan:"

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()
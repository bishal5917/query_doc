
import requests

def check_ollama_connection(host: str) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def generate_with_ollama(prompt: str, host: str, model: str) -> str:
    url = f"{host}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for answering questions from a document."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()

    if "choices" not in data or not data["choices"]:
        return "No answer generated. Check if model is loaded."

    return data["choices"][0]["message"]["content"].strip()


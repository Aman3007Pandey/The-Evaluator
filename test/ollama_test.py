import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:3b",
        "prompt": "Introduce yourself in 3 lines.",
        "stream": False,
    },
)

print(response.json()["response"])
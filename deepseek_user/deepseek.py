import requests


def deepseek_generate(prompt):
    api_url = 'http://localhost:11434/api/generate'

    payload = {
        "model": "deepseek-r1:7b",
        "prompt": f"{prompt}",
        "stream": False
    }

    response = requests.post(api_url, json=payload)
    result = response.json()
    return result['response']

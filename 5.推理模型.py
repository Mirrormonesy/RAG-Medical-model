import requests


prompt = "今天天气怎么样？"

response = requests.post(
    url="http://127.0.0.1:11434/api/generate",
    json={
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }
)
res = response.json()['response']
print(res)
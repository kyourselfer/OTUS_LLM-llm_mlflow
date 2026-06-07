import json
import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
headers = {'Content-Type': 'application/json','Authorization': 'Bearer local-qwen-key'}
data = {
    "model":"qwen2.5-3b-instruct-gptq-int8",
    "messages":
     [
         {"role":"user","content":"hi"}
     ],
    "max_tokens":4}

response = requests.post(url, headers=headers, json=data, timeout=5)
completion = response.json()
print(json.dumps(completion, indent=2))


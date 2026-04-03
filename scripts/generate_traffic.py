import requests
import json
import time

API_URL = "http://localhost:8080/generate"
HEADERS = {"Content-Type": "application/json"}

def send_request(prompt, max_new_tokens=8):
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens
    }
    try:
        r = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=10)
        r.raise_for_status()
        print(f"Prompt: {prompt} → {r.json()['output'][:50]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    for i in range(30):
        send_request(f"ping {i}")
        time.sleep(0.2)  # 5 запросов в секунду
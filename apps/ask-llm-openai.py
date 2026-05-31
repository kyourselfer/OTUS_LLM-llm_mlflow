from openai import OpenAI

client = OpenAI(
    api_key="DUMMY", # Any dummy string
    default_headers={
        "Authorization": "Bearer local-qwen-key",
    },
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="qwen2.5-3b-instruct-gptq-int8",
    messages=[
        {"role": "system", "content": "You are a Middle DevOps."},
        {"role": "user",   "content": "What is CI and CD. Give me a short answer."}
    ],
    max_tokens=128,
)

completion = response.to_json()
print(completion)


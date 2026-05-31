hf download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8 --local-dir models/Qwen2.5-3B-Instruct-GPTQ-Int8
docker compose up -d
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer local-qwen-key" | jq
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer local-qwen-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b-instruct-gptq-int8",
    "messages": [
      {
        "role": "user",
        "content": "Напиши краткое резюме: что такое vLLM?"
      }
    ],
    "temperature": 0.2,
    "max_tokens": 300
  }' | jq

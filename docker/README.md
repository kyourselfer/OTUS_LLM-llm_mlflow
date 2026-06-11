# Скачиваем модель локально для подгрузки в vLLM и проверяем доступность
hf download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8 --local-dir models/Qwen2.5-3B-Instruct-GPTQ-Int8
docker compose -f ./docker/docker-compose.yml up
curl -s http://192.168.1.200:8000/v1/models -H "Authorization: Bearer local-qwen-key" | jq
curl http://192.168.1.200:8000/v1/chat/completions \
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
    "max_tokens": 100
  }' | jq

# Разворот mlflow на Minikube
minikube start
bash ./scripts/deploy-mlflow.sh

# Проверяем через скрипты доступность LLM, через запросы requests,openai
python3 -m venv ~/AI/python12-venv
source ~/AI/python12-venv/bin/activate
pip install -r ./apps/requirements.txt

python3 ./apps/ask-llm-requests.py
python3 ./apps/ask-llm-openai.py

# Запуск оценки модели подгруженной в vLLM (модель оценивает сама себя как судья)
python3 ./apps/evaluate_vllm_direct_mlflow_judge.py


# Пробный запуск inference
k -n registry apply -f k8s/registry.yaml
docker buildx build -t minikube-registry.local/llm-ops-workshop:latest -f ./inference/Dockerfile
docker push minikube-registry.local/llm-ops-workshop:latest
bash ./scripts/deploy-app.sh


## Анализ результатов запуска ./apps/evaluate_vllm_direct_mlflow_judge.py

В evaluation используются два типа оценки:

1. Exact match - строгая строковая метрика. Она показывает, совпадает ли ответ модели с эталоном буквально. Для QA-задач эта метрика часто занижает качество, потому что смысловой правильный ответ может отличаться формулировкой.

2. make_genai_metric(LLM-as-a-Judge) - кастомная смысловая метрика. Она оценивает ответ по шкале 1-5 с учётом корректности, полноты, обоснованность и отсутствия галюцинаций.

Если exact_match низкий, а llm_as_judge_devops_quality_ru высокий, это означает, что модель отвечает близко к эталону по смыслу, но использует другую формулировку.

Если judge-score низкий, нужно смотреть justification в evaluation artifacts. Обычно низкая оценка возникает, если модель:
- добавила факт, которого не было в контексте;
- ответила слишком обобщённо;
- пропустила ключевую часть ответа;
- не сказала, что ответ отсутствует в контексте.

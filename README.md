Форкнуто от https://github.com/Ilia2704/llm_mlflow. Выражаю благодарность преподователю курса Илье Ящук.

# LLM Ops Workshop

## 📄 Состав проекта

### 1. `inference.py` — сервис инференса с метриками  
**Назначение:** API для работы с LLM с оптимизацией вычислений и экспортом метрик в Prometheus.  

**Ключевые моменты:**  
- **FastAPI** — REST API для генерации текста.  
- **transformers** — загрузка моделей HuggingFace.  
- **`torch.cuda.amp.autocast()`** — смешанная точность (mixed precision) для ускорения и экономии памяти.  
- **prometheus_client** — экспорт метрик (время ответа, количество токенов).  
- **DEVICE auto-detection** — автоматический выбор CPU или GPU.  
- **Latency tracking** — измерение задержки для SLA.  

**Что даёт:**  
Позволяет обрабатывать запросы и одновременно контролировать состояние инференса в продакшене.  

---

### 2. `Dockerfile` — контейнеризация сервиса  
**Назначение:** Сборка Docker-образа для запуска в Kubernetes.  

**Ключевые моменты:**  
- `FROM python:3.10-slim` — минимальный Python-образ.  
- `requirements.txt` — установка зависимостей.  
- `uvicorn` — ASGI-сервер для FastAPI.  
- `EXPOSE 8080` — порт сервиса.  

**Что даёт:**  
Воспроизводимая среда с фиксированными версиями Python и библиотек.  

---

### 3. `requirements.txt` — зависимости Python  
**Назначение:** Список библиотек для сервиса.  

**Ключевые моменты:**  
- `fastapi` — API.  
- `uvicorn` — сервер.  
- `transformers` — LLM.  
- `torch` — backend PyTorch.  
- `prometheus_client` — метрики.  

**Что даёт:**  
Быстрая установка окружения одной командой.  

---

### 4. `deployment.yaml` — деплой в Kubernetes  
**Назначение:** Описание запуска сервиса в Kubernetes.  

**Ключевые моменты:**  
- `replicas` — количество реплик.  
- `resources.limits` — лимиты по GPU, CPU, RAM.  
- `env` — переменные окружения (например, `MODEL_NAME`).  
- `selector.matchLabels` — связывание с сервисом.  

**Что даёт:**  
Контроль числа реплик и выделяемых ресурсов.  

---

### 5. `service.yaml` — внутренний доступ к сервису  
**Назначение:** Создание точки входа внутри кластера.  

**Ключевые моменты:**  
- `ClusterIP` — доступ только внутри кластера.  
- `targetPort` — порт контейнера.  
- `port` — порт сервиса.  

**Что даёт:**  
Доступ к LLM для других сервисов (например, Prometheus).  

---

### 6. `ingress.yaml` — внешний доступ  
**Назначение:** Подключение сервиса к интернету через домен.  

**Ключевые моменты:**  
- Ingress Controller (Nginx, Traefik) — маршрутизация.  
- `rules.host` — домен (например, `llm.example.com`).  
- `paths` — правила маршрутизации URL.  

**Что даёт:**  
Доступ пользователей и внешних систем к API.  

---

### 7. `hpa.yaml` — автоматическое масштабирование  
**Назначение:** Динамическое изменение числа реплик.  

**Ключевые моменты:**  
- `minReplicas` / `maxReplicas` — диапазон масштабирования.  
- `metrics` — условия изменения (например, CPU > 70%).  
- Horizontal Pod Autoscaler — встроенный контроллер Kubernetes.  

**Что даёт:**  
Экономия ресурсов и стабильность под нагрузкой.  

---

### 8. `prometheus-config.yaml` — сбор метрик  
**Назначение:** Настройка Prometheus для мониторинга LLM.  

**Ключевые моменты:**  
- `scrape_interval` — частота опроса (5s).  
- `targets` — адреса сервисов.  
- `job_name` — имя задания.  

**Что даёт:**  
Сбор данных для анализа и алертинга.  

---

### 9. `grafana-dashboard.json` — дашборд для мониторинга  
**Назначение:** Визуализация метрик в Grafana.  

**Ключевые моменты:**  
- **Request Latency** — задержка ответов.  
- **Tokens Processed** — нагрузка в токенах.  
- **expr** — Prometheus-запросы (PromQL).  

**Что даёт:**  
Быстрый старт мониторинга без ручной настройки.  

---

### 10. `filebeat-config.yaml` — логирование в ELK  
**Назначение:** Отправка логов в Elasticsearch через Filebeat.  

**Ключевые моменты:**  
- `type: container` — сбор логов контейнеров.  
- `paths` — пути к логам в Kubernetes.  
- `output.elasticsearch` — адрес Elasticsearch.  

**Что даёт:**  
Хранение, поиск и анализ логов.  

---

## 🚀 Запуск проекта

### 1. Сборка и публикация Docker-образа
```bash
bash scripts/create-kind-cluster.sh
bash scripts/enable-ingress-nginx.sh 
bash scripts/build-and-load.sh
bash scripts/deploy-app.sh
```

### 1.a MLflow, vLLM, Triton на kind
```bash
# MLflow UI: http://mlflow.localtest.me:8080
bash scripts/deploy-mlflow.sh

# vLLM OpenAI API: http://vllm.localtest.me:8080/v1
bash scripts/deploy-vllm.sh

# Triton Server: http://triton.localtest.me:8080  (metrics: http://triton.localtest.me:8080/metrics)
bash scripts/deploy-triton.sh

# Всё разом
bash scripts/deploy-all.sh
```

### 1.b Комплексное тестирование моделей (GLUE, SuperGLUE, SQuAD, LAMBADA, MMLU)
```bash
# Запуск полного бенчмарка с автоматической установкой зависимостей
bash scripts/run_benchmark.sh
```

Скрипт `scripts/run_benchmark.sh`:
- Скачивает датасеты: GLUE, SuperGLUE, SQuAD, LAMBADA, MMLU
- Тестирует модели: facebook/opt-125m, EleutherAI/gpt-neo-125M, gpt2, microsoft/DialoGPT-small
- Вычисляет метрики: Perplexity, BLEU, ROUGE, F1, Exact Match
- Логирует результаты в MLflow эксперимент "llm-benchmark"
- Показывает топ-модели по разным метрикам

Результаты доступны в MLflow UI: `http://mlflow.localtest.me:8080`

Примечание: домены *.localtest.me автоматически резолвятся в 127.0.0.1. В kind Ingress доступен на хостовом порту 8080 (см. `scripts/kind-config.yaml`), поэтому добавляйте `:8080` в URL.

Если DNS не резолвит `*.localtest.me`, добавь в `/etc/hosts`:
```bash
sudo sh -c 'printf "\n127.0.0.1 mlflow.localtest.me vllm.localtest.me triton.localtest.me\n" >> /etc/hosts'
```
---
#### **2. Тест API-запроса**
```markdown
### 2. Тест API-запроса (Теперь API доступен на http://localhost:8080)

```bash
curl -X POST http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a short haiku about Kubernetes", "max_new_tokens": 64}'
  
```
---

#### **3. Мониторинг**
```markdown
### 3. Мониторинг

#### 3.1 Проверка метрик API
```bash
curl http://localhost:8080/metrics
```
#### 3.2 Prometheus: 
```bash
docker run -p 9090:9090 -v $(pwd)/monitoring/prometheus-config.yaml:/etc/prometheus/prometheus.yml prom/prometheus
```
#### 3.3 Grafana: 
```bash
docker run -d -p 3000:3000 grafana/grafana
```
URL: http://localhost:3000, логин/пароль: admin/admin
Data Source Prometheus: http://host.docker.internal:9090
Импортируй `monitoring/grafana-dashboard.json`
Далее запускаем: `python3 scripts/generate_traffic.py`

---

## CI/CD (GitLab → local kind)

### Локальный GitLab Runner (shell executor)
1. Установить и запустить сервис (macOS):
```bash
brew install gitlab-runner
sudo gitlab-runner install && sudo gitlab-runner start
```

2. Регистрация раннера (тег должен совпадать с `.gitlab-ci.yml`):
```bash
gitlab-runner register \
  --url https://gitlab.com \
  --registration-token YOUR_TOKEN \
  --executor shell \
  --description "local-kind" \
  --tag-list "local-kind" \
  --run-untagged=false --locked=false
```

3. Требования окружения раннера:
- Docker Desktop запущен
- Установлены `kind`, `kubectl`, `jq` (`brew install kind kubectl jq`)
- Переменные окружения доступны раннеру: `PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin`, `KUBECONFIG=/Users/muse/.kube/config`

4. Что делает pipeline:
- `prepare`: создаёт кластер kind (`llm`), настраивает kubeconfig, включает Ingress NGINX
- `build`: собирает образ и загружает его в kind
- `deploy`: применяет манифесты (приложение, MLflow, vLLM, Triton)
- `test`: проверяет `/healthz` и базовые эндпоинты

В pipeline smoke-тест использует `kubectl port-forward`, чтобы не зависеть от готовности Ingress. Для доступа через Ingress, убедись, что `/etc/hosts` содержит хосты `mlflow.localtest.me vllm.localtest.me triton.localtest.me`.

Если job «stuck: no runners with tag local-kind» — убедитесь, что раннер онлайн и имеет тег `local-kind` (и отмечен как Protected для protected-веток).

### Переменные CI (GitLab → Settings → CI/CD → Variables)
- **GHCR_USERNAME / GHCR_TOKEN**: учётные данные GitHub (PAT с правом `read:packages`) для pull из `ghcr.io`
  - Применение: deploy job выполнит `docker login ghcr.io` и сможет тянуть образы (например, MLflow)
  - Рекомендации: `Masked=true`, `Protected=true` (если деплой с protected-веток)
- **NGC_API_KEY**: API-ключ NVIDIA NGC для pull из `nvcr.io`
  - Применение: при наличии ключа pipeline выполнит `docker login nvcr.io` и будет использовать `nvcr.io/nvidia/tritonserver:24.01-py3` для Triton
  - Рекомендации: `Masked=true`, `Protected=true`
- (Необязательно) **KUBECONFIG**: путь к kubeconfig, если раннер запускается как root
  - Пример: `/var/root/.kube/config`
  - По умолчанию pipeline пробует `$HOME/.kube/config` и `/var/root/.kube/config`

Поведение pipeline по переменным:
- Если задан `NGC_API_KEY` → используется образ Triton из `nvcr.io`
- Иначе, если заданы `GHCR_USERNAME/GHCR_TOKEN` → выполняется `docker login ghcr.io` (для образов из GHCR)
- Если ни то, ни другое не задано, pull приватных/ограниченных образов может падать (например, Triton из GHCR)

### Доступность сервисов
- Ingress NGINX деплоится и настраивается автоматически; домены доступны на `http://<host>:8080`
- Если домены `*.localtest.me` не резолвятся на твоей машине раннера, добавь в `/etc/hosts` запись `127.0.0.1 mlflow.localtest.me vllm.localtest.me triton.localtest.me`
- Всегда можно проверить сервисы через `kubectl port-forward`, минуя Ingress



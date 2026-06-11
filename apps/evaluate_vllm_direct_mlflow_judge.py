import os
import json
import time
from typing import Dict, Any, List, Optional

import pandas as pd
import mlflow
from openai import OpenAI

from mlflow.models import make_metric
from mlflow.metrics.base import MetricValue


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.1.200:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local-qwen-key")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen2.5-3b-instruct-gptq-int8")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.localtest.me",
)

# Если хотите включить встроенные QA-метрики MLflow, задайте:
# export MLFLOW_MODEL_TYPE=question-answering
#
# По умолчанию None, чтобы не скачивать лишние модели toxicity/readability,
# которые плохо подходят для русского языка.
MLFLOW_MODEL_TYPE = os.getenv("MLFLOW_MODEL_TYPE", "").strip() or None

SOURCE_EVAL_DF_FOR_JUDGE: Optional[pd.DataFrame] = None

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL,
)


EVAL_DATA = [
    {
        "context": (
            "В зрелой DevOps-практике инфраструктура описывается как код. "
            "Изменения в Terraform проходят code review, затем выполняется terraform plan. "
            "Команда применяет terraform apply только после проверки плана и согласования изменений."
        ),
        "question": "Зачем команда выполняет terraform plan перед terraform apply?",
        "ground_truth": (
            "Чтобы заранее увидеть планируемые изменения инфраструктуры, проверить их на code review "
            "и применять только согласованные изменения."
        ),
    },
    {
        "context": (
            "CI/CD pipeline должен автоматически запускать линтеры, unit-тесты и сборку контейнерного образа. "
            "Если тесты не проходят, pipeline должен остановиться и не публиковать образ в registry."
        ),
        "question": "Что должен сделать CI/CD pipeline, если unit-тесты не прошли?",
        "ground_truth": (
            "Он должен остановиться и не публиковать контейнерный образ в registry."
        ),
    },
    {
        "context": (
            "Blue-green deployment использует две одинаковые среды: blue и green. "
            "Новая версия выкатывается в неактивную среду, после проверки трафик переключается на неё. "
            "Если обнаружена проблема, трафик можно быстро вернуть на предыдущую среду."
        ),
        "question": "Как blue-green deployment помогает быстро откатить релиз?",
        "ground_truth": (
            "Трафик можно быстро вернуть на предыдущую рабочую среду, потому что старая версия остаётся доступной."
        ),
    },
    {
        "context": (
            "В Kubernetes liveness probe показывает, что контейнер завис и его нужно перезапустить. "
            "Readiness probe показывает, готов ли pod принимать пользовательский трафик. "
            "Если readiness probe не проходит, Kubernetes не должен направлять трафик на этот pod."
        ),
        "question": "Чем readiness probe отличается от liveness probe?",
        "ground_truth": (
            "Readiness probe определяет, готов ли pod принимать трафик, а liveness probe определяет, "
            "нужно ли перезапустить зависший контейнер."
        ),
    },
    {
        "context": (
            "Наблюдаемость в DevOps строится на трёх типах сигналов: метриках, логах и трассировках. "
            "Метрики помогают увидеть состояние системы во времени, логи дают детали событий, "
            "а трассировки показывают путь запроса через сервисы."
        ),
        "question": "Какие три типа сигналов используются для наблюдаемости?",
        "ground_truth": (
            "Метрики, логи и трассировки."
        ),
    },
    {
        "context": (
            "Секреты не должны храниться в Git-репозитории в открытом виде. "
            "Для production-среды команда использует secret manager, ограничивает доступ по ролям "
            "и регулярно ротирует токены и пароли."
        ),
        "question": "Почему нельзя хранить секреты в Git-репозитории в открытом виде?",
        "ground_truth": (
            "Потому что их могут увидеть посторонние; секреты нужно хранить в secret manager, "
            "ограничивать доступ и ротировать."
        ),
    },
    {
        "context": (
            "Immutable artifact означает, что один и тот же собранный контейнерный образ продвигается "
            "между dev, stage и prod. На каждом окружении не нужно пересобирать образ, меняются только "
            "конфигурация и параметры запуска."
        ),
        "question": "Что означает immutable artifact в CI/CD?",
        "ground_truth": (
            "Это означает, что один и тот же собранный артефакт или контейнерный образ продвигается "
            "между окружениями без пересборки."
        ),
    },
    {
        "context": (
            "После инцидента команда проводит blameless postmortem. "
            "Цель разбора - не найти виноватого, а понять причины сбоя, улучшить мониторинг, "
            "добавить автоматические проверки и зафиксировать конкретные action items."
        ),
        "question": "Какова цель blameless postmortem после инцидента?",
        "ground_truth": (
            "Понять причины сбоя без поиска виноватых и зафиксировать улучшения, такие как мониторинг, "
            "автоматические проверки и action items."
        ),
    },
]


def build_answer_prompt(context: str, question: str) -> str:
    return f"""
Ответьте на вопрос, используя только предоставленный контекст.

Правила:
- Не используйте внешние знания.
- Не добавляйте факты, которых нет в контексте.
- Если ответа нет в контексте, напишите: "Ответ отсутствует в контексте."
- Отвечайте кратко и по существу.

Контекст:
{context}

Вопрос:
{question}

Ответ:
""".strip()


def call_vllm(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> str:
    response = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    return "" if content is None else content.strip()


def generate_answer(context: str, question: str) -> Dict[str, Any]:
    prompt = build_answer_prompt(context, question)

    started_at = time.perf_counter()

    answer = call_vllm(
        messages=[
            {
                "role": "system",
                "content": (
                    "Вы опытный ассистент по DevOps-практикам. "
                    "Отвечайте только на основе переданного контекста. "
                    "Не выдумывайте детали."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=180,
    )

    latency = time.perf_counter() - started_at

    return {
        "inputs": prompt,
        "prediction": answer,
        "latency_seconds": latency,
    }


JUDGE_PROMPT_TEMPLATE = """
Ты судья с опытом более 10 лет в области LLM evaluator и ваш опыт наиболее компитентен в задачах ответов на вопросы по DevOps-практикам.

Оцени ответ модели на вопрос по шкале от 1 до 5.

Критерии оценки:ю
1 - Неверно. Ответ противоречит контексту, содержит галцинации или не отвечает на вопрос.
2 - В основном неверно. Есть небольшой релевантный фрагмент, но основной смысл ответа пропущен.
3 - Частично верно. Основная идея частично отражена, но есть важные пропуски или неподтверждённые детали.
4 - В основном верно. Ответ основан на контексте и отвечает на вопрос, но есть небольшая неточность или неполнота.
5 - Полностью верно. Ответ точный, полный, краткий и полностью основан на контексте.

Снижай оценку за:
- факты, которых нет в контексте;
- противоречие контексту;
- слишком общий или расплывчатый ответ;
- чрезмерную многословность;
- отсутствие ключевой информации.

Верните только валидный JSON без markdown и пояснений вокруг JSON:
{{
  "score": 1,
  "justification": "краткое обоснование оценки"
}}

Контекст:
{context}

Вопрос:
{question}

Эталонный ответ:
{ground_truth}

Ответ модели:
{prediction}
""".strip()


def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    """
    Qwen иногда добавляет текст вокруг JSON.
    Поэтому сначала пробуем json.loads, затем извлекаем JSON-подобный блок.
    """
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return {
                "score": 1,
                "justification": f"Судья вернул ответ не в JSON-формате: {raw_text[:300]}",
            }

        json_candidate = raw_text[start : end + 1]

        try:
            data = json.loads(json_candidate)
        except json.JSONDecodeError:
            return {
                "score": 1,
                "justification": f"Судья вернул невалидный JSON: {raw_text[:300]}",
            }

    try:
        score = int(data.get("score", 1))
    except (TypeError, ValueError):
        score = 1

    score = max(1, min(5, score))

    return {
        "score": score,
        "justification": str(data.get("justification", "")),
    }


def judge_one_row(row: pd.Series) -> Dict[str, Any]:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        context=row["context"],
        question=row["question"],
        ground_truth=row["ground_truth"],
        prediction=row["prediction"],
    )

    raw_judge_answer = call_vllm(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты строгий оценщик качества ответов. "
                    "Возвращай только валидный JSON. "
                    "Не добавляй markdown, списки или текст вне JSON."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    parsed = parse_judge_json(raw_judge_answer)
    parsed["raw_judge_answer"] = raw_judge_answer
    return parsed


def restore_judge_dataframe(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    MLflow default evaluator может передать в custom metric внутренний датафрейм
    без исходных колонок context/question/ground_truth.

    Поэтому восстанавливаем нужные поля из исходного eval_df,
    который сохраняется перед вызовом mlflow.evaluate().
    """
    global SOURCE_EVAL_DF_FOR_JUDGE

    required_columns = {"context", "question", "ground_truth", "prediction"}

    if required_columns.issubset(set(eval_df.columns)):
        return eval_df.copy()

    if SOURCE_EVAL_DF_FOR_JUDGE is None:
        raise ValueError(
            "SOURCE_EVAL_DF_FOR_JUDGE не инициализирован. "
            "Сохраните исходный eval_df перед вызовом mlflow.evaluate()."
        )

    try:
        restored_df = SOURCE_EVAL_DF_FOR_JUDGE.loc[eval_df.index].copy()
    except Exception:
        restored_df = SOURCE_EVAL_DF_FOR_JUDGE.iloc[: len(eval_df)].copy()

    restored_df = restored_df.reset_index(drop=True)
    internal_df = eval_df.reset_index(drop=True)

    if "prediction" in internal_df.columns:
        restored_df["prediction"] = internal_df["prediction"]

    if "target" in internal_df.columns:
        restored_df["ground_truth"] = internal_df["target"]

    return restored_df


def llm_as_judge_metric_fn(
    eval_df: pd.DataFrame,
    builtin_metrics: Dict[str, Any],
) -> MetricValue:
    judge_df = restore_judge_dataframe(eval_df)

    scores = []
    justifications = []

    for _, row in judge_df.iterrows():
        result = judge_one_row(row)
        scores.append(result["score"])
        justifications.append(result["justification"])

    score_series = pd.Series(scores, dtype="float")

    return MetricValue(
        scores=scores,
        justifications=justifications,
        aggregate_results={
            "mean": float(score_series.mean()),
            "median": float(score_series.median()),
            "min": float(score_series.min()),
            "max": float(score_series.max()),
            "p90": float(score_series.quantile(0.9)),
        },
    )


llm_as_judge_metric = make_metric(
    eval_fn=llm_as_judge_metric_fn,
    greater_is_better=True,
    name="make_genai_metric",
    long_name="LLM-as-a-Judge DevOps QA Quality RU",
    version="v1",
    metric_details=(
        "Кастомная make_genai_metric метрика для оценки ответов на русском языке "
        "по теме DevOps-практик. Оценивает корректность, groundedness, полноту, "
        "краткость и отсутствие галюцинаций по шкале от 1 до 5. "
        "Judge-модель вызывается через тот же vLLM OpenAI-compatible endpoint."
    ),
)


def log_endpoint_healthcheck() -> None:
    """
    Логирует список моделей vLLM как artifact.
    Это помогает показать, что evaluation был привязан к конкретному endpoint.
    """
    try:
        models_response = client.models.list()
        mlflow.log_text(str(models_response), "vllm_models_response.txt")
        mlflow.log_param("vllm_healthcheck_status", "ok")
    except Exception as exc:
        mlflow.log_param("vllm_healthcheck_status", "failed")
        mlflow.log_text(str(exc), "vllm_healthcheck_error.txt")


def main() -> None:
    global SOURCE_EVAL_DF_FOR_JUDGE

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("vllm-ru-llm-as-judge-eval")

    rows = []

    for item in EVAL_DATA:
        generated = generate_answer(
            context=item["context"],
            question=item["question"],
        )

        rows.append(
            {
                "question": item["question"],
                "context": item["context"],
                "ground_truth": item["ground_truth"],
                "inputs": generated["inputs"],
                "prediction": generated["prediction"],
                "latency_seconds": generated["latency_seconds"],
            }
        )

    eval_df = pd.DataFrame(rows)
    SOURCE_EVAL_DF_FOR_JUDGE = eval_df.copy()

    exact_match_manual = float(
        (
            eval_df["prediction"].str.strip()
            == eval_df["ground_truth"].str.strip()
        ).mean()
    )

    with mlflow.start_run(run_name="vllm-judge-eval"):
        mlflow.log_param("vllm_base_url", VLLM_BASE_URL)
        mlflow.log_param("vllm_model_name", VLLM_MODEL_NAME)
        mlflow.log_param("inference_temperature", 0.0)
        mlflow.log_param("judge_temperature", 0.0)
        mlflow.log_param("deployment_type", "vllm_openai_compatible")
        mlflow.log_param("judge_metric_name", "make_genai_metric")
        mlflow.log_param("dataset_language", "ru")
        mlflow.log_param("dataset_domain", "devops_practices")
        mlflow.log_param("eval_rows_count", len(eval_df))
        mlflow.log_param("mlflow_model_type", MLFLOW_MODEL_TYPE or "custom_metrics_only")

        log_endpoint_healthcheck()

        mlflow.log_metric("latency_mean_seconds", float(eval_df["latency_seconds"].mean()))
        mlflow.log_metric("latency_p90_seconds", float(eval_df["latency_seconds"].quantile(0.9)))
        mlflow.log_metric("exact_match_ru_manual", exact_match_manual)

        mlflow.log_table(
            data=eval_df,
            artifact_file="eval_predictions_ru.json",
        )

        mlflow.log_text(
            JUDGE_PROMPT_TEMPLATE,
            "judge_prompt_template_ru.txt",
        )

        mlflow.log_text(
            json.dumps(EVAL_DATA, ensure_ascii=False, indent=2),
            "eval_dataset_ru.json",
        )

        evaluate_kwargs = {
            "data": eval_df,
            "predictions": "prediction",
            "targets": "ground_truth",
            "evaluators": ["default"],
            "extra_metrics": [llm_as_judge_metric],
        }

        if MLFLOW_MODEL_TYPE is not None:
            evaluate_kwargs["model_type"] = MLFLOW_MODEL_TYPE

        result = mlflow.evaluate(**evaluate_kwargs)

        print("\n=== Evaluation metrics ===")
        for key, value in result.metrics.items():
            print(f"{key}: {value}")

        print("\n=== Predictions ===")
        print(
            eval_df[
                [
                    "question",
                    "prediction",
                    "ground_truth",
                    "latency_seconds",
                ]
            ]
        )


if __name__ == "__main__":
    main()

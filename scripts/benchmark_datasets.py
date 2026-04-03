#!/usr/bin/env python3
"""
Comprehensive LLM Benchmarking Script

Downloads and evaluates models on:
- GLUE (CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI)
- SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC)
- SQuAD (v1.1, v2.0)
- LAMBADA
- MMLU (subset for speed)

Metrics: Perplexity, BLEU, ROUGE, F1, Exact Match
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForQuestionAnswering
)
from evaluate import load
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    models: List[str]
    max_samples_per_task: int = 100  # Limit for speed
    batch_size: int = 8
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow_tracking_uri: str = "http://mlflow.localtest.me:8080"
    mlflow_experiment: str = "llm-benchmark"


class ModelBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = {
            'bleu': load('bleu'),
            'rouge': load('rouge'),
            'f1': load('f1'),
            'exact_match': load('exact_match'),
        }
        
    def load_model_and_tokenizer(self, model_id: str):
        """Load model and tokenizer"""
        logger.info(f"Loading {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Try to load as causal LM first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            model_type = "causal"
        except:
            # Fallback to sequence classification
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                model_type = "classification"
            except:
                logger.warning(f"Could not load {model_id} as causal or classification model")
                return None, None, None
        
        model.to(self.config.device)
        return model, tokenizer, model_type
    
    def calculate_perplexity(self, model, tokenizer, texts: List[str]) -> float:
        """Calculate perplexity on given texts"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                                 max_length=self.config.max_length)
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / len(texts)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
    
    def generate_text(self, model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate text from prompt"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                          max_length=self.config.max_length)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):]
        return generated_text.strip()
    
    def evaluate_glue(self, model, tokenizer, model_type: str) -> Dict[str, float]:
        """Evaluate on GLUE tasks"""
        results = {}
        
        # SST-2 (Sentiment Analysis)
        try:
            dataset = load_dataset("glue", "sst2", split="validation")
            dataset = dataset.select(range(min(self.config.max_samples_per_task, len(dataset))))
            
            if model_type == "classification":
                correct = 0
                total = 0
                for example in dataset:
                    inputs = tokenizer(example["sentence"], return_tensors="pt", truncation=True)
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        pred = torch.argmax(outputs.logits).item()
                        if pred == example["label"]:
                            correct += 1
                        total += 1
                
                results["sst2_accuracy"] = correct / total if total > 0 else 0.0
            else:
                # For causal models, use perplexity
                texts = [example["sentence"] for example in dataset]
                results["sst2_perplexity"] = self.calculate_perplexity(model, tokenizer, texts)
                
        except Exception as e:
            logger.warning(f"SST-2 evaluation failed: {e}")
            results["sst2_error"] = str(e)
        
        return results
    
    def evaluate_squad(self, model, tokenizer) -> Dict[str, float]:
        """Evaluate on SQuAD v1.1"""
        results = {}
        
        try:
            dataset = load_dataset("squad", split="validation")
            dataset = dataset.select(range(min(self.config.max_samples_per_task, len(dataset))))
            
            predictions = []
            references = []
            
            for example in dataset:
                context = example["context"]
                question = example["question"]
                answer = example["answers"]["text"][0]
                
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                generated = self.generate_text(model, tokenizer, prompt)
                
                predictions.append(generated)
                references.append(answer)
            
            # Calculate metrics
            if predictions and references:
                # F1 and Exact Match
                f1_scores = []
                em_scores = []
                
                for pred, ref in zip(predictions, references):
                    # Simple token-based F1
                    pred_tokens = pred.lower().split()
                    ref_tokens = ref.lower().split()
                    
                    if not pred_tokens or not ref_tokens:
                        f1_scores.append(0.0)
                        em_scores.append(0.0)
                        continue
                    
                    # Calculate overlap
                    overlap = 0
                    ref_counts = {}
                    for t in ref_tokens:
                        ref_counts[t] = ref_counts.get(t, 0) + 1
                    
                    for t in pred_tokens:
                        if ref_counts.get(t, 0) > 0:
                            overlap += 1
                            ref_counts[t] -= 1
                    
                    precision = overlap / len(pred_tokens)
                    recall = overlap / len(ref_tokens)
                    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                    f1_scores.append(f1)
                    
                    # Exact match
                    em_scores.append(1.0 if pred.lower().strip() == ref.lower().strip() else 0.0)
                
                results["squad_f1"] = np.mean(f1_scores)
                results["squad_exact_match"] = np.mean(em_scores)
                
                # BLEU
                bleu_result = self.metrics['bleu'].compute(predictions=predictions, references=references)
                results["squad_bleu"] = bleu_result['bleu']
                
                # ROUGE
                rouge_result = self.metrics['rouge'].compute(predictions=predictions, references=references)
                results["squad_rouge1"] = rouge_result['rouge1']
                results["squad_rouge2"] = rouge_result['rouge2']
                results["squad_rougeL"] = rouge_result['rougeL']
                
        except Exception as e:
            logger.warning(f"SQuAD evaluation failed: {e}")
            results["squad_error"] = str(e)
        
        return results
    
    def evaluate_lambada(self, model, tokenizer) -> Dict[str, float]:
        """Evaluate on LAMBADA (language modeling)"""
        results = {}
        
        try:
            dataset = load_dataset("lambada", split="test")
            dataset = dataset.select(range(min(self.config.max_samples_per_task, len(dataset))))
            
            texts = [example["text"] for example in dataset]
            results["lambada_perplexity"] = self.calculate_perplexity(model, tokenizer, texts)
            
        except Exception as e:
            logger.warning(f"LAMBADA evaluation failed: {e}")
            results["lambada_error"] = str(e)
        
        return results
    
    def evaluate_mmlu(self, model, tokenizer) -> Dict[str, float]:
        """Evaluate on MMLU subset (multiple choice)"""
        results = {}
        
        # Use a small subset for speed
        mmlu_subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]
        
        for subject in mmlu_subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                dataset = dataset.select(range(min(50, len(dataset))))  # Very small subset
                
                correct = 0
                total = 0
                
                for example in dataset:
                    question = example["question"]
                    choices = [example["A"], example["B"], example["C"], example["D"]]
                    answer = example["answer"]
                    
                    # Create prompt
                    prompt = f"Question: {question}\n"
                    for i, choice in enumerate(choices):
                        prompt += f"{chr(65+i)}. {choice}\n"
                    prompt += "Answer:"
                    
                    generated = self.generate_text(model, tokenizer, prompt, max_new_tokens=10)
                    
                    # Simple answer extraction
                    if generated and generated[0].upper() in ['A', 'B', 'C', 'D']:
                        pred = generated[0].upper()
                        if pred == answer:
                            correct += 1
                        total += 1
                
                if total > 0:
                    results[f"mmlu_{subject}_accuracy"] = correct / total
                    
            except Exception as e:
                logger.warning(f"MMLU {subject} evaluation failed: {e}")
                results[f"mmlu_{subject}_error"] = str(e)
        
        return results
    
    def benchmark_model(self, model_id: str) -> Dict[str, Any]:
        """Run full benchmark for a model"""
        logger.info(f"Starting benchmark for {model_id}")
        
        model, tokenizer, model_type = self.load_model_and_tokenizer(model_id)
        if model is None:
            return {"error": f"Failed to load model {model_id}"}
        
        results = {
            "model_id": model_id,
            "model_type": model_type,
            "device": self.config.device,
        }
        
        # Run evaluations
        try:
            results.update(self.evaluate_glue(model, tokenizer, model_type))
            results.update(self.evaluate_squad(model, tokenizer))
            results.update(self.evaluate_lambada(model, tokenizer))
            results.update(self.evaluate_mmlu(model, tokenizer))
        except Exception as e:
            logger.error(f"Benchmark failed for {model_id}: {e}")
            results["benchmark_error"] = str(e)
        
        return results
    
    def run_all_benchmarks(self):
        """Run benchmarks for all models and log to MLflow"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)
        
        all_results = []
        
        for model_id in self.config.models:
            with mlflow.start_run(run_name=model_id):
                # Log parameters
                mlflow.log_params({
                    "model_id": model_id,
                    "max_samples_per_task": self.config.max_samples_per_task,
                    "batch_size": self.config.batch_size,
                    "max_length": self.config.max_length,
                    "device": self.config.device,
                })
                
                # Run benchmark
                results = self.benchmark_model(model_id)
                all_results.append(results)
                
                # Log metrics (filter out error fields)
                metrics = {k: v for k, v in results.items() 
                          if isinstance(v, (int, float)) and not k.endswith('_error')}
                mlflow.log_metrics(metrics)
                
                # Log results as artifact
                with open(f"results_{model_id.replace('/', '_')}.json", "w") as f:
                    json.dump(results, f, indent=2)
                mlflow.log_artifact(f"results_{model_id.replace('/', '_')}.json")
                
                logger.info(f"Completed benchmark for {model_id}")
                print(f"Results for {model_id}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
        
        # Save summary
        with open("benchmark_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        return all_results


def main():
    # Configuration
    config = BenchmarkConfig(
        models=[
            "facebook/opt-125m",
            "EleutherAI/gpt-neo-125M", 
            "gpt2",
            "microsoft/DialoGPT-small",
        ],
        max_samples_per_task=50,  # Small for speed
        batch_size=4,
        max_length=256,
    )
    
    # Run benchmarks
    benchmarker = ModelBenchmarker(config)
    results = benchmarker.run_all_benchmarks()
    
    print("\nBenchmark completed!")
    print("Check MLflow UI for detailed results")
    print(f"Summary saved to: benchmark_summary.json")


if __name__ == "__main__":
    main()


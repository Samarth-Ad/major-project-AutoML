import requests
import json
from schema import Strategy
from utils.strategy_cache import StrategyCache

class StrategicLM:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.cache = StrategyCache()

    def build_prompt(self, profile):

        return f"""
You are an expert ML strategist.

CRITICAL RULES:
- Output STRICT JSON only
- No markdown
- No extra text outside JSON

Allowed models:
LogisticRegression, RandomForest, XGBoost

Allowed preprocessing:
mean_imputation, median_imputation, drop_column
one_hot, label_encoding
standard_scaler, none

Output format:

{{
  "preprocessing": {{
    "missing_value_handling": "",
    "encoding": "",
    "scaling": ""
  }},
  "model_candidates": [],
  "reasoning_summary": ""
}}

Dataset Profile:
Task Type: {profile['task_type']}
Rows: {profile['rows']}
Features: {profile['features']}
Numerical Features: {profile['numerical_features']}
Categorical Features: {profile['categorical_features']}
Missing Ratio: {profile['missing_ratio']}
Imbalance Ratio: {profile['imbalance_ratio']}
"""

    def extract_json(self, text):
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON found in model output")
        return text[start:end]

    def generate_strategy(self, profile, dataset_path=None):

    # ---------- Check Cache First ----------
        cached = self.cache.get_cached_strategy(dataset_path, profile)

        if cached:
            print("Strategy loaded from cache.")
            return cached

        # ---------- Call LLM ----------
        prompt = self.build_prompt(profile)

        response = requests.post(
            self.url,
            json={
                "model": "gpt-oss:120b-cloud",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2
            }
        )

        raw_output = response.json()["response"]
        json_text = self.extract_json(raw_output)
        strategy_dict = json.loads(json_text)

        strategy = Strategy.model_validate(strategy_dict)
        final_output = strategy.model_dump()

        # ---------- Save to Cache ----------
        file_path = self.cache.save_strategy(final_output, dataset_path, profile)
        print(f"Strategy saved at: {file_path}")

        return final_output

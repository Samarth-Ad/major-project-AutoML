import profile

import requests
import json
from schema import Strategy
from utils.strategy_cache import StrategyCache

class StrategicLM:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.cache = StrategyCache()

    def build_prompt(self, profile):

        import json
        profile_json = json.dumps(profile, indent=2)

        return f"""
You are an AutoML Planning Agent.

You must generate a valid JSON that matches EXACTLY this structure:

{{
  "task_type": "classification" or "regression",
  "target_column": string,

  "preprocessing": {{
    "missing": {{
      "strategy": "none" | "mean" | "median" | "mode",
      "threshold": float
    }},
    "encoding": {{
      "method": "label" | "one_hot" | "target",
      "drop_high_cardinality_ratio": float
    }},
    "scaling": {{
      "method": "none" | "standard" | "minmax"
    }},
    "skew_handling": {{
      "enabled": true or false,
      "threshold": float,
      "method": "log1p"
    }},
    "correlation_pruning": {{
      "enabled": true or false,
      "threshold": float
    }}
  }},

  "modeling": {{
    "candidates": [
      {{
        "name": string,
        "parameters": object
      }}
    ],
    "cross_validation_folds": int,
    "evaluation_metric": string,
    "split": {{
      "test_size": float,
      "random_state": int,
      "stratified": true or false
    }}
  }}
}}

Rules:
- Output ONLY valid JSON.
- No markdown.
- No explanation.
- No comments.
- Must be valid JSON.

Dataset profile (JSON):
{profile_json}
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

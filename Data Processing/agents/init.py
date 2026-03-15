"""
agents/__init__.py
------------------
The agents package now contains only one class: DynamicAgent.

Every pipeline step — load_dataset, remove_missing_values,
encode_categorical, normalize_features, feature_engineering,
train_model, or ANY custom step — is handled by the same class.

The intelligence lives in the LLM, not in Python code.

Usage
-----
    from agents.base_agent import DynamicAgent

    agent = DynamicAgent(
        step_name="remove_missing_values",
        pipeline_steps=["load_dataset", "remove_missing_values", "train_model"]
    )
    result = agent.execute(df)
    print(result["reasoning"])       # LLM explains its strategy
    print(result["code_equivalent"]) # exact code that was run
"""
from agents.base_agent import DynamicAgent, BaseAgent, AgentError, AgentRetryError

__all__ = ["DynamicAgent", "BaseAgent", "AgentError", "AgentRetryError"]
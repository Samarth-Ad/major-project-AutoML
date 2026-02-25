from dataset_profiler import profile_dataset
from strategic_lm import StrategicLM
from schema import Strategy
from strategy_executor import StrategyExecutor
import json

datasets = [
    "data/house_price.csv"
]

lm = StrategicLM()

for file in datasets:

    print("\n==============================")
    print(f"Analyzing: {file}")

    profile = profile_dataset(file)
    print("Dataset Profile:")
    print(profile)
    if profile["target_column"] is None:
        print("\nNo suitable target column detected.")
        print("Skipping modeling step.")
        continue
    raw_strategy = lm.generate_strategy(profile, dataset_path=file)

    try:
        strategy = Strategy(**raw_strategy)
    except Exception as e:
        print("Strategy validation failed:", e)
        continue

    print("\nValidated Strategy:")
    print(json.dumps(strategy.model_dump(), indent=2))

    # --------------------------------------------------
    # EXECUTE STRATEGY
    # --------------------------------------------------
    print("\nExecuting Strategy...")

    executor = StrategyExecutor(file, strategy)
    results = executor.execute()

    print("\nModel Results:")
    for result in results:
        print(f"{result['model']} → Score: {round(result['score'], 4)}")
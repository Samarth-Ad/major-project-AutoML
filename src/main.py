from dataset_profiler import profile_dataset
from strategic_lm import StrategicLM

datasets = [
    # "data/india_city_aqi_2015_2023.csv",
    # "data/stock_prices_daily.csv",
    "data/train.csv"
]

lm = StrategicLM()

for file in datasets:

    print("\n==============================")
    print(f"Analyzing: {file}")

    profile = profile_dataset(file)
    print("Dataset Profile:")
    print(profile)

    strategy = lm.generate_strategy(profile)

    print("\nGenerated Strategy:")
    print(strategy)
    print("\nReasoning:")
    print(strategy["reasoning_summary"])

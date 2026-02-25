import os
import json
import hashlib
from typing import Dict, Any


class StrategyCache:
    """
    Handles saving and loading of strategy JSON responses.
    Prevents duplicate generation for the same dataset.
    """

    def __init__(self, base_dir: str = "src/strat_responses"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _generate_hash(self, dataset_path: str | None, profile: Dict[str, Any] | None) -> str:
        """
        Generate deterministic hash using dataset path (preferred)
        or dataset profile as fallback.
        """
        if dataset_path:
            unique_string = dataset_path
        else:
            unique_string = json.dumps(profile, sort_keys=True)

        return hashlib.sha256(unique_string.encode()).hexdigest()[:12]

    def _build_filename(self, dataset_path: str | None, profile: Dict[str, Any] | None) -> str:
        """
        Create unique filename using dataset name + hash.
        """
        hash_id = self._generate_hash(dataset_path, profile)

        if dataset_path:
            dataset_name = os.path.basename(dataset_path).replace(".csv", "")
        else:
            dataset_name = "dataset"

        return f"{dataset_name}__{hash_id}.json"

    def get_cached_strategy(self, dataset_path: str | None, profile: Dict[str, Any] | None):
        """
        Returns cached strategy if exists, else None.
        """
        filename = self._build_filename(dataset_path, profile)
        file_path = os.path.join(self.base_dir, filename)

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)

        return None

    def save_strategy(self, strategy: Dict[str, Any], dataset_path: str | None, profile: Dict[str, Any] | None):
        """
        Saves strategy JSON to disk.
        """
        filename = self._build_filename(dataset_path, profile)
        file_path = os.path.join(self.base_dir, filename)

        with open(file_path, "w") as f:
            json.dump(strategy, f, indent=4)

        return file_path

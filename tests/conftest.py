import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# Canonical Kaggle Titanic train.csv mirror (891 rows x 12 cols, target='Survived').
# OpenML's titanic datasets ship different schemas (id 40704 is 2201x4 aggregated;
# id 40945 is 1309x14 combined train+test with lowercase 'survived'), so this stable
# mirror is used to satisfy the test fixture's strict shape and column-name asserts.
_TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
_TITANIC_CACHE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "experiments"
    / "titanic"
    / "titanic.csv"
)


@pytest.fixture(scope="session")
def titanic_df() -> pd.DataFrame:
    if not _TITANIC_CACHE.exists():
        _TITANIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(_TITANIC_URL)
        df.to_csv(_TITANIC_CACHE, index=False)
    return pd.read_csv(_TITANIC_CACHE)

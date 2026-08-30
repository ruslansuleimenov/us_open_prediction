"""Project paths anchored to the package location, not the working directory."""

from pathlib import Path

# <root>/src/usopen/paths.py -> parents[2] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
OUTPUTS = PROJECT_ROOT / "outputs"

MATCHES_CSV = DATA_RAW / "atp_tennis.csv"

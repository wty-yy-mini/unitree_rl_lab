from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SOURCE_ROOT.parent.parent
UNITREE_MODEL_DIR = REPO_ROOT / "unitree_model"

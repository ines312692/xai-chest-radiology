"""Make the package importable when a script is run directly.

On Kaggle the repository is copied next to the notebook and nothing is pip installed,
so the parent folder of this file is added to the import path. After an editable
install this module does nothing harmful and can stay.
"""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

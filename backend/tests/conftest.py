"""Test configuration.

Point the app at a throwaway data directory BEFORE any app module imports
``app.settings`` (settings are cached at import time). This keeps tests from
touching the real ``data/`` folder / DB.
"""

import os
import tempfile

os.environ.setdefault(
    "WRINKLE_DATA_DIR", tempfile.mkdtemp(prefix="wrinkle_test_")
)

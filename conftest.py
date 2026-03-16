"""
Root conftest — ensures the backend package is importable during tests.
"""
import sys
from pathlib import Path

# Add backend/ to sys.path so all backend modules are importable
sys.path.insert(0, str(Path(__file__).parent / "backend"))

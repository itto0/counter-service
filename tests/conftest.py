"""Pytest configuration for counter service tests.

Ensures proper sys.path and module aliasing whether run from the monorepo root
or directly from within standalone counter-service repo.
"""

from pathlib import Path
import sys
import types

COUNTER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = COUNTER_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(COUNTER_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTER_DIR))

# Ensure services and services.counter are importable in standalone repo mode
if "services" not in sys.modules:
    _services = types.ModuleType("services")
    _services.__path__ = []
    sys.modules["services"] = _services
else:
    _services = sys.modules["services"]

if "services.counter" not in sys.modules:
    _counter = types.ModuleType("services.counter")
    _counter.__path__ = [str(COUNTER_DIR)]
    _counter.__file__ = str(COUNTER_DIR / "__init__.py")
    sys.modules["services.counter"] = _counter
    _services.counter = _counter

# Fallback for shared contracts if running in standalone counter-service repo
if "shared.contracts" not in sys.modules:
    try:
        from src import contracts as _contracts
    except ImportError:
        import contracts as _contracts  # type: ignore

    _shared = types.ModuleType("shared")
    _shared.contracts = _contracts
    sys.modules["shared"] = _shared
    sys.modules["shared.contracts"] = _contracts



"""Local SQLite outbox storage."""

try:
    from .db import LocalStorage
except ImportError:
    try:
        from src.storage.db import LocalStorage
    except ImportError:
        from src.storage.db import LocalStorage

__all__ = ["LocalStorage"]

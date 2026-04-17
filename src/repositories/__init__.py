"""Repository layer — data access abstractions.

The existing PositionStore (src/models/position_store.py) serves as the
primary repository. This package provides re-exports and future expansion
points for signal_repository, trade_repository, etc.
"""

# Re-export the existing position store as the primary repository
from src.models.position_store import PositionStore

__all__ = ["PositionStore"]

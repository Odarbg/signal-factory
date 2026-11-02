"""Column roles that the signal spec enforces (spec §4)."""

from __future__ import annotations

ENTITY_KEYS: frozenset[str] = frozenset(
    {
        "uid",
        "card1",
        "card1_addr1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "id_30",
        "id_31",
        "addr1",
    }
)
"""Cross-row keys a signal may group by."""

FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {"isFraud", "TransactionID", "TransactionDT", "uid", "day", "card1_addr1"}
)
"""Never a signal ``column`` or ``filter`` column: the label, identifiers and derived keys."""

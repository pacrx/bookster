from typing import Final, FrozenSet

ALLOWED_REL_TYPES: Final[FrozenSet[str]] = frozenset({
    "KNOWS",
    "OWNS",
    "ALLY_OF",
    "ENEMY_OF",
    "LOCATED_IN",
    "MEMBER_OF",
    "HAS_TITLE",
})
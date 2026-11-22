"""Canonical record identity helpers."""

from typing import Any


def record_key(record: dict[str, Any]) -> tuple[str, str, str, int | str]:
    """Return the stable composite key for one inference record."""
    return (
        str(record["question_group"]),
        str(record["sample_id"]),
        str(record["version"]),
        record["template"],
    )


def record_id(record: dict[str, Any]) -> str:
    """Return a readable, deterministic identifier for one inference record."""
    group, sample, version, template = record_key(record)
    return f"{group}/{sample}/{version}/template-{template}"

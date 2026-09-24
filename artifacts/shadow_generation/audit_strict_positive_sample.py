#!/usr/bin/env python3
"""Recheck a slot-conditioned atlas from its saved PlanarMap rotation systems."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from structured_slot_detector import detect_structured_opened_trefoil_slots


def audit(ledger_path: Path, html_path: Path) -> dict:
    """Return a fresh exact-certificate audit of a saved publication sample."""
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    displayed = ledger["displayed"]
    if len(displayed) != ledger["target_panels"]:
        raise AssertionError("ledger panel count does not match target")

    html_counts = [
        int(value)
        for value in re.findall(
            r'<div class="usa-label">\d{2} <span>(\d+) certified slots?</span>',
            html,
        )
    ]
    overlay_count = html.count('class="usa-slot"')
    mask_count = html.count('class="usa-slot-mask"')
    strand_count = html.count('class="usa-slot-strand"')

    records = []
    for panel in displayed:
        certificates = detect_structured_opened_trefoil_slots(panel["rotation_system"])
        expected = panel["certificates"]
        if len(certificates) != len(expected):
            raise AssertionError(
                f"panel {panel['panel']}: detector count changed "
                f"from {len(expected)} to {len(certificates)}"
            )
        for certificate in certificates:
            if certificate["rotation_signature"] not in ((0, 1, 0), (1, 0, 1)):
                raise AssertionError("invalid structured-slot rotation signature")
            if certificate["pair_multiplicities"] != (1, 2, 2):
                raise AssertionError("invalid internal edge multiplicities")
        records.append(
            {
                "panel": panel["panel"],
                "seed": panel["seed"],
                "certified_count": len(certificates),
                "rotation_signatures": [
                    list(item["rotation_signature"]) for item in certificates
                ],
            }
        )

    counts = [record["certified_count"] for record in records]
    if counts != ledger["structured_slot_counts"] or counts != html_counts:
        raise AssertionError(
            f"count mismatch ledger={ledger['structured_slot_counts']} "
            f"html={html_counts} audit={counts}"
        )
    if overlay_count != sum(counts):
        raise AssertionError("orange overlay count does not match certificates")
    if mask_count != overlay_count or strand_count != overlay_count:
        raise AssertionError("each certificate must have one mask and canonical redraw")

    return {
        "status": "PASS",
        "detector": ledger["detector"],
        "selection_policy": ledger["selection_policy"],
        "proposals_examined": ledger["proposals_examined"],
        "one_component_proposals": ledger["one_component_proposals"],
        "certified_counts": counts,
        "certified_total": sum(counts),
        "html_overlay_total": overlay_count,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--html", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = audit(args.ledger, args.html)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

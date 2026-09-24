#!/usr/bin/env python3
"""Positive and adversarial tests for the structured slot detector."""

from __future__ import annotations

from structured_slot_detector import (
    certificates_from_gauss_word,
    detect_structured_opened_trefoil_slots,
    legacy_broad_candidates_from_gauss_word,
)


def rotations_from_word(
    word: list[int], local_signature: tuple[int, int, int]
) -> list[list[int]]:
    """Build a transition-compatible rotation system for a detector test."""

    occurrences: dict[int, list[int]] = {}
    for position, vertex in enumerate(word):
        occurrences.setdefault(vertex, []).append(position)
    if any(len(positions) != 2 for positions in occurrences.values()):
        raise ValueError("every test label must occur twice")
    n = len(word)
    bits = {0: local_signature[0], 1: local_signature[1], 2: local_signature[2]}
    rotations: list[list[int]] = []
    for vertex in range(max(word) + 1):
        first, second = occurrences[vertex]

        def incoming(position: int) -> int:
            return -(((position - 1) % n) + 1)

        def outgoing(position: int) -> int:
            return position + 1

        if bits.get(vertex, 0) == 0:
            row = [
                incoming(first),
                incoming(second),
                outgoing(first),
                outgoing(second),
            ]
        else:
            row = [
                incoming(first),
                outgoing(second),
                outgoing(first),
                incoming(second),
            ]
        rotations.append(row)
    return rotations


def assert_word_count(word: list[int], expected: int) -> None:
    actual = len(certificates_from_gauss_word(word))
    if actual != expected:
        raise AssertionError(f"word {word} gives {actual} candidates, expected {expected}")


def main() -> None:
    word = [0, 1, 2, 0, 1, 2, 3, 4, 3, 4]

    # The Gauss-word prefilter accepts abcabc and its reversal, but does not
    # claim to have reconstructed a missing ambient rotation system.
    assert_word_count(word, 1)
    assert_word_count([0, 2, 1, 0, 2, 1, 3, 4, 3, 4], 1)

    # Full positive controls: exactly the two mirror plane rotation signatures.
    for signature, orientation in (((0, 1, 0), "standard"), ((1, 0, 1), "mirror")):
        hits = detect_structured_opened_trefoil_slots(
            rotations_from_word(word, signature)
        )
        if len(hits) != 1:
            raise AssertionError((signature, hits))
        hit = hits[0]
        if hit["rotation_signature"] != signature:
            raise AssertionError(hit)
        if hit["orientation_class"] != orientation:
            raise AssertionError(hit)
        if hit["pair_multiplicities"] != (1, 2, 2):
            raise AssertionError(hit)
        if hit["incoming_edge"] == hit["outgoing_edge"]:
            raise AssertionError(hit)

    # Same word and abstract doubled triangle, wrong cyclic rows: reject.
    for signature in (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
    ):
        hits = detect_structured_opened_trefoil_slots(
            rotations_from_word(word, signature)
        )
        if hits:
            raise AssertionError((signature, hits))

    # Adversarial six-visit blocks admitted by the retired broad detector.
    for block in (
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 2, 2, 1],
        [0, 1, 1, 0, 2, 2],
        [0, 1, 2, 2, 1, 0],
    ):
        adversarial = block + [3, 4, 3, 4]
        assert_word_count(adversarial, 0)
        if not legacy_broad_candidates_from_gauss_word(adversarial):
            raise AssertionError(f"negative control {block} no longer triggers legacy detector")

    print("PASS: full detector accepts exactly the 010/101 mirror rotation signatures")
    print("PASS: full detector rejects all six non-trefoil cyclic-row signatures")
    print("PASS: word prefilter rejects four looped/non-trefoil broad-filter controls")
    print("PASS: each full certificate records the exact cut, 2,2,1 carrier, and pattern side")


if __name__ == "__main__":
    main()

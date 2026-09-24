#!/usr/bin/env python3
"""Exact detector for structured opened-trefoil subshadows.

The target is a one-strand, two-leg, three-crossing carrier.  Along the
straight-ahead component its six crossing visits have cyclic word ``abcabc``
with ``a``, ``b`` and ``c`` distinct.  The implementation does not stop at
that mnemonic: each candidate is checked against the complete ambient edge
ledger, the exact two-edge cut, the loop-free doubled-triangle multiplicities,
and the opposite-half-edge transition at every visit.

The module contains no sampling code and has no third-party dependency.  It is
used both by the archived PlanarMap session script and by the audit of the
frozen SVG panels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SlotCertificate:
    """Combinatorial certificate for one structured opened-trefoil carrier."""

    start: int
    vertices: tuple[int, int, int]
    gauss_block: tuple[int, int, int, int, int, int]
    incoming_edge: int
    outgoing_edge: int
    internal_edges: tuple[int, int, int, int, int]
    pair_multiplicities: tuple[int, int, int]
    rotation_signature: tuple[int, int, int] | None = None
    orientation_class: str | None = None
    pattern_side: str = "carrier side of the exact two-edge cut"
    certificate_level: str = "gauss_word"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def edge_vertices(rotations: Sequence[Sequence[int]]) -> dict[int, tuple[int, int]]:
    """Return the two endpoints of every signed-dart edge label."""

    endpoints: dict[int, list[int | None]] = defaultdict(lambda: [None, None])
    for vertex, row in enumerate(rotations):
        if len(row) != 4:
            raise ValueError(f"vertex {vertex} has {len(row)} darts, expected four")
        for dart in row:
            if dart == 0:
                raise ValueError("dart labels must be nonzero")
            slot = 0 if dart > 0 else 1
            label = abs(dart)
            if endpoints[label][slot] is not None:
                raise ValueError(f"signed dart {dart} occurs more than once")
            endpoints[label][slot] = vertex
    result: dict[int, tuple[int, int]] = {}
    for label, pair in endpoints.items():
        if pair[0] is None or pair[1] is None:
            raise ValueError(f"edge {label} is missing one signed dart")
        result[label] = (int(pair[0]), int(pair[1]))
    return result


def straight_ahead_components(
    rotations: Sequence[Sequence[int]],
) -> list[tuple[list[int], list[int]]]:
    """Return each straight-ahead vertex word and its outgoing edge labels.

    If ``vertices[k] == v``, then ``edges[k]`` is the edge traversed after
    passing straight through ``v``; it joins ``vertices[k]`` to
    ``vertices[k+1]`` cyclically.
    """

    position: dict[int, tuple[int, int]] = {}
    for vertex, row in enumerate(rotations):
        if len(row) != 4:
            raise ValueError(f"vertex {vertex} has {len(row)} darts, expected four")
        for index, dart in enumerate(row):
            if dart in position:
                raise ValueError(f"signed dart {dart} occurs more than once")
            position[dart] = (vertex, index)

    for dart in tuple(position):
        if -dart not in position:
            raise ValueError(f"signed dart {dart} has no opposite endpoint {-dart}")

    seen_edges: set[int] = set()
    components: list[tuple[list[int], list[int]]] = []
    for initial in position:
        if abs(initial) in seen_edges:
            continue
        current = initial
        word: list[int] = []
        traversed_edges: list[int] = []
        while abs(current) not in seen_edges:
            seen_edges.add(abs(current))
            vertex, index = position[current]
            through_dart = rotations[vertex][(index + 2) % 4]
            word.append(vertex)
            traversed_edges.append(abs(through_dart))
            current = -through_dart
        components.append((word, traversed_edges))

    if sum(len(edges) for _, edges in components) != len(position) // 2:
        raise AssertionError("straight-ahead traversal did not exhaust the edge set")
    return components


def _cyclic(values: Sequence[int], start: int, length: int) -> list[int]:
    n = len(values)
    return [values[(start + offset) % n] for offset in range(length)]


def _opposite_in_rotation(row: Sequence[int], edge_a: int, edge_b: int) -> bool:
    positions_a = [i for i, dart in enumerate(row) if abs(dart) == edge_a]
    positions_b = [i for i, dart in enumerate(row) if abs(dart) == edge_b]
    return any((i - j) % 4 == 2 for i in positions_a for j in positions_b)




def _unique_dart_for_edge(row: Sequence[int], edge: int) -> int | None:
    matches = [dart for dart in row if abs(dart) == edge]
    return matches[0] if len(matches) == 1 else None


def _rotation_signature(
    *,
    word: Sequence[int],
    traversal_edges: Sequence[int],
    start: int,
    rotations: Sequence[Sequence[int]],
) -> tuple[int, int, int] | None:
    """Return the three-bit rotation signature of an ``abcabc`` block.

    At each of the ordered vertices ``a,b,c``, start with the incoming dart
    of the first visit.  The next dart in the ambient cyclic order is either
    the incoming dart of the second visit (bit 0) or its outgoing dart
    (bit 1).  For the plane opened-trefoil carrier the two mirror signatures
    are ``010`` and ``101``.
    """

    n = len(word)
    bits: list[int] = []
    for offset in range(3):
        first = (start + offset) % n
        second = (start + offset + 3) % n
        vertex = word[first]
        if word[second] != vertex:
            return None
        row = rotations[vertex]
        incoming_first = traversal_edges[(first - 1) % n]
        incoming_second = traversal_edges[(second - 1) % n]
        outgoing_second = traversal_edges[second]
        dart_first = _unique_dart_for_edge(row, incoming_first)
        dart_second_in = _unique_dart_for_edge(row, incoming_second)
        dart_second_out = _unique_dart_for_edge(row, outgoing_second)
        if None in (dart_first, dart_second_in, dart_second_out):
            return None
        index = row.index(dart_first)
        next_dart = row[(index + 1) % 4]
        if next_dart == dart_second_in:
            bits.append(0)
        elif next_dart == dart_second_out:
            bits.append(1)
        else:
            return None
    return tuple(bits)  # type: ignore[return-value]


def _certify_candidate(
    *,
    word: Sequence[int],
    traversal_edges: Sequence[int],
    endpoints: Mapping[int, tuple[int, int]],
    start: int,
    rotations: Sequence[Sequence[int]] | None,
) -> SlotCertificate | None:
    n = len(word)
    if n < 7 or len(traversal_edges) != n:
        return None

    block = _cyclic(word, start, 6)
    if len(set(block[:3])) != 3 or block[:3] != block[3:]:
        return None
    vertices = frozenset(block[:3])

    # The six visits must be all visits to the three vertices.  In a
    # one-component four-regular shadow each crossing occurs twice, but this
    # explicit check prevents an incomplete carrier from being certified.
    positions = {i for i, vertex in enumerate(word) if vertex in vertices}
    block_positions = {(start + offset) % n for offset in range(6)}
    if positions != block_positions:
        return None

    incoming = traversal_edges[(start - 1) % n]
    internal_from_traversal = tuple(_cyclic(traversal_edges, start, 5))
    outgoing = traversal_edges[(start + 5) % n]
    if incoming == outgoing:
        return None

    internal = sorted(
        label
        for label, (u, v) in endpoints.items()
        if u in vertices and v in vertices
    )
    boundary = sorted(
        label
        for label, (u, v) in endpoints.items()
        if (u in vertices) ^ (v in vertices)
    )
    if len(internal) != 5 or set(internal) != set(internal_from_traversal):
        return None
    if boundary != sorted((incoming, outgoing)):
        return None

    # The opened trefoil carrier is loop-free and has edge multiplicities
    # 2,2,1 between its three vertex pairs.  This excludes the looped
    # three-vertex components admitted by the earlier broad detector.
    pair_counts: Counter[tuple[int, int]] = Counter()
    for label in internal:
        u, v = endpoints[label]
        if u == v:
            return None
        pair_counts[tuple(sorted((u, v)))] += 1
    multiplicities = tuple(sorted(pair_counts.values()))
    if len(pair_counts) != 3 or multiplicities != (1, 2, 2):
        return None

    # Verify directly that every incoming/outgoing pair used by the local
    # interval is opposite in the ambient rotation system.  The additional
    # three-bit signature distinguishes the two mirror plane embeddings of
    # the structured opened-trefoil carrier from the other abstract cyclic
    # orders on the same doubled-triangle multigraph.
    rotation_signature: tuple[int, int, int] | None = None
    orientation_class: str | None = None
    certificate_level = "gauss_word"
    if rotations is not None:
        for offset in range(6):
            index = (start + offset) % n
            vertex = word[index]
            edge_in = traversal_edges[(index - 1) % n]
            edge_out = traversal_edges[index]
            if not _opposite_in_rotation(rotations[vertex], edge_in, edge_out):
                return None
        rotation_signature = _rotation_signature(
            word=word,
            traversal_edges=traversal_edges,
            start=start,
            rotations=rotations,
        )
        if rotation_signature not in ((0, 1, 0), (1, 0, 1)):
            return None
        orientation_class = (
            "standard" if rotation_signature == (0, 1, 0) else "mirror"
        )
        certificate_level = "full_rotation_system"

    return SlotCertificate(
        start=start,
        vertices=tuple(sorted(vertices)),
        gauss_block=tuple(block),
        incoming_edge=incoming,
        outgoing_edge=outgoing,
        internal_edges=tuple(internal_from_traversal),
        pair_multiplicities=multiplicities,
        rotation_signature=rotation_signature,
        orientation_class=orientation_class,
        certificate_level=certificate_level,
    )


def detect_structured_opened_trefoil_slots(
    rotations: Sequence[Sequence[int]],
) -> list[dict[str, object]]:
    """Detect and certify all structured opened-trefoil carriers.

    Only one-component shadows are eligible.  Duplicates obtained from a
    cyclic reparametrization are suppressed by the three-vertex carrier.
    The returned dictionaries retain the keys expected by the historical
    drawing code: ``vertices`` and ordered ``boundary`` edges.
    """

    components = straight_ahead_components(rotations)
    if len(components) != 1:
        return []
    word, traversal_edges = components[0]
    endpoints = edge_vertices(rotations)

    hits: list[dict[str, object]] = []
    seen: set[frozenset[int]] = set()
    for start in range(len(word)):
        certificate = _certify_candidate(
            word=word,
            traversal_edges=traversal_edges,
            endpoints=endpoints,
            start=start,
            rotations=rotations,
        )
        if certificate is None:
            continue
        carrier = frozenset(certificate.vertices)
        if carrier in seen:
            continue
        seen.add(carrier)
        record = certificate.to_dict()
        record["vertices"] = carrier
        record["boundary"] = (
            certificate.incoming_edge,
            certificate.outgoing_edge,
        )
        hits.append(record)
    return hits


def certificates_from_gauss_word(word: Sequence[int]) -> list[SlotCertificate]:
    """Certify structured slots from a frozen straight-ahead Gauss word.

    Positional edge labels ``0,...,len(word)-1`` are used.  The word test is
    a complete prefilter for the carrier and exact cut, but it does not claim
    to reconstruct the ambient rotation system.  This distinction is enough
    for the frozen atlas audit: every displayed word fails already at the
    necessary ``abcabc`` gate, so no positive certificate depends on missing
    rotation rows.
    """

    n = len(word)
    if n == 0:
        return []
    traversal_edges = list(range(n))
    endpoints = {
        edge: (word[edge], word[(edge + 1) % n]) for edge in range(n)
    }
    hits: list[SlotCertificate] = []
    seen: set[frozenset[int]] = set()
    for start in range(n):
        certificate = _certify_candidate(
            word=word,
            traversal_edges=traversal_edges,
            endpoints=endpoints,
            start=start,
            rotations=None,
        )
        if certificate is None:
            continue
        carrier = frozenset(certificate.vertices)
        if carrier not in seen:
            seen.add(carrier)
            hits.append(certificate)
    return hits


def legacy_broad_candidates_from_gauss_word(
    word: Sequence[int],
) -> list[dict[str, object]]:
    """Reproduce the retired broad three-vertex/two-edge test.

    This function exists only for negative-control audits.  Its output must
    never be interpreted as a trefoil-slot certificate.
    """

    vertices = set(word)
    edges = {
        edge: (word[edge], word[(edge + 1) % len(word)])
        for edge in range(len(word))
    }
    hits: dict[frozenset[int], dict[str, object]] = {}
    for edge_1, edge_2 in combinations(edges, 2):
        adjacency: dict[int, set[int]] = defaultdict(set)
        for label, (u, v) in edges.items():
            if label in (edge_1, edge_2):
                continue
            adjacency[u].add(v)
            adjacency[v].add(u)

        unseen = set(vertices)
        components: list[set[int]] = []
        while unseen:
            initial = unseen.pop()
            component = {initial}
            queue = [initial]
            for u in queue:
                for v in adjacency[u]:
                    if v in unseen:
                        unseen.remove(v)
                        component.add(v)
                        queue.append(v)
            components.append(component)

        for component in components:
            if len(component) != 3:
                continue
            boundary = [
                label
                for label in (edge_1, edge_2)
                if (edges[label][0] in component) ^ (edges[label][1] in component)
            ]
            if len(boundary) != 2:
                continue
            internal = [
                label
                for label, (u, v) in edges.items()
                if u in component and v in component
            ]
            if len(internal) != 5:
                continue
            degrees = {vertex: 0 for vertex in component}
            loops: list[int] = []
            pair_counts: Counter[tuple[int, int]] = Counter()
            for label in internal:
                u, v = edges[label]
                if u == v:
                    degrees[u] += 2
                    loops.append(label)
                else:
                    degrees[u] += 1
                    degrees[v] += 1
                pair_counts[tuple(sorted((u, v)))] += 1
            for label in boundary:
                u, v = edges[label]
                degrees[u if u in component else v] += 1
            if not all(degrees[vertex] == 4 for vertex in component):
                continue
            carrier = frozenset(component)
            hits[carrier] = {
                "vertices": sorted(component),
                "boundary": boundary,
                "internal_edges": internal,
                "loops": loops,
                "pair_multiplicities": sorted(pair_counts.values()),
                "restricted_word": [vertex for vertex in word if vertex in component],
            }
    return [hits[key] for key in sorted(hits, key=lambda item: tuple(sorted(item)))]


def count_labels(word: Iterable[int]) -> Counter[int]:
    """Small public helper used by audit scripts."""

    return Counter(word)

#!/usr/bin/env python3
"""Exact finite-population workload calculus for adaptive fingerprint cascades.

The module implements the exact formulas used in the Supporting Information.  All theorem-level
quantities are computed with Python integers and ``fractions.Fraction``.
The source-controlled benchmark is mathematical, not an empirical knot census.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence


def _comb(n: int, k: int) -> int:
    """Return binomial(n,k), interpreted as zero outside 0 <= k <= n."""
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def normalize_profile(profile: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in profile if int(x) > 0)
    if not values:
        raise ValueError("a fiber profile must contain at least one positive size")
    return tuple(sorted(values, reverse=True))


def expected_singleton_objects(profile: Iterable[int], sample_size: int) -> Fraction:
    """Expected number of sampled objects in singleton sample buckets.

    A population fiber of size m contributes m*C(T-m,M-1)/C(T,M).
    """
    profile = normalize_profile(profile)
    t = sum(profile)
    m = int(sample_size)
    if not 0 <= m <= t:
        raise ValueError("sample_size must lie between 0 and the population size")
    if m == 0:
        return Fraction(0, 1)
    denominator = _comb(t, m)
    numerator = sum(size * _comb(t - size, m - 1) for size in profile)
    return Fraction(numerator, denominator)


def expected_surviving_objects(profile: Iterable[int], sample_size: int) -> Fraction:
    """Expected sampled objects lying in nonsingleton fingerprint buckets."""
    return Fraction(sample_size, 1) - expected_singleton_objects(profile, sample_size)


def population_unresolved_pairs(profile: Iterable[int]) -> int:
    profile = normalize_profile(profile)
    return sum(_comb(size, 2) for size in profile)


def expected_unresolved_pairs(profile: Iterable[int], sample_size: int) -> Fraction:
    """Expected equal-fingerprint pairs in a uniform sample without replacement."""
    profile = normalize_profile(profile)
    t = sum(profile)
    m = int(sample_size)
    if not 0 <= m <= t:
        raise ValueError("sample_size must lie between 0 and the population size")
    if t < 2 or m < 2:
        return Fraction(0, 1)
    return Fraction(_comb(m, 2) * population_unresolved_pairs(profile), _comb(t, 2))


def expected_cascade_workload(
    prefix_profiles: Sequence[Iterable[int]],
    per_object_costs: Sequence[int | Fraction],
    fallback_cost: int | Fraction,
    sample_size: int,
) -> Fraction:
    """Exact expected workload for a prefix cascade.

    ``prefix_profiles[j]`` is the population fiber profile of the joint prefix
    fingerprint J_{j+1}.  The first invariant is evaluated on every sampled
    object.  Invariant j+1 is evaluated only on objects remaining in a
    nonsingleton sample bucket of J_j.  The terminal fallback is charged per
    pair unresolved by the last prefix.
    """
    if len(prefix_profiles) != len(per_object_costs):
        raise ValueError("one prefix profile is required for each invariant cost")
    if not prefix_profiles:
        raise ValueError("a cascade must contain at least one invariant")
    profiles = [normalize_profile(p) for p in prefix_profiles]
    t_values = {sum(p) for p in profiles}
    if len(t_values) != 1:
        raise ValueError("all prefix profiles must describe the same population")
    costs = [Fraction(c) for c in per_object_costs]
    f = Fraction(fallback_cost)
    total = Fraction(sample_size) * costs[0]
    for j in range(1, len(costs)):
        total += costs[j] * expected_surviving_objects(profiles[j - 1], sample_size)
    total += f * expected_unresolved_pairs(profiles[-1], sample_size)
    return total


def two_stage_order_difference(
    profile_a: Iterable[int],
    profile_b: Iterable[int],
    cost_a: int | Fraction,
    cost_b: int | Fraction,
    sample_size: int,
) -> Fraction:
    """Expected cost(A then B) - expected cost(B then A), excluding common fallback."""
    s_a = expected_singleton_objects(profile_a, sample_size)
    s_b = expected_singleton_objects(profile_b, sample_size)
    return Fraction(cost_a) * s_b - Fraction(cost_b) * s_a


def marginal_break_even_gap(
    prefix_profile: Iterable[int],
    refined_profile: Iterable[int],
    new_invariant_cost: int | Fraction,
    fallback_cost: int | Fraction,
    sample_size: int,
) -> Fraction:
    """Fallback savings minus evaluation cost for adding one more invariant.

    A nonnegative value means the added invariant weakly reduces expected cost.
    """
    prefix = normalize_profile(prefix_profile)
    refined = normalize_profile(refined_profile)
    evaluation = Fraction(new_invariant_cost) * expected_surviving_objects(prefix, sample_size)
    saved_pairs = expected_unresolved_pairs(prefix, sample_size) - expected_unresolved_pairs(refined, sample_size)
    return Fraction(fallback_cost) * saved_pairs - evaluation


def critical_fallback_cost(
    prefix_profile: Iterable[int],
    refined_profile: Iterable[int],
    new_invariant_cost: int | Fraction,
    sample_size: int,
) -> Fraction | None:
    """Smallest per-pair fallback cost at which refinement does not increase expected workload.

    Returns None when the refinement removes no expected unresolved pairs.
    """
    prefix = normalize_profile(prefix_profile)
    refined = normalize_profile(refined_profile)
    reduction = expected_unresolved_pairs(prefix, sample_size) - expected_unresolved_pairs(refined, sample_size)
    if reduction <= 0:
        return None
    evaluation = Fraction(new_invariant_cost) * expected_surviving_objects(prefix, sample_size)
    return evaluation / reduction


def fiber_profile(records: Sequence[Mapping[str, str]], keys: Sequence[str]) -> tuple[int, ...]:
    if not keys:
        raise ValueError("at least one key is required")
    counts = Counter(tuple(row[k] for k in keys) for row in records)
    return normalize_profile(counts.values())


def compact_profile(profile: Iterable[int]) -> str:
    counts = Counter(normalize_profile(profile))
    terms: list[str] = []
    for size in sorted(counts, reverse=True):
        multiplicity = counts[size]
        terms.append(str(size) if multiplicity == 1 else f"{size}^{multiplicity}")
    return "(" + ",".join(terms) + ")"


@dataclass(frozen=True)
class Benchmark:
    records: tuple[dict[str, str], ...]
    profiles: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    sample_size: int = 100
    costs: tuple[int, int, int] = (1, 12, 40)
    fallback_cost: int = 100


def build_benchmark() -> Benchmark:
    """Construct a nested 1000-entry mathematical cascade.

    J1 has profile (101,1^899), J2 has (5^20,1^900), and J3 has
    (2^20,1^960).  The object-level labels make every refinement explicit.
    """
    records: list[dict[str, str]] = []

    # 899 entries are already unique at the first stage.
    for object_id in range(1, 900):
        records.append(
            {
                "object_id": f"x{object_id:04d}",
                "I1": f"A_s{object_id:04d}",
                "I2": f"B_s{object_id:04d}",
                "I3": f"C_s{object_id:04d}",
            }
        )

    # One 101-entry I1 fiber.  I2 divides 100 of its entries into twenty
    # quintuplets and leaves one singleton.  I3 turns each quintuplet into one
    # pair and three singletons.
    for object_id in range(900, 1000):
        offset = object_id - 900
        group = offset // 5 + 1
        position = offset % 5 + 1
        i3 = f"C_g{group:02d}_pair" if position <= 2 else f"C_g{group:02d}_s{position}"
        records.append(
            {
                "object_id": f"x{object_id:04d}",
                "I1": "A_heavy",
                "I2": f"B_g{group:02d}",
                "I3": i3,
            }
        )

    records.append(
        {
            "object_id": "x1000",
            "I1": "A_heavy",
            "I2": "B_heavy_singleton",
            "I3": "C_heavy_singleton",
        }
    )

    profiles = (
        fiber_profile(records, ("I1",)),
        fiber_profile(records, ("I1", "I2")),
        fiber_profile(records, ("I1", "I2", "I3")),
    )
    expected = (
        normalize_profile([101] + [1] * 899),
        normalize_profile([5] * 20 + [1] * 900),
        normalize_profile([2] * 20 + [1] * 960),
    )
    if profiles != expected:
        raise AssertionError(f"benchmark profile mismatch: {profiles!r}")
    return Benchmark(tuple(records), profiles)


def _fraction_fields(prefix: str, value: Fraction) -> dict[str, str]:
    return {
        f"{prefix}_numerator": str(value.numerator),
        f"{prefix}_denominator": str(value.denominator),
        f"{prefix}_decimal": f"{float(value):.12f}",
    }


def write_benchmark(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    records = benchmark.records
    profiles = benchmark.profiles
    m = benchmark.sample_size
    c1, c2, c3 = benchmark.costs
    f = benchmark.fallback_cost

    with (out_dir / "cascade_population.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object_id", "I1", "I2", "I3"])
        writer.writeheader()
        writer.writerows(records)

    summary_rows: list[dict[str, str]] = []
    for stage, keys, profile in [
        ("J1", "I1", profiles[0]),
        ("J2", "I1+I2", profiles[1]),
        ("J3", "I1+I2+I3", profiles[2]),
    ]:
        s = expected_singleton_objects(profile, m)
        u = expected_surviving_objects(profile, m)
        p = expected_unresolved_pairs(profile, m)
        row = {
            "stage": stage,
            "prefix_fingerprint": keys,
            "population_size_T": str(sum(profile)),
            "sample_size_M": str(m),
            "occupied_values_R": str(len(profile)),
            "fiber_profile": compact_profile(profile),
            "population_unresolved_pairs": str(population_unresolved_pairs(profile)),
        }
        row.update(_fraction_fields("expected_singletons_S", s))
        row.update(_fraction_fields("expected_survivors_U", u))
        row.update(_fraction_fields("expected_pairs_P", p))
        summary_rows.append(row)

    with (out_dir / "cascade_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    p1, p2, p3 = profiles
    u1 = expected_surviving_objects(p1, m)
    u2 = expected_surviving_objects(p2, m)
    u3 = expected_surviving_objects(p3, m)
    ep1 = expected_unresolved_pairs(p1, m)
    ep2 = expected_unresolved_pairs(p2, m)
    ep3 = expected_unresolved_pairs(p3, m)

    policies: list[tuple[str, Fraction, str]] = [
        ("I1 only", Fraction(m * c1) + f * ep1, "I1 on all objects; fallback after J1"),
        ("I2 only", Fraction(m * c2) + f * ep2, "I2 on all objects; fallback after J2"),
        (
            "I1 -> I2",
            expected_cascade_workload((p1, p2), (c1, c2), f, m),
            "I1 on all objects; I2 only on J1 survivors; fallback after J2",
        ),
        (
            "I2 -> I1",
            Fraction(m * c2) + c1 * u2 + f * ep2,
            "I2 on all objects; I1 only on I2 survivors; same terminal joint fingerprint",
        ),
        (
            "I1 -> I3",
            Fraction(m * c1) + c3 * u1 + f * ep3,
            "skip I2; evaluate I3 on J1 survivors; fallback after J3",
        ),
        (
            "I1 -> I2 -> I3",
            expected_cascade_workload((p1, p2, p3), (c1, c2, c3), f, m),
            "fully adaptive three-stage cascade",
        ),
        (
            "all three on all objects",
            Fraction(m * (c1 + c2 + c3)) + f * ep3,
            "nonadaptive full evaluation; fallback after J3",
        ),
    ]

    workload_rows: list[dict[str, str]] = []
    for name, total, description in policies:
        row = {
            "policy": name,
            "description": description,
            "sample_size_M": str(m),
            "cost_I1": str(c1),
            "cost_I2": str(c2),
            "cost_I3": str(c3),
            "fallback_cost_f": str(f),
        }
        row.update(_fraction_fields("expected_workload", total))
        workload_rows.append(row)

    with (out_dir / "cascade_workloads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(workload_rows[0]))
        writer.writeheader()
        writer.writerows(workload_rows)

    order_difference = two_stage_order_difference(p1, p2, c1, c2, m)
    break_even_2 = critical_fallback_cost(p1, p2, c2, m)
    break_even_3 = critical_fallback_cost(p2, p3, c3, m)
    decision_rows = [
        {
            "decision": "order I1 before I2",
            "left_quantity": "c1*S_J2",
            "right_quantity": "c2*S_J1",
            "left_decimal": f"{float(Fraction(c1) * expected_singleton_objects(p2, m)):.12f}",
            "right_decimal": f"{float(Fraction(c2) * expected_singleton_objects(p1, m)):.12f}",
            "difference_or_threshold": f"{float(order_difference):.12f}",
            "verdict": "I1 -> I2 is cheaper" if order_difference <= 0 else "I2 -> I1 is cheaper",
        },
        {
            "decision": "add I2 after J1",
            "left_quantity": "critical fallback cost",
            "right_quantity": "chosen fallback cost",
            "left_decimal": f"{float(break_even_2):.12f}" if break_even_2 is not None else "NA",
            "right_decimal": f"{f:.12f}",
            "difference_or_threshold": f"{float(marginal_break_even_gap(p1, p2, c2, f, m)):.12f}",
            "verdict": "reduces expected workload",
        },
        {
            "decision": "add I3 after J2",
            "left_quantity": "critical fallback cost",
            "right_quantity": "chosen fallback cost",
            "left_decimal": f"{float(break_even_3):.12f}" if break_even_3 is not None else "NA",
            "right_decimal": f"{f:.12f}",
            "difference_or_threshold": f"{float(marginal_break_even_gap(p2, p3, c3, f, m)):.12f}",
            "verdict": "reduces expected workload",
        },
    ]
    with (out_dir / "cascade_ordering_break_even.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    write_benchmark(args.out_dir)

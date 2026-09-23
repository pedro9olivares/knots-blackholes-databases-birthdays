#!/usr/bin/env python3
"""Exact finite-population collision geometry for the Supporting Information.

All calculations use Python integers and fractions.  The module is intentionally
small enough to audit directly and supports both the theorem regression suite
and the source-locked benchmark figure.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


def falling(x: int, k: int) -> int:
    """Return the falling factorial (x)_k."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k > x:
        return 0
    out = 1
    for j in range(k):
        out *= x - j
    return out


def validate_profile(profile: Sequence[int]) -> tuple[int, int]:
    if not profile:
        raise ValueError("profile must contain at least one occupied fiber")
    if any((not isinstance(m, int)) or m <= 0 for m in profile):
        raise ValueError("fiber sizes must be positive integers")
    return sum(profile), len(profile)


def elementary_symmetric(profile: Sequence[int], degree: int) -> int:
    """Compute e_degree(profile) by exact dynamic programming."""
    _, r = validate_profile(profile)
    if degree < 0:
        return 0
    if degree == 0:
        return 1
    if degree > r:
        return 0
    coeff = [0] * (degree + 1)
    coeff[0] = 1
    for m in profile:
        for j in range(degree, 0, -1):
            coeff[j] += m * coeff[j - 1]
    return coeff[degree]


def alpha_with_replacement(profile: Sequence[int]) -> Fraction:
    t, _ = validate_profile(profile)
    return Fraction(sum(m * m for m in profile), t * t)


def alpha_distinct(profile: Sequence[int]) -> Fraction:
    t, _ = validate_profile(profile)
    if t < 2:
        return Fraction(0, 1)
    return Fraction(sum(m * (m - 1) for m in profile), t * (t - 1))


def no_collision_without_replacement(profile: Sequence[int], sample_size: int) -> Fraction:
    t, _ = validate_profile(profile)
    if sample_size < 0 or sample_size > t:
        return Fraction(0, 1)
    if sample_size <= 1:
        return Fraction(1, 1)
    return Fraction(elementary_symmetric(profile, sample_size), math.comb(t, sample_size))


def no_collision_with_replacement(profile: Sequence[int], sample_size: int) -> Fraction:
    t, _ = validate_profile(profile)
    if sample_size < 0:
        return Fraction(0, 1)
    if sample_size <= 1:
        return Fraction(1, 1)
    numerator = math.factorial(sample_size) * elementary_symmetric(profile, sample_size)
    return Fraction(numerator, t**sample_size)


def collision_moments_without_replacement(profile: Sequence[int], sample_size: int) -> tuple[Fraction, Fraction]:
    """Return exact mean and variance of the pair-collision count."""
    t, _ = validate_profile(profile)
    if not (0 <= sample_size <= t):
        raise ValueError("sample size must lie between 0 and T")
    if t < 2:
        return Fraction(0, 1), Fraction(0, 1)

    s2 = sum(falling(m, 2) for m in profile)
    s3 = sum(falling(m, 3) for m in profile)
    a = Fraction(s2, falling(t, 2))
    b = Fraction(s3, falling(t, 3)) if t >= 3 else Fraction(0, 1)
    if t >= 4:
        d = Fraction(s2 * s2 - 4 * s3 - 2 * s2, falling(t, 4))
    else:
        d = Fraction(0, 1)

    c2 = math.comb(sample_size, 2)
    c3 = math.comb(sample_size, 3)
    c4 = math.comb(sample_size, 4)
    mean = c2 * a
    variance = c2 * (a - a * a) + 6 * c3 * (b - a * a) + 6 * c4 * (d - a * a)
    return mean, variance


def balanced_profile(t: int, r: int) -> list[int]:
    if not (1 <= r <= t):
        raise ValueError("require 1 <= R <= T")
    q, rem = divmod(t, r)
    return [q + 1] * rem + [q] * (r - rem)


def support_compression_bound(t: int, r: int) -> Fraction:
    if not (1 <= r <= t) or t < 2:
        raise ValueError("require 1 <= R <= T and T >= 2")
    return Fraction(t - r, r * (t - 1))


def median_collision_size(profile: Sequence[int]) -> int | None:
    """Smallest M for which collision probability is at least one half."""
    t, _ = validate_profile(profile)
    for m in range(2, t + 1):
        if no_collision_without_replacement(profile, m) <= Fraction(1, 2):
            return m
    return None


@dataclass(frozen=True)
class ProfileSummary:
    name: str
    description: str
    profile: tuple[int, ...]

    @property
    def t(self) -> int:
        return sum(self.profile)

    @property
    def r(self) -> int:
        return len(self.profile)

    @property
    def alpha_wr(self) -> Fraction:
        return alpha_with_replacement(self.profile)

    @property
    def alpha_neq(self) -> Fraction:
        return alpha_distinct(self.profile)

    @property
    def birthday_scale(self) -> float:
        a = float(self.alpha_neq)
        return math.inf if a == 0 else 1.0 / math.sqrt(a)

    @property
    def m50(self) -> int | None:
        return median_collision_size(self.profile)


def benchmark_profiles() -> list[ProfileSummary]:
    return [
        ProfileSummary(
            name="A",
            description="one fiber of size 101 and 899 singleton fibers",
            profile=tuple([101] + [1] * 899),
        ),
        ProfileSummary(
            name="B",
            description="500 doubleton fibers",
            profile=tuple([2] * 500),
        ),
    ]


def write_benchmark_data(out_dir: Path, max_m: int = 90) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = benchmark_profiles()

    summary_path = out_dir / "finite_profile_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profile", "description", "T", "R", "support_fraction",
            "alpha_wr_exact", "alpha_wr_decimal", "alpha_neq_exact",
            "alpha_neq_decimal", "alpha_neq_inverse_sqrt", "M50",
        ])
        for item in profiles:
            writer.writerow([
                item.name,
                item.description,
                item.t,
                item.r,
                f"{item.r/item.t:.12f}",
                f"{item.alpha_wr.numerator}/{item.alpha_wr.denominator}",
                f"{float(item.alpha_wr):.15g}",
                f"{item.alpha_neq.numerator}/{item.alpha_neq.denominator}",
                f"{float(item.alpha_neq):.15g}",
                f"{item.birthday_scale:.12f}",
                item.m50,
            ])

    curve_path = out_dir / "finite_no_collision_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "M", "p_no_collision_A", "p_collision_A",
            "p_no_collision_B", "p_collision_B",
        ])
        for m in range(0, max_m + 1):
            p_a = no_collision_without_replacement(profiles[0].profile, m)
            p_b = no_collision_without_replacement(profiles[1].profile, m)
            writer.writerow([
                m,
                f"{float(p_a):.15g}",
                f"{float(1-p_a):.15g}",
                f"{float(p_b):.15g}",
                f"{float(1-p_b):.15g}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--max-m", type=int, default=90)
    args = parser.parse_args()
    write_benchmark_data(args.out_dir, args.max_m)


if __name__ == "__main__":
    main()

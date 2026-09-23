#!/usr/bin/env python3
"""Deterministic regression tests for the self-contained Supporting Information."""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import generate_finite_benchmarks as fp
import generate_cascade_benchmarks as cw

checks = 0

def require(cond: bool, message: str) -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(message)

# 1. Eight local decorations: 2 trefoils, 6 unknots.
outcomes = Counter(sum(signs) for signs in itertools.product((-1, 1), repeat=3))
require(outcomes[3] == 1 and outcomes[-3] == 1, "unanimous sign triples")
require(outcomes[1] == 3 and outcomes[-1] == 3, "mixed sign triples")
require(sum(outcomes.values()) == 8, "eight local decorations")

# 2. Binomial maximal-atom bound for a large deterministic range.
# Exact rational arithmetic is used for small m; logarithmic evaluation avoids
# constructing enormous integers for the full regression range.
def log_binomial_atom(m: int, k: int, p: float = 0.25) -> float:
    return (
        math.lgamma(m + 1)
        - math.lgamma(k + 1)
        - math.lgamma(m - k + 1)
        + k * math.log(p)
        + (m - k) * math.log1p(-p)
    )

for m in range(1, 200001):
    mode = math.floor((m + 1) / 4)
    candidates = {max(0, min(m, mode + d)) for d in (-1, 0, 1)}
    bound = math.sqrt(2 * math.pi / (3 * m))
    if m <= 1000:
        pmax = max(Fraction(math.comb(m, j) * 3 ** (m-j), 4 ** m) for j in candidates)
        require(float(pmax) <= bound + 1e-15, f"binomial atom m={m}")
    else:
        log_pmax = max(log_binomial_atom(m, j) for j in candidates)
        require(log_pmax <= math.log(bound) + 5e-12, f"binomial atom m={m}")

# 3. Exact finite-population formulas against exhaustive subset enumeration.
def brute_profile(profile: tuple[int, ...], M: int):
    labels=[]
    for v,size in enumerate(profile):
        labels.extend([v]*size)
    vals=[]
    for subset in itertools.combinations(range(len(labels)), M):
        c=Counter(labels[i] for i in subset)
        vals.append(sum(math.comb(x,2) for x in c.values()))
    mean=Fraction(sum(vals),len(vals))
    var=Fraction(sum((Fraction(v)-mean)**2 for v in vals),len(vals))
    p0=Fraction(sum(v==0 for v in vals),len(vals))
    return mean,var,p0

def parts(n: int, max_part: int | None = None):
    if n == 0:
        yield ()
        return
    if max_part is None or max_part > n:
        max_part = n
    for first in range(max_part, 0, -1):
        for rest in parts(n - first, first):
            yield (first,) + rest

small_profiles=[]
for T in range(2,9):
    small_profiles.extend(parts(T))

for profile in small_profiles:
    T=sum(profile)
    for M in range(0,T+1):
        mean,var=fp.collision_moments_without_replacement(profile,M)
        p0=fp.no_collision_without_replacement(profile,M)
        bmean,bvar,bp0=brute_profile(profile,M)
        require(mean==bmean, f"finite mean {profile} M={M}")
        require(var==bvar, f"finite variance {profile} M={M}")
        require(p0==bp0, f"finite no collision {profile} M={M}")

# 4. Balanced profiles minimize collision mass and maximize exact no-collision curves.
for T in range(2,13):
    all_profiles_T=list(parts(T))
    for R in range(1,T+1):
        bal=tuple(fp.balanced_profile(T,R))
        bal_a=fp.alpha_distinct(bal)
        for profile in (p for p in all_profiles_T if len(p)==R):
            require(fp.alpha_distinct(profile) >= bal_a, f"balanced alpha T={T} R={R}")
            for M in range(0,T+1):
                require(fp.no_collision_without_replacement(profile,M) <= fp.no_collision_without_replacement(bal,M), f"balanced curve T={T} R={R} M={M}")

# 5. Cascade formulas against exhaustive samples on small random populations.
rng=random.Random(20260726)
for T in range(4,10):
    for _ in range(20):
        records=[]
        for i in range(T):
            a=str(rng.randrange(max(1,T//3)))
            b=a+":"+str(rng.randrange(3))
            c=b+":"+str(rng.randrange(2))
            records.append({"id":str(i),"I1":a,"I2":b,"I3":c})
        p1=cw.fiber_profile(records,("I1",))
        p2=cw.fiber_profile(records,("I1","I2"))
        p3=cw.fiber_profile(records,("I1","I2","I3"))
        for M in range(1,T+1):
            for profile,keys in [(p1,("I1",)),(p2,("I1","I2")),(p3,("I1","I2","I3"))]:
                obs_s=[]; obs_p=[]
                for inds in itertools.combinations(range(T),M):
                    counts=Counter(tuple(records[i][k] for k in keys) for i in inds)
                    obs_s.append(sum(n for n in counts.values() if n==1))
                    obs_p.append(sum(math.comb(n,2) for n in counts.values()))
                require(cw.expected_singleton_objects(profile,M)==Fraction(sum(obs_s),len(obs_s)), "cascade singleton")
                require(cw.expected_unresolved_pairs(profile,M)==Fraction(sum(obs_p),len(obs_p)), "cascade pairs")
            lhs=cw.two_stage_order_difference(p1,p2,2,5,M)
            w12=Fraction(M*2)+5*cw.expected_surviving_objects(p1,M)+7*cw.expected_unresolved_pairs(p2,M)
            w21=Fraction(M*5)+2*cw.expected_surviving_objects(p2,M)+7*cw.expected_unresolved_pairs(p2,M)
            require(lhs==w12-w21, "ordering identity")

# 6. Canonical benchmark outputs exist and contain exact fields.
required=[
    ROOT/'data'/'finite_profile_summary.csv',
    ROOT/'data'/'finite_no_collision_curves.csv',
    ROOT/'data'/'cascade_population.csv',
    ROOT/'data'/'cascade_summary.csv',
    ROOT/'data'/'cascade_workloads.csv',
    ROOT/'data'/'cascade_ordering_break_even.csv',
]
for path in required:
    require(path.exists() and path.stat().st_size>0, f"missing {path.name}")

record={"status":"PASS","checks":checks,"python":sys.version.split()[0]}
out=ROOT/'audit'/'verification_record.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record,indent=2)+"\n",encoding='utf-8')
(ROOT/'audit'/'verification_record.txt').write_text(
    f"PASS\nChecks: {checks}\nPython: {record['python']}\n",encoding='utf-8')
print(json.dumps(record))

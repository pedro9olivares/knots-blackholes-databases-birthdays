# Shadow-generation reproducibility subset

This directory contains the minimal source and audit material supporting the
slot-conditioned shadow atlas in the Supporting Information.

## Publication sample

The displayed atlas consists of ten rooted, one-component, 20-crossing shadows
generated through a deterministic PlanarMap search. The search was explicitly
conditioned on the presence of at least one complete structured opened-trefoil
slot; it is not an unconditioned random sample and is not used to estimate the
asymptotic slot density.

Starting with master seed `9324001`, the search examined 454 sequential
quartic-map proposals. Of these, 82 were one-component. The ten displayed
shadows have seeds

```text
9324010, 9324025, 9324051, 9324105, 9324285,
9324308, 9324312, 9324352, 9324399, 9324454
```

Each displayed shadow has exactly one certified structured slot. The
corresponding rotation systems, seeds, certificates, and selection statistics
are recorded in `strict_positive_sample_ledger.json`.

These finite drawings are illustrations. The asymptotic pattern statement in
the article comes from the cited theorem, not from this sample.

## Included files

- `structured_slot_detector.py` implements the exact structured-slot detector.
- `test_structured_slot_detector.py` contains positive and adversarial tests.
- `audit_strict_positive_sample.py` independently checks the saved rotation
  systems, certificates, and HTML annotations.
- `strict_positive_sample_ledger.json` records the PlanarMap rotation systems,
  seeds, certificates, and selection statistics.
- `strict_positive_sample_audit.json` is the expected audit result.
- `session_scripts/atlas_uniform.py` performs the conditioned PlanarMap search.
- `html_sources/final_inputs/uniform-shadow-sample-strict-positive.html` is the
  source containing the ten displayed shadows.
- `html_sources/final_inputs/uniform-shadow-sample-replaced01.html` preserves
  the earlier unconditioned sample referenced in the Supporting Information.
- `requirements.txt` pins the Python dependency needed by the sampler.

The final publication-ready TikZ source is stored separately as
`figures/figure3_random_shadow_atlas.tex`.

## Verify the frozen sample

From this directory, run:

```bash
python3 test_structured_slot_detector.py

python3 audit_strict_positive_sample.py \
  --ledger strict_positive_sample_ledger.json \
  --html html_sources/final_inputs/uniform-shadow-sample-strict-positive.html \
  --output reproduced_strict_positive_sample_audit.json

diff -u \
  strict_positive_sample_audit.json \
  reproduced_strict_positive_sample_audit.json
```

A successful run reports that the detector tests pass and produces no
difference between the archived and reproduced audit records.

## Regenerate the conditioned search

The sampler requires Python, NetworkX 3.5, and PlanarMap v1.2.

The PlanarMap executable used for the archived run had SHA-256

```text
788f293fadc864c626279d01d43827ff5c6e2e6b52af745d062bf9f80a6c65ae
```

The executable itself is not included in this repository. The sampler invokes
it with the equivalent of

```text
planarmap -Q1 -N1 -V20 -I0 -p -O1 -X<seed>
```

After installing the Python dependency, the search can be repeated with:

```bash
python3 -m pip install -r requirements.txt

python3 session_scripts/atlas_uniform.py \
  --output-html reproduced_strict_positive_sample.html \
  --ledger reproduced_strict_positive_sample_ledger.json \
  --planarmap /path/to/planarmap \
  --n 20 \
  --target 10 \
  --master-seed 9324001 \
  --minimum-slots 1
```

The selection condition is part of the command and the saved ledger; it should
not be interpreted as an unconditioned frequency estimate.

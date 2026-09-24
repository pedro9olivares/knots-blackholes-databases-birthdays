import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_slot_detector import detect_structured_opened_trefoil_slots


PLANARMAP = os.environ.get('PLANARMAP', 'planarmap')
N = 20
TARGET = 10
MASTER_SEED = 9324001
MIN_SLOTS = 1


def opened_trefoil(count=220):
    """A standard three-crossing trefoil shadow, cut open at two exterior points."""
    ta = 3.917873
    tb = 2.365313 + 2 * math.pi
    raw = []
    for i in range(count):
        t = ta + (tb - ta) * i / (count - 1)
        raw.append((math.sin(t) + 2 * math.sin(2*t), math.cos(t) - 2 * math.cos(2*t)))
    a, b = raw[0], raw[-1]
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    ex, ey = b[0] - a[0], b[1] - a[1]
    length = math.hypot(ex, ey)
    ex, ey = ex / length, ey / length
    nxv, nyv = -ey, ex
    result = []
    for x, y in raw:
        dx, dy = x - cx, y - cy
        u = 2 * (dx * ex + dy * ey) / length
        v = 1.72 * (dx * nxv + dy * nyv) / length
        result.append((u, v))
    return result


def polyline_crossings(points):
    hits = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        rx, ry = b[0] - a[0], b[1] - a[1]
        for j in range(i + 2, len(points) - 1):
            if j == i + 1:
                continue
            c, d = points[j], points[j + 1]
            sx, sy = d[0] - c[0], d[1] - c[1]
            denominator = rx * sy - ry * sx
            if abs(denominator) < 1e-9:
                continue
            qx, qy = c[0] - a[0], c[1] - a[1]
            t = (qx * sy - qy * sx) / denominator
            u = (qx * ry - qy * rx) / denominator
            if 1e-5 < t < 1 - 1e-5 and 1e-5 < u < 1 - 1e-5:
                point = (a[0] + t * rx, a[1] + t * ry)
                if all(math.hypot(point[0] - q[0], point[1] - q[1]) > .02 for q in hits):
                    hits.append(point)
    return hits


TREFOIL = opened_trefoil()
TREFOIL_CROSSINGS = polyline_crossings(TREFOIL)
TREFOIL_CENTER = (
    (min(x for x, y in TREFOIL) + max(x for x, y in TREFOIL)) / 2,
    (min(y for x, y in TREFOIL) + max(y for x, y in TREFOIL)) / 2,
)


def run_sample(seed):
    out = subprocess.check_output([
        PLANARMAP, '-Q1', '-N1', f'-V{N}', '-I0', '-p', '-O1', f'-X{seed}'
    ], text=True)
    rotations = []
    for line in out.splitlines():
        match = re.match(r'Vertex\s+(\d+):\s+(.+)', line)
        if match:
            rotations.append([int(x) for x in match.group(2).split()])
    if len(rotations) != N or any(len(row) != 4 for row in rotations):
        raise RuntimeError(f'could not parse seed {seed}')
    return rotations


def gauss_components(rotations):
    position = {}
    for v, row in enumerate(rotations):
        for i, dart in enumerate(row):
            position[dart] = (v, i)
    seen = set()
    components = []
    for dart in list(position):
        if dart in seen:
            continue
        current = dart
        sequence = []
        while current not in seen:
            seen.add(current)
            seen.add(-current)
            v, i = position[current]
            sequence.append(v)
            current = -rotations[v][(i + 2) % 4]
        components.append(sequence)
    return components


def edge_vertices(rotations):
    result = defaultdict(lambda: [None, None])
    for v, row in enumerate(rotations):
        for dart in row:
            if dart > 0:
                result[dart][0] = v
            else:
                result[-dart][1] = v
    return dict(result)


def faces(rotations):
    position = {}
    for v, row in enumerate(rotations):
        for i, dart in enumerate(row):
            position[dart] = (v, i)
    face_of = {}
    cycles = []
    for dart in position:
        if dart in face_of:
            continue
        current = dart
        face = []
        fid = len(cycles)
        while current not in face_of:
            face_of[current] = fid
            face.append(current)
            v, i = position[-current]
            current = rotations[v][(i - 1) % 4]
        cycles.append(face)
    return cycles, face_of


def graph_distance(adj, sources):
    dist = {s: 0 for s in sources}
    queue = list(sources)
    for u in queue:
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def detect_trefoil_separators(rotations):
    """Return only fully certified structured opened-trefoil carriers."""

    return detect_structured_opened_trefoil_slots(rotations)


def layout(rotations, seed):
    edges = edge_vertices(rotations)
    simple = nx.Graph()
    simple.add_nodes_from(range(N))
    simple.add_edges_from(tuple(sorted(pair)) for pair in edges.values() if pair[0] != pair[1])
    planar, embedding = nx.check_planarity(simple)
    if not planar:
        raise RuntimeError('PlanarMap output parsed as nonplanar')
    pos = nx.combinatorial_embedding_to_pos(embedding)
    positions = {v: [float(pos[v][0]), float(pos[v][1])] for v in simple.nodes}
    return edges, positions


def edge_curves(graph, positions):
    grouped = defaultdict(list)
    for key, (u, v) in graph.items():
        grouped[tuple(sorted((u, v)))].append((u, v, key))
    curves = {}
    for pair, items in grouped.items():
        count = len(items)
        for index, (u, v, key) in enumerate(items):
            p = positions[u]; q = positions[v]
            if u == v:
                angle = 2 * math.pi * (index + 1) / max(1, count)
                radius = .12 + .035 * index
                c1 = [p[0] + radius * math.cos(angle), p[1] + radius * math.sin(angle)]
                c2 = [p[0] + radius * math.cos(angle + 1.6), p[1] + radius * math.sin(angle + 1.6)]
                curves[key] = (p, c1, c2, q)
                continue
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = math.hypot(dx, dy) or 1
            nxv, nyv = -dy / length, dx / length
            offset = (index - (count - 1) / 2) * .095
            c1 = [p[0] + .34 * dx + offset * nxv, p[1] + .34 * dy + offset * nyv]
            c2 = [p[0] + .66 * dx + offset * nxv, p[1] + .66 * dy + offset * nyv]
            curves[key] = (p, c1, c2, q)
    return curves


def draw_path_from_gauss(sample, pos, project):
    rotations = sample['rotations']
    position = {}
    for v, row in enumerate(rotations):
        for i, dart in enumerate(row):
            position[dart] = (v, i)
    current = rotations[0][0]
    darts = []
    seen = set()
    while current not in seen:
        seen.add(current)
        darts.append(current)
        v, i = position[current]
        current = -rotations[v][(i + 2) % 4]
    points = []
    for index, dart in enumerate(darts):
        v, i = position[dart]
        next_v, _ = position[-dart]
        a = pos[v]; b = pos[next_v]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1
        nxv, nyv = -dy / length, dx / length
        bend = .10 * math.sin((abs(dart) * 1.618 + sample['seed']) % (2 * math.pi))
        for j in range(8):
            t = j / 8
            ease = t * t * (3 - 2 * t)
            bulge = math.sin(math.pi * t) * bend
            points.append([a[0] + ease * dx + bulge * nxv, a[1] + ease * dy + bulge * nyv])
    projected = [project(p) for p in points]
    return ' '.join((('M' if i == 0 else 'L') + f'{p[0]:.1f} {p[1]:.1f}') for i, p in enumerate(projected)) + ' Z'


def knotty_layout(sample):
    rng = random.Random(sample['seed'] * 17 + 91)
    planar = sample['pos']
    initial = {v: [float(planar[v][0]), float(planar[v][1])] for v in range(N)}
    unique = sorted(set(tuple(sorted(pair)) for pair in sample['graph'].values()))
    for iteration in range(900):
        force = {v: [0.0, 0.0] for v in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                dx = initial[i][0] - initial[j][0]; dy = initial[i][1] - initial[j][1]
                d2 = dx*dx + dy*dy + .015
                strength = .0048 / d2
                force[i][0] += strength * dx; force[i][1] += strength * dy
                force[j][0] -= strength * dx; force[j][1] -= strength * dy
        for u, v in unique:
            if u == v:
                continue
            dx = initial[v][0] - initial[u][0]; dy = initial[v][1] - initial[u][1]
            length = math.hypot(dx, dy) or 1
            target = 2.2
            strength = .035 * (length - target) / length
            force[u][0] += strength * dx; force[u][1] += strength * dy
            force[v][0] -= strength * dx; force[v][1] -= strength * dy
        step = .18 * (1 - iteration / 1100)
        for v in range(N):
            initial[v][0] += step * max(-.20, min(.20, force[v][0]))
            initial[v][1] += step * max(-.20, min(.20, force[v][1]))
    angle = rng.uniform(-math.pi, math.pi)
    aspect = rng.uniform(.72, 1.38)
    ca, sa = math.cos(angle), math.sin(angle)
    warped = {}
    phase1, phase2 = rng.uniform(-2, 2), rng.uniform(-2, 2)
    for v, p in initial.items():
        x, y = float(p[0]), float(p[1])
        xr = (ca * x - sa * y) * aspect
        yr = (sa * x + ca * y) / aspect
        x1 = xr + .18 * math.sin(1.8 * yr + phase1)
        y1 = yr + .20 * math.sin(1.7 * x1 + phase2)
        warped[v] = [x1, y1]
    return warped


def sample_all():
    accepted = []
    attempt = 0
    one_component_proposals = 0
    while len(accepted) < TARGET:
        seed = MASTER_SEED + attempt
        attempt += 1
        rotations = run_sample(seed)
        comps = gauss_components(rotations)
        if len(comps) != 1:
            continue
        one_component_proposals += 1
        sequence = comps[0]
        slots = detect_trefoil_separators(rotations)
        if len(slots) < MIN_SLOTS:
            continue
        graph, pos = layout(rotations, seed)
        accepted.append({
            'seed': seed,
            'rotations': rotations,
            'sequence': sequence,
            'slots': slots,
            'graph': graph,
            'pos': pos,
            'curves': edge_curves(graph, pos),
        })
    return accepted, attempt, one_component_proposals


def svg_for(sample, number):
    pos = knotty_layout(sample)
    curves = edge_curves(sample['graph'], pos)
    allx = [p[0] for p in pos.values()]; ally = [p[1] for p in pos.values()]
    xmin, xmax = min(allx), max(allx); ymin, ymax = min(ally), max(ally)
    pad = .18 * max(xmax - xmin, ymax - ymin, 1)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    def project(p):
        return (12 + 216 * (p[0] - xmin) / (xmax - xmin), 10 + 170 * (p[1] - ymin) / (ymax - ymin))
    shadow_path = draw_path_from_gauss(sample, pos, project)
    nodes = '\n'.join(f'<circle class="usa-node" cx="{project(pos[v])[0]:.1f}" cy="{project(pos[v])[1]:.1f}" r="2.7" />' for v in range(N))
    masks = []
    canonical_slots = []
    circles = []
    for slot_index, slot in enumerate(sample['slots']):
        original = [project(pos[v]) for v in slot['vertices']]
        cx = sum(x for x, y in original) / 3
        cy = sum(y for x, y in original) / 3

        anchors = []
        for edge in slot['boundary']:
            u, v = sample['graph'][edge]
            outside = v if u in slot['vertices'] else u
            anchors.append(project(pos[outside]))
        ax, ay = anchors[0]
        bx, by = anchors[1]
        dx, dy = bx - ax, by - ay
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            farthest = max(
                ((p, q) for i, p in enumerate(original) for q in original[i + 1:]),
                key=lambda pair: math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]),
            )
            dx, dy = farthest[1][0] - farthest[0][0], farthest[1][1] - farthest[0][1]
            distance = math.hypot(dx, dy) or 1
        ex, ey = dx / distance, dy / distance
        nxv, nyv = -ey, ex
        if (sample['seed'] + slot_index) % 2:
            nxv, nyv = -nxv, -nyv

        horizontal_scale = 10.3
        vertical_scale = 13.0
        qcx, qcy = TREFOIL_CENTER

        def place(point):
            u, v = point
            u -= qcx
            v -= qcy
            return (cx + horizontal_scale * u * ex + vertical_scale * v * nxv,
                    cy + horizontal_scale * u * ey + vertical_scale * v * nyv)

        trefoil_points = [place(point) for point in TREFOIL]
        first, last = trefoil_points[0], trefoil_points[-1]
        direct = math.hypot(first[0] - ax, first[1] - ay) + math.hypot(last[0] - bx, last[1] - by)
        reverse = math.hypot(last[0] - ax, last[1] - ay) + math.hypot(first[0] - bx, first[1] - by)
        if reverse < direct:
            anchors.reverse()

        trefoil_path = ' '.join((('M' if i == 0 else 'L') + f'{p[0]:.1f} {p[1]:.1f}') for i, p in enumerate(trefoil_points))
        connector_path = (
            f'M{anchors[0][0]:.1f} {anchors[0][1]:.1f} L{first[0]:.1f} {first[1]:.1f} '
            f'M{last[0]:.1f} {last[1]:.1f} L{anchors[1][0]:.1f} {anchors[1][1]:.1f}'
        )
        crossing_nodes = ''.join(
            f'<circle class="usa-slot-node" cx="{x:.1f}" cy="{y:.1f}" r="3.0" />'
            for x, y in (place(point) for point in TREFOIL_CROSSINGS)
        )
        angle = math.degrees(math.atan2(ey, ex))
        original_u = max(abs((x - cx) * ex + (y - cy) * ey) for x, y in original)
        original_v = max(abs((x - cx) * nxv + (y - cy) * nyv) for x, y in original)
        ring_rx = max(26.0, original_u + 8.0)
        ring_ry = max(18.5, original_v + 8.0)
        transform = f'rotate({angle:.1f} {cx:.1f} {cy:.1f})'
        masks.append(f'<ellipse class="usa-slot-mask" cx="{cx:.1f}" cy="{cy:.1f}" rx="{ring_rx:.1f}" ry="{ring_ry:.1f}" transform="{transform}" />')
        canonical_slots.append(f'<path class="usa-slot-connector" d="{connector_path}" /><path class="usa-slot-strand" d="{trefoil_path}" />{crossing_nodes}')
        circles.append(f'<ellipse class="usa-slot" cx="{cx:.1f}" cy="{cy:.1f}" rx="{ring_rx:.1f}" ry="{ring_ry:.1f}" transform="{transform}" />')
    return f'''<article class="usa-item">
      <div class="usa-label">{number:02d} <span>{len(sample['slots'])} certified slot{'s' if len(sample['slots']) != 1 else ''}</span></div>
      <svg viewBox="0 0 240 190" role="img" aria-label="Slot-conditioned rooted knot shadow {number} with {len(sample['slots'])} certified structured opened-trefoil slots">
        <path class="usa-edge" d="{shadow_path}" />
        {nodes}
        {''.join(masks)}
        {''.join(canonical_slots)}
        {''.join(circles)}
      </svg>
    </article>'''


def normalize_certificate(certificate):
    normalized = {}
    for key, value in certificate.items():
        if isinstance(value, frozenset):
            normalized[key] = sorted(value)
        elif isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


def main():
    global PLANARMAP, N, TARGET, MASTER_SEED, MIN_SLOTS

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-html', required=True, type=Path)
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--planarmap', default=PLANARMAP)
    parser.add_argument('--n', default=N, type=int)
    parser.add_argument('--target', default=TARGET, type=int)
    parser.add_argument('--master-seed', default=MASTER_SEED, type=int)
    parser.add_argument('--minimum-slots', default=MIN_SLOTS, type=int)
    args = parser.parse_args()

    PLANARMAP = args.planarmap
    N = args.n
    TARGET = args.target
    MASTER_SEED = args.master_seed
    MIN_SLOTS = args.minimum_slots
    if TARGET < 1 or N < 4 or MIN_SLOTS < 1:
        raise ValueError('target, n, and minimum-slots must be positive')

    samples, attempts, one_component_proposals = sample_all()
    counts = [len(sample['slots']) for sample in samples]
    items = '\n'.join(svg_for(sample, i + 1) for i, sample in enumerate(samples))
    html = f'''<div id="uniform-shadow-atlas" class="usa-root">
  <h2>Strictly certified random knot-shadow atlas</h2>
  <div class="usa-subtitle">{TARGET} rooted one-component shadows · n = {N} · conditioned on at least {MIN_SLOTS} certified slot{'s' if MIN_SLOTS != 1 else ''}</div>
  <div class="usa-grid">{items}</div>
  <div class="usa-key">
    <span><i class="usa-line"></i> shadow</span>
    <span><i class="usa-dot"></i> crossing</span>
    <span><i class="usa-ring"></i> certified structured slot</span>
    <span>{attempts} uniform quartic-map proposals searched · {one_component_proposals} one-component · certified counts {', '.join(map(str, counts))}</span>
  </div>
</div>

<style>
  #uniform-shadow-atlas {{ color:var(--foreground); width:100%; }}
  #uniform-shadow-atlas h2 {{ font-weight:500; margin:0; text-align:center; }}
  #uniform-shadow-atlas .usa-subtitle {{ color:var(--muted-foreground); margin:4px 0 12px; text-align:center; }}
  #uniform-shadow-atlas .usa-grid {{ display:grid; gap:15px 12px; grid-template-columns:repeat(5,minmax(0,1fr)); }}
  #uniform-shadow-atlas .usa-item {{ min-width:0; }}
  #uniform-shadow-atlas .usa-label {{ align-items:baseline; display:flex; font-weight:500; justify-content:space-between; margin:0 3px 2px; }}
  #uniform-shadow-atlas .usa-label span {{ color:var(--muted-foreground); font-weight:400; min-width:0; overflow-wrap:anywhere; text-align:right; }}
  #uniform-shadow-atlas svg {{ display:block; height:auto; width:100%; }}
  #uniform-shadow-atlas .usa-edge {{ fill:none; stroke:var(--foreground); stroke-linecap:round; stroke-width:2.2; }}
  #uniform-shadow-atlas .usa-node {{ fill:var(--viz-series-1); stroke:var(--foreground); stroke-width:.9; }}
  #uniform-shadow-atlas .usa-slot-mask {{ fill:var(--background); stroke:var(--background); stroke-width:7; }}
  #uniform-shadow-atlas .usa-slot-connector {{ fill:none; stroke:var(--foreground); stroke-linecap:round; stroke-width:2.2; }}
  #uniform-shadow-atlas .usa-slot-strand {{ fill:none; stroke:var(--viz-series-1); stroke-linecap:round; stroke-linejoin:round; stroke-width:2.9; }}
  #uniform-shadow-atlas .usa-slot-node {{ fill:var(--viz-series-1); stroke:var(--foreground); stroke-width:.9; }}
  #uniform-shadow-atlas .usa-slot {{ fill:none; stroke:var(--viz-series-3); stroke-dasharray:6 5; stroke-width:2.1; }}
  #uniform-shadow-atlas .usa-key {{ align-items:center; display:flex; flex-wrap:wrap; gap:8px 18px; justify-content:center; margin-top:13px; }}
  #uniform-shadow-atlas .usa-key span {{ align-items:center; display:inline-flex; gap:6px; }}
  #uniform-shadow-atlas .usa-key span:last-child {{ display:block; flex-basis:100%; overflow-wrap:anywhere; text-align:center; }}
  #uniform-shadow-atlas .usa-line {{ background:var(--foreground); display:inline-block; height:3px; width:18px; }}
  #uniform-shadow-atlas .usa-dot {{ background:var(--viz-series-1); border:1px solid var(--foreground); border-radius:50%; display:inline-block; height:8px; width:8px; }}
  #uniform-shadow-atlas .usa-ring {{ border:2px dashed var(--viz-series-3); border-radius:50%; display:inline-block; height:13px; width:18px; }}
  @media (max-width:760px) {{ #uniform-shadow-atlas .usa-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  @media (max-width:620px) {{
    #uniform-shadow-atlas .usa-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    #uniform-shadow-atlas .usa-key {{ justify-content:flex-start; }}
  }}
</style>
'''
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding='utf-8')

    executable = Path(PLANARMAP).resolve()
    ledger = {
        'status': 'PASS',
        'sampler': 'PlanarMap v1.2 general rooted quartic maps',
        'planarmap_command': [PLANARMAP, '-Q1', '-N1', f'-V{N}', '-I0', '-p', '-O1', '-X<seed>'],
        'planarmap_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
        'detector': 'structured_opened_trefoil_v2',
        'n': N,
        'target_panels': TARGET,
        'master_seed': MASTER_SEED,
        'selection_policy': {
            'one_straight_ahead_component': True,
            'minimum_certified_slots': MIN_SLOTS,
            'disclosure': 'The displayed atlas is explicitly conditioned on certified-slot count.',
        },
        'proposals_examined': attempts,
        'one_component_proposals': one_component_proposals,
        'displayed_seeds': [sample['seed'] for sample in samples],
        'structured_slot_counts': counts,
        'structured_slot_total': sum(counts),
        'displayed': [
            {
                'panel': index,
                'seed': sample['seed'],
                'rotation_system': sample['rotations'],
                'certificates': [normalize_certificate(item) for item in sample['slots']],
            }
            for index, sample in enumerate(samples, start=1)
        ],
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': 'PASS',
        'proposals_examined': attempts,
        'one_component_proposals': one_component_proposals,
        'counts': counts,
        'seeds': ledger['displayed_seeds'],
    }))


if __name__ == '__main__':
    main()

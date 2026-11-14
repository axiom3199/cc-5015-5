"""Island seam optimization: choose seam edges from mesh topology, driven by
measured UV stretching and shared-edge (seam) error rather than guesses.

The kernel is used by both the API and the 3D viewport controls ("由服饰展开
内核驱动控件"). It returns structured reasons on failure so the UI can feed
them back to the unwrap panel instead of failing silently.

Metrics
-------
stretch       per-chart SVD-style angle/area distortion summary computed from
              the LSCM parameterization (reuses unwrap.triangle_distortion).
seam error    for every seam edge, the 3D length vs. the UV-space length on
              each of its two cut sides, and the mismatch between the two
              cut sides (shared-edge error after sewing).
"""
from __future__ import annotations

import numpy as np

from .geometry import Mesh
from .unwrap import (chart_topology, cut_charts, find_overlaps, lscm,
                     normalize_chart_scale, triangle_distortion)


class SeamOptError(Exception):
    pass


def _edge_lengths_3d(mesh: Mesh, edges):
    out = {}
    for a, b in edges:
        out[(a, b)] = float(np.linalg.norm(
            mesh.positions[a] - mesh.positions[b]))
    return out


def seam_edge_error(mesh: Mesh, charts, seam_set: set) -> list:
    """Shared-edge error for every cut edge.

    For seam edge (a,b) the two chart sides each carry their own UV segment
    (vertices are duplicated at cuts). We collect every local UV segment
    whose underlying mesh edge is (a,b), then report:
      len_3d            true 3D edge length
      uv_sides          UV lengths of each cut side in geometry units
      stretch_ratio     per-side uv_len / len_3d (1 = isometric)
      shared_error      |uv_len_1 - uv_len_2| / max(len_3d, eps): mismatch
                        that would remain when sewing the two island borders
    """
    rows = []
    lens3d = _edge_lengths_3d(mesh, sorted(seam_set))
    for (a, b), l3 in lens3d.items():
        sides = []
        for ch in charts:
            for tri in ch.faces:
                for k in range(3):
                    x, y = int(tri[k]), int(tri[(k + 1) % 3])
                    va, vb = ch.vertices[x], ch.vertices[y]
                    key = (min(va, vb), max(va, vb))
                    if key == (a, b):
                        duv = ch.uv[y] - ch.uv[x]
                        sides.append(float(np.linalg.norm(duv)))
        sides = sorted(sides)
        if not sides:
            continue
        stretch = [s / max(l3, 1e-12) for s in sides]
        if len(sides) >= 2:
            shared = abs(sides[-1] - sides[0]) / max(l3, 1e-12)
        else:
            shared = 0.0
        rows.append(dict(edge=[a, b], len_3d=l3, uv_sides=sides,
                         stretch_ratio=stretch, shared_error=float(shared)))
    rows.sort(key=lambda r: -r["shared_error"])
    return rows


def chart_stretch(charts) -> dict:
    """Aggregate stretch per chart from per-triangle distortion."""
    summary = {}
    for ch in charts:
        dist = triangle_distortion(ch)
        angles = [d["angle"] for d in dist
                  if np.isfinite(d["angle"]) and not d["flipped"]]
        areas = [d["area_rel"] for d in dist if d["area_rel"] > 0]
        flipped = sum(1 for d in dist if d["flipped"])
        summary[ch.index] = dict(
            max_angle=float(max(angles) if angles else float("inf")),
            mean_angle=float(np.mean(angles) if angles else float("inf")),
            max_area_dev=float(max(abs(np.log2(a)) for a in areas)
                               if areas else float("inf")),
            n_flipped=flipped,
            n_overlaps=len(find_overlaps(ch)),
        )
    return summary


def _disk_after_toggle(mesh: Mesh, seam_set: set, edge) -> bool:
    """Would every chart remain disk-topology if `edge` is toggled?"""
    trial = set(seam_set)
    if edge in trial:
        trial.discard(edge)
    else:
        trial.add(edge)
    charts = cut_charts(mesh, trial)
    for ch in charts:
        topo = chart_topology(ch, trial, mesh.edge_faces)
        if topo["problems"]:
            return False
    return True


def _score(mesh: Mesh, seam_set: set):
    """Full objective: stretch + shared seam error + fold-over penalty."""
    charts = cut_charts(mesh, seam_set)
    for ch in charts:
        ch.topology = chart_topology(ch, seam_set, mesh.edge_faces)
        lscm(ch, {})
        normalize_chart_scale(ch)
    stretch = chart_stretch(charts)
    errors = seam_edge_error(mesh, charts, seam_set)
    s_angle = sum(s["max_angle"] for s in stretch.values())
    s_flip = sum(s["n_flipped"] for s in stretch.values()) * 10.0
    s_overlap = sum(s["n_overlaps"] for s in stretch.values()) * 5.0
    s_seam = sum(e["shared_error"] for e in errors)
    s_long = sum(abs(r - 1.0)
                 for e in errors for r in e["stretch_ratio"]) * 0.5
    return dict(score=s_angle + s_flip + s_overlap + s_seam + s_long,
                charts=charts, stretch=stretch, errors=errors)


def optimize_seams(mesh: Mesh, initial_seams: list | None = None,
                   max_iterations: int = 12,
                   max_seams: int | None = None) -> dict:
    """Greedy seam optimization driven by measured stretch / seam error.

    Each iteration evaluates every topologically legal edge toggle (adding
    or removing one seam while keeping all charts disk-topology) and applies
    the one with the best objective. Deterministic: ties break on the
    canonical edge key so results are reproducible.
    """
    seam_set = {(min(a, b), max(a, b)) for a, b in (initial_seams or [])}
    # any initial seam must exist in topology
    for e in seam_set:
        if e not in mesh.edge_faces:
            raise SeamOptError(f"初始接缝边 {e} 不在网格拓扑中")
    if mesh.nonmanifold_edges:
        raise SeamOptError(
            f"网格含 {len(mesh.nonmanifold_edges)} 条非流形边，"
            f"接缝优化要求先修复网格")

    current = _score(mesh, seam_set)
    history = [dict(step=0, score=current["score"], n_seams=len(seam_set))]
    for step in range(1, max_iterations + 1):
        best_toggle, best_state = None, current
        # candidate edges: current seams (removal) plus edges incident to the
        # most distorted chart's vertices (targeted additions, not a full
        # O(E) re-unwrap per edge on big meshes)
        candidates = set(seam_set)
        worst = max(current["stretch"].items(),
                    key=lambda kv: kv[1]["max_angle"])[0]
        worst_chart = next(c for c in current["charts"] if c.index == worst)
        for li, vid in enumerate(worst_chart.vertices):
            for other in mesh.vertex_adj[vid]:
                candidates.add((min(vid, other), max(vid, other)))
        for edge in sorted(candidates):
            if len(edge) != 2 or edge not in mesh.edge_faces:
                continue
            if edge[0] == edge[1]:
                continue
            if max_seams is not None and edge not in seam_set \
                    and len(seam_set) >= max_seams:
                continue
            if not _disk_after_toggle(mesh, seam_set, edge):
                continue
            trial = set(seam_set)
            trial.symmetric_difference_update([edge])
            state = _score(mesh, trial)
            if state["score"] < best_state["score"] - 1e-9:
                best_state, best_toggle = state, edge
        if best_toggle is None:
            break
        seam_set.symmetric_difference_update([best_toggle])
        current = best_state
        history.append(dict(step=step, edge=list(best_toggle),
                            score=current["score"], n_seams=len(seam_set)))

    final = _score(mesh, seam_set)
    return dict(
        seams=[list(e) for e in sorted(seam_set)],
        score=float(final["score"]),
        history=history,
        stretch=final["stretch"],
        seam_errors=final["errors"],
        charts=final["charts"],
        converged=len(history) <= max_iterations,
    )

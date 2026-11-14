"""Chart cutting, LSCM parameterization, packing and distortion analysis.

Real parameterization: each chart is flattened with LSCM (Levy et al. 2002),
solving the discrete Cauchy-Riemann equations in least squares with pinned
vertices removed from the system. No planar projection of source coordinates.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Chart:
    def __init__(self, index, vertices, faces, positions):
        self.index = index
        self.vertices = vertices        # original vertex ids, local index -> vid
        self.faces = faces              # (F,3) local indices
        self.positions = positions      # (V,3) 3D positions of local verts
        self.uv = None                  # (V,2) after unwrap + pack
        self.tile = 0
        self.locked = False             # user-pinned charts skip the packer
        self.topology = None
        self.tri_dist = None            # per-tri distortion dicts
        self.overlaps = []              # (tri_i, tri_j) pairs


# ---------------------------------------------------------------- cutting
class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cut_charts(mesh, seam_set: set) -> list:
    """Split into charts along seams and build the *cut* mesh per chart:
    vertices on seam edges are duplicated (corner union-find glued only
    across non-seam edges), so chart topology reflects the cut."""
    n_f = len(mesh.faces)
    adj = [[] for _ in range(n_f)]
    for edge, fs in mesh.edge_faces.items():
        if len(fs) == 2 and edge not in seam_set:
            f0, f1 = fs
            adj[f0].append(f1)
            adj[f1].append(f0)
    # union corners (face*3+slot) that share a vertex across a non-seam edge
    dsu = _DSU(n_f * 3)
    for edge, fs in mesh.edge_faces.items():
        if len(fs) != 2 or edge in seam_set:
            continue
        f0, f1 = fs
        for v in edge:
            k0 = int(np.nonzero(mesh.faces[f0] == v)[0][0])
            k1 = int(np.nonzero(mesh.faces[f1] == v)[0][0])
            dsu.union(f0 * 3 + k0, f1 * 3 + k1)

    seen = np.zeros(n_f, bool)
    charts = []
    for f0 in range(n_f):
        if seen[f0]:
            continue
        stack, comp = [f0], []
        seen[f0] = True
        while stack:
            f = stack.pop()
            comp.append(f)
            for g in adj[f]:
                if not seen[g]:
                    seen[g] = True
                    stack.append(g)
        comp = np.asarray(comp, dtype=np.int64)
        roots = {}
        local_vid = []
        faces_local = np.empty((len(comp), 3), dtype=np.int64)
        for ci, f in enumerate(comp):
            for k in range(3):
                r = dsu.find(int(f) * 3 + k)
                if r not in roots:
                    roots[r] = len(local_vid)
                    local_vid.append(int(mesh.faces[f][k]))
                faces_local[ci, k] = roots[r]
        vids = np.asarray(local_vid, dtype=np.int64)
        charts.append(Chart(len(charts), vids.tolist(), faces_local,
                            mesh.positions[vids].copy()))
    return charts


def chart_topology(chart: Chart, seam_set: set = frozenset(),
                   mesh_edge_faces: dict | None = None) -> dict:
    """Topology of the cut chart. A face edge-slot is boundary when its mesh
    edge is a seam or a mesh boundary edge (both sides of a seam count)."""
    n_boundary_slots = 0
    dir_edges = []  # directed boundary slots in local vertex ids
    for tri in chart.faces:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            va, vb = chart.vertices[a], chart.vertices[b]
            key = (min(va, vb), max(va, vb))
            is_seam = key in seam_set
            n_incident = (len(mesh_edge_faces[key])
                          if mesh_edge_faces is not None and key in mesh_edge_faces
                          else None)
            if is_seam or n_incident == 1 or n_incident is None:
                n_boundary_slots += 1
                dir_edges.append((a, b))
    # loops = connected components of the graph that boundary slots form
    # between local vertices (orientation-agnostic: user meshes may have
    # inconsistent winding)
    dsu = _DSU(len(chart.vertices))
    for a, b in dir_edges:
        dsu.union(a, b)
    loops = len({dsu.find(v) for e_ in dir_edges for v in e_})
    v = len(chart.vertices)
    f = len(chart.faces)
    e = (3 * f + n_boundary_slots) // 2
    chi = v - e + f
    problems = []
    if loops == 0:
        problems.append("闭合曲面（无边界）：需要接缝将其切开")
    if not (chi == 1 and loops == 1):
        problems.append(
            f"拓扑不是圆盘（欧拉数 χ={chi}，边界环 {loops} 条）：需要更多接缝")
    boundary_verts = sorted({x for e_ in dir_edges for x in e_})
    return dict(n_boundary_edges=n_boundary_slots, n_boundary_loops=loops,
                euler=chi, problems=problems, boundary_verts=boundary_verts,
                boundary_edges=dir_edges)


# ------------------------------------------------------------------- LSCM
def _tri_local_frame(p0, p1, p2):
    e1 = p1 - p0
    L = float(np.linalg.norm(e1))
    if L < 1e-14:
        return None
    e1 /= L
    v2 = p2 - p0
    x2 = float(v2 @ e1)
    w = v2 - x2 * e1
    y2 = float(np.linalg.norm(w))
    if y2 < 1e-14:
        return None
    return L, x2, y2


def lscm(chart: Chart, pins: dict) -> list:
    """pins: {local_vid: (u,v)}. Returns list of degenerate face indices."""
    P, F = chart.positions, chart.faces
    V = len(P)
    n_user_pins = len(pins)
    rows, cols, vals = [], [], []
    degenerate = []
    row = 0
    for fi, tri in enumerate(F):
        fr = _tri_local_frame(P[tri[0]], P[tri[1]], P[tri[2]])
        if fr is None:
            degenerate.append(fi)
            continue
        L, x2, y2 = fr
        dA = L * y2
        xs = (0.0, L, x2)
        ys = (0.0, 0.0, y2)
        ddx = np.array([ys[1] - ys[2], ys[2] - ys[0], ys[0] - ys[1]]) / dA
        ddy = np.array([xs[2] - xs[1], xs[0] - xs[2], xs[1] - xs[0]]) / dA
        w = np.sqrt(dA / 2.0)
        for k in range(3):
            i = int(tri[k])
            # du/dx - dv/dy = 0
            rows += [row, row]
            cols += [2 * i, 2 * i + 1]
            vals += [w * ddx[k], -w * ddy[k]]
            # du/dy + dv/dx = 0
            rows += [row + 1, row + 1]
            cols += [2 * i, 2 * i + 1]
            vals += [w * ddy[k], w * ddx[k]]
        row += 2
    A = sp.coo_matrix((vals, (rows, cols)), shape=(row, 2 * V)).tocsr()

    pins = dict(pins)
    topo = chart.topology or {}
    bedges = topo.get("boundary_edges") or []
    if len(pins) < 2 and bedges:
        # anchor one real boundary edge at its true 3D length: locally
        # consistent with the isometric map, gives well-shaped unwraps
        if len(pins) == 1:
            first = next(iter(pins))
            nbrs = [b if a == first else a for a, b in bedges
                    if a == first or b == first]
            if nbrs:
                second = max(nbrs,
                             key=lambda v: float(np.linalg.norm(P[v] - P[first])))
                if second not in pins:
                    d = float(np.linalg.norm(P[second] - P[first]))
                    pins[second] = (pins[first][0] + d, pins[first][1])
        if len(pins) == 0:
            a, b = max(bedges,
                       key=lambda e: float(np.linalg.norm(P[e[0]] - P[e[1]])))
            pins[a] = (0.0, 0.0)
            pins[b] = (float(np.linalg.norm(P[a] - P[b])), 0.0)
    if len(pins) < 2:
        # fallback: two boundary vertices farthest apart
        bnd = topo.get("boundary_verts") or list(range(V))
        bp = P[np.asarray(bnd)]
        if len(pins) == 1:
            first = next(iter(pins))
        else:
            first = bnd[0]
            pins[first] = (0.0, 0.0)
        d = np.linalg.norm(bp - P[first], axis=1)
        second = bnd[int(np.argmax(d))]
        if second != first and second not in pins:
            pins[second] = (float(d.max()), 0.0)

    pin_ids = sorted(pins)
    pin_cols = [2 * i for i in pin_ids] + [2 * i + 1 for i in pin_ids]
    pin_vals = np.array([pins[i][0] for i in pin_ids]
                        + [pins[i][1] for i in pin_ids], dtype=float)
    free = np.ones(2 * V, bool)
    free[pin_cols] = False
    Af = A[:, free]
    b = -A[:, pin_cols] @ pin_vals
    C = (Af.T @ Af).tocsc()
    rhs = Af.T @ b
    with np.errstate(all="ignore"):
        x = spla.spsolve(C, rhs)
    if not np.all(np.isfinite(x)):
        x = spla.lsqr(Af, b, iter_lim=2000)[0]
    flat = np.zeros(2 * V)
    flat[pin_cols] = pin_vals
    flat[free] = x
    uv = np.column_stack([flat[0::2], flat[1::2]])
    if n_user_pins < 2:
        # canonicalize auto-pinned charts only; user-pinned charts keep
        # the exact placement their anchors define
        signed = 0.0
        for tri in F:
            t = uv[tri]
            e1 = t[1] - t[0]
            e2 = t[2] - t[0]
            signed += e1[0] * e2[1] - e1[1] * e2[0]
        if signed < 0:
            uv[:, 1] *= -1.0
        span = uv.max(axis=0) - uv.min(axis=0)
        if span[1] > span[0]:  # longest axis horizontal
            uv = np.column_stack([-uv[:, 1], uv[:, 0]])
    chart.uv = uv
    return degenerate


# ---------------------------------------------------------------- packing
def normalize_chart_scale(chart: Chart):
    """Re-anchor LSCM's arbitrary output scale to geometry units, so the
    global texel density means the same thing on every chart."""
    P, F, UV = chart.positions, chart.faces, chart.uv
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    ga = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum()
    t0, t1, t2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    ua = 0.5 * np.abs((t1[:, 0] - t0[:, 0]) * (t2[:, 1] - t0[:, 1])
                      - (t1[:, 1] - t0[:, 1]) * (t2[:, 0] - t0[:, 0])).sum()
    if ua > 1e-30 and ga > 0:
        chart.uv = chart.uv * np.sqrt(ga / ua)


def pack_charts(charts, uv_scale: float, texture_size: int, pad_px: int) -> dict:
    """Uniform global scale for every chart; charts are only translated.
    Overflow goes to the next UDIM tile - charts are never rescaled to fit.
    Locked charts (pinned by the user) keep their exact placement."""
    pad = pad_px / float(texture_size)
    movable = []
    for c in charts:
        if c.locked:
            continue
        c.uv = c.uv * uv_scale
        movable.append(c)
    order = sorted(movable,
                   key=lambda c: -(c.uv[:, 1].max() - c.uv[:, 1].min()))
    x = y = pad
    shelf_h = 0.0
    tile = 0
    overflow_notes = []
    for c in order:
        mn = c.uv.min(axis=0)
        mx = c.uv.max(axis=0)
        w, h = float(mx[0] - mn[0]), float(mx[1] - mn[1])
        if x + w + pad > 1.0 and x > pad:
            y += shelf_h + pad
            x = pad
            shelf_h = 0.0
        if y + h + pad > 1.0:
            tile += 1
            x = y = pad
            shelf_h = 0.0
        if w + 2 * pad > 1.0:
            overflow_notes.append(
                f"图表 {c.index} 宽度 {w:.3f} 超过单张纹理，未缩放，已放入平铺 {tile}")
        c.uv = c.uv + np.array([tile + x - mn[0], y - mn[1]])
        c.tile = tile
        x += w + pad
        shelf_h = max(shelf_h, h)
    return dict(tiles=tile + 1, notes=overflow_notes)


# -------------------------------------------------------------- distortion
def triangle_distortion(chart: Chart):
    """Per-triangle Jacobian SVD: angle ratio (1 = conformal), signed area
    ratio, flip flag. Area ratio is also normalized per chart (mean = 1)."""
    P, F, UV = chart.positions, chart.faces, chart.uv
    out = []
    for tri in F:
        fr = _tri_local_frame(P[tri[0]], P[tri[1]], P[tri[2]])
        if fr is None:
            out.append(dict(angle=float("inf"), area=0.0, area_rel=0.0,
                            flipped=False, degenerate=True))
            continue
        L, x2, y2 = fr
        dA = L * y2
        xs = (0.0, L, x2)
        ys = (0.0, 0.0, y2)
        ddx = np.array([ys[1] - ys[2], ys[2] - ys[0], ys[0] - ys[1]]) / dA
        ddy = np.array([xs[2] - xs[1], xs[0] - xs[2], xs[1] - xs[0]]) / dA
        t = UV[tri]
        dudx, dudy = float(ddx @ t[:, 0]), float(ddy @ t[:, 0])
        dvdx, dvdy = float(ddx @ t[:, 1]), float(ddy @ t[:, 1])
        m00 = dudx * dudx + dvdx * dvdx
        m01 = dudx * dudy + dvdx * dvdy
        m11 = dudy * dudy + dvdy * dvdy
        tr = m00 + m11
        det = dudx * dvdy - dudy * dvdx
        disc = max(tr * tr / 4.0 - det * det, 0.0)
        sq = np.sqrt(disc)
        s1 = np.sqrt(max(tr / 2.0 + sq, 1e-30))
        s2 = np.sqrt(max(tr / 2.0 - sq, 1e-30))
        out.append(dict(angle=float(s1 / s2), area=float(det),
                        area_rel=0.0, flipped=bool(det < 0), degenerate=False))
    areas = np.array([d["area"] for d in out])
    geom_areas = np.array([
        0.5 * np.linalg.norm(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]))
        for t in F])
    pos = areas > 0
    if pos.any():
        mean_ratio = float((areas[pos] * geom_areas[pos]).sum()
                           / geom_areas[pos].sum())
        for d, g in zip(out, geom_areas):
            if d["area"] > 0 and mean_ratio > 0:
                d["area_rel"] = d["area"] / mean_ratio
    return out


# --------------------------------------------------------------- overlaps
def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _seg_intersect(p1, p2, p3, p4, eps=1e-12):
    d1 = _cross2(p4 - p3, p1 - p3)
    d2 = _cross2(p4 - p3, p2 - p3)
    d3 = _cross2(p2 - p1, p3 - p1)
    d4 = _cross2(p2 - p1, p4 - p1)
    return ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
           ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))


def _point_in_tri(p, t, eps=1e-12):
    d0 = _cross2(t[1] - t[0], p - t[0])
    d1 = _cross2(t[2] - t[1], p - t[1])
    d2 = _cross2(t[0] - t[2], p - t[2])
    has_neg = (d0 < -eps) or (d1 < -eps) or (d2 < -eps)
    has_pos = (d0 > eps) or (d1 > eps) or (d2 > eps)
    return not (has_neg and has_pos)


def _tris_overlap(t1, t2):
    for i in range(3):
        for j in range(3):
            if _seg_intersect(t1[i], t1[(i + 1) % 3], t2[j], t2[(j + 1) % 3]):
                return True
    return _point_in_tri(t1[0], t2) or _point_in_tri(t2[0], t1)


def find_overlaps(chart: Chart):
    """Overlapping triangle pairs that share no vertex (adjacent fold-overs
    are already reported as flipped triangles)."""
    F, UV = chart.faces, chart.uv
    n = len(F)
    bboxes = []
    for tri in F:
        t = UV[tri]
        bboxes.append((t[:, 0].min(), t[:, 1].min(), t[:, 0].max(), t[:, 1].max()))
    order = sorted(range(n), key=lambda i: bboxes[i][0])
    pairs = []
    for oi, i in enumerate(order):
        x1max = bboxes[i][2]
        for j in order[oi + 1:]:
            if bboxes[j][0] > x1max:
                break
            if bboxes[j][1] > bboxes[i][3] or bboxes[j][3] < bboxes[i][1]:
                continue
            if set(F[i]) & set(F[j]):
                continue
            if _tris_overlap(UV[F[i]], UV[F[j]]):
                pairs.append((i, j))
    return pairs

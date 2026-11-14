"""Procedural validation meshes: closed sleeve cylinder, thin long strip,
multi-component panels, tessellated cube (seam-loop test), non-manifold fan.

Each generator returns (positions, faces, suggested_seams) where
suggested_seams are canonical edge pairs usable directly as seam input.
"""
from __future__ import annotations

import numpy as np


def cylinder(n=24, m=8, radius=0.35, height=1.6):
    """Closed sleeve tube: annulus topology until one axial seam is marked."""
    verts = []
    for j in range(m + 1):
        y = height * j / m
        for i in range(n):
            a = 2 * np.pi * i / n
            verts.append([radius * np.cos(a), y, radius * np.sin(a)])
    faces = []
    for j in range(m):
        for i in range(n):
            i2 = (i + 1) % n
            a, b = j * n + i, j * n + i2
            c, d = (j + 1) * n + i2, (j + 1) * n + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    seams = [[j * n, (j + 1) * n] for j in range(m)]  # axial cut at i=0
    verts, faces = orient_outward(np.array(verts), np.array(faces))
    return verts, faces, seams


def strip(n=60, half_w=0.12, arc_radius=1.1, arc=2.4):
    """Thin long curved garment piece (developable band, S-curve in plane)."""
    verts = []
    cols = [-half_w, 0.0, half_w]
    for i in range(n + 1):
        t = i / n
        ang = arc * (t - 0.5)
        r = arc_radius + 0.25 * np.sin(3 * np.pi * t)
        cx, cy = r * np.cos(ang), r * np.sin(ang)
        # tangent-normal in plane for the width direction
        nx, ny = np.cos(ang), np.sin(ang)
        for w in cols:
            verts.append([cx + nx * w, cy + ny * w, 0.0])
    nc = len(cols)
    faces = []
    for i in range(n):
        for k in range(nc - 1):
            a, b = i * nc + k, i * nc + k + 1
            c, d = (i + 1) * nc + k + 1, (i + 1) * nc + k
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.array(verts), np.array(faces), []


def panels(n_u=14, n_v=18):
    """Two separate gently curved panels (front/back): two components."""
    verts, faces = [], []

    def panel(u0, z_bulge, phase):
        base = len(verts)
        for j in range(n_v + 1):
            v = j / n_v
            for i in range(n_u + 1):
                u = i / n_u
                x = u0 + 0.55 * (u - 0.5)
                y = 1.4 * (v - 0.5)
                z = z_bulge * (np.cos(np.pi * (u - 0.5)) - 0.6) \
                    + 0.05 * np.sin(phase + 2 * np.pi * v)
                verts.append([x, y, z])
        for j in range(n_v):
            for i in range(n_u):
                a = base + j * (n_u + 1) + i
                b = a + 1
                c = a + (n_u + 1) + 1
                d = a + (n_u + 1)
                faces.append([a, b, c])
                faces.append([a, c, d])

    panel(-0.45, 0.22, 0.0)
    panel(0.45, 0.18, 1.3)
    return np.array(verts), np.array(faces), []


def cube(k=2, size=1.0):
    """Tessellated closed cube; the equator seam loop splits it into 2 disks."""
    verts = []
    vid = {}

    def add_vert(p):
        key = tuple(np.round(p, 9))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(list(p))
        return vid[key]

    faces = []
    s = size
    # (origin, u_dir, v_dir) for each of the 6 faces
    specs = [
        ((0, 0, 0), (s, 0, 0), (0, s, 0)),  # bottom? use as -z face below
    ]
    specs = [
        ((0, 0, 0), (s, 0, 0), (0, s, 0)),          # z = 0
        ((0, 0, s), (s, 0, 0), (0, s, 0)),          # z = s
        ((0, 0, 0), (s, 0, 0), (0, 0, s)),          # y = 0
        ((0, s, 0), (s, 0, 0), (0, 0, s)),          # y = s
        ((0, 0, 0), (0, s, 0), (0, 0, s)),          # x = 0
        ((s, 0, 0), (0, s, 0), (0, 0, s)),          # x = s
    ]
    for origin, du, dv in specs:
        o = np.array(origin, float)
        du = np.array(du, float)
        dv = np.array(dv, float)
        grid = {}
        for j in range(k + 1):
            for i in range(k + 1):
                grid[(i, j)] = add_vert(o + du * (i / k) + dv * (j / k))
        for j in range(k):
            for i in range(k):
                a = grid[(i, j)]
                b = grid[(i + 1, j)]
                c = grid[(i + 1, j + 1)]
                d = grid[(i, j + 1)]
                faces.append([a, b, c])
                faces.append([a, c, d])
    verts = np.array(verts)
    # equator seam loop: edges with both endpoints at y = size/2
    seams = []
    edge_set = {}
    for tri in faces:
        for kk in range(3):
            a, b = tri[kk], tri[(kk + 1) % 3]
            key = (min(a, b), max(a, b))
            edge_set[key] = edge_set.get(key, 0) + 1
    for (a, b) in edge_set:
        ya, yb = verts[a][1], verts[b][1]
        if abs(ya - s / 2) < 1e-9 and abs(yb - s / 2) < 1e-9:
            seams.append([a, b])
    verts, faces = orient_outward(verts, np.array(faces))
    return verts, faces, seams


def mirror_panels(n_u=10, n_v=16, height=2.0):
    """Two mirrored curved cloth panels sewn along one straight center seam.

    Left and right halves are mirror images; without the center seam they are
    one connected panel whose curved outer borders force measurable stretch,
    cutting the seam yields two mirror islands whose shared border must have
    matching UV lengths (shared-edge error check)."""
    verts, faces = [], []
    seam_edges = []

    def half(sign):
        base = len(verts)
        for j in range(n_v + 1):
            v = j / n_v
            y = height * (v - 0.5)
            for i in range(n_u + 1):
                u = i / n_u
                # curved panel: depth bulge grows toward the outer (sewn) side
                x = sign * 0.6 * u
                z = 0.28 * np.sin(np.pi * v) * u
                verts.append([x, y, z])
        for j in range(n_v):
            for i in range(n_u):
                a = base + j * (n_u + 1) + i
                b = a + 1
                c = a + (n_u + 1) + 1
                d = a + (n_u + 1)
                faces.append([a, b, c])
                faces.append([a, c, d])
        return base

    base_l = half(-1.0)
    base_r = half(1.0)
    # sew center columns (i = 0 of each half): identical x,y,z -> weld them
    weld = {}
    for j in range(n_v + 1):
        vl = base_l + j * (n_u + 1)
        vr = base_r + j * (n_u + 1)
        weld[vr] = vl
    remap = {}
    out_verts = []
    for vi, p in enumerate(verts):
        root = weld.get(vi, vi)
        if root not in remap:
            remap[root] = len(out_verts)
            out_verts.append(verts[root])
    out_faces = [[remap[weld.get(a, a)], remap[weld.get(b, b)],
                  remap[weld.get(c, c)]] for a, b, c in faces]
    for j in range(n_v):
        a = remap[base_l + j * (n_u + 1)]
        b = remap[base_l + (j + 1) * (n_u + 1)]
        seam_edges.append([min(a, b), max(a, b)])
    return np.array(out_verts), np.array(out_faces), seam_edges


def nonmanifold():
    """Three triangles glued along one edge: classic non-manifold fan."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.0, 1.0],
    ])
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    return verts, faces, []


def orient_outward(verts, faces):
    """Flip triangles so normals point away from the mesh centroid."""
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, dtype=np.int64).copy()
    c = verts.mean(axis=0)
    for i, tri in enumerate(faces):
        p = verts[tri]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if n @ (p.mean(axis=0) - c) < 0:
            faces[i] = [tri[0], tri[2], tri[1]]
    return verts, faces


GENERATORS = {
    "cylinder": ("闭合袖筒（圆筒）", cylinder),
    "strip": ("细长弧形衣片", strip),
    "panels": ("前后双衣片（多连通块）", panels),
    "cube": ("立方体（缝线环切割）", cube),
    "mirror": ("镜像双图岛（共享接缝误差）", mirror_panels),
    "nonmanifold": ("非流形扇形（诊断演示）", nonmanifold),
}

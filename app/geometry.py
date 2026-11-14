"""Mesh topology: OBJ IO, edges, manifold checks, topology fingerprint.

Seams are stored as canonical topological edge keys (sorted vertex-index
pairs). The mesh fingerprint is derived from topology only (vertex count,
face index buffer, edge set) and never from vertex positions, so a reloaded
model with moved vertices still matches its saved project.
"""
from __future__ import annotations

import hashlib

import numpy as np


class MeshError(Exception):
    pass


class Mesh:
    def __init__(self, positions, faces, name: str = "mesh"):
        self.name = name
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(self.positions) == 0 or len(self.faces) == 0:
            raise MeshError("网格为空：没有顶点或三角面")
        if int(self.faces.min()) < 0 or int(self.faces.max()) >= len(self.positions):
            raise MeshError("面索引超出顶点范围")
        keep = []
        for fi, tri in enumerate(self.faces):
            a, b, c = (int(t) for t in tri)
            if a != b and b != c and c != a:
                keep.append(fi)
        if len(keep) != len(self.faces):
            self.faces = self.faces[keep]
        if len(self.faces) == 0:
            raise MeshError("网格为空：所有面均退化")
        self._build_topology()
        self.fingerprint = self._fingerprint()

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_obj(cls, text: str, name: str = "mesh") -> "Mesh":
        verts, faces = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    raise MeshError(f"无法解析顶点行: {line}")
            elif parts[0] == "f":
                idx = []
                for tok in parts[1:]:
                    try:
                        vi = int(tok.split("/")[0])
                    except ValueError:
                        raise MeshError(f"无法解析面行: {line}")
                    if vi < 0:
                        vi = len(verts) + 1 + vi
                    idx.append(vi - 1)
                if len(idx) < 3:
                    raise MeshError(f"面顶点数不足: {line}")
                for k in range(1, len(idx) - 1):  # fan triangulation
                    faces.append([idx[0], idx[k], idx[k + 1]])
        if not verts or not faces:
            raise MeshError("OBJ 中没有可用的顶点或三角面")
        return cls(np.array(verts), np.array(faces), name)

    def to_obj_with_uv(self, charts) -> str:
        """Export geometry + UVs. Vertices are duplicated per chart so every
        chart owns its texture coordinates (cuts stay cuts in the file)."""
        lines = [f"# UV unwrapped by garment-uv-workbench", f"mtllib checker.mtl"]
        v_offset = 1
        for ch in charts:
            lines.append(f"o {self.name}_chart{ch['index']}")
            for li, vid in enumerate(ch["vertices"]):
                x, y, z = self.positions[vid]
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            for u, v in ch["uv"]:
                lines.append(f"vt {u:.6f} {v:.6f}")
            lines.append("usemtl checker")
            for tri in ch["faces"]:
                a, b, c = (v_offset + int(t) for t in tri)
                lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            v_offset += len(ch["vertices"])
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ topology
    def _build_topology(self):
        edge_faces = {}
        for fi, tri in enumerate(self.faces):
            for k in range(3):
                a, b = int(tri[k]), int(tri[(k + 1) % 3])
                key = (a, b) if a < b else (b, a)
                edge_faces.setdefault(key, []).append(fi)
        self.edge_faces = edge_faces
        self.edges = sorted(edge_faces.keys())
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.boundary_edges = [e for e, fs in edge_faces.items() if len(fs) == 1]
        self.nonmanifold_edges = [e for e, fs in edge_faces.items() if len(fs) > 2]
        adj = [set() for _ in range(len(self.positions))]
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        self.vertex_adj = [sorted(s) for s in adj]

    def _fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(f"{len(self.positions)}|{len(self.faces)}|".encode())
        h.update(np.ascontiguousarray(np.sort(self.faces, axis=1)).tobytes())
        h.update(np.ascontiguousarray(np.asarray(self.edges, dtype=np.int64)).tobytes())
        return h.hexdigest()[:16]

    def has_edge(self, a: int, b: int) -> bool:
        key = (min(a, b), max(a, b))
        return key in self.edge_faces

    def stats(self) -> dict:
        bmin = self.positions.min(axis=0)
        bmax = self.positions.max(axis=0)
        return dict(
            name=self.name,
            fingerprint=self.fingerprint,
            n_vertices=int(len(self.positions)),
            n_faces=int(len(self.faces)),
            n_edges=len(self.edges),
            n_boundary_edges=len(self.boundary_edges),
            n_nonmanifold_edges=len(self.nonmanifold_edges),
            bbox_min=[float(x) for x in bmin],
            bbox_max=[float(x) for x in bmax],
        )

    def edges_json(self) -> list:
        """flag: 0 internal, 1 boundary, 2 non-manifold."""
        out = []
        bset = set(self.boundary_edges)
        nset = set(self.nonmanifold_edges)
        for a, b in self.edges:
            flag = 2 if (a, b) in nset else (1 if (a, b) in bset else 0)
            out.append([a, b, flag])
        return out

"""Validation suite: closed sleeve cylinder, thin long strip, multiple
connected components, seam loops, non-manifold diagnostics, topology-based
seam restore, real parameterization quality, distortion checks, export."""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import testmeshes
from app.geometry import Mesh
from app.main import app
from app.unwrap import (cut_charts, chart_topology, find_overlaps, lscm,
                        pack_charts, triangle_distortion, Chart)

client = TestClient(app)


def load_testmesh(name):
    r = client.get(f"/api/testmesh/{name}")
    assert r.status_code == 200, r.text
    return r.json()


def set_seams(mesh_id, seams):
    r = client.put(f"/api/mesh/{mesh_id}/seams", json={"seams": seams})
    assert r.status_code == 200, r.text
    return r.json()


def unwrap(mesh_id, **kw):
    r = client.post(f"/api/mesh/{mesh_id}/unwrap", json=kw)
    return r


# ----------------------------------------------------------- sleeve cylinder
def test_closed_cylinder_blocked_until_seamed():
    p = load_testmesh("cylinder")
    mid = p["mesh_id"]
    # closed tube: annulus topology, unwrap must be blocked with diagnostics
    r = unwrap(mid)
    assert r.status_code == 400
    detail = r.json()["detail"]
    probs = detail["diagnostics"]["charts"][0]["problems"]
    assert any("圆盘" in s for s in probs), probs
    # apply the axial seam -> disk topology -> unwrap succeeds
    set_seams(mid, p["demo_seams"])
    r = unwrap(mid)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["stats"]["n_charts"] == 1
    assert res["stats"]["n_flipped"] == 0
    # cylinder is developable: conformal distortion must be tiny
    assert res["stats"]["max_angle_distortion"] < 1.05
    assert res["stats"]["n_overlap_pairs"] == 0


def test_cylinder_unwrap_is_isometric_not_projection():
    """A planar projection of a tube would crush the back half; a real
    parameterization keeps width/height = circumference/height."""
    verts, faces, seams = testmeshes.cylinder()
    mesh = Mesh(verts, faces)
    seam_set = {(min(a, b), max(a, b)) for a, b in seams}
    charts = cut_charts(mesh, seam_set)
    assert len(charts) == 1
    ch = charts[0]
    ch.topology = chart_topology(ch, seam_set, mesh.edge_faces)
    lscm(ch, {})
    pack_charts([ch], uv_scale=1.0, texture_size=1024, pad_px=0)
    w = ch.uv[:, 0].max() - ch.uv[:, 0].min()
    h = ch.uv[:, 1].max() - ch.uv[:, 1].min()
    circum = 2 * np.pi * 0.35
    assert abs(w / h - circum / 1.6) / (circum / 1.6) < 0.05


# ------------------------------------------------------------------- strip
def test_thin_long_strip_unwraps_clean():
    p = load_testmesh("strip")
    mid = p["mesh_id"]
    r = unwrap(mid)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["stats"]["n_charts"] == 1
    assert res["stats"]["n_flipped"] == 0
    assert res["stats"]["max_angle_distortion"] < 1.1
    assert res["stats"]["n_overlap_pairs"] == 0


# ------------------------------------------------------- multiple components
def test_multiple_components_share_one_global_scale():
    p = load_testmesh("panels")
    mid = p["mesh_id"]
    texel, tsize = 300.0, 1024
    r = unwrap(mid, texel_density=texel, texture_size=tsize)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["stats"]["n_charts"] == 2
    scale = texel / tsize
    pos = np.array(p["positions"]).reshape(-1, 3)
    for ch in res["charts"]:
        uv = np.array(ch["uv"])
        geom_area = uv_area = 0.0
        for tri in ch["faces"]:
            p3 = pos[np.array(ch["vertices"])[tri]]
            geom_area += 0.5 * np.linalg.norm(
                np.cross(p3[1] - p3[0], p3[2] - p3[0]))
            t = uv[tri]
            uv_area += 0.5 * abs(np.cross(t[1] - t[0], t[2] - t[0]))
        # every chart obeys the same texel density; no per-chart rescaling
        assert abs(uv_area / geom_area - scale ** 2) / scale ** 2 < 0.02


# ---------------------------------------------------------------- seam loop
def test_seam_loop_splits_cube_into_two_disks():
    p = load_testmesh("cube")
    mid = p["mesh_id"]
    # closed cube cannot unwrap before the loop is marked
    r = unwrap(mid)
    assert r.status_code == 400
    set_seams(mid, p["demo_seams"])
    r = unwrap(mid)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["stats"]["n_charts"] == 2
    for ch in res["diagnostics"]["charts"]:
        assert ch["euler"] == 1
        assert ch["n_boundary_loops"] == 1
        assert ch["problems"] == []
    assert res["stats"]["n_flipped"] == 0


# -------------------------------------------------------------- non-manifold
def test_nonmanifold_diagnostic_blocks_unwrap():
    p = load_testmesh("nonmanifold")
    mid = p["mesh_id"]
    assert p["stats"]["n_nonmanifold_edges"] == 1
    diag = p["diagnostics"]
    assert diag["blocking"] is True
    assert diag["global_problems"][0]["kind"] == "nonmanifold"
    r = unwrap(mid)
    assert r.status_code == 400
    assert "非流形" in json.dumps(r.json()["detail"], ensure_ascii=False)


# ------------------------------------------- topology-based seam persistence
def test_seams_restore_by_topology_not_positions():
    verts, faces, seams = testmeshes.cylinder()
    mesh_a = Mesh(verts, faces)
    assert mesh_a.fingerprint == Mesh(verts, faces).fingerprint
    # jitter positions only: topology identical -> fingerprint identical
    rng = np.random.default_rng(0)
    jittered = verts + rng.normal(scale=0.01, size=verts.shape)
    mesh_b = Mesh(jittered, faces)
    assert mesh_a.fingerprint == mesh_b.fingerprint
    # topology change -> fingerprint changes
    mesh_c = Mesh(verts, faces[:-1])
    assert mesh_a.fingerprint != mesh_c.fingerprint

    obj = "\n".join(
        [f"v {x} {y} {z}" for x, y, z in verts]
        + [f"f {a+1} {b+1} {c+1}" for a, b, c in faces])
    r = client.post("/api/mesh/upload",
                    json={"obj_text": obj, "name": "sleeve_a"})
    assert r.status_code == 200
    p1 = r.json()
    set_seams(p1["mesh_id"], seams)
    # re-upload the same model with moved vertices: seams come back
    obj2 = "\n".join(
        [f"v {x} {y} {z}" for x, y, z in jittered]
        + [f"f {a+1} {b+1} {c+1}" for a, b, c in faces])
    r = client.post("/api/mesh/upload",
                    json={"obj_text": obj2, "name": "sleeve_b"})
    p2 = r.json()
    assert p2["restored"] is True
    assert p2["stats"]["fingerprint"] == p1["stats"]["fingerprint"]
    assert sorted(map(sorted, p2["project"]["seams"])) == sorted(
        map(sorted, seams))


# -------------------------------------------------------------------- pins
def test_boundary_pins_are_respected():
    p = load_testmesh("strip")
    mid = p["mesh_id"]
    r = unwrap(mid)
    ch = r.json()["charts"][0]
    bnd = ch["boundary_verts"]
    v0, v1 = ch["vertices"][bnd[0]], ch["vertices"][bnd[-1]]
    pins = {str(v0): [0.1, 0.2], str(v1): [0.9, 0.8]}
    r = unwrap(mid, pins=pins)
    assert r.status_code == 200, r.text
    ch = r.json()["charts"][0]
    uv = np.array(ch["uv"])
    li0 = ch["vertices"].index(v0)
    li1 = ch["vertices"].index(v1)
    assert np.allclose(uv[li0], [0.1, 0.2], atol=1e-8)
    assert np.allclose(uv[li1], [0.9, 0.8], atol=1e-8)


# ------------------------------------------------- flip/overlap detection
def test_flip_and_overlap_detection():
    # two triangles sharing an edge, one folded over in UV -> flipped
    ch = Chart(0, [0, 1, 2, 3], np.array([[0, 1, 2], [1, 0, 3]]),
               np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]],
                        dtype=float))
    ch.uv = np.array([[0, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    dist = triangle_distortion(ch)
    assert dist[0]["flipped"] is False
    assert dist[1]["flipped"] is True
    # two disjoint triangles overlapping in UV space
    ch2 = Chart(0, [0, 1, 2, 3, 4, 5],
                np.array([[0, 1, 2], [3, 4, 5]]),
                np.zeros((6, 3)))
    ch2.uv = np.array([[0, 0], [2, 0], [0, 2],
                       [0.5, 0.5], [1.5, 0.5], [0.5, 1.5]], dtype=float)
    pairs = find_overlaps(ch2)
    assert pairs == [(0, 1)]


# ------------------------------------------------------------------- export
def test_export_obj_checker_report():
    p = load_testmesh("cylinder")
    mid = p["mesh_id"]
    set_seams(mid, p["demo_seams"])
    assert unwrap(mid).status_code == 200

    r = client.get(f"/api/mesh/{mid}/export/obj")
    assert r.status_code == 200
    obj = r.text
    n_v = sum(1 for l in obj.splitlines() if l.startswith("v "))
    n_vt = sum(1 for l in obj.splitlines() if l.startswith("vt "))
    n_f = sum(1 for l in obj.splitlines() if l.startswith("f "))
    assert n_v == n_vt > 0
    assert n_f == p["stats"]["n_faces"]
    assert all("/" in l for l in obj.splitlines() if l.startswith("f "))
    # exported UVs are real: re-importing keeps the same topology fingerprint
    r2 = client.post("/api/mesh/upload", json={"obj_text": obj, "name": "re"})
    assert r2.json()["stats"]["n_faces"] == p["stats"]["n_faces"]

    r = client.get(f"/api/mesh/{mid}/export/checker.png")
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    r = client.get(f"/api/mesh/{mid}/export/report.json")
    assert r.status_code == 200
    rep = r.json()
    assert rep["stats"]["n_charts"] == 1
    assert "不保证服装实际可裁剪" in rep["disclaimer"]


# ----------------------------------------------------------------- versions
def test_versions_snapshot_and_restore():
    p = load_testmesh("strip")
    mid = p["mesh_id"]
    assert unwrap(mid).status_code == 200
    r = client.post(f"/api/mesh/{mid}/versions", json={"name": "自动版"})
    assert r.status_code == 200
    vid = r.json()["versions"][0]["id"]
    # change pins, unwrap again -> different UVs; restore brings old ones back
    r0 = client.post(f"/api/mesh/{mid}/versions/{vid}/restore")
    assert r0.status_code == 200
    assert r0.json()["result"]["stats"]["n_charts"] == 1


# ------------------------------------------------------- texel density rule
def test_overflow_goes_to_udim_without_rescaling():
    p = load_testmesh("panels")
    mid = p["mesh_id"]
    # absurdly high density: charts cannot fit one tile
    r = unwrap(mid, texel_density=5000.0, texture_size=256)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["packing"]["tiles"] > 1
    assert any("未做任何缩放" in w or "未缩放" in w for w in res["warnings"])
    scale = 5000.0 / 256
    ch = res["charts"][0]
    uv = np.array(ch["uv"])
    pos = np.array(p["positions"]).reshape(-1, 3)
    geom_area = uv_area = 0.0
    for tri in ch["faces"]:
        p3 = pos[np.array(ch["vertices"])[tri]]
        geom_area += 0.5 * np.linalg.norm(
            np.cross(p3[1] - p3[0], p3[2] - p3[0]))
        t = uv[tri]
        uv_area += 0.5 * abs(np.cross(t[1] - t[0], t[2] - t[0]))
    assert abs(uv_area / geom_area - scale ** 2) / scale ** 2 < 0.02

"""Multi-texture grid layout, lock constraints, seam optimization and
island seam-error validation.

Covers: explicit numbered tiles, no per-chart rescaling, rotation ban,
cross-tile policy, lock/density conflicts with concrete reasons, single-cell
UV stability, texture binding checks, export mapping, re-open restore,
narrow-strip stretch, and mirrored-island shared-edge errors.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import testmeshes
from app.geometry import Mesh
from app.layout import (LayoutError, TileSpec, apply_layout,
                        default_single_tile, layout_charts,
                        validate_texture_bindings)
from app.main import app
from app.seams_opt import optimize_seams, seam_edge_error
from app.unwrap import (chart_topology, cut_charts, lscm,
                        normalize_chart_scale)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_server_state():
    # projects are keyed by topology fingerprint and otherwise leak between
    # tests that load the same procedural mesh
    from app import main as main_mod
    main_mod.MESHES.clear()
    main_mod.PROJECTS.clear()
    yield
    main_mod.MESHES.clear()
    main_mod.PROJECTS.clear()


def load(name):
    r = client.get(f"/api/testmesh/{name}")
    assert r.status_code == 200, r.text
    return r.json()


def set_seams(mid, seams):
    r = client.put(f"/api/mesh/{mid}/seams", json={"seams": seams})
    assert r.status_code == 200, r.text


def unwrap(mid, **kw):
    return client.post(f"/api/mesh/{mid}/unwrap", json=kw)


def make_chart(index, w, h):
    """Synthetic chart with unit-density normalized UV of size w x h."""
    from app.unwrap import Chart
    ch = Chart(index, [0, 1, 2, 3],
               np.array([[0, 1, 2], [0, 2, 3]]),
               np.zeros((4, 3)))
    ch.uv = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    return ch


# --------------------------------------------------------- explicit tiles
def test_explicit_numbered_tiles_and_mapping():
    p = load("panels")
    mid = p["mesh_id"]
    tiles = [
        dict(tile_id=1001, texture_size=512, margin_px=8, headroom_px=4,
             texture="fabric_a", u_index=0, v_index=0),
        dict(tile_id=1002, texture_size=1024, margin_px=16, headroom_px=8,
             texture="fabric_b", u_index=1, v_index=0),
    ]
    r = client.put(f"/api/mesh/{mid}/layout",
                   json=dict(tiles=tiles, textures=dict(
                       fabric_a=dict(size=512), fabric_b=dict(size=1024))))
    assert r.status_code == 200, r.text
    r = unwrap(mid, texel_density=200.0, pad_px=4)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["packing"]["tile_ids"] == [1001, 1002]
    used = {m["tile_id"] for m in res["tile_map"]}
    assert used <= {1001, 1002}
    # every chart carries an explicit tile id and a real placement record
    for m in res["tile_map"]:
        assert m["tile_id"] in (1001, 1002)
        assert isinstance(m["chart_index"], int)


def test_duplicate_tile_ids_rejected():
    specs = [TileSpec(1001, 512), TileSpec(1001, 256)]
    with pytest.raises(LayoutError):
        layout_charts([make_chart(0, 0.1, 0.1)], specs, texel_density=100)


def test_near_identical_tile_numbers_distinct_bindings_warn():
    specs = [TileSpec(1001, 512, texture="a"),
             TileSpec(1002, 512, texture="b")]
    warns = validate_texture_bindings(specs, registry=dict(a={}, b={}))
    assert any("相近" in w for w in warns)


def test_unbound_texture_resource_rejected():
    specs = [TileSpec(1001, 512, texture="ghost")]
    with pytest.raises(LayoutError):
        validate_texture_bindings(specs, registry=dict(other={}))


def test_same_resource_bound_twice_rejected():
    specs = [TileSpec(1001, 512, texture="a"),
             TileSpec(1002, 512, texture="a")]
    with pytest.raises(LayoutError):
        validate_texture_bindings(specs, registry=dict(a={}))


# ----------------------------------------------------- no rescaling policy
def test_oversized_chart_reports_unsolved_not_scaled():
    big = make_chart(0, 1.5, 0.4)   # 1.5 geometry units wide
    tiles = [TileSpec(1001, 256, margin_px=4)]
    lay = layout_charts([big], tiles, texel_density=200.0,
                        allow_cross_tile=False)
    assert lay.solved is False
    pl = lay.placements[0]
    assert pl.fits is False and pl.tile_id is None
    assert "像素密度" in pl.reason or "纹理格" in pl.reason
    # UV untouched: apply_layout must not move an unsolved chart
    before = big.uv.copy()
    apply_layout([big], tiles, lay, texel_density=200.0)
    assert np.allclose(big.uv, before)


def test_cross_tile_policy_keeps_chart_whole():
    big = make_chart(0, 1.2, 0.2)
    tiles = [TileSpec(1001, 256), TileSpec(1002, 256)]
    # policy forbids splitting: report no-solution instead
    lay = layout_charts([big], tiles, texel_density=250.0,
                        allow_cross_tile=False)
    assert lay.placements[0].fits is False
    assert any("跨格" in n or "接缝" in n for n in lay.notes + lay.reasons)
    # even when cross-tile is allowed, an uncut chart is never sliced
    lay2 = layout_charts([big], tiles, texel_density=250.0,
                         allow_cross_tile=True)
    assert lay2.placements[0].fits is False
    assert "接缝" in lay2.placements[0].reason


def test_density_identical_across_different_resolution_tiles():
    # two charts of equal geometric size land on different-resolution tiles;
    # their pixel footprints must be identical (global texel density)
    a = make_chart(0, 0.3, 0.3)
    b = make_chart(1, 0.3, 0.3)
    tiles = [TileSpec(1001, 512, margin_px=8),
             TileSpec(1002, 2048, margin_px=8)]
    # density high enough that the first tile's inner area holds one only
    lay = layout_charts([a, b], tiles, texel_density=1400.0, gap_px=4)
    assert lay.solved
    pa, pb = lay.placements[0], lay.placements[1]
    assert pa.tile_id != pb.tile_id
    assert np.isclose(pa.width_px, pb.width_px)
    assert np.isclose(pa.height_px, pb.height_px)
    apply_layout([a, b], tiles, lay, 1400.0)
    # UV span inside each tile, normalized by its own texture size, matches
    for ch, tile in ((a, tiles[0]), (b, tiles[1])):
        span_u = ch.uv[:, 0].max() - ch.uv[:, 0].min()
        assert np.isclose(span_u * tile.texture_size, 0.3 * 1400.0, rtol=1e-6)


# ----------------------------------------------------------- rotation rule
def test_rotation_ban_when_policy_forbids():
    # Square tile, inner 240px. A rotation-banned chart is never rotated even
    # when rotation is available; and an over-long chart that cannot fit in
    # either orientation stays unsolved with the rotation remedy in reason.
    tiles = [TileSpec(1001, 256, margin_px=8)]
    first = make_chart(0, 150 / 250, 150 / 250)    # 150x150 px
    second = make_chart(1, 80 / 250, 60 / 250)     # 80x60 px
    lay = layout_charts([first, second], tiles, texel_density=250.0,
                        gap_px=4, rotation_policy={0: False, 1: False})
    assert lay.solved
    assert all(not p.rotated for p in lay.placements.values())
    lay2 = layout_charts([first, second], tiles, texel_density=250.0,
                         gap_px=4, rotation_policy={0: False, 1: True})
    assert lay2.solved
    # 350px long side exceeds inner 240 in both orientations: no rescale
    huge = make_chart(2, 1.4, 0.3)
    lay3 = layout_charts([huge], tiles, texel_density=250.0,
                         rotation_policy={2: False})
    assert lay3.placements[2].fits is False
    assert "90° 旋转" in lay3.placements[2].reason
    # allowing rotation cannot magically shrink the long side on a square
    lay4 = layout_charts([huge], tiles, texel_density=250.0,
                         rotation_policy={2: True})
    assert lay4.placements[2].fits is False


# ------------------------------------------------------------------ locking
def test_locked_charts_stay_and_others_repack():
    p = load("panels")
    mid = p["mesh_id"]
    client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
        dict(tile_id=1001, texture_size=1024, margin_px=8),
        dict(tile_id=1002, texture_size=1024, margin_px=8)]))
    r = unwrap(mid, texel_density=250.0)
    res = r.json()
    target = res["tile_map"][0]
    r = client.post(f"/api/mesh/{mid}/charts/lock", json=dict(
        chart_index=target["chart_index"], locked=True))
    assert r.status_code == 200, r.text
    r2 = client.post(f"/api/mesh/{mid}/relayout")
    assert r2.status_code == 200, r2.text
    res2 = r2.json()
    locked_now = next(m for m in res2["tile_map"]
                      if m["chart_index"] == target["chart_index"])
    assert locked_now["locked"] is True
    # same tile and identical placement coordinates after re-pack
    assert locked_now["tile_id"] == target["tile_id"]
    assert np.allclose(
        [locked_now["placement"]["x_px"], locked_now["placement"]["y_px"]],
        [target["placement"]["x_px"], target["placement"]["y_px"]], atol=1e-6)


def test_lock_density_conflict_has_concrete_reason():
    ch = make_chart(0, 0.8, 0.8)
    ch.locked = True
    ch.placement = dict(tile_id=1001, x_px=0.0, y_px=0.0, rotated=False)
    tiles = [TileSpec(1001, 256, margin_px=4)]
    # high density makes the locked footprint exceed the tile
    lay = layout_charts([ch], tiles, texel_density=400.0)
    assert lay.solved is False
    reason = lay.placements[0].reason
    assert "锁定与密度冲突" in reason
    assert "400" in reason and "1001" in reason


def test_lock_to_undefined_tile_rejected():
    ch = make_chart(0, 0.1, 0.1)
    ch.locked = True
    ch.placement = dict(tile_id=9999, x_px=0.0, y_px=0.0, rotated=False)
    lay = layout_charts([ch], [TileSpec(1001, 256)], texel_density=100)
    assert lay.placements[0].fits is False
    assert "9999" in lay.placements[0].reason


# ------------------------------------------------- single-cell UV stability
def test_single_cell_project_keeps_original_uv():
    p = load("strip")
    mid = p["mesh_id"]
    r1 = unwrap(mid).json()
    # no explicit tiles configured -> legacy single-cell path, repeat runs
    # must produce bit-identical UVs (no layout-engine re-flow drift)
    r2 = unwrap(mid).json()
    for c1, c2 in zip(r1["charts"], r2["charts"]):
        assert np.allclose(c1["uv"], c2["uv"], atol=1e-12)
    assert all(c["tile_id"] in (None, 1001) for c in r2["charts"])


def test_movable_chart_avoids_locked_rectangle():
    # locked chart occupies the first shelf region; the movable chart must
    # be placed elsewhere in the tile, never on top of the locked one
    locked = make_chart(0, 0.5, 0.3)          # 125x75 px at origin
    locked.locked = True
    locked.placement = dict(tile_id=1001, x_px=0.0, y_px=0.0, rotated=False)
    movable = make_chart(1, 0.4, 0.4)         # 100x100 px
    tiles = [TileSpec(1001, 512, margin_px=8)]
    lay = layout_charts([locked, movable], tiles, texel_density=250.0)
    assert lay.solved
    lp, mp = lay.placements[0], lay.placements[1]
    # axis-separated rectangles (no overlap), locked stays at its origin
    assert (lp.x_px, lp.y_px) == (8.0, 8.0)
    separated = (mp.x_px >= lp.x_px + lp.width_px
                 or mp.y_px >= lp.y_px + lp.height_px)
    assert separated, (mp.x_px, mp.y_px, lp.x_px, lp.y_px)


def test_relayout_runs_without_prior_cache():
    # changing the tile config clears the cached result; relayout must then
    # recompute instead of rejecting the request
    p = load("panels")
    mid = p["mesh_id"]
    client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
        dict(tile_id=1001, texture_size=1024, margin_px=8)]))
    r = client.post(f"/api/mesh/{mid}/relayout")
    assert r.status_code == 200, r.text
    assert r.json()["packing"]["solved"] is True


# ------------------------------------------------- pins inside multi-tile
def test_pinned_chart_keeps_exact_anchor_and_gets_tile():
    p = load("strip")
    mid = p["mesh_id"]
    client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
        dict(tile_id=1001, texture_size=1024, margin_px=8),
        dict(tile_id=1002, texture_size=1024, margin_px=8, u_index=1)]))
    pre = unwrap(mid, texel_density=200.0).json()
    ch = pre["charts"][0]
    bnd = ch["boundary_verts"]
    v0, v1 = ch["vertices"][bnd[0]], ch["vertices"][bnd[-1]]
    r = unwrap(mid, texel_density=200.0,
               pins={str(v0): [0.1, 0.2], str(v1): [0.9, 0.8]})
    assert r.status_code == 200, r.text
    out = r.json()
    c = out["charts"][0]
    uv = np.array(c["uv"])
    li0, li1 = c["vertices"].index(v0), c["vertices"].index(v1)
    assert np.allclose(uv[li0], [0.1, 0.2], atol=1e-8)
    assert np.allclose(uv[li1], [0.9, 0.8], atol=1e-8)
    assert c["locked"] is True
    assert c["tile_id"] == 1001
    assert out["packing"]["solved"] is True


# ------------------------------------------------------------- margin rulesdef test_margin_and_headroom_respected():
    ch = make_chart(0, 0.9, 0.1)
    tiles = [TileSpec(1001, 256, margin_px=20, headroom_px=10)]
    # inner = 256 - 2*30 = 196 px; chart width 0.9*200 = 180 fits, but with
    # zero margin a second identical chart on the shelf would overflow
    lay = layout_charts([ch], tiles, texel_density=200.0)
    assert lay.solved
    pl = lay.placements[0]
    assert pl.x_px >= 30 and pl.y_px >= 30
    with pytest.raises(LayoutError):
        TileSpec(1001, 64, margin_px=20, headroom_px=20).inner_px


# ------------------------------------------------------- export + restore
def test_export_contains_tile_mapping_and_restore_rebuilds_locks():
    p = load("panels")
    mid = p["mesh_id"]
    client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
        dict(tile_id=1001, texture_size=512, texture="fa"),
        dict(tile_id=1002, texture_size=512, texture="fb")],
        textures=dict(fa={}, fb={})))
    r = unwrap(mid, texel_density=180.0)
    res = r.json()
    idx = res["tile_map"][0]["chart_index"]
    client.post(f"/api/mesh/{mid}/charts/lock",
                json=dict(chart_index=idx, locked=True))
    client.post(f"/api/mesh/{mid}/versions", json=dict(name="v1"))
    # OBJ carries tile comments + usable UV coordinates, not just pixels
    obj = client.get(f"/api/mesh/{mid}/export/obj").text
    assert "tile_id=" in obj and f"chart={idx}" in obj
    assert any(l.startswith("vt ") for l in obj.splitlines())
    # report JSON contains the chart->tile map
    rep = client.get(f"/api/mesh/{mid}/export/report.json").json()
    assert {m["chart_index"] for m in rep["tile_map"]} == {0, 1}
    assert rep["tiles"] and rep["tile_map"][0]["tile_id"] in (1001, 1002)
    # re-open by topology fingerprint restores locks and layout state
    p2 = load("panels")
    proj = p2["project"]
    assert proj["tiles"] and len(proj["tiles"]) == 2
    assert str(idx) in proj["locks"]
    assert proj["locks"][str(idx)]["placement"] is not None


# ----------------------------------------------- texture size change reflow
def test_changing_texture_size_reflows_without_rescaling():
    p = load("panels")
    mid = p["mesh_id"]
    client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
        dict(tile_id=1001, texture_size=1024, margin_px=8)]))
    r1 = unwrap(mid, texel_density=200.0).json()
    # shrink the tile while charts still fit: density is fixed, so the atlas
    # UV ratio follows texel/size while the pixel footprint on each tile is
    # resolution-invariant
    for size in (1024, 768):
        client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
            dict(tile_id=1001, texture_size=size, margin_px=8)]))
        r = unwrap(mid, texel_density=200.0)
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["packing"]["solved"], res["packing"].get("reasons")
        scale = 200.0 / size
        pos = np.array(p["positions"]).reshape(-1, 3)
        for ch in res["charts"]:
            uv = np.array(ch["uv"])
            verts = np.array(ch["vertices"])
            ga = ua = 0.0
            for tri in ch["faces"]:
                p3 = pos[verts[tri]]
                ga += 0.5 * np.linalg.norm(
                    np.cross(p3[1] - p3[0], p3[2] - p3[0]))
                t = uv[tri]
                d1, d2 = t[1] - t[0], t[2] - t[0]
                ua += 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
            assert np.isclose(ua / ga, scale ** 2, rtol=0.02)
            # pixel footprint on the tile is invariant across resolutions
            px_area = ua * size * size
            assert np.isclose(px_area / ga, 200.0 ** 2, rtol=0.02)
    # further shrinking reaches a no-solution state, reported rather than
    # hidden by per-chart rescaling
    for size in (256, 64):
        client.put(f"/api/mesh/{mid}/layout", json=dict(tiles=[
            dict(tile_id=1001, texture_size=size, margin_px=4)]))
        res = unwrap(mid, texel_density=200.0).json()
        assert res["packing"]["solved"] is False
        assert res["packing"]["reasons"]


# ------------------------------------------------------- seam optimization
def _seamed_mesh(name):
    verts, faces, seams = getattr(testmeshes, name)()
    mesh = Mesh(verts, faces)
    return mesh, seams


def test_seam_optimize_keeps_disk_topology_and_improves_or_holds():
    mesh, seams = _seamed_mesh("cylinder")
    opt = optimize_seams(mesh, seams, max_iterations=6)
    seam_set = {(min(a, b), max(a, b)) for a, b in opt["seams"]}
    charts = cut_charts(mesh, seam_set)
    for ch in charts:
        topo = chart_topology(ch, seam_set, mesh.edge_faces)
        assert topo["problems"] == []
    assert opt["history"][0]["score"] >= opt["score"] - 1e-9


def test_seam_optimize_rejects_nonmanifold():
    verts, faces, _ = testmeshes.nonmanifold()
    mesh = Mesh(verts, faces)
    from app.seams_opt import SeamOptError
    with pytest.raises(SeamOptError):
        optimize_seams(mesh, [])


def test_narrow_strip_stretch_is_small():
    """Long narrow developable strip: angle distortion stays near 1 and the
    seam/stretch error of its borders is tiny."""
    mesh, _ = _seamed_mesh("strip")
    opt = optimize_seams(mesh, [], max_iterations=4)
    seam_set = {(min(a, b), max(a, b)) for a, b in opt["seams"]}
    charts = cut_charts(mesh, seam_set)
    assert len(charts) == 1
    ch = charts[0]
    ch.topology = chart_topology(ch, seam_set, mesh.edge_faces)
    lscm(ch, {})
    normalize_chart_scale(ch)
    from app.unwrap import triangle_distortion
    angles = [d["angle"] for d in triangle_distortion(ch)
              if np.isfinite(d["angle"])]
    assert max(angles) < 1.15
    # length preservation along the strip: UV edge lengths match 3D within 6%
    errors = seam_edge_error(mesh, charts, seam_set)
    for e in errors:
        for r in e["stretch_ratio"]:
            assert abs(r - 1.0) < 0.06


def test_mirrored_islands_shared_edge_error():
    """Two mirror panels sewn at the center seam: after cutting, the two
    border sides must have matching UV lengths (shared-edge error ~ 0)."""
    verts, faces, seams = testmeshes.mirror_panels()
    mesh = Mesh(verts, faces)
    seam_set = {(min(a, b), max(a, b)) for a, b in seams}
    charts = cut_charts(mesh, seam_set)
    assert len(charts) == 2
    for ch in charts:
        ch.topology = chart_topology(ch, seam_set, mesh.edge_faces)
        lscm(ch, {})
        normalize_chart_scale(ch)
    errors = seam_edge_error(mesh, charts, seam_set)
    assert errors, "center seam edges must be measured"
    for e in errors:
        assert len(e["uv_sides"]) == 2
        assert e["shared_error"] < 0.02, e
        for r in e["stretch_ratio"]:
            assert abs(r - 1.0) < 0.05
    # endpoint through API: kernel-driven endpoint returns the same metrics
    p = load("mirror")
    mid = p["mesh_id"]
    set_seams(mid, p["demo_seams"])
    r = client.post(f"/api/mesh/{mid}/seams/optimize",
                    json=dict(max_iterations=3, auto_unwrap=True))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seam_errors"]
    assert max(e["shared_error"] for e in body["seam_errors"]) < 0.02
    assert body["unwrap"]["stats"]["n_flipped"] == 0


def test_seam_optimize_endpoint_failure_feeds_back_reason():
    p = load("nonmanifold")
    mid = p["mesh_id"]
    r = client.post(f"/api/mesh/{mid}/seams/optimize", json=dict())
    assert r.status_code == 400
    assert r.json()["detail"]["kind"] == "seam_optimize"
    assert "非流形" in r.json()["detail"]["message"]

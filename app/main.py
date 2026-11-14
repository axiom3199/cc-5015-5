"""FastAPI backend for the garment UV unwrapping workbench."""
from __future__ import annotations

import struct
import time
import urllib.parse
import zlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               Response)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import testmeshes
from .geometry import Mesh, MeshError
from .layout import (ChartPlacement, LayoutError, TileSpec, apply_layout,
                     default_single_tile, layout_charts,
                     validate_texture_bindings)
from .seams_opt import SeamOptError, optimize_seams
from .unwrap import (Chart, chart_topology, cut_charts, find_overlaps, lscm,
                     normalize_chart_scale, pack_charts, triangle_distortion)

DISCLAIMER = ("自动展开结果仅供 UV 布局参考，不保证服装实际可裁剪性；"
              "请结合角度/面积畸变、翻转与重叠检查以及人工调整确认。")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Garment UV Workbench")

MESHES: dict[str, Mesh] = {}
PROJECTS: dict[str, "Project"] = {}


class Project:
    def __init__(self, mesh_id: str):
        self.mesh_id = mesh_id
        self.seams: list[list[int]] = []
        self.pins: dict[int, list[float]] = {}
        self.params = dict(texel_density=256.0, texture_size=1024, pad_px=8)
        self.versions: list[dict] = []
        self.last_result: dict | None = None
        self.last_charts: list[Chart] | None = None
        # multi-texture grid layout state
        self.tiles: list[dict] = []          # explicit numbered tile specs
        self.allow_cross_tile = False        # project policy
        self.rotation_policy: dict[str, bool] = {}
        self.locks: dict[str, dict] = {}     # chart index -> lock constraint
        self.textures: dict[str, dict] = {}  # resource key -> metadata


def _new_id() -> str:
    return f"{time.time_ns():x}{np.random.randint(0, 1 << 30):09x}"


def _get_mesh(mesh_id: str) -> Mesh:
    if mesh_id not in MESHES:
        raise HTTPException(404, "网格不存在，请重新上传")
    return MESHES[mesh_id]


def _get_project(mesh: Mesh) -> Project:
    proj = PROJECTS.get(mesh.fingerprint)
    if proj is None:
        raise HTTPException(404, "项目不存在")
    return proj


def _seam_set(proj: Project) -> set:
    return {(min(a, b), max(a, b)) for a, b in proj.seams}


def _build_tiles(proj: Project) -> list[TileSpec]:
    """Materialize explicit numbered tile specs; an empty config means the
    legacy single-cell project (one 1001 tile)."""
    if not proj.tiles:
        return [default_single_tile(int(proj.params["texture_size"]),
                                    int(proj.params["pad_px"]))]
    specs = []
    for t in proj.tiles:
        specs.append(TileSpec(
            tile_id=int(t["tile_id"]),
            texture_size=int(t["texture_size"]),
            margin_px=int(t.get("margin_px", 0)),
            headroom_px=int(t.get("headroom_px", 0)),
            texture=t.get("texture"),
            u_index=int(t.get("u_index", 0)),
            v_index=int(t.get("v_index", 0))))
    return specs


def _apply_lock_state(charts, proj: Project):
    """Restore saved lock constraints onto freshly parameterized charts."""
    for ch in charts:
        saved = proj.locks.get(str(ch.index))
        if saved is None:
            continue
        ch.locked = True
        ch.rotation_locked = not bool(saved.get("allow_rotation", False))
        ch.placement = saved.get("placement")


def _diagnostics(mesh: Mesh, proj: Project) -> dict:
    """Topology diagnostics computed before any parameterization attempt."""
    glob = []
    if mesh.nonmanifold_edges:
        glob.append(dict(
            kind="nonmanifold",
            message=f"发现 {len(mesh.nonmanifold_edges)} 条非流形边"
                    f"（被 3 个或以上面共享），请先修复网格",
            edges=[[int(a), int(b)] for a, b in mesh.nonmanifold_edges]))
    charts = cut_charts(mesh, _seam_set(proj))
    chart_diags = []
    for ch in charts:
        topo = chart_topology(ch, _seam_set(proj), mesh.edge_faces)
        ch.topology = topo
        chart_diags.append(dict(
            index=ch.index, n_faces=len(ch.faces),
            n_vertices=len(ch.vertices),
            n_boundary_loops=topo["n_boundary_loops"],
            euler=topo["euler"], problems=topo["problems"]))
    blocking = bool(glob) or any(c["problems"] for c in chart_diags)
    return dict(global_problems=glob, charts=chart_diags, blocking=blocking)


def _run_unwrap(mesh: Mesh, proj: Project) -> dict:
    diag = _diagnostics(mesh, proj)
    if diag["blocking"]:
        raise HTTPException(400, detail=dict(
            message="存在阻塞性问题，未执行展开", diagnostics=diag))
    seam_set = _seam_set(proj)
    charts = cut_charts(mesh, seam_set)
    p = proj.params
    texel_density = float(p["texel_density"])
    texture_size = int(p["texture_size"])
    pad_px = int(p["pad_px"])
    uv_scale = texel_density / float(texture_size)

    degenerate_faces = []
    raw_uvs = {}
    pinned_idx = set()
    for ch in charts:
        ch.topology = chart_topology(ch, seam_set, mesh.edge_faces)
        user_pins = {}
        for li, vid in enumerate(ch.vertices):
            if vid in proj.pins:
                user_pins[li] = tuple(proj.pins[vid])
        distinct = {tuple(np.round(uv, 9)) for uv in user_pins.values()}
        if len(user_pins) >= 2 and len(distinct) >= 2:
            # pinned points are anchors in final UV space: solve in raw
            # space, then only the global scale maps back - no re-normalize,
            # no packer offset; pinned vertices land exactly where placed
            pins_local = {li: (uv[0] / uv_scale, uv[1] / uv_scale)
                          for li, uv in user_pins.items()}
            degen = lscm(ch, pins_local)
            ch.uv = ch.uv * uv_scale
            ch.locked = True
            pinned_idx.add(ch.index)
        elif user_pins:
            # single anchor: solve naturally, then translate the chart so
            # the pinned vertex lands on its target
            degen = lscm(ch, {})
            normalize_chart_scale(ch)
            li, target = next(iter(user_pins.items()))
            ch.uv = ch.uv * uv_scale
            ch.uv = ch.uv + (np.asarray(target) - ch.uv[li])
            ch.locked = True
            pinned_idx.add(ch.index)
        else:
            degen = lscm(ch, {})
            normalize_chart_scale(ch)
        # geometry-unit UVs (density-normalized) are the layout engine input
        raw_uvs[ch.index] = ch.uv.copy()
        for d in degen:
            degenerate_faces.append([ch.index, int(d)])

    multi_tile = bool(proj.tiles)
    layout_conflicts = []
    if multi_tile:
        # explicit numbered texture grid: density/rotation/lock hard rules,
        # per-tile texture resolution, no per-chart rescaling. The layout
        # works in geometry units + global texel density; final atlas UVs
        # are emitted per tile resolution inside apply_layout.
        for ch in charts:
            ch.uv = raw_uvs[ch.index]
        _apply_lock_state(charts, proj)
        tiles = _build_tiles(proj)
        warnings_bind = validate_texture_bindings(tiles, proj.textures or None)
        rotation_policy = {int(k): bool(v)
                           for k, v in proj.rotation_policy.items()}
        # user-pinned charts keep their exact anchored UVs and only need a
        # tile assignment; the layout engine packs the remaining charts.
        packable = [ch for ch in charts if ch.index not in pinned_idx]
        layout = layout_charts(
            packable, tiles, texel_density, gap_px=pad_px,
            rotation_policy=rotation_policy,
            allow_cross_tile=proj.allow_cross_tile)
        apply_layout(packable, tiles, layout, texel_density)
        # assign pinned charts to the tile containing their anchor, without
        # moving any vertex (anchors must remain bit-exact)
        for ch in charts:
            if ch.index in pinned_idx:
                anchor = ch.uv
                u0, v0 = float(anchor[:, 0].min()), float(anchor[:, 1].min())
                tile = next((t for t in tiles
                             if t.u_index <= u0 < t.u_index + 1
                             and t.v_index <= v0 < t.v_index + 1), tiles[0])
                ch.tile = tile.u_index + 10 * tile.v_index
                ch.tile_id = tile.tile_id
                ch.placement = dict(tile_id=tile.tile_id, x_px=0.0,
                                    y_px=0.0, rotated=False)
                layout.placements[ch.index] = ChartPlacement(
                    ch.index, tile_id=tile.tile_id, locked=True, fits=True)
        # unsolved charts keep clean geometry-unit UVs (no stale atlas
        # offset), tile_id/tile marked as unassigned
        for ch in packable:
            pl = layout.placements.get(ch.index)
            if pl is None or not pl.fits:
                ch.uv = raw_uvs[ch.index]
                ch.tile = -1
                ch.tile_id = None
                ch.placement = None
        packing = dict(
            tiles=len(tiles),
            tile_ids=[t.tile_id for t in tiles],
            notes=list(layout.notes) + warnings_bind,
            solved=layout.solved,
            reasons=layout.reasons,
            placements={
                str(idx): dict(tile_id=p.tile_id, rotated=p.rotated,
                               locked=p.locked, fits=p.fits, reason=p.reason,
                               x_px=p.x_px, y_px=p.y_px,
                               width_px=p.width_px, height_px=p.height_px)
                for idx, p in layout.placements.items()},
            cross_tile_allowed=proj.allow_cross_tile)
        layout_conflicts = layout.reasons
    else:
        packing = pack_charts(charts, uv_scale, texture_size, pad_px)

    charts_json = []
    max_angle, sum_angle, n_tris = 0.0, 0.0, 0
    n_flipped = 0
    for ch in charts:
        ch.tri_dist = triangle_distortion(ch)
        ch.overlaps = find_overlaps(ch)
        for d in ch.tri_dist:
            if d["flipped"]:
                n_flipped += 1
            elif np.isfinite(d["angle"]):
                max_angle = max(max_angle, d["angle"])
                sum_angle += d["angle"]
                n_tris += 1
        charts_json.append(dict(
            index=ch.index,
            vertices=ch.vertices,
            faces=ch.faces.tolist(),
            uv=[[float(u), float(v)] for u, v in ch.uv],
            tile=ch.tile,
            tile_id=getattr(ch, "tile_id", None),
            locked=bool(getattr(ch, "locked", False)),
            rotation_locked=bool(getattr(ch, "rotation_locked", False)),
            placement=getattr(ch, "placement", None),
            boundary_verts=ch.topology["boundary_verts"],
            distortion=[dict(angle=d["angle"], area=d["area"],
                             area_rel=d["area_rel"], flipped=d["flipped"])
                        for d in ch.tri_dist],
            overlaps=[[int(a), int(b)] for a, b in ch.overlaps],
        ))
    n_overlap_pairs = sum(len(c["overlaps"]) for c in charts_json)
    warnings = list(packing["notes"])
    if multi_tile:
        if not packing["solved"]:
            warnings.append(
                "布局无解：以下约束冲突未通过缩放掩盖，已保留整片图表——"
                + "；".join(layout_conflicts))
    elif packing["tiles"] > 1:
        warnings.append(
            f"图表超出单张纹理，已排入 {packing['tiles']} 个 UDIM 平铺；"
            f"所有图表保持同一像素密度，未做任何缩放")
    if degenerate_faces:
        warnings.append(f"跳过 {len(degenerate_faces)} 个退化三角形（零面积）")
    if n_flipped:
        warnings.append(f"存在 {n_flipped} 个翻转三角形（UV 面积为负），已在 2D 视图标红")
    if n_overlap_pairs:
        warnings.append(f"检测到 {n_overlap_pairs} 对重叠三角形，已在 2D 视图高亮")

    result = dict(
        ok=True,
        diagnostics=diag,
        charts=charts_json,
        params=dict(texel_density=texel_density, texture_size=texture_size,
                    pad_px=pad_px, uv_scale=uv_scale),
        packing=packing,
        tile_map=[dict(chart_index=c["index"], tile=c["tile"],
                       tile_id=c["tile_id"], locked=c["locked"],
                       rotation_locked=c["rotation_locked"],
                       placement=c["placement"])
                  for c in charts_json],
        tiles=proj.tiles,
        allow_cross_tile=proj.allow_cross_tile,
        rotation_policy={str(k): v for k, v in proj.rotation_policy.items()},
        warnings=warnings,
        stats=dict(
            max_angle_distortion=float(max_angle),
            mean_angle_distortion=float(sum_angle / max(n_tris, 1)),
            n_flipped=int(n_flipped),
            n_overlap_pairs=int(n_overlap_pairs),
            n_charts=len(charts),
        ),
        disclaimer=DISCLAIMER,
    )
    if multi_tile:
        # persist actual placements of locked charts so re-open restores
        # every lock constraint and the real layout, not just a screenshot
        for ch in charts:
            if ch.locked and getattr(ch, "placement", None) is not None:
                prev = proj.locks.get(str(ch.index), {})
                proj.locks[str(ch.index)] = dict(
                    placement=ch.placement,
                    allow_rotation=not bool(
                        getattr(ch, "rotation_locked",
                                not prev.get("allow_rotation", False))),
                    tile_id=ch.placement.get("tile_id"))
    proj.last_result = result
    proj.last_charts = charts
    return result


# ------------------------------------------------------------------ models
class UploadBody(BaseModel):
    obj_text: str
    name: str = "mesh"


class SeamsBody(BaseModel):
    seams: list[list[int]]


class UnwrapBody(BaseModel):
    texel_density: float | None = None
    texture_size: int | None = None
    pad_px: int | None = None
    pins: dict[int, list[float]] | None = None


class TileBody(BaseModel):
    tile_id: int
    texture_size: int
    margin_px: int = 0
    headroom_px: int = 0
    texture: str | None = None
    u_index: int = 0
    v_index: int = 0


class LayoutBody(BaseModel):
    tiles: list[TileBody] | None = None
    allow_cross_tile: bool | None = None
    rotation_policy: dict[str, bool] | None = None
    textures: dict[str, dict] | None = None


class LockBody(BaseModel):
    chart_index: int
    locked: bool
    allow_rotation: bool = False
    tile_id: int | None = None


class SeamOptBody(BaseModel):
    max_iterations: int = 12
    max_seams: int | None = None
    auto_unwrap: bool = True


class VersionBody(BaseModel):
    name: str


# ----------------------------------------------------------------- helpers
def _project_json(proj: Project) -> dict:
    return dict(
        seams=proj.seams,
        pins={str(k): v for k, v in proj.pins.items()},
        params=proj.params,
        tiles=proj.tiles,
        allow_cross_tile=proj.allow_cross_tile,
        rotation_policy={str(k): v for k, v in proj.rotation_policy.items()},
        locks=proj.locks,
        textures=proj.textures,
        versions=[dict(id=v["id"], name=v["name"], created=v["created"])
                  for v in proj.versions],
    )


def _register_mesh(mesh: Mesh) -> tuple[str, Project, bool]:
    mesh_id = _new_id()
    MESHES[mesh_id] = mesh
    proj = PROJECTS.get(mesh.fingerprint)
    restored = proj is not None
    if not restored:
        proj = Project(mesh_id)
        PROJECTS[mesh.fingerprint] = proj
    proj.mesh_id = mesh_id
    return mesh_id, proj, restored


def _mesh_payload(mesh: Mesh, mesh_id: str, proj: Project, restored: bool,
                  demo_seams=None) -> dict:
    payload = dict(
        mesh_id=mesh_id,
        stats=mesh.stats(),
        positions=mesh.positions.reshape(-1).tolist(),
        faces=mesh.faces.reshape(-1).tolist(),
        edges=mesh.edges_json(),
        restored=restored,
        project=_project_json(proj),
        diagnostics=_diagnostics(mesh, proj),
    )
    if demo_seams is not None:
        payload["demo_seams"] = demo_seams
    return payload


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/mesh/upload")
def upload(body: UploadBody):
    try:
        mesh = Mesh.from_obj(body.obj_text, body.name or "mesh")
    except MeshError as e:
        raise HTTPException(400, str(e))
    mesh_id, proj, restored = _register_mesh(mesh)
    return _mesh_payload(mesh, mesh_id, proj, restored)


@app.get("/api/testmesh/{name}")
def testmesh(name: str):
    if name not in testmeshes.GENERATORS:
        raise HTTPException(404, "未知测试网格")
    label, gen = testmeshes.GENERATORS[name]
    verts, faces, seams = gen()
    mesh = Mesh(verts, faces, name=label)
    mesh_id, proj, restored = _register_mesh(mesh)
    demo = [[int(a), int(b)] for a, b in seams]
    return _mesh_payload(mesh, mesh_id, proj, restored, demo_seams=demo)


@app.get("/api/mesh/{mesh_id}")
def get_mesh(mesh_id: str):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    return _mesh_payload(mesh, mesh_id, proj, restored=True)


@app.put("/api/mesh/{mesh_id}/seams")
def set_seams(mesh_id: str, body: SeamsBody):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    seams = []
    for pair in body.seams:
        if len(pair) != 2:
            raise HTTPException(400, "接缝边必须是顶点对")
        a, b = int(pair[0]), int(pair[1])
        if not mesh.has_edge(a, b):
            raise HTTPException(400, f"边 ({a},{b}) 不存在于网格拓扑中")
        seams.append([min(a, b), max(a, b)])
    proj.seams = sorted({tuple(s) for s in seams})
    proj.seams = [list(s) for s in proj.seams]
    proj.last_result = None
    return dict(ok=True, diagnostics=_diagnostics(mesh, proj),
                project=_project_json(proj))


@app.post("/api/mesh/{mesh_id}/unwrap")
def unwrap(mesh_id: str, body: UnwrapBody):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    if body.texel_density is not None:
        proj.params["texel_density"] = float(body.texel_density)
    if body.texture_size is not None:
        proj.params["texture_size"] = int(body.texture_size)
    if body.pad_px is not None:
        proj.params["pad_px"] = int(body.pad_px)
    if body.pins is not None:
        proj.pins = {int(k): [float(x), float(y)]
                     for k, (x, y) in
                     ((k, v) for k, v in body.pins.items())}
    try:
        return _run_unwrap(mesh, proj)
    except HTTPException:
        raise


@app.put("/api/mesh/{mesh_id}/layout")
def set_layout(mesh_id: str, body: LayoutBody):
    """Configure the explicit numbered texture grid and project policies."""
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    if body.tiles is not None:
        raw = [t.model_dump() for t in body.tiles]
        try:
            specs = _build_tiles_from(raw)
            validate_texture_bindings(
                specs, body.textures or proj.textures or None)
        except LayoutError as e:
            raise HTTPException(400, str(e))
        proj.tiles = raw
    if body.allow_cross_tile is not None:
        proj.allow_cross_tile = bool(body.allow_cross_tile)
    if body.rotation_policy is not None:
        proj.rotation_policy = {str(k): bool(v)
                                for k, v in body.rotation_policy.items()}
    if body.textures is not None:
        proj.textures = dict(body.textures)
    proj.last_result = None
    return dict(ok=True, project=_project_json(proj))


def _build_tiles_from(raw: list[dict]) -> list[TileSpec]:
    specs = []
    for t in raw:
        specs.append(TileSpec(
            tile_id=int(t["tile_id"]),
            texture_size=int(t["texture_size"]),
            margin_px=int(t.get("margin_px", 0)),
            headroom_px=int(t.get("headroom_px", 0)),
            texture=t.get("texture"),
            u_index=int(t.get("u_index", 0)),
            v_index=int(t.get("v_index", 0))))
    return specs


@app.post("/api/mesh/{mesh_id}/charts/lock")
def lock_chart(mesh_id: str, body: LockBody):
    """Lock/unlock one chart. Locked charts keep their exact placement on
    subsequent unwraps; only the other charts are re-packed."""
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    key = str(body.chart_index)
    if body.locked:
        if proj.last_charts is None:
            raise HTTPException(400, "请先执行一次展开，再锁定图表位置")
        ch = next((c for c in proj.last_charts
                   if c.index == body.chart_index), None)
        if ch is None:
            raise HTTPException(404, f"图表 {body.chart_index} 不存在")
        placement = getattr(ch, "placement", None)
        if placement is None:
            # derive placement from the current atlas UV + tile spec
            tiles = _build_tiles(proj)
            tile = next((t for t in tiles
                         if t.tile_id == (body.tile_id or
                                         getattr(ch, "tile_id", 1001))),
                        tiles[0])
            mn = ch.uv.min(axis=0)
            local_u = mn[0] - tile.u_index
            local_v = mn[1] - tile.v_index
            placement = dict(
                tile_id=tile.tile_id,
                x_px=float(local_u * tile.texture_size
                           - tile.margin_px - tile.headroom_px),
                y_px=float(local_v * tile.texture_size
                           - tile.margin_px - tile.headroom_px),
                rotated=False)
        proj.locks[key] = dict(
            placement=placement, allow_rotation=body.allow_rotation,
            tile_id=placement["tile_id"])
        if not body.allow_rotation:
            proj.rotation_policy[key] = False
    else:
        proj.locks.pop(key, None)
    return dict(ok=True, project=_project_json(proj))


@app.post("/api/mesh/{mesh_id}/relayout")
def relayout(mesh_id: str):
    """Re-run the layout only: locked charts stay put, the rest re-pack."""
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    # relayout re-runs the layout with current constraints; a missing cache
    # simply means we compute fresh instead of rejecting the request
    return _run_unwrap(mesh, proj)


@app.post("/api/mesh/{mesh_id}/seams/optimize")
def optimize(mesh_id: str, body: SeamOptBody):
    """Island seam optimization driven by the unwrap kernel. Failures return
    structured reasons that the 3D viewport surfaces directly."""
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    try:
        opt = optimize_seams(mesh, proj.seams,
                             max_iterations=body.max_iterations,
                             max_seams=body.max_seams)
    except SeamOptError as e:
        raise HTTPException(400, dict(message=str(e), kind="seam_optimize"))
    proj.seams = opt["seams"]
    proj.last_result = None
    payload = dict(
        ok=True, seams=opt["seams"], score=opt["score"],
        history=opt["history"], converged=opt["converged"],
        stretch={str(k): v for k, v in opt["stretch"].items()},
        seam_errors=[dict(edge=e["edge"], len_3d=e["len_3d"],
                          uv_sides=e["uv_sides"],
                          stretch_ratio=e["stretch_ratio"],
                          shared_error=e["shared_error"])
                     for e in opt["seam_errors"]],
        diagnostics=_diagnostics(mesh, proj))
    if body.auto_unwrap and not payload["diagnostics"]["blocking"]:
        payload["unwrap"] = _run_unwrap(mesh, proj)
    return payload


@app.post("/api/mesh/{mesh_id}/versions")
def save_version(mesh_id: str, body: VersionBody):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    if proj.last_result is None:
        raise HTTPException(400, "请先执行展开，再保存版本")
    vid = _new_id()[:12]
    proj.versions.append(dict(
        id=vid, name=body.name or f"版本 {len(proj.versions) + 1}",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        seams=[list(s) for s in proj.seams],
        pins={str(k): list(v) for k, v in proj.pins.items()},
        params=dict(proj.params),
        tiles=[dict(t) for t in proj.tiles],
        allow_cross_tile=proj.allow_cross_tile,
        rotation_policy=dict(proj.rotation_policy),
        locks={k: dict(v) for k, v in proj.locks.items()},
        textures=dict(proj.textures),
        result=proj.last_result,
    ))
    return dict(ok=True, versions=_project_json(proj)["versions"])


@app.post("/api/mesh/{mesh_id}/versions/{version_id}/restore")
def restore_version(mesh_id: str, version_id: str):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    ver = next((v for v in proj.versions if v["id"] == version_id), None)
    if ver is None:
        raise HTTPException(404, "版本不存在")
    proj.seams = [list(s) for s in ver["seams"]]
    proj.pins = {int(k): list(v) for k, v in ver["pins"].items()}
    proj.params = dict(ver["params"])
    proj.tiles = [dict(t) for t in ver.get("tiles", [])]
    proj.allow_cross_tile = bool(ver.get("allow_cross_tile", False))
    proj.rotation_policy = dict(ver.get("rotation_policy", {}))
    proj.locks = {k: dict(v) for k, v in ver.get("locks", {}).items()}
    proj.textures = dict(ver.get("textures", {}))
    proj.last_result = ver["result"]
    return dict(ok=True, project=_project_json(proj), result=ver["result"])


@app.get("/api/mesh/{mesh_id}/export/obj")
def export_obj(mesh_id: str):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    if proj.last_result is None:
        raise HTTPException(400, "请先执行展开再导出")
    text = mesh.to_obj_with_uv(proj.last_result["charts"])
    fname = urllib.parse.quote(f"{mesh.name}_uv.obj")
    return PlainTextResponse(
        text, media_type="text/plain",
        headers={"Content-Disposition":
                 f"attachment; filename=uv_export.obj; "
                 f"filename*=UTF-8''{fname}"})


def _checker_png(size: int = 512, cells: int = 8) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell = size // cells
    for j in range(cells):
        for i in range(cells):
            even = (i + j) % 2 == 0
            color = (235, 235, 235) if even else (40, 40, 40)
            img[j * cell:(j + 1) * cell, i * cell:(i + 1) * cell] = color
    # red grid lines at cell borders to spot stretching
    img[::cell, :, 0] = 200
    img[:, ::cell, 0] = 200

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(size))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    return png


@app.get("/api/mesh/{mesh_id}/export/checker.png")
def export_checker(mesh_id: str):
    _get_mesh(mesh_id)
    return Response(content=_checker_png(), media_type="image/png",
                    headers={"Content-Disposition":
                             'attachment; filename="checker.png"'})


@app.get("/api/mesh/{mesh_id}/export/report.json")
def export_report(mesh_id: str):
    mesh = _get_mesh(mesh_id)
    proj = _get_project(mesh)
    if proj.last_result is None:
        raise HTTPException(400, "请先执行展开再导出")
    res = proj.last_result
    report = dict(
        mesh=mesh.stats(),
        params=res["params"],
        packing=res["packing"],
        tiles=res.get("tiles", []),
        tile_map=res.get("tile_map", []),
        allow_cross_tile=res.get("allow_cross_tile", False),
        stats=res["stats"],
        warnings=res["warnings"],
        charts=[dict(index=c["index"], n_faces=len(c["faces"]),
                     n_vertices=len(c["vertices"]), tile=c["tile"],
                     tile_id=c.get("tile_id"),
                     locked=c.get("locked", False),
                     placement=c.get("placement"),
                     n_flipped=sum(1 for d in c["distortion"] if d["flipped"]),
                     n_overlap_pairs=len(c["overlaps"]))
                for c in res["charts"]],
        disclaimer=DISCLAIMER,
    )
    return JSONResponse(report, headers={
        "Content-Disposition": 'attachment; filename="uv_report.json"'})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

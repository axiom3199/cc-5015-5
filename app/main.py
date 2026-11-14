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
        elif user_pins:
            # single anchor: solve naturally, then translate the chart so
            # the pinned vertex lands on its target
            degen = lscm(ch, {})
            normalize_chart_scale(ch)
            li, target = next(iter(user_pins.items()))
            ch.uv = ch.uv * uv_scale
            ch.uv = ch.uv + (np.asarray(target) - ch.uv[li])
            ch.locked = True
        else:
            degen = lscm(ch, {})
            normalize_chart_scale(ch)
        for d in degen:
            degenerate_faces.append([ch.index, int(d)])

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
            boundary_verts=ch.topology["boundary_verts"],
            distortion=[dict(angle=d["angle"], area=d["area"],
                             area_rel=d["area_rel"], flipped=d["flipped"])
                        for d in ch.tri_dist],
            overlaps=[[int(a), int(b)] for a, b in ch.overlaps],
        ))
    n_overlap_pairs = sum(len(c["overlaps"]) for c in charts_json)
    warnings = list(packing["notes"])
    if packing["tiles"] > 1:
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


class VersionBody(BaseModel):
    name: str


# ----------------------------------------------------------------- helpers
def _project_json(proj: Project) -> dict:
    return dict(
        seams=proj.seams,
        pins={str(k): v for k, v in proj.pins.items()},
        params=proj.params,
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
        stats=res["stats"],
        warnings=res["warnings"],
        charts=[dict(index=c["index"], n_faces=len(c["faces"]),
                     n_vertices=len(c["vertices"]), tile=c["tile"],
                     n_flipped=sum(1 for d in c["distortion"] if d["flipped"]),
                     n_overlap_pairs=len(c["overlaps"]))
                for c in res["charts"]],
        disclaimer=DISCLAIMER,
    )
    return JSONResponse(report, headers={
        "Content-Disposition": 'attachment; filename="uv_report.json"'})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

"""Multi-texture grid layout: explicit numbered texture tiles with hard
pixel-density, rotation and lock constraints.

Charts are never per-tile rescaled. Every chart's footprint is computed from
its already density-normalized UVs; the layout only translates (and rotates
in 90 degree steps when the chart policy allows it). A chart that does not
fit any tile is reported as an unsolved placement with a concrete reason —
the packer never "solves" overflow by shrinking it.

All geometry inside this module is handled in *pixels*: tile canvas size in
px, chart footprints in px, margins/sampling headroom in px. The final UV
mapping is produced from the pixel placement, so texel density is identical
on every chart regardless of the tile's texture resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class LayoutError(Exception):
    """Raised when constraints cannot be satisfied at all."""


@dataclass
class TileSpec:
    """One explicitly numbered texture grid cell."""
    tile_id: int                     # explicit user-visible number (e.g. 1001)
    texture_size: int                # square canvas px for this tile
    margin_px: int = 0               # border kept empty (px)
    headroom_px: int = 0             # extra sampling headroom inside the tile
    texture: str | None = None       # bound texture resource key
    u_index: int = 0                 # integer U offset in the exported atlas
    v_index: int = 0

    @property
    def inner_px(self) -> int:
        """Usable square after margin on both sides plus sampling headroom."""
        used = 2 * (self.margin_px + self.headroom_px)
        inner = self.texture_size - used
        if inner <= 0:
            raise LayoutError(
                f"纹理格 {self.tile_id}：边距 {self.margin_px}px 与采样余量 "
                f"{self.headroom_px}px 已占满或超出 {self.texture_size}px 画布")
        return inner


@dataclass
class ChartPlacement:
    chart_index: int
    tile_id: int | None = None
    x_px: float = 0.0                # min corner inside the tile canvas
    y_px: float = 0.0
    width_px: float = 0.0
    height_px: float = 0.0
    rotated: bool = False
    locked: bool = False
    fits: bool = True
    reason: str | None = None        # concrete no-fit / conflict explanation


@dataclass
class LayoutResult:
    placements: dict = field(default_factory=dict)   # chart index -> placement
    tile_ids: list = field(default_factory=list)
    solved: bool = True
    reasons: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _footprint_px(uv: np.ndarray, scale_px: float):
    """Axis-aligned footprint in texture pixels at the global texel density."""
    mn = uv.min(axis=0)
    mx = uv.max(axis=0)
    w = float((mx[0] - mn[0]) * scale_px)
    h = float((mx[1] - mn[1]) * scale_px)
    return mn, w, h


def _validate_tiles(tiles: list[TileSpec]):
    if not tiles:
        raise LayoutError("至少需要一个纹理格")
    ids = [t.tile_id for t in tiles]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise LayoutError(f"纹理格编号必须唯一，重复编号：{dup}")
    textures = [t.texture for t in tiles if t.texture]
    if len(set(textures)) != len(textures):
        dup = sorted({x for x in textures if textures.count(x) > 1})
        raise LayoutError(f"同一纹理资源被多个格绑定：{dup}")
    for t in tiles:
        if t.texture_size <= 0:
            raise LayoutError(f"纹理格 {t.tile_id} 的纹理尺寸必须为正数")
        _ = t.inner_px  # raises on impossible margins
    # near-identical ids are legal but must not alias texture resources
    by_tex = {}
    for t in tiles:
        if t.texture:
            by_tex.setdefault(t.texture, []).append(t.tile_id)


def validate_texture_bindings(tiles: list[TileSpec],
                              registry: dict | None = None) -> list:
    """Check that every bound texture key resolves in the project's texture
    registry and that near-equal tile numbers never share a resource by typo.
    Returns a list of human-readable warnings; raises LayoutError on hard
    binding errors."""
    _validate_tiles(tiles)
    warnings_ = []
    registry = registry or {}
    for t in tiles:
        if t.texture and registry and t.texture not in registry:
            raise LayoutError(
                f"纹理格 {t.tile_id} 绑定的纹理资源 '{t.texture}' "
                f"不在项目纹理清单中（可用：{sorted(registry)}）")
    ids = sorted(t.tile_id for t in tiles)
    for a, b in zip(ids, ids[1:]):
        if abs(a - b) <= 2:
            ta = next(t for t in tiles if t.tile_id == a)
            tb = next(t for t in tiles if t.tile_id == b)
            if ta.texture and tb.texture and ta.texture != tb.texture:
                warnings_.append(
                    f"格号 {a} 与 {b} 数字相近但绑定不同纹理 "
                    f"('{ta.texture}' / '{tb.texture}')，请确认未引用错资源")
    return warnings_


class _Bin:
    """One tile's remaining shelf state in pixels (shelf packing).

    Coordinates are absolute canvas pixels: the usable region starts at
    (origin, origin) where origin = margin + headroom, so every placement
    automatically respects border margin and sampling headroom."""
    def __init__(self, spec: TileSpec):
        self.spec = spec
        self.origin = spec.margin_px + spec.headroom_px
        self.limit = spec.inner_px              # usable side length
        self.x = float(self.origin)
        self.y = float(self.origin)
        self.shelf_h = 0.0
        self.rects = []   # occupied rects for locked-chart collision checks

    def _overlaps_reserved(self, x, y, w, h):
        for rx, ry, rw, rh in self.rects:
            if (x < rx + rw and x + w > rx and y < ry + rh
                    and y + h > ry):
                return True
        return False

    def place(self, w, h, gap):
        if self.x + w > self.origin + self.limit and self.x > self.origin:
            self.y += self.shelf_h + gap
            self.x = float(self.origin)
            self.shelf_h = 0.0
        if w > self.limit or h > self.limit:
            return None
        if self.y + h > self.origin + self.limit:
            return None
        # walk shelves forward until clear of every locked rectangle
        guard = 0
        while self._overlaps_reserved(self.x, self.y, w, h):
            self.x += w + gap
            if self.x + w > self.origin + self.limit:
                self.y += self.shelf_h + gap
                self.x = float(self.origin)
                self.shelf_h = 0.0
            if self.y + h > self.origin + self.limit:
                return None
            guard += 1
            if guard > 10000:
                return None
        rect = (self.x, self.y, w, h)
        self.rects.append(rect)
        self.x += w + gap
        self.shelf_h = max(self.shelf_h, h)
        return rect

    def reserve(self, x, y, w, h):
        """Register an externally fixed rectangle (absolute canvas px).
        Fails on out-of-bounds or overlap with another locked rectangle."""
        if x < self.origin or y < self.origin:
            return False
        if x + w > self.origin + self.limit or y + h > self.origin + self.limit:
            return False
        for rx, ry, rw, rh in self.rects:
            if (x < rx + rw and x + w > rx and y < ry + rh
                    and y + h > ry):
                return False
        self.rects.append((x, y, w, h))
        return True


def _rotated_uv(uv: np.ndarray) -> np.ndarray:
    # 90 degree CCW rotation about the origin; translation is normalized later
    return np.column_stack([-uv[:, 1], uv[:, 0]])


def layout_charts(charts, tiles: list[TileSpec], texel_density: float,
                  gap_px: int = 0,
                  rotation_policy: dict | None = None,
                  allow_cross_tile: bool = False) -> LayoutResult:
    """Place charts into explicit numbered tiles.

    charts: objects carrying .index, .uv (density-normalized geometry-unit
            UVs, i.e. *before* the texel scale), .locked and optional
            .rotation_locked / .target_tile / .placement (saved transform).
    texel_density: px per geometry unit — global, identical for every chart.
    rotation_policy: {chart_index: bool allow_90deg_rotation}; default False.
    allow_cross_tile: project policy. When False a chart bigger than every
            tile is reported unsolved instead of being split across tiles.
    """
    _validate_tiles(tiles)
    rotation_policy = rotation_policy or {}
    result = LayoutResult(tile_ids=[t.tile_id for t in tiles])
    gap = float(gap_px)
    bins = {t.tile_id: _Bin(t) for t in tiles}
    by_id = {t.tile_id: t for t in tiles}

    movable, locked = [], []
    for ch in charts:
        (locked if getattr(ch, "locked", False) else movable).append(ch)

    # ------------------------------------------------------------- locked
    for ch in locked:
        spec = ch.placement if hasattr(ch, "placement") else None
        scale = texel_density
        mn, w, h = _footprint_px(ch.uv, scale)
        pl = ChartPlacement(ch.index, locked=True, width_px=w, height_px=h)
        if spec is None:
            pl.fits = False
            pl.reason = (f"图表 {ch.index} 已锁定但没有保存的布局位置，"
                         f"无法在不移动它的情况下参与布局")
            result.reasons.append(pl.reason)
            result.solved = False
            result.placements[ch.index] = pl
            continue
        tile_id = spec.get("tile_id")
        if tile_id not in by_id:
            pl.fits = False
            pl.reason = (f"锁定冲突：图表 {ch.index} 锁定在纹理格 {tile_id}，"
                         f"但当前工程只定义了格 {[t.tile_id for t in tiles]}")
            result.reasons.append(pl.reason)
            result.solved = False
            result.placements[ch.index] = pl
            continue
        tile = by_id[tile_id]
        if spec.get("rotated"):
            w, h = h, w
            pl.rotated = True
        # density conflict: locked footprint must still fit its tile at the
        # *current* global density — the lock cannot override texel density
        if w > tile.inner_px or h > tile.inner_px:
            pl.fits = False
            pl.tile_id = tile_id
            pl.reason = (
                f"锁定与密度冲突：图表 {ch.index} 在 {texel_density:g} px/单位下"
                f"占用 {w:.1f}×{h:.1f}px，超过纹理格 {tile_id} 可用区域 "
                f"{tile.inner_px}×{tile.inner_px}px（纹理 {tile.texture_size}px，"
                f"边距 {tile.margin_px}px，采样余量 {tile.headroom_px}px）；"
                f"本系统不会缩放图表，请降低像素密度、放大纹理格或解除锁定")
            result.reasons.append(pl.reason)
            result.solved = False
            result.placements[ch.index] = pl
            continue
        x = float(spec["x_px"]) + tile.margin_px + tile.headroom_px
        y = float(spec["y_px"]) + tile.margin_px + tile.headroom_px
        if not bins[tile_id].reserve(x, y, w, h):
            pl.fits = False
            pl.tile_id = tile_id
            pl.reason = (f"锁定冲突：图表 {ch.index} 的保存位置 "
                         f"({x:.1f},{y:.1f})px 超出纹理格 {tile_id} 边界")
            result.reasons.append(pl.reason)
            result.solved = False
            result.placements[ch.index] = pl
            continue
        pl.tile_id = tile_id
        pl.x_px, pl.y_px = x, y
        result.placements[ch.index] = pl

    # ------------------------------------------------------------- movable
    order = sorted(
        movable,
        key=lambda c: -max(_footprint_px(c.uv, texel_density)[1:]))
    for ch in order:
        _, w0, h0 = _footprint_px(ch.uv, texel_density)
        allow_rot = bool(rotation_policy.get(ch.index, False))
        best = None
        for tile in tiles:
            bin_ = bins[tile.tile_id]
            for rotated in ([False, True] if allow_rot else [False]):
                w, h = (h0, w0) if rotated else (w0, w0)
                if w > tile.inner_px or h > tile.inner_px:
                    continue
                state = (bin_.x, bin_.y, bin_.shelf_h, len(bin_.rects))
                rect = bin_.place(w, h, gap)
                if rect is None:
                    bin_.x, bin_.y, bin_.shelf_h, n = state
                    bin_.rects = bin_.rects[:n]
                    continue
                best = (tile, rect, rotated, w, h)
                break
            if best:
                break
        pl = ChartPlacement(ch.index, width_px=w0, height_px=h0)
        if best is None:
            fits_some_tile = any(
                (w0 <= t.inner_px and h0 <= t.inner_px)
                or (allow_rot and h0 <= t.inner_px and w0 <= t.inner_px)
                for t in tiles)
            if not fits_some_tile:
                if not allow_cross_tile:
                    pl.reason = (
                        f"图表 {ch.index} 最小外接尺寸 {w0:.1f}×{h0:.1f}px "
                        f"大于任何纹理格的可用尺寸 "
                        f"{sorted({t.inner_px for t in tiles})}；跨格策略已禁止，"
                        f"保持整片不切断：请降低像素密度、加大纹理"
                        + ("或允许 90° 旋转" if not allow_rot else ""))
                else:
                    pl.reason = (
                        f"图表 {ch.index} 尺寸 {w0:.1f}×{h0:.1f}px 无法整体放入"
                        f"任何纹理格；即使允许跨格也不能切断未选择的接缝，报告无解")
            else:
                pl.reason = (
                    f"图表 {ch.index}（{w0:.1f}×{h0:.1f}px）本身可放入纹理格，"
                    f"但当前剩余空间不足：所有格已被锁定图表占满或边距/余量不足；"
                    f"已锁定图表不会被缩放或移动")
            pl.fits = False
            result.solved = False
            result.reasons.append(pl.reason)
            result.placements[ch.index] = pl
            continue
        tile, (x, y, w, h), rotated, _, _ = best
        pl.tile_id = tile.tile_id
        pl.x_px, pl.y_px = x, y
        pl.rotated = rotated
        result.placements[ch.index] = pl

    # cross-tile policy note for charts that could only be solved by splitting
    if not allow_cross_tile:
        for ch in order:
            _, w, h = _footprint_px(ch.uv, texel_density)
            if all(w > t.inner_px or h > t.inner_px for t in tiles):
                result.notes.append(
                    f"图表 {ch.index} 超过单格且跨格被禁止：未切断任何接缝，"
                    f"整片保留并报告无解")
    return result


def apply_layout(charts, tiles: list[TileSpec], layout: LayoutResult,
                 texel_density: float):
    """Write final atlas UVs from geometry-unit UVs and pixel placements.

    The global texel density (px per geometry unit) is fixed for every
    chart. A chart placed on a tile of resolution R occupies
    (density * geometry_size / R) in that tile's unit UV square, so equal
    geometry gets equal pixel footprints on tiles of different resolution.
    Nothing is ever rescaled to make a chart fit.

    Tile (u_index, v_index) maps to [u_index, u_index+1] × [v_index,
    v_index+1]; a single 1001 tile at origin reproduces the classic 0..1 UV.
    """
    by_id = {t.tile_id: t for t in tiles}
    for ch in charts:
        pl = layout.placements.get(ch.index)
        if pl is None or not pl.fits:
            continue
        tile = by_id[pl.tile_id]
        uv = ch.uv.copy()
        if pl.rotated:
            uv = _rotated_uv(uv)
        # geometry units -> fraction of THIS tile's texture
        uv = uv * (texel_density / float(tile.texture_size))
        mn = uv.min(axis=0)
        origin_u = tile.u_index + pl.x_px / float(tile.texture_size)
        origin_v = tile.v_index + pl.y_px / float(tile.texture_size)
        ch.uv = uv + np.array([origin_u - mn[0], origin_v - mn[1]])
        ch.tile = tile.u_index + 10 * tile.v_index
        ch.tile_id = tile.tile_id
        ch.placement = dict(tile_id=tile.tile_id,
                            x_px=pl.x_px - tile.margin_px - tile.headroom_px,
                            y_px=pl.y_px - tile.margin_px - tile.headroom_px,
                            rotated=pl.rotated)


def default_single_tile(texture_size: int, pad_px: int = 0,
                        headroom_px: int = 0) -> TileSpec:
    """The legacy single-cell project: one numbered tile, UDIM 1001 at 0..1."""
    return TileSpec(tile_id=1001, texture_size=texture_size,
                    margin_px=pad_px, headroom_px=headroom_px,
                    texture="default", u_index=0, v_index=0)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id);

const S = {
  meshId: null,
  stats: null,
  positions: null,   // Float32Array
  faces: null,       // Uint32Array
  edges: [],         // [[a,b,flag]]
  seams: new Set(),  // "a_b", a<b
  pins: {},          // vid -> [u,v]
  versions: [],
  result: null,
  demoSeams: null,
  mode: 'toggle',
  pathStart: null,   // vid
  cam2d: { cx: 0.5, cy: 0.5, s: 400 },
  fitted: false,
  layout: {
    enabled: false,
    allowCrossTile: false,
    tiles: [],        // {tile_id, texture_size, margin_px, headroom_px, texture}
    locks: {},        // chart index -> {locked, allow_rotation, tile_id}
  },
  checker3d: null,
};

const edgeKey = (a, b) => a < b ? `${a}_${b}` : `${b}_${a}`;
const notice = msg => {
  const n = $('notice');
  n.textContent = msg;
  n.style.display = 'block';
  clearTimeout(notice._t);
  notice._t = setTimeout(() => n.style.display = 'none', 3500);
};

// ============================================================ 3D viewport
const view3d = $('view3d');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
view3d.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x232a33);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
dirLight.position.set(2, 3, 4);
scene.add(dirLight);

let meshObj = null, wireObj = null, seamObj = null, bndObj = null, nmObj = null, pathMark = null;

function makeCheckerTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const n = 8, s = 256 / n;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      g.fillStyle = (i + j) % 2 ? '#3a3f4a' : '#dfe6f0';
      g.fillRect(i * s, j * s, s, s);
    }
  g.strokeStyle = 'rgba(220,60,60,.6)';
  for (let i = 0; i <= n; i++) {
    g.beginPath(); g.moveTo(i * s + .5, 0); g.lineTo(i * s + .5, 256); g.stroke();
    g.beginPath(); g.moveTo(0, i * s + .5); g.lineTo(256, i * s + .5); g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function apply3dChecker() {
  if (!meshObj) return;
  const on = $('show3dChecker').checked && S.result;
  if (on) {
    if (!S.checker3d) S.checker3d = makeCheckerTexture();
    const nV = S.positions.length / 3;
    const uv = new Float32Array(nV * 2);
    // charts duplicate UVs per cut vertex; map every local vertex by its
    // position in the chart's uv array (vertices are duplicated per chart)
    for (const ch of S.result.charts) {
      ch.vertices.forEach((vid, li) => {
        uv[2 * vid] = ch.uv[li][0];
        uv[2 * vid + 1] = ch.uv[li][1];
      });
    }
    meshObj.geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    meshObj.material = new THREE.MeshStandardMaterial({
      map: S.checker3d, side: THREE.DoubleSide, flatShading: true,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
  } else {
    meshObj.material = new THREE.MeshStandardMaterial({
      color: 0x8fa8c8, side: THREE.DoubleSide, flatShading: true,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
  }
}

function disposeObj(o) {
  if (!o) return;
  scene.remove(o);
  o.geometry?.dispose();
  o.material?.dispose();
}

function buildMesh() {
  [meshObj, wireObj, seamObj, bndObj, nmObj, pathMark].forEach(disposeObj);
  meshObj = wireObj = seamObj = bndObj = nmObj = pathMark = null;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(S.positions, 3));
  geo.setIndex(new THREE.BufferAttribute(S.faces, 1));
  geo.computeVertexNormals();
  meshObj = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: 0x8fa8c8, side: THREE.DoubleSide, flatShading: true,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  }));
  scene.add(meshObj);

  const wirePos = [], bndPos = [], nmPos = [];
  for (const [a, b, flag] of S.edges) {
    const pa = S.positions.subarray(3 * a, 3 * a + 3);
    const pb = S.positions.subarray(3 * b, 3 * b + 3);
    if (flag === 2) nmPos.push(...pa, ...pb);
    else if (flag === 1) bndPos.push(...pa, ...pb);
    else wirePos.push(...pa, ...pb);
  }
  const mkLines = (arr, color, opacity) => {
    if (!arr.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(arr), 3));
    const o = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      color, transparent: true, opacity }));
    scene.add(o);
    return o;
  };
  wireObj = mkLines(wirePos, 0x4a5568, 0.35);
  bndObj = mkLines(bndPos, 0x5aa0e0, 0.8);
  nmObj = mkLines(nmPos, 0xff8800, 1.0);
  rebuildSeamLines();

  const c = new THREE.Box3().setFromBufferAttribute(
    new THREE.BufferAttribute(S.positions, 3)).getCenter(new THREE.Vector3());
  const size = new THREE.Box3().setFromBufferAttribute(
    new THREE.BufferAttribute(S.positions, 3)).getSize(new THREE.Vector3()).length();
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0.6, 0.5, 1).normalize().multiplyScalar(size * 1.4));
  camera.near = size / 100; camera.far = size * 10;
  camera.updateProjectionMatrix();
  controls.update();
}

function rebuildSeamLines() {
  disposeObj(seamObj);
  seamObj = null;
  const pos = [];
  for (const key of S.seams) {
    const [a, b] = key.split('_').map(Number);
    pos.push(...S.positions.subarray(3 * a, 3 * a + 3),
             ...S.positions.subarray(3 * b, 3 * b + 3));
  }
  if (pos.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    seamObj = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      color: 0xff3333, depthTest: false, transparent: true, opacity: 0.95 }));
    seamObj.renderOrder = 5;
    scene.add(seamObj);
  }
  $('seamCount').textContent = `接缝边：${S.seams.size}`;
}

function markPathStart(vid) {
  disposeObj(pathMark);
  pathMark = null;
  if (vid == null) return;
  const g = new THREE.SphereGeometry(1, 16, 12);
  pathMark = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffdd33 }));
  const p = S.positions;
  pathMark.position.set(p[3 * vid], p[3 * vid + 1], p[3 * vid + 2]);
  const s = camera.position.distanceTo(pathMark.position) * 0.008;
  pathMark.scale.setScalar(s);
  pathMark.renderOrder = 6;
  scene.add(pathMark);
}

const raycaster = new THREE.Raycaster();
let downXY = null;
renderer.domElement.addEventListener('pointerdown', e => { downXY = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!downXY || !meshObj) return;
  const moved = Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]);
  downXY = null;
  if (moved > 5) return;  // was an orbit drag
  const r = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((e.clientX - r.left) / r.width) * 2 - 1,
    -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = raycaster.intersectObject(meshObj)[0];
  if (!hit) return;
  const f = hit.faceIndex;
  const tri = [S.faces[3 * f], S.faces[3 * f + 1], S.faces[3 * f + 2]];
  const p = hit.point;
  // nearest edge of the hit face
  let best = null, bestD = Infinity;
  const tmp = new THREE.Vector3();
  for (let k = 0; k < 3; k++) {
    const a = tri[k], b = tri[(k + 1) % 3];
    const pa = new THREE.Vector3().fromArray(S.positions, 3 * a);
    const pb = new THREE.Vector3().fromArray(S.positions, 3 * b);
    const seg = new THREE.Line3(pa, pb);
    seg.closestPointToPoint(p, true, tmp);
    const d = tmp.distanceTo(p);
    if (d < bestD) { bestD = d; best = [a, b]; }
  }
  if (!best) return;
  if (S.mode === 'toggle') {
    const key = edgeKey(...best);
    S.seams.has(key) ? S.seams.delete(key) : S.seams.add(key);
    afterSeamEdit();
  } else {
    // path mode: nearest endpoint of the picked edge
    const pa = new THREE.Vector3().fromArray(S.positions, 3 * best[0]);
    const pb = new THREE.Vector3().fromArray(S.positions, 3 * best[1]);
    const vid = pa.distanceTo(p) < pb.distanceTo(p) ? best[0] : best[1];
    if (S.pathStart == null) {
      S.pathStart = vid;
      markPathStart(vid);
    } else {
      const path = shortestPath(S.pathStart, vid);
      for (let i = 0; i + 1 < path.length; i++)
        S.seams.add(edgeKey(path[i], path[i + 1]));
      S.pathStart = null;
      markPathStart(null);
      afterSeamEdit();
    }
  }
});

function shortestPath(v0, v1) {
  const adj = new Map();
  for (const [a, b] of S.edges.map(e => [e[0], e[1]])) {
    const pa = S.positions, w = Math.hypot(
      pa[3 * a] - pa[3 * b], pa[3 * a + 1] - pa[3 * b + 1], pa[3 * a + 2] - pa[3 * b + 2]);
    (adj.get(a) || adj.set(a, []).get(a)).push([b, w]);
    (adj.get(b) || adj.set(b, []).get(b)).push([a, w]);
  }
  const dist = new Map([[v0, 0]]), prev = new Map(), done = new Set();
  const pq = [[0, v0]];
  while (pq.length) {
    pq.sort((x, y) => x[0] - y[0]);
    const [d, u] = pq.shift();
    if (done.has(u)) continue;
    done.add(u);
    if (u === v1) break;
    for (const [v, w] of adj.get(u) || []) {
      const nd = d + w;
      if (nd < (dist.get(v) ?? Infinity)) {
        dist.set(v, nd); prev.set(v, u); pq.push([nd, v]);
      }
    }
  }
  const path = [v1];
  while (path[0] !== v0) {
    const u = prev.get(path[0]);
    if (u == null) break;
    path.unshift(u);
  }
  return path;
}

let seamPushTimer = null;
function afterSeamEdit() {
  rebuildSeamLines();
  S.result = null;
  draw2D();
  clearTimeout(seamPushTimer);
  seamPushTimer = setTimeout(pushSeams, 250);
}

async function pushSeams() {
  if (!S.meshId) return;
  const seams = [...S.seams].map(k => k.split('_').map(Number));
  const r = await fetch(`/api/mesh/${S.meshId}/seams`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seams }),
  });
  const j = await r.json();
  if (r.ok) renderDiag(j.diagnostics);
}

// Driven by the unwrap kernel: requests seam optimization, then lets the
// backend re-unwrap so both the 3D model and 2D charts update together.
async function optimizeSeams() {
  if (!S.meshId) return;
  $('optSeams').disabled = true;
  try {
    const r = await fetch(`/api/mesh/${S.meshId}/seams/optimize`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_iterations: 10, auto_unwrap: true }),
    });
    const j = await r.json();
    if (!r.ok) {
      // kernel failure is fed back into the seam/unwrap panel, not swallowed
      const msg = j.detail?.message || '接缝优化失败';
      notice(msg);
      if (j.detail?.kind === 'seam_optimize') renderWarnings([msg]);
      return;
    }
    S.seams = new Set(j.seams.map(e => edgeKey(e[0], e[1])));
    rebuildSeamLines();
    renderDiag(j.diagnostics);
    const worst = Math.max(0, ...j.seam_errors.map(e => e.shared_error));
    notice(`接缝优化完成：评分 ${j.score.toFixed(3)} · 最大共享边误差 ${worst.toFixed(4)}`);
    if (j.unwrap) {
      S.result = j.unwrap;
      S.fitted = false;
      renderDiag(j.unwrap.diagnostics);
      renderWarnings(j.unwrap.warnings);
      renderChartList();
      apply3dChecker();
      draw2D();
    }
  } finally {
    $('optSeams').disabled = false;
  }
}

// ============================================================ 2D viewport
const canvas = $('view2d');
const ctx2d = canvas.getContext('2d');

const checker = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const n = 8, s = 256 / n;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      g.fillStyle = (i + j) % 2 ? '#2b2b2b' : '#ebebeb';
      g.fillRect(i * s, j * s, s, s);
    }
  g.strokeStyle = 'rgba(220,60,60,.55)';
  for (let i = 0; i <= n; i++) {
    g.beginPath(); g.moveTo(i * s + .5, 0); g.lineTo(i * s + .5, 256); g.stroke();
    g.beginPath(); g.moveTo(0, i * s + .5); g.lineTo(256, i * s + .5); g.stroke();
  }
  return c;
})();

function uv2screen(u, v) {
  const { cx, cy, s } = S.cam2d;
  return [(u - cx) * s + canvas.width / 2, canvas.height / 2 - (v - cy) * s];
}
function screen2uv(x, y) {
  const { cx, cy, s } = S.cam2d;
  return [(x - canvas.width / 2) / s + cx, cy - (y - canvas.height / 2) / s];
}

function metricColor(d) {
  const m = $('metric').value;
  if (d.flipped) return 'rgba(255,0,0,0.85)';
  if (m === 'none') return null;
  if (m === 'angle') {
    const t = Math.min(Math.max(Math.log2(Math.max(d.angle, 1)) / 2, 0), 1);
    return `rgba(${Math.round(255 * t)},${Math.round(200 * (1 - t))},40,0.5)`;
  }
  if (m === 'area') {
    const r = d.area_rel > 0 ? d.area_rel : 1e-6;
    const t = Math.min(Math.max(Math.log2(r) / 1.5, -1), 1);
    return t >= 0 ? `rgba(255,${Math.round(180 * (1 - t))},40,0.5)`
                  : `rgba(40,${Math.round(180 * (1 + t))},255,0.5)`;
  }
  return null;
}

function draw2D() {
  const w = canvas.width, h = canvas.height;
  ctx2d.setTransform(1, 0, 0, 1, 0, 0);
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.fillStyle = '#fafafa';
  ctx2d.fillRect(0, 0, w, h);
  const legend = $('legend');
  if (!S.result) {
    ctx2d.fillStyle = '#999';
    ctx2d.font = '14px sans-serif';
    ctx2d.fillText(S.meshId ? '点击「执行展开」生成 UV' : '请先载入网格', 20, 30);
    legend.textContent = '';
    return;
  }
  const res = S.result;
  if (!S.fitted) { fitView(res); S.fitted = true; }

  // tile boxes: explicit numbered tiles take precedence; each maps to its
  // own unit square (u_index, v_index) so near-equal ids never alias
  const tileSpecs = res.tiles && res.tiles.length
    ? res.tiles
    : [{ tile_id: 1001, u_index: 0, v_index: 0, texture_size: res.params.texture_size }];
  for (const t of tileSpecs) {
    const u0 = t.u_index ?? 0, v0 = t.v_index ?? 0;
    const [x0, y0] = uv2screen(u0, v0), [x1, y1] = uv2screen(u0 + 1, v0 + 1);
    ctx2d.strokeStyle = '#2f6fdd';
    ctx2d.lineWidth = 2;
    ctx2d.strokeRect(x0, y1, x1 - x0, y0 - y1);
    ctx2d.fillStyle = '#44556a';
    ctx2d.font = 'bold 12px sans-serif';
    ctx2d.fillText(`纹理格 ${t.tile_id}${t.texture ? ' · ' + t.texture : ''} · ${t.texture_size}px`,
      x0 + 4, y0 - 5);
  }

  const { s } = S.cam2d;
  const m = [s, 0, 0, -s, canvas.width / 2 - S.cam2d.cx * s,
             canvas.height / 2 + S.cam2d.cy * s];
  const useChecker = $('showChecker').checked;
  const overlapTris = new Set();
  for (const ch of res.charts)
    for (const [a, b] of ch.overlaps) { overlapTris.add(`${ch.index}_${a}`); overlapTris.add(`${ch.index}_${b}`); }

  for (const ch of res.charts) {
    const uv = ch.uv;
    for (let fi = 0; fi < ch.faces.length; fi++) {
      const tri = ch.faces[fi];
      const t = [uv[tri[0]], uv[tri[1]], uv[tri[2]]];
      const sp = t.map(([u, v]) => uv2screen(u, v));
      if (useChecker) {
        ctx2d.save();
        ctx2d.beginPath();
        ctx2d.moveTo(...sp[0]); ctx2d.lineTo(...sp[1]); ctx2d.lineTo(...sp[2]);
        ctx2d.closePath();
        ctx2d.clip();
        ctx2d.setTransform(...m);
        ctx2d.drawImage(checker, 0, 0, 1, 1);
        ctx2d.restore();
      }
      const color = metricColor(ch.distortion[fi]);
      ctx2d.beginPath();
      ctx2d.moveTo(...sp[0]); ctx2d.lineTo(...sp[1]); ctx2d.lineTo(...sp[2]);
      ctx2d.closePath();
      if (color) { ctx2d.fillStyle = color; ctx2d.fill(); }
      else if (!useChecker) { ctx2d.fillStyle = 'rgba(140,170,210,0.35)'; ctx2d.fill(); }
      ctx2d.strokeStyle = 'rgba(60,70,80,0.5)';
      ctx2d.lineWidth = 0.5;
      ctx2d.stroke();
      if (overlapTris.has(`${ch.index}_${fi}`)) {
        ctx2d.strokeStyle = '#d000d0';
        ctx2d.lineWidth = 2.5;
        ctx2d.stroke();
      }
    }
    // chart label with lock + tile number
    let cu = 0, cv = 0;
    for (const [u, v] of uv) { cu += u; cv += v; }
    cu /= uv.length; cv /= uv.length;
    const [lx, ly] = uv2screen(cu, cv);
    ctx2d.fillStyle = ch.locked ? '#1d4fa3' : 'rgba(30,40,55,0.85)';
    ctx2d.font = ch.locked ? 'bold 12px sans-serif' : 'bold 12px sans-serif';
    const lockTag = ch.locked ? ' 🔒' : '';
    const tileTag = ch.tile_id != null ? ` [格${ch.tile_id}]` : '';
    ctx2d.fillText(`图表 ${ch.index}${tileTag}${lockTag}`, lx - 24, ly);
  }

  // boundary vertices + pins
  if (s > 60) {
    for (const ch of res.charts) {
      for (const li of ch.boundary_verts) {
        const vid = ch.vertices[li];
        const [x, y] = uv2screen(...ch.uv[li]);
        const pinned = S.pins[vid] != null;
        ctx2d.beginPath();
        ctx2d.arc(x, y, pinned ? 5 : 2.5, 0, 7);
        ctx2d.fillStyle = pinned ? '#2f6fdd' : 'rgba(90,110,130,0.5)';
        ctx2d.fill();
        if (pinned) { ctx2d.strokeStyle = '#fff'; ctx2d.lineWidth = 1.5; ctx2d.stroke(); }
      }
    }
  }
  legend.textContent =
    `像素密度 ${res.params.texel_density} px/单位 · 纹理格 ${(res.tiles || [{ tile_id: 1001 }]).map(t => t.tile_id).join(',')} · ` +
    `布局 ${res.packing.solved === false ? '无解（见侧栏原因）' : '已容纳'}\n角度畸变 max ${res.stats.max_angle_distortion.toFixed(3)} · ` +
    `翻转 ${res.stats.n_flipped} · 重叠对 ${res.stats.n_overlap_pairs}`;
}

function fitView(res) {
  let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity;
  for (const ch of res.charts)
    for (const [u, v] of ch.uv) {
      u0 = Math.min(u0, u); v0 = Math.min(v0, v);
      u1 = Math.max(u1, u); v1 = Math.max(v1, v);
    }
  u0 = Math.min(u0, 0); v0 = Math.min(v0, 0);
  u1 = Math.max(u1, 1); v1 = Math.max(v1, 1);
  S.cam2d.cx = (u0 + u1) / 2;
  S.cam2d.cy = (v0 + v1) / 2;
  S.cam2d.s = Math.min(canvas.width / (u1 - u0), canvas.height / (v1 - v0)) * 0.9;
}

// 2D interaction: pan / zoom / pin
let drag = null;  // {type:'pan'} | {type:'pin', vid}
function nearestBoundaryVert(mx, my) {
  if (!S.result) return null;
  let best = null, bestD = 9;
  for (const ch of S.result.charts)
    for (const li of ch.boundary_verts) {
      const [x, y] = uv2screen(...ch.uv[li]);
      const d = Math.hypot(x - mx, y - my);
      if (d < bestD) { bestD = d; best = { vid: ch.vertices[li], uv: ch.uv[li] }; }
    }
  return best;
}

function eventXY(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * (canvas.width / r.width),
          (e.clientY - r.top) * (canvas.height / r.height)];
}

canvas.addEventListener('pointerdown', e => {
  const [mx, my] = eventXY(e);
  const hit = nearestBoundaryVert(mx, my);
  if (hit && S.pins[hit.vid]) {
    drag = { type: 'pin', vid: hit.vid };
    canvas.setPointerCapture(e.pointerId);
  } else {
    drag = { type: 'pan', x: mx, y: my };
    canvas.setPointerCapture(e.pointerId);
  }
});
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const [mx, my] = eventXY(e);
  if (drag.type === 'pan') {
    S.cam2d.cx -= (mx - drag.x) / S.cam2d.s;
    S.cam2d.cy += (my - drag.y) / S.cam2d.s;
    drag.x = mx; drag.y = my;
    draw2D();
  } else {
    S.pins[drag.vid] = screen2uv(mx, my);
    draw2D();
  }
});
canvas.addEventListener('pointerup', e => {
  if (drag?.type === 'pin') unwrap();
  drag = null;
});
canvas.addEventListener('dblclick', e => {
  const [mx, my] = eventXY(e);
  const hit = nearestBoundaryVert(mx, my);
  if (!hit) return;
  if (S.pins[hit.vid]) delete S.pins[hit.vid];
  else S.pins[hit.vid] = [...hit.uv];
  unwrap();
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const [mx, my] = eventXY(e);
  const [u, v] = screen2uv(mx, my);
  S.cam2d.s *= e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const [u2, v2] = screen2uv(mx, my);
  S.cam2d.cx += u - u2;
  S.cam2d.cy += v - v2;
  draw2D();
}, { passive: false });

// ==================================================== texture grid layout
function syncLayoutFromProject(project) {
  S.layout.enabled = !!(project.tiles && project.tiles.length);
  S.layout.allowCrossTile = !!project.allow_cross_tile;
  S.layout.tiles = (project.tiles || []).map(t => ({ ...t }));
  S.layout.locks = {};
  for (const [k, v] of Object.entries(project.locks || {}))
    S.layout.locks[k] = { locked: true, allow_rotation: !!v.allow_rotation,
                          tile_id: v.tile_id ?? (v.placement && v.placement.tile_id) };
  $('multiTile').checked = S.layout.enabled;
  $('crossTile').checked = S.layout.allowCrossTile;
  renderTileEditor();
}

function renderTileEditor() {
  const el = $('tileEditor');
  if (!S.layout.enabled) { el.innerHTML = ''; return; }
  if (!S.layout.tiles.length) {
    S.layout.tiles = [{ tile_id: 1001, texture_size: +$('texsize').value || 1024,
      margin_px: +$('pad').value || 0, headroom_px: 0, texture: 'default' }];
  }
  el.innerHTML = '';
  S.layout.tiles.forEach((t, i) => {
    const row = document.createElement('div');
    row.className = 'tile-row';
    row.innerHTML = `
      <input data-f="tile_id" type="number" value="${t.tile_id}" title="显式格号（如 UDIM 1001）">
      <input data-f="texture_size" type="number" value="${t.texture_size}" title="该格纹理分辨率 px">
      <button type="button" data-act="del">删除</button>
      <input data-f="margin_px" type="number" value="${t.margin_px ?? 0}" title="边距 px">
      <input data-f="headroom_px" type="number" value="${t.headroom_px ?? 0}" title="采样余量 px">
      <input data-f="texture" type="text" value="${t.texture ?? ''}" placeholder="纹理资源键">
      <div class="meta">格 ${t.tile_id} · 可用 ${(t.texture_size - 2 * ((t.margin_px ?? 0) + (t.headroom_px ?? 0)))}px · 资源 ${t.texture || '未绑定'}</div>`;
    row.querySelectorAll('input').forEach(inp => inp.onchange = () => {
      const f = inp.dataset.f;
      t[f] = f === 'texture' ? inp.value.trim() : +inp.value;
      renderTileEditor();
    });
    row.querySelector('[data-act=del]').onclick = () => {
      S.layout.tiles.splice(i, 1);
      renderTileEditor();
    };
    el.appendChild(row);
  });
}

async function pushLayout() {
  const body = {
    allow_cross_tile: $('crossTile').checked,
    tiles: S.layout.enabled ? S.layout.tiles : [],
  };
  const r = await fetch(`/api/mesh/${S.meshId}/layout`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) { notice(typeof j.detail === 'string' ? j.detail : '格布局配置失败'); return false; }
  return true;
}

function renderChartList() {
  const el = $('chartList');
  if (!S.result) { el.innerHTML = '<span style="color:#889">执行展开后显示</span>'; return; }
  const tileIds = (S.result.tiles || [{ tile_id: 1001 }]).map(t => t.tile_id);
  el.innerHTML = '';
  S.result.charts.forEach(ch => {
    const lock = S.layout.locks[ch.index] || { locked: !!ch.locked };
    const placement = S.result.packing.placements?.[ch.index];
    const row = document.createElement('div');
    row.className = 'chart-row' + (lock.locked ? ' locked' : '');
    const curTile = ch.tile_id ?? tileIds[0];
    row.innerHTML = `
      <div class="line"><b>图表 ${ch.index}</b>
        <select data-f="locked">
          <option value="0" ${lock.locked ? '' : 'selected'}>可重排</option>
          <option value="1" ${lock.locked ? 'selected' : ''}>锁定布局</option>
        </select></div>
      <div class="line"><span>格号</span>
        <select data-f="tile_id">${tileIds.map(id =>
          `<option value="${id}" ${id === (lock.tile_id ?? curTile) ? 'selected' : ''}>${id}</option>`).join('')}
        </select></div>
      <label><input type="checkbox" data-f="allow_rotation" ${lock.allow_rotation ? 'checked' : ''}>
        允许 90° 旋转（默认禁止）</label>
      ${placement && placement.fits === false
        ? `<div class="reason">✗ ${placement.reason || '无法容纳'}</div>` : ''}
      ${placement && placement.fits ? `<div class="meta">占用 ${Math.round(placement.width_px)}×${Math.round(placement.height_px)}px · 格 ${placement.tile_id}</div>` : ''}`;
    row.querySelector('[data-f=locked]').onchange = async e => {
      const locked = e.target.value === '1';
      await setLock(ch.index, locked,
        row.querySelector('[data-f=allow_rotation]').checked,
        +row.querySelector('[data-f=tile_id]').value);
    };
    row.querySelector('[data-f=tile_id]').onchange = async e => {
      if (S.layout.locks[ch.index]?.locked)
        await setLock(ch.index, true,
          row.querySelector('[data-f=allow_rotation]').checked, +e.target.value);
    };
    row.querySelector('[data-f=allow_rotation]').onchange = async e => {
      if (S.layout.locks[ch.index]?.locked)
        await setLock(ch.index, true, e.target.checked,
          +row.querySelector('[data-f=tile_id]').value);
    };
    el.appendChild(row);
  });
  const reasons = S.result.packing.reasons || [];
  $('layoutReasons').innerHTML = reasons.map(r => `✗ ${r}`).join('<br>');
}

async function setLock(chartIndex, locked, allowRotation, tileId) {
  const r = await fetch(`/api/mesh/${S.meshId}/charts/lock`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chart_index: chartIndex, locked,
      allow_rotation: allowRotation, tile_id: tileId }),
  });
  const j = await r.json();
  if (!r.ok) { notice(typeof j.detail === 'string' ? j.detail : '锁定失败'); return; }
  if (locked) S.layout.locks[chartIndex] = { locked: true, allow_rotation: allowRotation, tile_id: tileId };
  else delete S.layout.locks[chartIndex];
  await unwrap();
}

// ============================================================ data flow
async function loadPayload(p) {
  S.meshId = p.mesh_id;
  S.stats = p.stats;
  S.positions = new Float32Array(p.positions);
  S.faces = new Uint32Array(p.faces);
  S.edges = p.edges;
  S.seams = new Set((p.project.seams || []).map(e => edgeKey(e[0], e[1])));
  S.pins = {};
  for (const [k, v] of Object.entries(p.project.pins || {})) S.pins[k] = v;
  S.versions = p.project.versions || [];
  S.result = null;
  S.pathStart = null;
  S.fitted = false;
  S.demoSeams = p.demo_seams || null;
  syncLayoutFromProject(p.project);
  $('applyDemo').style.display = S.demoSeams?.length ? '' : 'none';
  buildMesh();
  renderDiag(p.diagnostics);
  renderVersions();
  draw2D();
  const st = p.stats;
  $('meshStats').textContent =
    `${st.name} · 顶点 ${st.n_vertices} · 面 ${st.n_faces} · 边 ${st.n_edges}` +
    ` · 边界边 ${st.n_boundary_edges} · 非流形边 ${st.n_nonmanifold_edges}` +
    ` · 指纹 ${st.fingerprint}`;
  const diag = Math.hypot(
    st.bbox_max[0] - st.bbox_min[0], st.bbox_max[1] - st.bbox_min[1],
    st.bbox_max[2] - st.bbox_min[2]);
  $('texel').value = Math.max(1, Math.round(0.55 * +$('texsize').value / diag));
  for (const [id, path] of [['expObj', 'export/obj'], ['expPng', 'export/checker.png'],
                            ['expRep', 'export/report.json']])
    $(id).href = `/api/mesh/${S.meshId}/${path}`;
  if (p.restored && (p.project.seams?.length || Object.keys(p.project.pins || {}).length))
    notice('检测到相同拓扑指纹：已按拓扑边恢复接缝与固定点（未使用顶点位置）');
  else if (p.restored)
    notice('检测到相同拓扑指纹：已恢复项目');
}

function renderDiag(d) {
  const el = $('diag');
  let html = '';
  for (const g of d.global_problems || [])
    html += `<div class="item bad">✗ ${g.message}</div>`;
  for (const c of d.charts || []) {
    const ok = c.problems.length === 0;
    html += `<div class="item ${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'} 图表 ${c.index}：` +
      `${c.n_faces} 面 · χ=${c.euler} · 边界环 ${c.n_boundary_loops}` +
      (ok ? ' · 圆盘拓扑，可展开' : '') + '</div>';
    for (const p of c.problems)
      html += `<div class="item bad">&nbsp;&nbsp;↳ ${p}</div>`;
  }
  html += `<div class="item ${d.blocking ? 'bad' : 'ok'}">` +
    (d.blocking ? '存在阻塞问题：修复后才能展开' : '诊断通过，可以展开') + '</div>';
  el.innerHTML = html;
}

function renderWarnings(list) {
  $('warnBox').style.display = list.length ? '' : 'none';
  $('warnings').innerHTML = list.map(w => `<div class="item">⚠ ${w}</div>`).join('');
}

function renderVersions() {
  const el = $('verList');
  el.innerHTML = S.versions.map(v =>
    `<div class="ver"><span>${v.name} <small style="color:#889">${v.created}</small></span>` +
    `<button data-vid="${v.id}">恢复</button></div>`).join('') ||
    '<div style="color:#99a;font-size:12px">暂无版本</div>';
  el.querySelectorAll('button').forEach(b => b.onclick = () => restoreVersion(b.dataset.vid));
}

async function unwrap() {
  if (!S.meshId) return;
  const body = {
    texel_density: +$('texel').value,
    texture_size: +$('texsize').value,
    pad_px: +$('pad').value,
    pins: S.pins,
  };
  const r = await fetch(`/api/mesh/${S.meshId}/unwrap`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) {
    if (j.detail?.diagnostics) {
      renderDiag(j.detail.diagnostics);
      notice(j.detail.message || '存在阻塞问题');
    } else notice(typeof j.detail === 'string' ? j.detail : '展开失败');
    return;
  }
  S.result = j;
  S.fitted = false;
  renderDiag(j.diagnostics);
  renderWarnings(j.warnings);
  renderChartList();
  apply3dChecker();
  $('uvStats').textContent =
    `图表 ${j.stats.n_charts} · 角度畸变 max ${j.stats.max_angle_distortion.toFixed(3)} ` +
    `(均值 ${j.stats.mean_angle_distortion.toFixed(3)}) · 翻转 ${j.stats.n_flipped} · ` +
    `重叠对 ${j.stats.n_overlap_pairs}`;
  draw2D();
}

async function restoreVersion(vid) {
  const r = await fetch(`/api/mesh/${S.meshId}/versions/${vid}/restore`, { method: 'POST' });
  const j = await r.json();
  if (!r.ok) return notice('版本恢复失败');
  S.seams = new Set((j.project.seams || []).map(e => edgeKey(e[0], e[1])));
  S.pins = {};
  for (const [k, v] of Object.entries(j.project.pins || {})) S.pins[k] = v;
  $('texel').value = j.project.params.texel_density;
  $('texsize').value = j.project.params.texture_size;
  $('pad').value = j.project.params.pad_px;
  syncLayoutFromProject(j.project);
  S.result = j.result;
  S.fitted = false;
  rebuildSeamLines();
  pushSeams();
  if (S.result) { renderChartList(); apply3dChecker(); draw2D(); }
  notice('已恢复版本快照（含纹理格、锁定与实际布局）');
}

// ============================================================ UI wiring
$('loadTest').onclick = async () => {
  const r = await fetch(`/api/testmesh/${$('testmesh').value}`);
  loadPayload(await r.json());
};
$('objFile').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  const text = await f.text();
  const r = await fetch('/api/mesh/upload', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ obj_text: text, name: f.name.replace(/\.obj$/i, '') }),
  });
  const j = await r.json();
  if (!r.ok) return notice(j.detail || 'OBJ 解析失败');
  loadPayload(j);
};
$('clearSeams').onclick = () => { S.seams.clear(); afterSeamEdit(); };
$('applyDemo').onclick = () => {
  for (const [a, b] of S.demoSeams) S.seams.add(edgeKey(a, b));
  afterSeamEdit();
};
document.querySelectorAll('input[name=mode]').forEach(r =>
  r.onchange = () => { S.mode = r.value; S.pathStart = null; markPathStart(null); });
$('unwrapBtn').onclick = unwrap;
$('metric').onchange = draw2D;
$('showChecker').onchange = draw2D;
$('show3dChecker').onchange = apply3dChecker;
$('optSeams').onclick = optimizeSeams;

$('multiTile').onchange = e => {
  S.layout.enabled = e.target.checked;
  if (S.layout.enabled && !S.layout.tiles.length) renderTileEditor();
  if (!S.layout.enabled) $('tileEditor').innerHTML = '';
};
$('crossTile').onchange = e => { S.layout.allowCrossTile = e.target.checked; };
$('addTile').onclick = () => {
  if (!S.layout.enabled) { notice('请先勾选「启用显式编号纹理格」'); return; }
  const nextId = 1001 + S.layout.tiles.length;
  S.layout.tiles.push({ tile_id: nextId,
    texture_size: +$('texsize').value || 1024,
    margin_px: +$('pad').value || 0, headroom_px: 0,
    texture: `fabric_${nextId}` });
  renderTileEditor();
};
$('applyLayout').onclick = async () => {
  if (!(await pushLayout())) return;
  await unwrap();
  notice('格布局已应用，已按当前像素密度重排');
};
$('relayoutBtn').onclick = async () => {
  const r = await fetch(`/api/mesh/${S.meshId}/relayout`, { method: 'POST' });
  const j = await r.json();
  if (!r.ok) { notice(typeof j.detail === 'string' ? j.detail : '重排失败'); return; }
  S.result = j;
  S.fitted = false;
  renderWarnings(j.warnings);
  renderChartList();
  apply3dChecker();
  draw2D();
  notice('已重排：锁定图表保持原位，其余图表重新布局');
};
$('saveVer').onclick = async () => {
  if (!S.result) return notice('请先执行展开');
  const name = prompt('版本名称', `版本 ${S.versions.length + 1}`);
  if (name == null) return;
  const r = await fetch(`/api/mesh/${S.meshId}/versions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const j = await r.json();
  if (r.ok) { S.versions = j.versions; renderVersions(); notice('版本已保存'); }
  else notice(j.detail || '保存失败');
};

function resize() {
  const r3 = view3d.getBoundingClientRect();
  renderer.setSize(r3.width, r3.height);
  camera.aspect = r3.width / Math.max(r3.height, 1);
  camera.updateProjectionMatrix();
  const r2 = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r2.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(r2.height * devicePixelRatio));
  draw2D();
}
new ResizeObserver(resize).observe(view3d);
new ResizeObserver(resize).observe(canvas);
window.addEventListener('resize', resize);
resize();

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// canvas 2D works in device pixels; cam2d.s is in device px per UV unit
loadPayload(await (await fetch('/api/testmesh/cylinder')).json());

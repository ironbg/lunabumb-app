"""A quick look at a week's sculpt: marching cubes straight to a GLB (no Blender, no colours), for
shots.mjs clay. PARTS=head,torso builds only those parts.
  python preview.py <week> <out.glb> [grid spacing, default 0.004]"""
import os, sys, time
import numpy as np
t0 = time.time()
moddir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
sys.path.insert(0, moddir)
import build_embryo as be
from build_fetus import sculpt_grid, GlbWriter, eval_sdf
from skimage import measure
week, out = int(sys.argv[1]), sys.argv[2]
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004
emb = {8: be.Week8, 9: be.Week9, 10: be.Week10}[week]().build()
if os.environ.get('PARTS'):
    keep = os.environ['PARTS'].split(',')
    junc = emb.ops.junctions
    kept = [o for o in emb.ops if o.group in keep]
    del emb.ops[:]
    emb.ops.extend(kept)
    emb.ops.junctions = junc
d, lo = sculpt_grid(emb.ops, h)
v, f, n, _ = measure.marching_cubes(d, level=0.0, spacing=(h, h, h), gradient_direction='ascent')
v = v + lo
# outward normals: along the field's gradient
p = v[:2000]
e = 0.002
g = np.stack([(eval_sdf(emb.ops, p + np.eye(3)[i] * e) - eval_sdf(emb.ops, p - np.eye(3)[i] * e)) for i in range(3)], axis=1)
if np.mean(np.einsum('ij,ij->i', g, n[:2000])) < 0:
    n = -n
fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
if np.mean(np.einsum('ij,ij->i', fn, n[f[:, 0]])) < 0:
    f = f[:, ::-1]
center = np.zeros(3)
w = GlbWriter()
pos = (v - center).astype(np.float32)
attrs = {"POSITION": w.dense(pos, {"componentType": 5126, "type": "VEC3", "count": len(pos), "min": pos.min(0).tolist(), "max": pos.max(0).tolist()}),
         "NORMAL": w.dense(n.astype(np.float32), {"componentType": 5126, "type": "VEC3", "count": len(pos)})}
idx = w.dense(f.astype(np.uint32).ravel()[:, None], {"componentType": 5125, "type": "SCALAR", "count": int(f.size)}, indices=True)
w.save({"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "Skin", "mesh": 0}],
        "meshes": [{"name": "Skin", "primitives": [{"attributes": attrs, "indices": idx}]}]}, out)
print(f"{len(v)} verts, {len(f)} tris, bbox {v.min(0).round(3)} {v.max(0).round(3)}, {time.time() - t0:.1f}s")

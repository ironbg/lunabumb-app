"""Build a Luna View embryo model for one early week (glTF), from a signed-distance-field sculpt.

The fetus model (build_fetus.py) morphs one mesh through all the weeks. An embryo looks too
different for that: its own sculpt per week reads better. Each week here is a static mesh plus
the two dark eyes, with per-vertex data for the skin shader:

  COLOR_0   rgb: soft occlusion and tints, a: local thickness (for the light shining through)
  _DETAIL   x: how much of the light through the skin the organs inside block (heart, liver,
             vertebrae, ribs), seen as dark shapes when the light is behind the embryo;
            y: where the fine surface vessels of the head show (the shader draws them)

Coordinates are those of build_fetus.py: pose space, x = the embryo's right, y = up, z = front,
in crown-rump units (CRL = 1), written out centred on the bounding box. A small JSON rig holds
the anchor points the page needs (face, heart, hands, feet, the cord root).

Anatomy follows the Carnegie stages: week 8 of pregnancy (counted from the last period) is about
6 weeks after conception, Carnegie stage 18 to 19, crown-rump length about 14-17 mm. The head is
nearly half the body and bent onto the chest over the heart and liver bulges; the eyes are dark
with pigment; the ear is a low ring of small hillocks behind the jaw; the hands and feet are
plates with the rays of the fingers and toes and notches between them; the gut still loops into
the base of the cord (physiological herniation); a short tail is left at the bottom.

Run (needs Blender's `bpy` module, numpy, scipy and scikit-image):
  python build_embryo.py --week 8 --out ../models/embryo-w08.glb --rig ../models/embryo-w08.json
"""

import argparse
import json
import math
import time

import numpy as np

from build_fetus import (Chain, Ellipsoid, GlbWriter, Op, RoundCone, Sculpt, adjacency, ambient_occlusion,
                         blender_mesh, eval_sdf, gradient, marching_cubes, project, read_mesh, rotation_between,
                         sculpt_grid, skin_thickness, smoothstep, unit, vec, vertex_normals)


def frame(normal, up):
    """Rotation whose columns are (u, v, n): a plate lying across n, with v towards up."""
    n = unit(np.asarray(normal, float))
    v = np.asarray(up, float) - n * (np.asarray(up, float) @ n)
    v = unit(v)
    u = np.cross(v, n)
    return np.stack([u, v, n], axis=1)


class Week8:
    """Carnegie stage 18-19."""

    def __init__(self):
        self.ops = Sculpt()
        self.anchors = {}
        self.eyes = []
        self.organs = []     # (Ellipsoid, absorbance per CRL)
        self.lid_lines = []

    # the head bends forward onto the chest round the neck (the cervical flexure)
    HEAD_PIVOT = vec(0, -0.02, -0.06)
    HEAD_PITCH = 0.2

    def hp(self, v):
        c, sn = math.cos(self.HEAD_PITCH), math.sin(self.HEAD_PITCH)
        d = np.asarray(v, float) - self.HEAD_PIVOT
        return self.HEAD_PIVOT + vec(d[0], d[1] * c - d[2] * sn, d[1] * sn + d[2] * c)

    def head_shape(self, shape):
        c, sn = math.cos(self.HEAD_PITCH), math.sin(self.HEAD_PITCH)
        Rx = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
        if isinstance(shape, Ellipsoid):
            return Ellipsoid(self.hp(shape.c), shape.r, Rx @ shape.R)
        return RoundCone(self.hp(shape.a), self.hp(shape.b), shape.ra, shape.rb)

    def add(self, name, shape, k, group):
        if group == "head":
            shape = self.head_shape(shape)
        self.ops.append(Op(name, shape, "add", k, group))

    def sub(self, name, shape, k, group):
        if group == "head":
            shape = self.head_shape(shape)
        self.ops.append(Op(name, shape, "sub", k, group))

    def build(self):
        add, sub = self.add, self.sub
        ops = self.ops
        ops.junctions = {
            "head": (vec(0, -0.02, -0.06), 0.18, 0.06),
            "armL": (vec(-0.12, -0.1, 0.03), 0.06, 0.04),
            "armR": (vec(0.12, -0.1, 0.03), 0.06, 0.04),
            "legL": (vec(-0.1, -0.36, 0.01), 0.07, 0.045),
            "legR": (vec(0.1, -0.36, 0.01), 0.07, 0.045),
        }

        # ---- head: forebrain in front, the midbrain dome on top, the hindbrain sloping to the back
        # the big forebrain bulges far forward over the small face; the eye sits about halfway
        # along the head
        add("cranium", Ellipsoid(vec(0, 0.19, 0.03), vec(0.215, 0.24, 0.34)), 0.1, "head")
        add("midbrain", Ellipsoid(vec(0, 0.26, -0.08), vec(0.18, 0.185, 0.2)), 0.1, "head")
        add("forebrain", Ellipsoid(vec(0, 0.15, 0.2), vec(0.19, 0.185, 0.2)), 0.1, "head")
        add("hindbrain", Ellipsoid(vec(0, 0.07, -0.19), vec(0.15, 0.17, 0.13)), 0.09, "head")
        # face: the upper jaw and nose region under the forehead, just in front of and below the eye
        add("midface", Ellipsoid(vec(0, -0.04, 0.16), vec(0.13, 0.09, 0.11)), 0.06, "head")
        add("snout", Ellipsoid(vec(0, -0.05, 0.22), vec(0.07, 0.048, 0.05)), 0.05, "head")
        add("jaw", Ellipsoid(vec(0, -0.115, 0.13), vec(0.1, 0.05, 0.075)), 0.05, "head")
        for s, side in ((1, "R"), (-1, "L")):
            # nasal pits and the nostril rims
            sub(f"nasalPit{side}", Ellipsoid(vec(0.026 * s, -0.035, 0.27), vec(0.009, 0.011, 0.014)), 0.008, "head")
            # the eye sits in a low swelling on the side of the head (the eye itself is placed on
            # the finished surface, below)
            add(f"eyeBulge{side}", Ellipsoid(vec(0.15 * s, 0.03, 0.09), vec(0.045, 0.048, 0.048)), 0.04, "head")
            # a soft fold of skin above the eye: the eyelids are just starting
            self.lid_lines.append(self.head_shape(Ellipsoid(vec(0.175 * s, 0.062, 0.095), vec(0.03, 0.006, 0.03))))
            # the ear: a ring of small hillocks round the first groove, low behind the jaw
            ear_c = vec(0.17 * s, 0.03, -0.075)
            add(f"earBase{side}", Ellipsoid(ear_c, vec(0.03, 0.036, 0.03)), 0.03, "head")
            add(f"earKnob{side}", Ellipsoid(ear_c + vec(0.012 * s, 0.012, -0.004), vec(0.016, 0.02, 0.016)), 0.014, "head")
            sub(f"earGroove{side}", Ellipsoid(ear_c + vec(0.03 * s, -0.012, 0.008), vec(0.01, 0.012, 0.008)), 0.01, "head")
        # the mouth: a wide slit between the jaws
        sub("mouth", Ellipsoid(vec(0, -0.088, 0.225), vec(0.055, 0.006, 0.03)), 0.01, "head")

        # ---- trunk: the heart and liver bulge out in front, the back curls round
        add("upperBack", Ellipsoid(vec(0, -0.1, -0.075), vec(0.145, 0.16, 0.125)), 0.08, "torso")
        add("heart", Ellipsoid(vec(0, -0.175, 0.1), vec(0.15, 0.12, 0.13)), 0.08, "torso")
        add("liver", Ellipsoid(vec(0, -0.3, 0.07), vec(0.17, 0.14, 0.15)), 0.08, "torso")
        add("lowerBack", Ellipsoid(vec(0, -0.29, -0.045), vec(0.135, 0.15, 0.115)), 0.08, "torso")
        add("rump", Ellipsoid(vec(0, -0.41, 0.0), vec(0.11, 0.085, 0.095)), 0.07, "torso")
        add("tail", RoundCone(vec(0, -0.45, -0.04), vec(0, -0.505, 0.04), 0.038, 0.016), 0.03, "torso")
        # neck: the cervical bend joins the hindbrain to the back
        add("neck", Ellipsoid(vec(0, 0.0, -0.1), vec(0.13, 0.12, 0.12)), 0.08, "torso")
        # the spine: a row of small swellings along the back (somites)
        back = []
        for y in (0.0, -0.07, -0.14, -0.21, -0.28, -0.35, -0.41):
            lo_z, hi_z = -0.45, 0.0   # bisect for the skin of the back at this height
            for _ in range(30):
                mid = 0.5 * (lo_z + hi_z)
                if eval_sdf(ops, np.array([[0.0, y, mid]]))[0] < 0:
                    hi_z = mid
                else:
                    lo_z = mid
            back.append((y, hi_z))
        for i, (y, z) in enumerate(back):
            add(f"somite{i}", Ellipsoid(vec(0, y, z + 0.014), vec(0.048, 0.022, 0.014)), 0.025, "torso")
        # the root of the cord, swollen by the loop of gut in it at this age
        cord_a, cord_b = vec(0, -0.39, 0.13), vec(0, -0.45, 0.26)
        add("cordRoot", RoundCone(cord_a, cord_b, 0.07, 0.058), 0.05, "torso")

        # ---- limbs: short, bent, with plates for hands and feet
        for s, side in ((1, "R"), (-1, "L")):
            arm = f"arm{side}"
            sh = vec(0.12 * s, -0.1, 0.03)
            el = vec(0.165 * s, -0.2, 0.12)
            wr = vec(0.13 * s, -0.16, 0.215)
            hand_c = vec(0.11 * s, -0.115, 0.27)
            add(f"upperArm{side}", Chain([RoundCone(sh, el, 0.058, 0.05)]), 0.03, arm)
            add(f"forearm{side}", Chain([RoundCone(el, wr, 0.05, 0.044)]), 0.03, arm)
            self.plate(add, sub, f"hand{side}", arm, centre=hand_c,
                       normal=vec(1.0 * s, 0.1, -0.15), up=vec(0, 0.85, 0.5), r=(0.085, 0.074), thick=0.019, rays=5,
                       spread=2.2, from_dir=unit(wr - hand_c))
            leg = f"leg{side}"
            hip = vec(0.1 * s, -0.36, 0.01)
            kn = vec(0.15 * s, -0.37, 0.13)
            an = vec(0.1 * s, -0.45, 0.16)
            foot_c = vec(0.072 * s, -0.49, 0.205)
            add(f"thigh{side}", Chain([RoundCone(hip, kn, 0.056, 0.047)]), 0.035, leg)
            add(f"shin{side}", Chain([RoundCone(kn, an, 0.047, 0.039)]), 0.03, leg)
            self.plate(add, sub, f"foot{side}", leg, centre=foot_c,
                       normal=vec(1.0 * s, -0.1, 0.2), up=vec(0, -0.3, 1.0), r=(0.062, 0.052), thick=0.018, rays=5,
                       spread=1.9, from_dir=unit(an - foot_c))
            self.anchors[f"hand{side}"] = hand_c
            self.anchors[f"foot{side}"] = foot_c
            self.anchors[f"knee{side}"] = kn + vec(0.04 * s, 0, 0.02)


        # ---- what blocks the light inside: heart, liver, the spine and ribs, the eye pigment
        self.organs = [
            (Ellipsoid(vec(0, -0.18, 0.1), vec(0.075, 0.07, 0.07)), 32.0),     # heart
            (Ellipsoid(vec(0, -0.3, 0.06), vec(0.12, 0.1, 0.1)), 26.0),       # liver
            (Ellipsoid(vec(0, -0.12, -0.1), vec(0.05, 0.12, 0.05)), 10.0),    # upper spine cord
        ]
        for i, (y, z) in enumerate(back):
            self.organs.append((Ellipsoid(vec(0, y, z + 0.03), vec(0.05, 0.016, 0.03)), 30.0))   # vertebrae
            if -0.35 < y < 0.0:
                for s in (1, -1):   # ribs curving round the sides
                    self.organs.append((Ellipsoid(vec(0.1 * s, y - 0.02, z + 0.12), vec(0.035, 0.012, 0.09),
                                                  rotation_between(vec(0, 0, 1), unit(vec(0.6 * s, 0, 1)))), 18.0))

        # ---- anchors for the page
        self.anchors.update({
            "head": self.hp(vec(0.1, 0.4, 0.06)),
            "face": self.hp(vec(0, -0.04, 0.26)),
            "faceFront": self.hp(vec(0, -0.04, 0.6)),
            "ear": self.hp(vec(0.2, 0.03, -0.075)),
            "heart": vec(0.05, -0.17, 0.23),
            "navel": cord_b + vec(0, -0.01, 0.04),
            "groin": vec(0, -0.47, 0.1),
            "core": vec(0, -0.04, 0.03),
        })
        self.anchors["hand"] = self.anchors["handL"]
        self.anchors["knee"] = self.anchors["kneeL"]
        self.anchors["footL"] = self.anchors.pop("footL")
        self.anchors["footR"] = self.anchors.pop("footR")
        return self

    def plate(self, add, sub, name, group, centre, normal, up, r, thick, rays, spread, from_dir):
        """A hand or foot plate: a flat oval with raised rays fanning out to the rim, and small
        notches between the rays where the fingers or toes will separate."""
        R = frame(normal, up)
        add(name, Ellipsoid(centre, vec(r[0], r[1], thick), R), 0.02, group)
        u, v, n = R[:, 0], R[:, 1], R[:, 2]
        # the rays fan away from the wrist or ankle
        back = unit(from_dir - n * (from_dir @ n))
        angle0 = math.atan2(back @ v, back @ u) + math.pi
        for i in range(rays):
            a = angle0 + (i - (rays - 1) / 2) * spread / (rays - 1)
            d = math.cos(a) * u + math.sin(a) * v
            tip = centre + d * 0.86 * min(r)
            add(f"{name}Ray{i}", RoundCone(centre + d * 0.2 * min(r), tip, 0.011, 0.0125), 0.016, group)
            if i < rays - 1:
                a2 = a + spread / (rays - 1) / 2
                d2 = math.cos(a2) * u + math.sin(a2) * v
                sub(f"{name}Notch{i}", Ellipsoid(centre + d2 * 1.0 * min(r), vec(0.006, 0.008, 0.03)), 0.008, group)


def organ_shadow(organs, p, nrm, depth, steps=20):
    """How much of the light passing through the skin at p the organs inside absorb, from a march
    inward along the normal through the body (Beer-Lambert over the organs' soft volumes)."""
    total = np.zeros(len(p))
    for k in range(steps):
        t = (k + 0.5) / steps * depth
        q = p - nrm * t[:, None]
        dens = np.zeros(len(p))
        for shape, absorb in organs:
            d = shape.sdf(q)
            dens += absorb * np.clip(-d / 0.02 + 0.5, 0.0, 1.0)
        total += dens * depth / steps
    return 1.0 - np.exp(-total)


def sphere(c, r, rings=18, segs=28):
    verts, normals = [], []
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segs + 1):
            ph = 2 * math.pi * j / segs
            n = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
            verts.append(c + r * n)
            normals.append(n)
    tris = []
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j
            b = a + segs + 1
            tris += [(a, a + 1, b), (a + 1, b + 1, b)]   # counter-clockwise seen from outside
    return np.array(verts), np.array(normals), np.array(tris)


def write_glb(path, skin, eyes):
    """skin: dict(position, normal, color (uint8 RGBA), detail (uint8 VEC2), tris); eyes the same
    without colour and detail. Plain floats: the file is small enough without quantization."""
    out = GlbWriter()

    def prim(m, extra=True):
        n = len(m["position"])
        pos = m["position"].astype(np.float32)
        attrs = {
            "POSITION": out.dense(pos, {"componentType": 5126, "type": "VEC3", "count": n,
                                        "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()}),
            "NORMAL": out.dense(m["normal"].astype(np.float32), {"componentType": 5126, "type": "VEC3", "count": n}),
        }
        if extra:
            attrs["COLOR_0"] = out.dense(m["color"], {"componentType": 5121, "normalized": True, "type": "VEC4", "count": n})
            attrs["_DETAIL"] = out.dense(m["detail"], {"componentType": 5121, "normalized": True, "type": "VEC2", "count": n})
        kind, dtype = (5123, np.uint16) if n < 65536 else (5125, np.uint32)
        idx = out.dense(m["tris"].astype(dtype).ravel()[:, None], {"componentType": kind, "type": "SCALAR",
                                                                   "count": int(m["tris"].size)}, indices=True)
        return {"attributes": attrs, "indices": idx}

    gltf = {
        "asset": {"version": "2.0", "generator": "Lunabump build_embryo.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "Skin", "mesh": 0}, {"name": "Eyes", "mesh": 1}],
        "meshes": [{"name": "Skin", "primitives": [prim(skin)]}, {"name": "Eyes", "primitives": [prim(eyes, False)]}],
    }
    out.save(gltf, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--h", type=float, default=0.0024, help="grid spacing for marching cubes")
    ap.add_argument("--tris", type=int, default=30000, help="triangle budget outside the face, ears, hands and feet")
    args = ap.parse_args()
    t0 = time.time()
    if args.week != 8:
        raise SystemExit("only week 8 so far")
    emb = Week8().build()
    ops = emb.ops

    d, lo = sculpt_grid(ops, args.h)
    verts, faces = marching_cubes(d, lo, args.h)
    print(f"grid {d.shape}, marching cubes {len(verts)} verts ({time.time() - t0:.1f}s)")

    def near(prefixes, dist):
        sel = [op for op in ops if op.name.startswith(prefixes)]
        return lambda p: np.min([op.shape.sdf(p) for op in sel], axis=0) < dist

    for s in (1, -1):
        _, y, z = emb.hp(vec(0, 0.03, 0.095))
        r = 0.028
        lo_x, hi_x = 0.0, 0.4
        for _ in range(40):   # bisect for the skin along x at the eye's height
            mid = 0.5 * (lo_x + hi_x)
            if eval_sdf(ops, np.array([[mid * s, y, z]]))[0] < 0:
                lo_x = mid
            else:
                hi_x = mid
        emb.eyes.append((vec((lo_x - 0.45 * r) * s, y, z), r))
    obj = blender_mesh(verts, faces, args.tris, details=(
        (near(("ear",), 0.01), 0.6), (near(("hand", "foot"), 0.012), 0.5),
        (near(("snout", "nasalPit", "mouth", "eyeBulge", "jaw"), 0.01), 0.6)))
    pos, tris, _ = read_mesh(obj)
    pos = project(ops, pos, iterations=2, max_step=0.003)
    nrm = gradient(ops, pos, eps=0.0015)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    # mesh normals where the SDF gradient is unreliable (inside tight creases)
    mesh_n = vertex_normals(pos, tris)
    nrm = np.where((np.einsum("ij,ij->i", nrm, mesh_n) < 0.3)[:, None], mesh_n, nrm)
    print(f"mesh: {len(pos)} verts, {len(tris)} tris ({time.time() - t0:.1f}s)")

    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    A, deg = adjacency(edges, len(pos))

    # ---- colour: soft occlusion, a darker lid fold, a little more red round the mouth and nose
    ao = ambient_occlusion(ops, pos, nrm, dist=0.01)
    for _ in range(4):
        ao = 0.5 * ao + 0.5 * (A @ ao) / deg
    lid = np.zeros(len(pos))
    for line in emb.lid_lines:
        lid = np.maximum(lid, np.exp(-np.maximum(line.sdf(pos), 0) / 0.004))
    by = {op.name: op for op in ops}
    warm = np.zeros(len(pos))
    for name in ("snout", "jaw", "midface"):
        warm = np.maximum(warm, 0.5 * np.exp(-np.maximum(by[name].shape.sdf(pos), 0) / 0.01))
    shade = (0.6 + 0.4 * ao) * (1 - 0.25 * lid)
    rgb = np.stack([shade, shade * (1 - 0.08 * warm), shade * (1 - 0.1 * warm)], axis=1)

    # ---- thickness for the light through the skin: an embryo is thin all over, so the scale is wide
    thick = skin_thickness(ops, pos, nrm, A, deg)
    print(f"thickness: min {thick.min():.3f}, median {np.median(thick):.3f}, max {thick.max():.3f}")
    alpha = np.array([smoothstep(0.02, 0.5, x) for x in thick])

    # ---- organs in the way of the light, and where the head's surface vessels run
    organ = organ_shadow(emb.organs, pos, nrm, np.minimum(thick, 0.45))
    for _ in range(3):
        organ = 0.5 * organ + 0.5 * (A @ organ) / deg
    head_ops = [by[n] for n in ("cranium", "forebrain", "midbrain", "hindbrain")]
    on_head = np.exp(-np.maximum(np.min([o.shape.sdf(pos) for o in head_ops], axis=0), 0) / 0.01)
    away_from_face = np.array([smoothstep(0.02, 0.12, y) for y in pos[:, 1]])
    vessel = on_head * away_from_face
    for _ in range(3):
        vessel = 0.5 * vessel + 0.5 * (A @ vessel) / deg
    print(f"organ shadow: mean {organ.mean():.2f}, max {organ.max():.2f}")

    center = (pos.min(axis=0) + pos.max(axis=0)) / 2
    skin = {
        "position": pos - center,
        "normal": nrm,
        "color": np.round(np.clip(np.concatenate([rgb, alpha[:, None]], axis=1), 0, 1) * 255).astype(np.uint8),
        "detail": np.round(np.clip(np.stack([organ, vessel], axis=1), 0, 1) * 255).astype(np.uint8),
        "tris": tris,
    }
    ev, en, et = [], [], []
    for c, r in emb.eyes:
        v_, n_, t_ = sphere(c - center, r)
        et.append(t_ + sum(len(x) for x in ev))
        ev.append(v_)
        en.append(n_)
    eyes = {"position": np.concatenate(ev), "normal": np.concatenate(en), "tris": np.concatenate(et)}
    write_glb(args.out, skin, eyes)

    rig = {"week": args.week, "center": center.round(5).tolist(),
           "anchors": {k: (v - center).round(5).tolist() for k, v in emb.anchors.items()}}
    with open(args.rig, "w") as fh:
        json.dump(rig, fh, separators=(",", ":"))
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) and {args.rig} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

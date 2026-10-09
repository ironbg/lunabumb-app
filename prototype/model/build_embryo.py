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

    # the eye: height and depth on the side of the face
    EYE = (0.04, 0.1)

    def add(self, name, shape, k, group):
        self.ops.append(Op(name, shape, "add", k, group))

    def sub(self, name, shape, k, group):
        self.ops.append(Op(name, shape, "sub", k, group))

    def surface_x(self, y, z, hi=0.4):
        """How far out along +x the skin is at height y and depth z (the sculpt so far)."""
        lo = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if eval_sdf(self.ops, np.array([[mid, y, z]]))[0] < 0:
                lo = mid
            else:
                hi = mid
        return lo

    def build(self):
        add, sub = self.add, self.sub
        ops = self.ops
        ops.junctions = {
            "head": (vec(0, -0.03, -0.1), 0.2, 0.07),
            "armL": (vec(-0.13, -0.14, -0.15), 0.07, 0.045),
            "armR": (vec(0.13, -0.14, -0.15), 0.07, 0.045),
            "legL": (vec(-0.09, -0.49, 0.07), 0.07, 0.045),
            "legR": (vec(0.09, -0.49, 0.07), 0.07, 0.045),
        }

        # ---- head: nearly half the embryo. The forebrain bulges far forward over the small face,
        # the midbrain makes the dome on top and the hindbrain slopes into the back of the neck.
        # Proportions follow the side view of the reference clip (measured in model units)
        add("cranium", Ellipsoid(vec(0, 0.215, 0.06), vec(0.2, 0.225, 0.29)), 0.1, "head")
        add("midbrain", Ellipsoid(vec(0, 0.285, 0.1), vec(0.175, 0.165, 0.21)), 0.1, "head")
        add("forebrain", Ellipsoid(vec(0, 0.14, 0.2), vec(0.18, 0.16, 0.17)), 0.1, "head")
        add("hindbrain", Ellipsoid(vec(0, 0.13, -0.15), vec(0.15, 0.17, 0.15)), 0.09, "head")
        # the face is small and tucked under the forebrain, looking down onto the chest
        add("midface", Ellipsoid(vec(0, 0.0, 0.13), vec(0.11, 0.065, 0.085)), 0.05, "head")
        add("snout", Ellipsoid(vec(0, -0.02, 0.15), vec(0.065, 0.04, 0.045)), 0.04, "head")
        add("jaw", Ellipsoid(vec(0, -0.035, 0.095), vec(0.105, 0.032, 0.065)), 0.04, "head")
        sub("mouth", Ellipsoid(vec(0, -0.05, 0.155), vec(0.048, 0.005, 0.03)), 0.008, "head")
        for s, side in ((1, "R"), (-1, "L")):
            sub(f"nasalPit{side}", Ellipsoid(vec(0.024 * s, -0.045, 0.175), vec(0.008, 0.01, 0.01)), 0.006, "head")
            # the cheek (upper jaw swelling) fills in under and behind the eye, towards the ear
            add(f"cheek{side}", Ellipsoid(vec(0.095 * s, -0.005, 0.06), vec(0.065, 0.05, 0.085)), 0.04, "head")
            # the eye sits in a low swelling on the side of the face (the eye itself is placed on the
            # finished surface in main())
            add(f"eyeBulge{side}", Ellipsoid(vec(0.115 * s, self.EYE[0], self.EYE[1]), vec(0.045, 0.045, 0.045)), 0.035, "head")

        # ---- trunk: a deep body curled round the heart and liver; the back is one smooth curve
        add("upperBack", Ellipsoid(vec(0, -0.05, -0.1), vec(0.15, 0.13, 0.165)), 0.08, "torso")
        add("heart", Ellipsoid(vec(0, -0.155, 0.04), vec(0.15, 0.11, 0.16)), 0.08, "torso")
        add("midBack", Ellipsoid(vec(0, -0.2, -0.1), vec(0.145, 0.16, 0.15)), 0.08, "torso")
        add("liver", Ellipsoid(vec(0, -0.3, 0.06), vec(0.155, 0.12, 0.16)), 0.08, "torso")
        add("lowerBack", Ellipsoid(vec(0, -0.35, -0.03), vec(0.13, 0.12, 0.12)), 0.08, "torso")
        add("rump", Ellipsoid(vec(0, -0.465, 0.1), vec(0.115, 0.095, 0.12)), 0.07, "torso")
        # what is left of the tail curls under between the legs
        add("tail", RoundCone(vec(0, -0.51, 0.04), vec(0, -0.55, 0.13), 0.035, 0.015), 0.03, "torso")
        # the root of the cord, swollen by the loop of gut in it at this age
        cord_a, cord_b = vec(0, -0.29, 0.15), vec(0, -0.3, 0.27)
        add("cordRoot", RoundCone(cord_a, cord_b, 0.065, 0.056), 0.05, "torso")

        # the ear: a small rounded knob level with the eye and well behind it, where the head meets
        # the neck (the outer ear only starts to fold later)
        ear_y, ear_z = 0.037, -0.068
        ear_x = self.surface_x(ear_y, ear_z)
        for s, side in ((1, "R"), (-1, "L")):
            c = vec((ear_x + 0.002) * s, ear_y, ear_z)
            add(f"ear{side}", Ellipsoid(c, vec(0.016, 0.027, 0.019)), 0.01, "head")
            sub(f"earPit{side}", Ellipsoid(c + vec(0.016 * s, -0.006, 0.008), vec(0.004, 0.009, 0.005)), 0.004, "head")
        self.anchors["ear"] = vec(ear_x + 0.02, ear_y, ear_z)
        # where the vessels of the head fan out from, high on each side (the shader draws them), and
        # the middle of the skull they fan round
        vy, vz = 0.25, -0.02
        self.anchors["vessels"] = vec(self.surface_x(vy, vz), vy, vz)
        self.anchors["skull"] = vec(0, 0.215, 0.06)

        # where the spine runs under the skin of the back: the back itself stays smooth, the vertebrae
        # and ribs only show as shadows inside (see the organs below). Bisect for the skin of the back
        # along a few lines across it, so the shadows follow its curve
        def back_z(x, y):
            lo_z, hi_z = -0.45, 0.0
            for _ in range(30):
                mid = 0.5 * (lo_z + hi_z)
                if eval_sdf(ops, np.array([[x, y, mid]]))[0] < 0:
                    hi_z = mid
                else:
                    lo_z = mid
            return hi_z
        back = [(float(y), [(float(x), back_z(x, y)) for x in np.linspace(-0.1, 0.1, 9)])
                for y in np.arange(-0.03, -0.45, -0.05)]

        # ---- limbs: short, thick buds with no elbow or knee yet, ending in plates for the hands and
        # feet. The arm leaves the side of the body high up near the back and reaches forwards under
        # the ear; the hand plate hangs down from it against the chest. The leg leaves the rump and
        # reaches forwards, the foot plate under the cord
        for s, side in ((1, "R"), (-1, "L")):
            arm = f"arm{side}"
            sh = vec(0.12 * s, -0.14, -0.15)
            wr = vec(0.21 * s, -0.08, -0.01)
            add(f"armBud{side}", Chain([RoundCone(sh, wr, 0.05, 0.043)]), 0.03, arm)
            hand_c = vec(0.225 * s, -0.15, 0.02)
            self.plate(add, sub, f"hand{side}", arm, centre=hand_c,
                       normal=vec(1.0 * s, 0.0, 0.25), up=unit(wr - hand_c), r=(0.06, 0.085), thick=0.018, rays=5,
                       spread=1.7, from_dir=unit(wr - hand_c))
            leg = f"leg{side}"
            hip = vec(0.08 * s, -0.49, 0.07)
            an = vec(0.12 * s, -0.46, 0.23)
            add(f"legBud{side}", Chain([RoundCone(hip, an, 0.06, 0.05)]), 0.03, leg)
            foot_c = vec(0.125 * s, -0.43, 0.29)
            self.plate(add, sub, f"foot{side}", leg, centre=foot_c,
                       normal=vec(1.0 * s, -0.15, 0.1), up=unit(foot_c - an), r=(0.05, 0.06), thick=0.018, rays=5,
                       spread=1.6, from_dir=unit(an - foot_c))
            self.anchors[f"hand{side}"] = hand_c
            self.anchors[f"foot{side}"] = foot_c
            self.anchors[f"knee{side}"] = 0.5 * (hip + an) + vec(0.04 * s, 0, 0)

        # ---- what blocks the light inside, just under the skin: the heart and liver faintly from the
        # front; the spinal cord and the vertebrae with their ribs much more, as the dark column and
        # bars that show through the back. The head's darker middle comes from its thickness alone
        self.organs = [
            (Ellipsoid(vec(0, -0.15, 0.06), vec(0.08, 0.07, 0.08)), 6.0),     # heart
            (Ellipsoid(vec(0, -0.3, 0.06), vec(0.12, 0.09, 0.11)), 5.0),      # liver
        ]
        # (these only show from behind: seen from the side they would streak the flanks)
        behind = vec(0, 0, -1)
        for y, line in back:
            z0 = line[len(line) // 2][1]
            self.organs.append((Ellipsoid(vec(0, y, z0 + 0.035), vec(0.036, 0.034, 0.022)), 50.0, behind))   # spinal cord
            if -0.36 < y < -0.05:
                # a vertebra and its ribs: a bar across the back, fading out to the sides
                for x, z in line:
                    self.organs.append((Ellipsoid(vec(x, y, z + 0.03), vec(0.022, 0.016, 0.022)),
                                        70.0 * (1.0 - 0.45 * abs(x) / 0.1), behind))

        # ---- anchors for the page
        self.anchors.update({
            "head": vec(0.1, 0.42, 0.1),
            "face": vec(0, 0.0, 0.16),
            "faceFront": vec(0, -0.1, 0.5),
            "heart": vec(0.05, -0.13, 0.2),
            "navel": cord_b + vec(0, 0, 0.03),
            "groin": vec(0, -0.48, 0.12),
            "core": vec(0, -0.05, 0.0),
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
    inward along the normal through the body (Beer-Lambert over the organs' soft volumes). An organ
    may carry a direction it only shows from (its absorption fades where the skin faces away)."""
    total = np.zeros(len(p))
    facing = [np.ones(len(p)) if len(o) < 3 else np.clip((nrm @ o[2] - 0.25) / 0.45, 0.0, 1.0) for o in organs]
    for k in range(steps):
        t = (k + 0.5) / steps * depth
        q = p - nrm * t[:, None]
        dens = np.zeros(len(p))
        for (shape, absorb, *_), f in zip(organs, facing):
            d = shape.sdf(q)
            dens += absorb * f * np.clip(-d / 0.01 + 0.5, 0.0, 1.0)
        total += dens * depth / steps
    return 1.0 - np.exp(-total)


def sphere(c, r, rings=32, segs=48):
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


def eye_colours(normals, outward):
    """Black pigment with a soft grey ring round the front, where the lens shows through."""
    cosang = normals @ outward
    ring = np.exp(-((cosang - 0.8) / 0.09) ** 2)
    shade = 0.03 + 0.32 * ring
    return np.round(np.clip(np.stack([shade, shade * 0.97, shade], axis=1), 0, 1) * 255).astype(np.uint8)


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
        if "color" in m:
            nc = m["color"].shape[1]
            attrs["COLOR_0"] = out.dense(m["color"], {"componentType": 5121, "normalized": True, "type": f"VEC{nc}", "count": n})
        if extra:
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
        y, z = emb.EYE
        r = 0.028
        sx = emb.surface_x(y, z)
        emb.eyes.append((vec((sx - 0.45 * r) * s, y, z), r))
        # a soft fold of skin above the eye: the eyelids are just starting
        emb.lid_lines.append(Ellipsoid(vec((sx - 0.012) * s, y + 0.033, z), vec(0.03, 0.006, 0.03)))
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
    # only what lies just under the skin shows as a shape: from the front the heart and liver,
    # from behind the spine and ribs (deeper organs only add to the overall darkening by thickness)
    organ = organ_shadow(emb.organs, pos, nrm, np.minimum(thick, 0.16))
    for _ in range(1):
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
    ev, en, et, ec = [], [], [], []
    for c, r in emb.eyes:
        v_, n_, t_ = sphere(c - center, r)
        et.append(t_ + sum(len(x) for x in ev))
        ev.append(v_)
        en.append(n_)
        # the eye looks out sideways and a little forwards
        ec.append(eye_colours(n_, unit(vec(np.sign(c[0]), 0.0, 0.45))))
    eyes = {"position": np.concatenate(ev), "normal": np.concatenate(en), "tris": np.concatenate(et),
            "color": np.concatenate(ec)}
    write_glb(args.out, skin, eyes)

    rig = {"week": args.week, "center": center.round(5).tolist(),
           "anchors": {k: (v - center).round(5).tolist() for k, v in emb.anchors.items()}}
    with open(args.rig, "w") as fh:
        json.dump(rig, fh, separators=(",", ":"))
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) and {args.rig} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

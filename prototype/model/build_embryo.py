"""Build a Luna View embryo model for one early week (glTF), from a signed-distance-field sculpt.

The fetus model (build_fetus.py) morphs one mesh through all the weeks. An embryo looks too
different for that: its own sculpt per week reads better. Each week here is a static mesh plus
the two dark eyes, with per-vertex data for the skin shader:

  COLOR_0   rgb: soft occlusion and tints, a: local thickness (for the light shining through)
  _VESSEL   where the fine surface vessels of the head may show (the shader draws them)
  _PALE     how much paler the skin is: the hand and foot plates and, less, the limb buds have
            little blood in them yet and look almost white-pink next to the red body
  _MOVE     x: which part a vertex belongs to (1 left arm, 2 right arm, 3 left leg, 4 right leg,
            5 head), y: how much it follows that part when it moves (0 at the joint, 1 beyond);
            the page swings a part round its joint (in the rig) when it is touched. The eyes carry
            it too, so they nod with the head

The organs that block the light coming through the body (heart, liver, spinal cord, vertebrae and
ribs) are not painted on the skin: they are baked into a small 3D texture that goes in the rig, and
the shader looks into it along the line of sight, so they sit deep inside and shift with the view.

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
                         blender_mesh, eval_group, eval_sdf, gradient, marching_cubes, project, read_mesh, rotation_between,
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
            "armL": (vec(-0.13, -0.12, -0.13), 0.07, 0.045),
            "armR": (vec(0.13, -0.12, -0.13), 0.07, 0.045),
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
            add(f"eyeBulge{side}", Ellipsoid(vec(0.1 * s, self.EYE[0], self.EYE[1]), vec(0.04, 0.04, 0.04)), 0.03, "head")

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

        # ---- limbs: short, thick, soft buds with no elbow or knee yet. The arm leaves the side of the
        # body high up near the back and reaches straight forwards and a little down, beside the jaw;
        # the hand is a thin, broad, rounded plate carried on in line with it, flat side out, so from
        # the side it shows whole, a pale plate in front of the chin with a softly scalloped edge. The
        # leg leaves the rump and reaches forwards under the cord; the foot is a smaller, narrower plate
        # of the same kind in line with it, its sole turned in, so from the front or behind it shows
        # edge-on as a thin, leaf-like blade. Each limb can swing a little at its root when touched (the
        # joints below go in the rig)
        self.joints = {}
        for s, side in ((1, "R"), (-1, "L")):
            arm = f"arm{side}"
            sh = vec(0.125 * s, -0.12, -0.13)
            reach = unit(vec(0.22 * s, -0.1, 0.97))
            wr = sh + reach * 0.15
            add(f"armBud{side}", Chain([RoundCone(sh, wr, 0.052, 0.042)]), 0.03, arm)
            self.limb_plate(add, f"hand{side}", arm, root=wr, axis=reach, normal=vec(1.0 * s, 0.0, 0.0),
                            length=0.15, width=0.078, thick=0.011, root_r=0.042, lobes=5, spread=2.3)
            hand_c = wr + reach * 0.09
            leg = f"leg{side}"
            hip = vec(0.08 * s, -0.49, 0.07)
            an = vec(0.12 * s, -0.465, 0.25)
            add(f"legBud{side}", Chain([RoundCone(hip, an, 0.062, 0.046)]), 0.03, leg)
            axis = unit(an - hip)
            self.limb_plate(add, f"foot{side}", leg, root=an, axis=axis, normal=vec(1.0 * s, 0.0, 0.0),
                            length=0.12, width=0.052, thick=0.012, root_r=0.046, lobes=5, spread=1.7)
            foot_c = an + axis * 0.07
            self.anchors[f"hand{side}"] = hand_c
            self.anchors[f"foot{side}"] = foot_c
            up = vec(0, 1, 0)
            self.joints[arm] = {"pivot": sh, "axis": unit(np.cross(reach, up)), "dir": reach, "plate": wr}
            self.joints[leg] = {"pivot": hip, "axis": unit(np.cross(axis, up)), "dir": axis, "plate": an}
            self.anchors[f"knee{side}"] = 0.5 * (hip + an) + vec(0.04 * s, 0, 0)

        # ---- what blocks the light inside the body (baked into a small 3D texture, see main()): the
        # heart and liver, the brain, the spinal cord and the vertebrae with their ribs. They sit well
        # under the skin, so they show as soft dark shapes deep inside when light comes through the
        # body, and shift with the viewing angle instead of looking drawn on
        self.organs = [
            (Ellipsoid(vec(0, -0.15, 0.06), vec(0.08, 0.07, 0.08)), 3.0),     # heart
            (Ellipsoid(vec(0, -0.3, 0.06), vec(0.12, 0.09, 0.11)), 2.5),      # liver
            # the medulla, running down from the brain into the spinal cord
            (RoundCone(vec(0, 0.12, -0.16), vec(0, -0.01, -0.2), 0.042, 0.032), 14.0),
        ]
        # the brain: against the light the big forebrain shows as a large, soft dark mass in the front
        # of the head, in front of and above the eye, fading out towards a lighter rim (nested shells
        # give the soft edge); the midbrain and hindbrain behind it are only faint
        for c in (vec(0.07, 0.19, 0.19), vec(-0.07, 0.19, 0.19)):
            for k, absorb in ((1.0, 2.5), (0.75, 3.0), (0.5, 3.5)):
                self.organs.append((Ellipsoid(c, vec(0.085, 0.11, 0.12) * k), absorb))
        self.organs.append((Ellipsoid(vec(0, 0.29, 0.03), vec(0.09, 0.08, 0.1)), 2.5))
        self.organs.append((Ellipsoid(vec(0, 0.17, -0.13), vec(0.08, 0.09, 0.08)), 3.0))
        # the spinal cord: one tube along the back from the medulla to the tail
        cord_pts = [vec(0, -0.01, -0.2)] + [vec(0, y, line[len(line) // 2][1] + 0.06) for y, line in back]
        for a_, b_ in zip(cord_pts[:-1], cord_pts[1:]):
            self.organs.append((RoundCone(a_, b_, 0.025, 0.025), 30.0))
        # four ribs low on the trunk: each an arc under the skin from beside the spine round the back
        # and the side of the body, so against the light they show as four separate dark bars on the
        # flank. Bisect for the skin along rays out from the middle of the trunk
        def skin_along(c, d):
            lo_t, hi_t = 0.0, 0.45
            for _ in range(30):
                mid = 0.5 * (lo_t + hi_t)
                if eval_sdf(ops, (c + d * mid)[None])[0] < 0:
                    lo_t = mid
                else:
                    hi_t = mid
            return lo_t
        for y in (-0.2, -0.245, -0.29, -0.335):
            c = vec(0, y, -0.03)
            for s_ in (1, -1):
                for th in np.linspace(0.3, 1.75, 9):
                    d = vec(s_ * math.sin(th), 0, -math.cos(th))
                    self.organs.append((Ellipsoid(c + d * (skin_along(c, d) - 0.04), vec(0.018, 0.012, 0.018)), 60.0))

        # the head nods forwards round the neck
        self.joints["head"] = {"pivot": vec(0, -0.03, -0.12), "axis": vec(1, 0, 0), "dir": unit(vec(0, 0.23, 0.18))}

        # ---- anchors for the page
        self.anchors.update({
            "head": vec(0.1, 0.42, 0.1),
            "face": vec(0, 0.0, 0.16),
            "faceFront": vec(0, -0.1, 0.5),
            "heart": vec(0.05, -0.13, 0.2),
            "navel": cord_b + vec(0, 0, 0.03),
            # the cord leaves along the root's axis
            "cordBase": cord_a,
            "groin": vec(0, -0.48, 0.12),
            "core": vec(0, -0.05, 0.0),
        })
        self.anchors["hand"] = self.anchors["handL"]
        self.anchors["knee"] = self.anchors["kneeL"]
        self.anchors["footL"] = self.anchors.pop("footL")
        self.anchors["footR"] = self.anchors.pop("footR")
        return self

    def limb_plate(self, add, name, group, root, axis, normal, length, width, thick, root_r, lobes, spread):
        """A hand or foot plate carried on from its limb in line with it (no bend at the wrist or
        ankle). The round limb flattens and widens gradually into a thin, broad plate (there is no
        narrowing at the wrist), whose far edge is made of soft, flat, overlapping lobes where the
        fingers or toes are starting, so the outline is gently scalloped and nothing stands out as a
        finger yet. root_r is the limb's radius where the plate begins."""
        n = np.asarray(normal, float)
        n = unit(n - axis * (n @ axis))
        R = frame(n, -axis)
        u, v = R[:, 0], R[:, 1]
        # the flattening: from the limb's own round section to the plate's
        for i, (t, w, th) in enumerate(((0.0, 0.0, 1.0), (0.16, 0.35, 0.55), (0.34, 0.7, 0.25), (0.52, 0.92, 0.0))):
            ru = root_r + (width - root_r) * w
            rn = thick + (root_r * 0.9 - thick) * th
            add(f"{name}Palm{i}", Ellipsoid(root + axis * length * t, vec(ru, length * 0.2, rn), R), 0.025, group)
        far = root + axis * length * 0.68
        add(f"{name}Far", Ellipsoid(far, vec(width, length * 0.26, thick), R), 0.02, group)
        lobe = 0.46 * width * spread / (lobes - 1)
        # (a little uneven, as a real plate is: the middle lobes reach a touch further)
        for i, (k, r) in enumerate(((0.96, 0.9), (1.02, 1.0), (1.04, 1.05), (1.01, 0.95), (0.95, 0.85))[:lobes]):
            a = -math.pi / 2 + (i - (lobes - 1) / 2) * spread / (lobes - 1)
            d = math.cos(a) * u + math.sin(a) * v
            rim = math.hypot(math.cos(a) * width, math.sin(a) * length * 0.26)
            c = far + d * (rim - 0.55 * lobe) * k
            add(f"{name}Lobe{i}", Ellipsoid(c, vec(lobe * r, lobe * r, thick * 0.65), R), 0.01, group)


def organ_volume(organs, h=0.012, blur=1.0):
    """The organs' light absorbance as a small 3D texture: soft-edged and blurred a little more, so
    they read as shapes inside the body. Returns the texture's lower corner and size (covering whole
    texels), the uint8 volume with x varying fastest, and the absorbance per unit length at 255."""
    from scipy.ndimage import gaussian_filter
    boxes = [shape.aabb() for shape, _ in organs]
    lo = np.min([b[0] for b in boxes], axis=0) - 0.04
    hi = np.max([b[1] for b in boxes], axis=0) + 0.04
    dims = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[i] + h * np.arange(dims[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    dens = np.zeros(len(pts))
    for shape, absorb in organs:
        dens += absorb * np.clip(-shape.sdf(pts) / 0.01 + 0.5, 0.0, 1.0)
    dens = gaussian_filter(dens.reshape(dims), blur)
    scale = float(dens.max())
    vol = np.round(dens / scale * 255).astype(np.uint8).transpose(2, 1, 0)
    # texel i is centred on lo + i * h
    return lo - 0.5 * h, dims * h, vol, scale


def sphere(c, r, rings=32, segs=48, squash=1.0):
    """A UV sphere, optionally flattened in y (an ellipsoid)."""
    verts, normals = [], []
    k = np.array([1.0, squash, 1.0])
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segs + 1):
            ph = 2 * math.pi * j / segs
            n = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
            verts.append(c + r * n * k)
            normals.append(unit(n / k))
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
    # (the eye is set deep, so only the cap round the outward axis shows: keep the ring small in it)
    ring = np.exp(-((cosang - 0.95) / 0.03) ** 2)
    shade = 0.03 + 0.32 * ring
    return np.round(np.clip(np.stack([shade, shade * 0.97, shade], axis=1), 0, 1) * 255).astype(np.uint8)


def write_glb(path, skin, eyes):
    """skin: dict(position, normal, color (uint8 RGBA), vessel (float), tris); eyes the same with
    an RGB colour and no vessel mask. Plain floats: the file is small enough without quantization."""
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
            attrs["_VESSEL"] = out.dense(m["vessel"].astype(np.float32)[:, None], {"componentType": 5126, "type": "SCALAR", "count": n})
            attrs["_PALE"] = out.dense(m["pale"].astype(np.float32)[:, None], {"componentType": 5126, "type": "SCALAR", "count": n})
        if "move" in m:
            attrs["_MOVE"] = out.dense(m["move"].astype(np.float32), {"componentType": 5126, "type": "VEC2", "count": n})
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
        # set well into the head, so only a flat, almond-shaped cap shows (it doesn't bulge out)
        r = 0.032
        sx = emb.surface_x(y, z)
        emb.eyes.append((vec((sx - 0.72 * r) * s, y, z), r))
        # a soft fold of skin above the eye: the eyelids are just starting
        emb.lid_lines.append(Ellipsoid(vec((sx - 0.012) * s, y + 0.033, z), vec(0.03, 0.006, 0.03)))
    obj = blender_mesh(verts, faces, args.tris, details=(
        (near(("ear",), 0.01), 0.6), (near(("hand", "foot"), 0.01), 0.7),
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
    # the root of the cord takes the same value the cord tube is shaded with in the page (0.3), so the
    # two meet without a change of colour; it blends into the belly's own value where they join
    rest = np.min([op.shape.sdf(pos) for op in ops if op.group == "torso" and op.mode == "add" and op.name != "cordRoot"], axis=0)
    t = np.clip(rest / 0.03, 0.0, 1.0)
    w = np.clip(1.0 - by["cordRoot"].shape.sdf(pos) / 0.012, 0.0, 1.0) * t * t * (3 - 2 * t)
    alpha = alpha * (1 - w) + 0.3 * w

    # ---- where the head's surface vessels run
    head_ops = [by[n] for n in ("cranium", "forebrain", "midbrain", "hindbrain")]
    on_head = np.exp(-np.maximum(np.min([o.shape.sdf(pos) for o in head_ops], axis=0), 0) / 0.01)
    away_from_face = np.array([smoothstep(0.02, 0.12, y) for y in pos[:, 1]])
    vessel = on_head * away_from_face
    for _ in range(3):
        vessel = 0.5 * vessel + 0.5 * (A @ vessel) / deg
    # ---- which part each vertex moves with, and how much (see _MOVE)
    parts = {"armL": 1, "armR": 2, "legL": 3, "legR": 4, "head": 5}
    groups = ["torso"] + list(parts)
    gd = np.stack([eval_group([op for op in ops if op.group == g], pos) for g in groups], axis=1)
    owner = np.argmin(gd, axis=1)
    move = np.zeros((len(pos), 2))
    # a part's vertices next to the body (where its surface meets another part's) must not move at
    # all, or the skin there would tear: each part's ramp starts just beyond where it leaves the body
    other = np.zeros(len(pos), bool)
    cross = owner[edges[:, 0]] != owner[edges[:, 1]]
    other[edges[cross, 0]] = True
    other[edges[cross, 1]] = True
    for gi, g in enumerate(groups[1:], start=1):
        j = emb.joints[g]
        sel = owner == gi
        along = (pos - j["pivot"]) @ j["dir"]
        start = np.percentile(along[sel & other], 97) if (sel & other).any() else 0.0
        length = 0.12 if g == "head" else 0.07
        t = np.clip((along - start) / length, 0, 1)
        move[sel, 0] = parts[g]
        move[sel, 1] = (t * t * (3 - 2 * t))[sel]
    for _ in range(3):   # soften the bend a little more
        lim = move[:, 0] > 0
        move[:, 1] = np.where(lim, 0.5 * move[:, 1] + 0.5 * (A @ move[:, 1]) / deg, 0)
        move[other, 1] = 0
    print("moving parts: " + ", ".join(f"{g} {int((move[:, 0] == parts[g]).sum())}" for g in parts))

    # ---- the paler hands and feet. Both the pale colour and the thickness the light comes through
    # run along each limb on one long, soft curve: from where the limb leaves the body (the body's own
    # tone and thickness) all the way to a little past the wrist or ankle (the plate's). There is no
    # second step at the shoulder or the wrist, so the limb shades evenly from the body into the plate
    pale = np.zeros(len(pos))
    limb = np.zeros(len(pos), bool)
    for gi, g in enumerate(groups[1:], start=1):
        if g == "head":
            continue
        j = emb.joints[g]
        sel = owner == gi
        limb |= sel
        along = (pos[sel] - j["pivot"]) @ j["dir"]
        start = np.percentile(along[other[sel]], 97) if other[sel].any() else along.min()
        end = (j["plate"] - j["pivot"]) @ j["dir"] + 0.05
        t = np.clip((along - start) / (end - start), 0, 1)
        t = t * t * t * (t * (6 * t - 15) + 10)
        # (the paleness picks up a little later than the thickness: the upper arm and thigh keep
        # nearly the body's tone)
        pale[sel] = t ** 1.6
        # the limb starts with the thickness of the body next to it (a thin bud would otherwise glow
        # paler than the trunk it grows from), and the plate keeps more than half of that
        from scipy.spatial import cKDTree
        rim = pos[sel & other]
        body_rim = (owner == 0) & other
        near = cKDTree(rim).query(pos[body_rim], distance_upper_bound=0.04)[0] < np.inf if len(rim) else np.zeros(int(body_rim.sum()), bool)
        a_root = np.median(alpha[body_rim][near]) if near.any() else alpha[sel].max()
        a_plate = np.median(alpha[sel][along > end]) if (along > end).any() else alpha[sel].min()
        a_plate = a_root + (a_plate - a_root) * 0.4
        alpha[sel] = a_root + (a_plate - a_root) * t
    # soften both over the surface, the thickness also a little way into the body round each limb
    near_limb = limb.astype(float)
    for _ in range(6):
        near_limb = np.maximum(near_limb, (A @ near_limb) / deg)
    near_limb = near_limb > 0.01
    for _ in range(40):
        pale = 0.5 * pale + 0.5 * (A @ pale) / deg
        alpha = np.where(near_limb, 0.5 * alpha + 0.5 * (A @ alpha) / deg, alpha)

    # ---- the organs inside, as a 3D texture
    vol_lo, vol_size, vol, vol_scale = organ_volume(emb.organs)
    print(f"organ volume {vol.shape[::-1]} ({vol.size / 1e3:.0f} kB), absorbance up to {vol_scale:.0f}")

    center = (pos.min(axis=0) + pos.max(axis=0)) / 2
    skin = {
        "position": pos - center,
        "normal": nrm,
        "color": np.round(np.clip(np.concatenate([rgb, alpha[:, None]], axis=1), 0, 1) * 255).astype(np.uint8),
        "vessel": np.clip(vessel, 0, 1).round(3),
        "pale": np.clip(pale, 0, 1).round(3),
        "move": move.round(3),
        "tris": tris,
    }
    ev, en, et, ec = [], [], [], []
    for c, r in emb.eyes:
        v_, n_, t_ = sphere(c - center, r, squash=0.8)
        et.append(t_ + sum(len(x) for x in ev))
        ev.append(v_)
        en.append(n_)
        # the eye looks out sideways and a little forwards
        ec.append(eye_colours(n_, unit(vec(np.sign(c[0]), 0.0, 0.45))))
    eyes = {"position": np.concatenate(ev), "normal": np.concatenate(en), "tris": np.concatenate(et),
            "color": np.concatenate(ec)}
    eyes["move"] = np.tile([5.0, 1.0], (len(eyes["position"]), 1))
    write_glb(args.out, skin, eyes)

    import base64
    rig = {"week": args.week, "center": center.round(5).tolist(),
           "anchors": {k: (v - center).round(5).tolist() for k, v in emb.anchors.items()},
           # what each part swings round when touched: the joint, and the axis it turns about
           "joints": {g: {"part": parts[g], "pivot": (j["pivot"] - center).round(5).tolist(),
                          "axis": np.asarray(j["axis"]).round(4).tolist()} for g, j in emb.joints.items()},
           # x varies fastest, then y, then z; "scale" is the absorbance per CRL at a texel of 255
           "organs": {"min": (vol_lo - center).round(5).tolist(), "size": vol_size.round(5).tolist(),
                      "dims": list(vol.shape[::-1]), "scale": round(vol_scale, 2),
                      "data": base64.b64encode(vol.tobytes()).decode("ascii")}}
    with open(args.rig, "w") as fh:
        json.dump(rig, fh, separators=(",", ":"))
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) and {args.rig} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

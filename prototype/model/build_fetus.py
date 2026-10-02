"""Build the Luna View fetus model (glTF) from a signed-distance-field sculpt.

Pipeline
  1. A parametric skeleton for any gestational week (key poses at 8, 20 and 40
     weeks, proportions from approximate fetal biometry tables).
  2. An SDF sculpt on top of it: smooth unions of ellipsoids and round cones
     (head, face, ears, torso, buttocks, limbs, fingers, toes) and smooth
     subtractions for small creases (eyelids, mouth, nostrils, ear concha,
     gluteal cleft).
  3. Marching cubes on the week-24 sculpt, cleaned up and decimated in Blender,
     keeping more vertices in the ears, face and groin. This is the basis mesh.
  4. Growth shape keys (W8 ... W40): the basis vertices are carried to each week
     by blending per-primitive frame mappings, then projected onto that week's
     SDF surface. All keys share one topology, so they morph smoothly.
  5. Action shape keys (KICK_L, KICK_R with half-way KICK_*_MID keys, WAVE_L)
     built the same way from posed skeletons at week 24; kicks are joint
     rotations at the hip, knee and ankle. Only the moving limb is relaxed,
     so these keys are sparse.
  6. Sex keys (SEX_8, SEX_12, F16 ... F40, M16 ... M40): external genitals grown
     out of each week's neutral body in steps, as deltas on that body.
  7. Ambient occlusion and soft skin tints computed from the SDF and stored as
     vertex colours, local thickness in the colour alpha (for translucency).
  8. GLB export, gltfpack quantization, then a repack that stores the sex keys
     and the action keys as sparse morph targets. Plus a small JSON rig with
     anchor points per key week.

Coordinates: "pose space" is x = baby's right, y = up, z = front, in
crown–rump units (CRL = 1). It is written to Blender as (x, -z, y) so the
glTF export (+Y up) comes back out as pose space.

Run (needs Blender's `bpy` module, numpy, scipy and scikit-image, and gltfpack):
  python build_fetus.py --out ../models/fetus.glb --rig ../models/fetus-rig.json
"""

import argparse
import json
import math
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------


def vec(*a):
    return np.array(a, dtype=np.float64)


def unit(a):
    return a / np.linalg.norm(a)


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def table(pairs, g):
    xs, ys = zip(*pairs)
    return float(np.interp(g, xs, ys))


def rotation_between(a, b):
    """Minimal rotation matrix taking unit vector a to unit vector b."""
    a = unit(a)
    b = unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        axis = unit(np.cross(a, vec(1, 0, 0)) if abs(a[0]) < 0.9 else np.cross(a, vec(0, 1, 0)))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


# --------------------------------------------------------------------------
# SDF primitives
# --------------------------------------------------------------------------


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def ssub(a, b, k):
    """Smoothly subtract shape b from shape a."""
    return -smin(-a, b, k)


class Ellipsoid:
    kind = "ell"

    def __init__(self, c, r, R=None):
        self.c = np.asarray(c, dtype=np.float64)
        self.r = np.asarray(r, dtype=np.float64)
        self.R = np.eye(3) if R is None else R

    def sdf(self, p):
        q = (p - self.c) @ self.R
        k0 = np.linalg.norm(q / self.r, axis=-1)
        k1 = np.linalg.norm(q / (self.r * self.r), axis=-1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    def aabb(self):
        m = self.r.max()
        return self.c - m, self.c + m

    def size(self):
        return float(self.r.min())

    def map_to(self, other, p):
        q = (p - self.c) @ self.R / self.r
        return other.c + (q * other.r) @ other.R.T


class RoundCone:
    kind = "cone"

    def __init__(self, a, b, ra, rb):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.ra = float(ra)
        self.rb = float(rb)
        # keep the cone well defined (|ra - rb| < length)
        length = np.linalg.norm(self.b - self.a)
        if abs(self.ra - self.rb) > 0.95 * length:
            mid = 0.5 * (self.ra + self.rb)
            half = 0.475 * length
            self.ra, self.rb = (mid + half, mid - half) if self.ra > self.rb else (mid - half, mid + half)

    def sdf(self, p):
        # Inigo Quilez, exact round cone
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        ba = b - a
        l2 = float(ba @ ba)
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pa = p - a
        y = pa @ ba
        z = y - l2
        xv = pa * l2 - y[:, None] * ba
        x2 = np.einsum("ij,ij->i", xv, xv)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d_end = np.sqrt(x2 + z2) * il2 - r2
        d_start = np.sqrt(x2 + y2) * il2 - r1
        d_side = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
        return np.where(np.sign(z) * a2 * z2 > k, d_end, np.where(np.sign(y) * a2 * y2 < k, d_start, d_side))

    def aabb(self):
        m = max(self.ra, self.rb)
        return np.minimum(self.a, self.b) - m, np.maximum(self.a, self.b) + m

    def size(self):
        return min(self.ra, self.rb)

    def map_to(self, other, p):
        l0 = np.linalg.norm(self.b - self.a)
        l1 = np.linalg.norm(other.b - other.a)
        u0 = (self.b - self.a) / l0
        u1 = (other.b - other.a) / l1
        R = rotation_between(u0, u1)
        rel = p - self.a
        s = rel @ u0
        radial = rel - s[:, None] * u0
        t = np.clip(s / l0, 0.0, 1.0)
        ratio = (1 - t) * (other.ra / max(self.ra, 1e-6)) + t * (other.rb / max(self.rb, 1e-6))
        s1 = np.where(s < 0, s * other.ra / max(self.ra, 1e-6),
                      np.where(s > l0, l1 + (s - l0) * other.rb / max(self.rb, 1e-6), s * l1 / l0))
        return other.a + s1[:, None] * u1 + (radial @ R.T) * ratio[:, None]


class Chain:
    """A tube made of round cones joined with an exact union, so the joints don't bulge."""
    kind = "chain"

    def __init__(self, cones):
        self.cones = cones

    def sdf(self, p):
        return np.min([c.sdf(p) for c in self.cones], axis=0)

    def aabb(self):
        boxes = [c.aabb() for c in self.cones]
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def size(self):
        return min(c.size() for c in self.cones)

    def map_to(self, other, p):
        nearest = np.stack([c.sdf(p) for c in self.cones], axis=1).argmin(axis=1)
        out = np.empty_like(p)
        for i, (a, b) in enumerate(zip(self.cones, other.cones)):
            m = nearest == i
            if m.any():
                out[m] = a.map_to(b, p[m])
        return out


class Op:
    def __init__(self, name, shape, mode, k, group):
        self.name = name
        self.shape = shape
        self.mode = mode  # "add" or "sub"
        self.k = k
        self.group = group


class Sculpt(list):
    """Ops grouped by body part. Parts are joined smoothly only around their joint
    (neck, shoulder, hip); elsewhere they meet with a hard edge, so a hand resting on
    a cheek or two feet touching stay separate surfaces that can move apart."""

    GROUPS = ("torso", "head", "armL", "armR", "legL", "legR")
    K_MIN = 0.004

    def __init__(self):
        super().__init__()
        self.junctions = {}   # group -> (point, radius, k)

    def junction_k(self, group, p):
        c, rho, k = self.junctions[group]
        d2 = np.einsum("ij,ij->i", p - c, p - c)
        return self.K_MIN + k * np.exp(-d2 / (rho * rho))


def eval_group(ops, p):
    d = np.full(len(p), 1.0)
    for op in ops:
        if op.shape.size() < 1e-4:
            continue
        s = op.shape.sdf(p)
        d = smin(d, s, op.k) if op.mode == "add" else ssub(d, s, op.k)
    return d


def eval_sdf(sculpt, p):
    total = None
    for group in Sculpt.GROUPS:
        ops = [op for op in sculpt if op.group == group]
        g = eval_group(ops, p)
        total = g if total is None else smin(total, g, sculpt.junction_k(group, p))
    return total


# --------------------------------------------------------------------------
# Skeleton: key poses + growth tables
# --------------------------------------------------------------------------

# Joint positions in pose space (x right, y up, z front), CRL units.
POSE_8 = {
    "curlL": (1.1, 1.25, 0), "curlR": (1.1, 1.25, 0),
    "skull": (0, 0.235, 0.105), "neck": (0, 0.04, 0.02),
    "backTop": (0, 0.04, -0.09), "backMid": (0, -0.12, -0.1), "backLow": (0, -0.28, -0.07),
    "chest": (0, -0.03, 0.01), "belly": (0, -0.165, 0.06), "pelvis": (0, -0.3, -0.02),
    "glutL": (-0.03, -0.34, -0.05), "glutR": (0.03, -0.34, -0.05),
    "tailA": (0, -0.36, -0.03), "tailB": (0, -0.44, 0.05),
    "shL": (-0.115, 0.0, 0.03), "elL": (-0.15, -0.07, 0.1), "wrL": (-0.13, -0.05, 0.165), "tipL": (-0.11, -0.02, 0.23), "palmL": (0.6, 0.3, 0.6),
    "shR": (0.115, 0.0, 0.03), "elR": (0.15, -0.07, 0.1), "wrR": (0.13, -0.05, 0.165), "tipR": (0.11, -0.02, 0.23), "palmR": (-0.6, 0.3, 0.6),
    "hipL": (-0.085, -0.28, 0.0), "knL": (-0.12, -0.3, 0.09), "anL": (-0.11, -0.33, 0.15), "toeL": (-0.1, -0.34, 0.21),
    "hipR": (0.085, -0.28, 0.0), "knR": (0.12, -0.3, 0.09), "anR": (0.11, -0.33, 0.15), "toeR": (0.1, -0.34, 0.21),
}
POSE_20 = {
    "curlL": (1.1, 1.25, 0), "curlR": (1.1, 1.25, 0),
    "skull": (0, 0.385, 0.07), "neck": (0, 0.2, 0.0),
    "backTop": (0, 0.16, -0.055), "backMid": (0, -0.04, -0.075), "backLow": (0, -0.24, -0.08),
    "chest": (0, 0.07, 0.005), "belly": (0, -0.105, 0.025), "pelvis": (0, -0.27, -0.035),
    "glutL": (-0.05, -0.32, -0.065), "glutR": (0.05, -0.32, -0.065),
    "tailA": (0, -0.33, -0.06), "tailB": (0, -0.36, -0.04),
    "shL": (-0.105, 0.165, -0.015), "elL": (-0.15, 0.03, 0.085), "wrL": (-0.15, 0.16, 0.165), "tipL": (-0.145, 0.275, 0.19), "palmL": (1.0, 0.0, 0.3),
    "shR": (0.105, 0.165, -0.015), "elR": (0.145, 0.015, 0.095), "wrR": (0.07, 0.125, 0.19), "tipR": (0.045, 0.205, 0.22), "palmR": (-1.0, 0.0, 0.2),
    "hipL": (-0.075, -0.262, 0.01), "knL": (-0.088, -0.08, 0.19), "anL": (-0.046, -0.258, 0.19), "toeL": (-0.05, -0.28, 0.32),
    "hipR": (0.075, -0.262, 0.01), "knR": (0.088, -0.08, 0.19), "anR": (0.046, -0.258, 0.19), "toeR": (0.05, -0.28, 0.32),
}
POSE_40 = {
    "curlL": (1.1, 1.25, 0), "curlR": (1.1, 1.25, 0),
    "skull": (0, 0.4, 0.06), "neck": (0, 0.215, 0.0),
    "backTop": (0, 0.17, -0.06), "backMid": (0, -0.04, -0.085), "backLow": (0, -0.24, -0.09),
    "chest": (0, 0.075, 0.005), "belly": (0, -0.105, 0.03), "pelvis": (0, -0.27, -0.035),
    "glutL": (-0.058, -0.325, -0.075), "glutR": (0.058, -0.325, -0.075),
    "tailA": (0, -0.33, -0.07), "tailB": (0, -0.36, -0.05),
    "shL": (-0.12, 0.175, -0.02), "elL": (-0.162, 0.04, 0.09), "wrL": (-0.162, 0.17, 0.17), "tipL": (-0.157, 0.29, 0.2), "palmL": (1.0, 0.0, 0.3),
    "shR": (0.12, 0.175, -0.02), "elR": (0.155, 0.02, 0.1), "wrR": (0.075, 0.13, 0.195), "tipR": (0.05, 0.215, 0.225), "palmR": (-1.0, 0.0, 0.2),
    "hipL": (-0.082, -0.262, 0.015), "knL": (-0.095, -0.075, 0.2), "anL": (-0.05, -0.264, 0.2), "toeL": (-0.055, -0.288, 0.345),
    "hipR": (0.082, -0.262, 0.015), "knR": (0.095, -0.075, 0.2), "anR": (0.05, -0.264, 0.2), "toeR": (0.055, -0.288, 0.345),
}

# Mean head radius from head circumference / CRL; limb sizes from femur length, all approximate
HEAD_R = [(8, 0.25), (12, 0.205), (16, 0.178), (20, 0.17), (28, 0.165), (36, 0.158), (40, 0.153)]
PITCH = [(8, 0.95), (12, 0.72), (20, 0.42), (40, 0.45)]
NECK_R = [(8, 0.12), (12, 0.09), (20, 0.068), (40, 0.075)]
BACK_R = [(8, 0.075), (20, 0.058), (40, 0.068)]
ARM_SH = [(8, 0.03), (12, 0.03), (20, 0.033), (28, 0.042), (40, 0.056)]
ARM_EL = [(8, 0.026), (12, 0.024), (20, 0.026), (28, 0.033), (40, 0.043)]
ARM_WR = [(8, 0.024), (12, 0.02), (20, 0.021), (28, 0.026), (40, 0.032)]
LEG_HIP = [(8, 0.04), (12, 0.04), (20, 0.05), (28, 0.063), (40, 0.08)]
LEG_KN = [(8, 0.032), (12, 0.03), (20, 0.034), (28, 0.044), (40, 0.055)]
LEG_AN = [(8, 0.026), (12, 0.022), (20, 0.024), (28, 0.029), (40, 0.035)]
HAND_L = [(8, 0.1), (12, 0.13), (20, 0.15), (40, 0.165)]
FOOT_L = [(8, 0.1), (12, 0.15), (20, 0.19), (40, 0.21)]
CHEST = [(8, (0.135, 0.125, 0.125)), (20, (0.11, 0.105, 0.095)), (40, (0.128, 0.115, 0.105))]
BELLY = [(8, (0.13, 0.12, 0.145)), (20, (0.115, 0.11, 0.11)), (40, (0.132, 0.115, 0.125))]
PELVIS = [(8, (0.095, 0.085, 0.085)), (20, (0.1, 0.09, 0.085)), (40, (0.115, 0.095, 0.1))]
GLUT_R = [(8, 0.05), (20, 0.065), (40, 0.08)]
TAIL_R = [(8, 0.05), (10, 0.03), (11, 0.0)]
# Crown–rump length in mm, to turn measured sizes into CRL units
CRL_MM = [(8, 16), (12, 55), (16, 116), (20, 165), (24, 210), (28, 250), (32, 285), (36, 320), (40, 360)]


def table3(pairs, g):
    xs = [p[0] for p in pairs]
    return np.array([np.interp(g, xs, [p[1][i] for p in pairs]) for i in range(3)])


def joints_at(g):
    a, b, t = (POSE_8, POSE_20, (g - 8) / 12) if g <= 20 else (POSE_20, POSE_40, (g - 20) / 20)
    t = min(max(t, 0.0), 1.0)
    return {k: (1 - t) * vec(*a[k]) + t * vec(*b[k]) for k in a}


def skeleton(g, overrides=None):
    j = joints_at(g)
    if overrides:
        j.update(overrides(dict(j)))
    return j


ARM_PARTS = ("upperArm", "forearm", "palm", "finger", "thumb")
LEG_PARTS = ("thigh", "shin", "foot", "heel", "toe")
HEAD_PARTS = ("cranium", "face", "cheek", "chin", "nose", "eye", "ear", "lip", "mouth", "nostril", "concha")


def groups_of(name):
    """Body parts a primitive belongs to; only primitives that share one are blended."""
    side = "L" if "L" in name[-2:] else ("R" if "R" in name[-2:] else "")
    if name.startswith(ARM_PARTS):
        return {"arm" + side} | ({"torso"} if name.startswith("upperArm") else set())
    if name.startswith(LEG_PARTS):
        return {"leg" + side} | ({"torso"} if name.startswith("thigh") else set())
    if name == "neck":
        return {"torso", "head"}
    if name == "shoulders":
        return {"torso", "armL", "armR"}
    if name.startswith(HEAD_PARTS):
        return {"head"}
    return {"torso"}


# --------------------------------------------------------------------------
# The sculpt
# --------------------------------------------------------------------------


def add_chain(add, name, pts, radii, k, per_span=3):
    """A smooth tube through pts (Catmull-Rom), made of short round cones (helix rim, antihelix)."""
    pts = [np.asarray(p, dtype=np.float64) for p in pts]
    ext = [2 * pts[0] - pts[1]] + pts + [2 * pts[-1] - pts[-2]]
    samples, rads = [], []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for t in np.linspace(0, 1, per_span, endpoint=False):
            samples.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                  + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
            rads.append(radii[i] + (radii[i + 1] - radii[i]) * t)
    samples.append(pts[-1])
    rads.append(radii[-1])
    add(name, Chain([RoundCone(samples[i], samples[i + 1], rads[i], rads[i + 1]) for i in range(len(samples) - 1)]), k)


def add_ears(add, sub, at, X, UP, FWD, R, g, feat):
    """Both ears: plate, helix rim, antihelix with its crus, concha, tragus, antitragus and
    lobule. On the Ballard maturity scale the pinna is flat and soft around 24 weeks, well
    curved by 32-34 and firm at term, so the rim and the antihelix stand out more with age.
    Returns the left ear's centre for the rig."""
    curl = smoothstep(18, 36, g)
    H = 0.31 * R * (0.45 + 0.55 * feat)      # half the ear's height
    a, b = math.radians(22) * (0.3 + 0.7 * feat), math.radians(15)   # early ears lie flatter
    anchor = None
    for sx, s in ((-1, "L"), (1, "R")):
        out = X * sx
        n_e = out * math.cos(a) + FWD * math.sin(a)        # faces out and a little forward
        back0 = -FWD * math.cos(a) + out * math.sin(a)     # so the back edge stands off the head
        up_e = UP * math.cos(b) + back0 * math.sin(b)      # and the top leans back
        back = back0 * math.cos(b) - UP * math.sin(b)
        frame = np.stack([back, up_e, n_e], axis=1)
        E = at(0.88 * sx, -0.15 - 0.3 * (1 - feat), -0.14)

        def ep(u, v, w):
            return E + back * u * H + up_e * v * H + n_e * w * H

        soft = 1 - smoothstep(10, 15, g)          # before ~14 weeks the ear is a low mound on the head
        detail = smoothstep(12, 18, g)            # the folds come in gradually; earlier a smooth rounded shell
        add("earRoot" + s, Ellipsoid(ep(-0.3, -0.05, 0.0), np.array([0.22, 0.72, 0.2]) * H, frame), (0.3 + 0.5 * soft) * H)
        add("earPlate" + s, Ellipsoid(ep(0.12, 0.0, 0.17), np.array([0.6, 1.0, 0.12 + 0.08 * (1 - detail)]) * H, frame), (0.25 + 0.6 * soft) * H)
        rim_w = 0.2 + 0.12 * curl
        rim = [(-0.12, 0.04, 0.2), (-0.28, 0.45, rim_w), (-0.08, 0.9, rim_w), (0.34, 0.9, rim_w),
               (0.64, 0.5, rim_w), (0.7, 0.0, 0.92 * rim_w), (0.52, -0.48, 0.65 * rim_w)]
        rr = (0.085 + 0.035 * curl) * H
        add_chain(add, "earHelix" + s, [ep(*p) for p in rim], [0.55 * rr, 0.85 * rr, rr, rr, rr, 0.95 * rr, 0.8 * rr], 0.1 * H)
        ah_w = (0.2 + 0.08 * curl) * (0.6 + 0.4 * detail)
        ar = (0.05 + 0.03 * curl) * H * (0.25 + 0.75 * detail)
        anti = [(0.12, -0.4, 0.85 * ah_w), (0.33, -0.04, ah_w), (0.34, 0.3, ah_w), (0.18, 0.62, 0.9 * ah_w)]
        add_chain(add, "earAntihelix" + s, [ep(*p) for p in anti], [0.9 * ar, ar, ar, 0.8 * ar], 0.1 * H)
        add_chain(add, "earCrus" + s, [ep(0.32, 0.28, ah_w), ep(0.12, 0.42, 0.95 * ah_w), ep(-0.06, 0.46, 0.9 * ah_w)],
                  [0.9 * ar, 0.8 * ar, 0.65 * ar], 0.1 * H)
        add("earLobule" + s, Ellipsoid(ep(0.2, -0.72, 0.12), np.array([0.3, 0.26, 0.11]) * H, frame), 0.2 * H)
        sub("earConcha" + s, Ellipsoid(ep(-0.02, -0.12, 0.42 + 0.1 * (1 - detail)), np.array([0.3, 0.36, 0.3]) * H * (0.8 + 0.2 * detail), frame),
            (0.08 + 0.06 * (1 - detail)) * H)
        add("earTragus" + s, Ellipsoid(ep(-0.36, -0.2, 0.24), np.array([0.11, 0.15, 0.1]) * H * (0.4 + 0.6 * detail), frame), 0.1 * H)
        add("earAntitragus" + s, Ellipsoid(ep(0.12, -0.5, 0.25), np.array([0.1, 0.08, 0.08]) * H * (0.4 + 0.6 * detail), frame), 0.1 * H)
        if s == "L":
            anchor = ep(0.1, 0.05, 0.4)
    return anchor


def add_face(add, sub, at, HR, R, feat, fat):
    """Cheeks, chin, closed eyes, a small button nose and lips, in head units (R).
    feat (0-1) is how formed the face is: early on the features are small and soft.
    Returns the eyelid lines (for darkening the lash line in the vertex colours)."""
    fs = 0.45 + 0.55 * feat
    soft = 1 + 0.6 * (1 - feat)           # blend radii: features melt into the face early on
    cheek = (0.22 + 0.13 * fat) * R * fs
    for sx, s in ((-1, "L"), (1, "R")):
        add("cheek" + s, Ellipsoid(at(0.4 * sx, -0.52, 0.6), np.full(3, cheek)), 0.15 * R)
    add("chin", Ellipsoid(at(0, -0.87, 0.66), np.array([0.17, 0.15, 0.17]) * R, HR), 0.12 * R)

    # Closed eyes: the lid bulges over the eyeball, a soft upper fold, and the lash line
    # curving gently down at the outer corner
    lid_lines = []
    for sx, s in ((-1, "L"), (1, "R")):
        add("eye" + s, Ellipsoid(at(0.32 * sx, -0.21, 0.81 + 0.03 * fs), np.array([0.17, 0.12, 0.1 * fs]) * R, HR), 0.07 * R * soft)
        add("eyeFold" + s, Ellipsoid(at(0.31 * sx, -0.165, 0.86 + 0.02 * fs), np.array([0.15, 0.055, 0.05 * fs]) * R, HR), 0.04 * R * soft)
        line = [at(0.18 * sx, -0.238, 0.905), at(0.27 * sx, -0.252, 0.925), at(0.37 * sx, -0.247, 0.905), at(0.46 * sx, -0.222, 0.85)]
        lash = Chain([RoundCone(line[i], line[i + 1], 0.011 * R, 0.011 * R) for i in range(3)])
        if feat > 0.3:
            sub("eyeLine" + s, lash, 0.012 * R * soft)
            sub("eyeCorner" + s, Ellipsoid(at(0.165 * sx, -0.24, 0.88), np.full(3, 0.022 * R * feat)), 0.015 * R)
        lid_lines.append(lash)

    # Nose: low flat bridge, round tip, soft alae, small nostrils
    add("noseBridge", RoundCone(at(0, -0.2, 0.86), at(0, -0.37, 0.93), 0.055 * R * fs, 0.075 * R * fs), 0.06 * R * soft)
    add("noseTip", Ellipsoid(at(0, -0.41, 0.94 + 0.04 * fs), np.array([0.08, 0.07, 0.065]) * R * fs, HR), 0.05 * R * soft)
    for sx, s in ((-1, "L"), (1, "R")):
        add("noseAla" + s, Ellipsoid(at(0.08 * sx, -0.455, 0.9 + 0.02 * fs), np.array([0.055, 0.045, 0.05]) * R * fs, HR), 0.05 * R * soft)
    for sx, s in ((-1, "L"), (1, "R")):
        sub("nostril" + s, Ellipsoid(at(0.048 * sx, -0.5, 0.94 + 0.02 * fs), np.array([0.022, 0.012, 0.02]) * R * fs * feat, HR), 0.01 * R)

    # Mouth: philtrum, upper lip with a cupid's bow, full lower lip, a curved mouth line
    sub("lipPhiltrum", RoundCone(at(0, -0.5, 0.955), at(0, -0.6, 0.94), 0.018 * R * feat, 0.022 * R * feat), 0.03 * R)
    add("lipUpperC", Ellipsoid(at(0, -0.632, 0.875), np.array([0.05, 0.042, 0.05]) * R * fs, HR), 0.05 * R * soft)
    for sx, s in ((-1, "L"), (1, "R")):
        add("lipUpper" + s, Ellipsoid(at(0.072 * sx, -0.642, 0.855), np.array([0.085, 0.042, 0.05]) * R * fs, HR), 0.05 * R * soft)
    add("lipLower", Ellipsoid(at(0, -0.73, 0.845), np.array([0.11, 0.048, 0.06]) * R * fs, HR), 0.05 * R * soft)
    crease = max(0.008 * R, 0.0014) * feat
    mouth = [at(-0.145, -0.692, 0.815), at(-0.08, -0.686, 0.865), at(0, -0.684, 0.885), at(0.08, -0.686, 0.865), at(0.145, -0.692, 0.815)]
    sub("mouth", Chain([RoundCone(mouth[i], mouth[i + 1], crease, crease) for i in range(4)]), 0.012 * R)
    sub("chinSulcus", Ellipsoid(at(0, -0.795, 0.87), np.array([0.1, 0.02, 0.03]) * R * feat, HR), 0.03 * R)
    return lid_lines


def genital_frame(ops, j):
    """Pubic point on the skin, between the thighs, plus a frame: X across the body,
    U up the belly, N out of the skin."""
    P = j["pelvis"]
    d = unit(vec(0, -0.5, 1.0))
    t = 0.0
    while eval_sdf(ops, (P + d * t)[None])[0] < 0 and t < 0.3:
        t += 0.002
    lo, hi = t - 0.002, t
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if eval_sdf(ops, (P + d * mid)[None])[0] < 0:
            lo = mid
        else:
            hi = mid
    O = P + d * hi
    N = unit(gradient(ops, O[None])[0])
    U = unit(vec(0, 1, 0) - N * N[1])
    return O, unit(np.cross(U, N)), U, N


def add_genitals(add, sub, gframe, g, sex, grow, fat):
    """External genitals, sized from fetal measurements and staged like the Ballard scale.
    "X": genital tubercle and labioscrotal swellings, alike in girls and boys until ~12 weeks.
    "M": penis (length 0.81 * GA - 8.8 mm) over a scrotum that fills out once the testes
         descend (28-33 weeks).
    "F": labia majora, labia minora and clitoris; the clitoris and minora are prominent in
         mid-pregnancy and the majora close over them by term.
    grow (0-1) scales the shapes so they can be grown onto the mesh in steps."""
    O, Xg, U, N = gframe
    frame = np.stack([Xg, U, N], axis=1)
    s = grow

    def gp(a, b, c):
        return O + Xg * a + U * b + N * c

    def along(d):
        e2 = unit(np.cross(d, Xg))
        return np.stack([unit(np.cross(e2, d)), e2, d], axis=1)

    if sex == "X":
        L = table([(8, 0.045), (12, 0.036)], g)
        d = unit(N * math.cos(0.35) - U * math.sin(0.35))
        add("genTubercle", RoundCone(O - N * 0.25 * L, O + d * 0.7 * L * s, 0.28 * L * s, 0.24 * L * s), 0.3 * L)
        for sx, side in ((-1, "L"), (1, "R")):
            add("genSwelling" + side, Ellipsoid(gp(0.45 * L * sx, -0.6 * L, -0.08 * L), np.array([0.36, 0.55, 0.2]) * L * s, frame), 0.4 * L)
    elif sex == "M":
        Lp = max(0.8068 * g - 8.84, 3.0) / table(CRL_MM, g)
        desc = smoothstep(26, 33, g)
        hw = table([(16, 0.017), (24, 0.022), (32, 0.03), (40, 0.034)], g)   # scrotum half-width
        C = gp(0, -0.95 * hw - 0.15 * Lp, 0.0)
        for sx, side in ((-1, "L"), (1, "R")):
            rad = np.array([0.64, 0.8 + 0.22 * desc, 0.4 + 0.33 * desc]) * hw * s
            add("genScrotum" + side, Ellipsoid(C + Xg * 0.4 * hw * sx, rad, frame), 0.6 * hw)
        sub("genRaphe", RoundCone(C + U * 0.7 * hw + N * 0.5 * hw * s, C - U * 0.85 * hw + N * 0.45 * hw * s,
                                  0.035 * hw, 0.015 * hw), 0.1 * hw, group="torso")
        d = unit(N * math.cos(0.7) - U * math.sin(0.7))
        r = 0.2 * Lp
        base = O - N * 0.5 * r
        tip = base + d * (0.25 + 0.62 * s) * Lp
        add("genShaft", RoundCone(base, tip, r * (0.6 + 0.4 * s), r * (0.55 + 0.4 * s)), 0.5 * r)
        add("genGlans", Ellipsoid(tip + d * 0.08 * Lp * s, np.array([1.1, 1.1, 1.35]) * r * s, along(d)), 0.3 * r)
    elif sex == "F":
        Lv = table([(16, 0.05), (24, 0.055), (40, 0.058)], g)          # mons to fourchette
        maj = smoothstep(26, 40, g)
        clit = table([(16, 1.0), (28, 0.85), (40, 0.5)], g)
        mino = table([(16, 0.75), (24, 1.0), (32, 0.8), (40, 0.4)], g)
        V = gp(0, -0.3 * Lv, 0.0)
        for sx, side in ((-1, "L"), (1, "R")):
            rad = np.array([0.17 + 0.07 * maj, 0.5, 0.08 + 0.12 * maj]) * Lv * s
            add("genMajora" + side, Ellipsoid(V + Xg * (0.19 + 0.04 * maj) * Lv * sx, rad, frame), 0.14 * Lv)
        add("genMons", Ellipsoid(gp(0, 0.3 * Lv, -0.1 * Lv), np.array([0.45, 0.32, 0.1 + 0.1 * fat]) * Lv * s, frame), 0.1 * Lv)
        sub("genCleft", RoundCone(V + U * 0.4 * Lv + N * 0.07 * Lv * s, V - U * 0.48 * Lv + N * 0.05 * Lv * s,
                                  0.03 * Lv, 0.03 * Lv), 0.05 * Lv, group="torso")
        for sx, side in ((-1, "L"), (1, "R")):
            rad = np.array([0.035, 0.3, 0.08 * mino]) * Lv * s
            add("genMinora" + side, Ellipsoid(V + Xg * 0.035 * Lv * sx + N * 0.03 * Lv - U * 0.04 * Lv, rad, frame), 0.03 * Lv)
        add("genClitoris", Ellipsoid(V + U * 0.36 * Lv + N * 0.06 * Lv, np.full(3, 0.1 * Lv * clit * s)), 0.05 * Lv)


def build(g, overrides=None, sex=None, grow=1.0):
    """The sculpt for gestational week g. sex is None (no external genitals), "X" (the
    indifferent genital tubercle, same for everyone before ~12 weeks), "F" or "M";
    grow scales the genitals from 0 to full size, used to grow them in steps."""
    j = skeleton(g, overrides)
    fat = smoothstep(22, 40, g)
    feat = smoothstep(8.5, 14, g)          # how formed the face, hands and feet are
    R = table(HEAD_R, g)
    pitch = table(PITCH, g)
    ops = Sculpt()

    def group_for(name):
        if name in ("neck", "shoulders"):
            return "torso"
        gs = groups_of(name)
        for grp in ("armL", "armR", "legL", "legR", "head"):
            if grp in gs:
                return grp
        return "torso"

    def add(name, shape, k):
        ops.append(Op(name, shape, "add", k, group_for(name)))

    def sub(name, shape, k, group="head"):
        ops.append(Op(name, shape, "sub", k, group))

    # --- torso --------------------------------------------------------------
    add("chest", Ellipsoid(j["chest"], table3(CHEST, g) + 0.014 * fat), 0.07)
    add("belly", Ellipsoid(j["belly"], table3(BELLY, g) + 0.018 * fat), 0.07)
    add("pelvis", Ellipsoid(j["pelvis"], table3(PELVIS, g) + 0.012 * fat), 0.07)
    br = table(BACK_R, g)
    add("backUpper", RoundCone(j["backTop"], j["backMid"], br, br * 1.05), 0.06)
    add("backLower", RoundCone(j["backMid"], j["backLow"], br * 1.05, br), 0.06)
    # Buttocks: two soft, fairly flat masses tucked under the pelvis, parted by the gluteal cleft
    gr = 0.8 * table(GLUT_R, g) + 0.008 * fat
    glut_shift = vec(0, 0.012, 0.008)
    for s in "LR":
        add("glut" + s, Ellipsoid(j["glut" + s] + glut_shift, np.array([0.95, 0.88, 0.72]) * gr), 0.042 + 0.012 * (1 - feat))
    tr = table(TAIL_R, g)
    add("tail", RoundCone(j["tailA"], j["tailB"], tr, tr * 0.45), 0.04)
    cleft = 0.0055 * smoothstep(11, 18, g) * (1 + 0.3 * fat)
    if cleft > 0:
        # follow the valley between the buttocks, from the sacrum round to the perineum
        torso = [op for op in ops if op.group == "torso"]
        gc = 0.5 * (j["glutL"] + j["glutR"]) + glut_shift
        pts = []
        for th in np.radians(np.linspace(28, -75, 9)):
            d = vec(0, math.sin(th), -math.cos(th))
            t = 0.0
            while eval_group(torso, (gc + d * t)[None])[0] < 0 and t < 0.3:
                t += 0.002
            pts.append(gc + d * (t - 0.45 * cleft))
        taper = (0.15, 0.6, 0.95, 1.0, 1.0, 1.0, 0.9, 0.6, 0.3)
        cones = [RoundCone(pts[i], pts[i + 1], cleft * taper[i], cleft * taper[i + 1]) for i in range(len(pts) - 1)]
        sub("glutCleft", Chain(cones), 0.014, group="torso")
    navel_dir = unit(vec(0, -0.25, 1))
    navel = j["belly"] + navel_dir * table3(BELLY, g)[2] * 0.93
    add("navel", RoundCone(navel - navel_dir * 0.01, navel + navel_dir * 0.018, 0.021, 0.019), 0.012)

    # --- arms and hands -----------------------------------------------------
    sh_r, el_r, wr_r = table(ARM_SH, g), table(ARM_EL, g), table(ARM_WR, g)
    add("shoulders", RoundCone(j["shL"], j["shR"], sh_r * 1.1, sh_r * 1.1), 0.05)
    hand_len = table(HAND_L, g)
    finger_scale = 0.35 + 0.65 * smoothstep(8, 12, g)
    finger_k = 0.02 + (0.0035 - 0.02) * smoothstep(8, 11, g)
    for s in "LR":
        add("upperArm" + s, RoundCone(j["sh" + s], j["el" + s], sh_r, el_r), 0.035)
        add("forearm" + s, RoundCone(j["el" + s], j["wr" + s], el_r, wr_r), 0.02)
        W = j["wr" + s]
        d = unit(j["tip" + s] - W)
        n = unit(j["palm" + s] - d * np.dot(j["palm" + s], d))   # palm normal
        side = np.cross(n, d)
        L = hand_len
        add("palm" + s, Ellipsoid(W + d * 0.27 * L, (0.22 * L, 0.085 * L, 0.27 * L), np.stack([side, n, d], axis=1)), 0.014)
        # The thumb is on the radial side: cross(n, d) for the right hand, mirrored for the left
        # (check: right hand hanging, palm forward -> n = +z, d = -y, cross = +x = lateral)
        radial = side if s == "R" else -side
        lengths = (0.42, 0.47, 0.44, 0.36)          # index, middle, ring, little
        offsets = (0.16, 0.055, -0.055, -0.16)      # along the radial direction
        for i in range(4):
            base = W + d * 0.5 * L + radial * offsets[i] * L
            fl = lengths[i] * L * finger_scale
            c1, c2 = j["curl" + s][0], j["curl" + s][1]
            d1 = d * math.cos(c1) + n * math.sin(c1)
            d2 = d * math.cos(c1 + c2) + n * math.sin(c1 + c2)
            mid = base + d1 * fl * 0.55
            tip = mid + d2 * fl * 0.45
            fr = 0.06 * L
            add(f"finger{i}{s}a", RoundCone(base, mid, fr, fr * 0.92), finger_k)
            add(f"finger{i}{s}b", RoundCone(mid, tip, fr * 0.92, fr * 0.82), finger_k)
        thumb_side = radial
        t0 = W + d * 0.12 * L + thumb_side * 0.2 * L + n * 0.03 * L
        t1 = unit(d * 0.65 + thumb_side * 0.35 + n * 0.45)
        t2 = unit(d * 0.4 + thumb_side * 0.05 + n * 0.75)
        tm = t0 + t1 * 0.24 * L * finger_scale
        tt = tm + t2 * 0.2 * L * finger_scale
        add("thumb" + s + "a", RoundCone(t0, tm, 0.078 * L, 0.07 * L), finger_k)
        add("thumb" + s + "b", RoundCone(tm, tt, 0.07 * L, 0.062 * L), finger_k)

    # --- legs and feet ------------------------------------------------------
    hip_r, kn_r, an_r = table(LEG_HIP, g), table(LEG_KN, g), table(LEG_AN, g)
    foot_len = table(FOOT_L, g)
    for s in "LR":
        add("thigh" + s, RoundCone(j["hip" + s], j["kn" + s], hip_r, kn_r), 0.045)
        add("shin" + s, RoundCone(j["kn" + s], j["an" + s], kn_r, an_r), 0.02)
        A = j["an" + s]
        f = unit(j["toe" + s] - A)
        up = foot_up(j, s, f)
        side = np.cross(up, f)
        Lf = foot_len
        F = np.stack([side, up, f], axis=1)
        # big toe sits on the medial side (towards +x for the left foot)
        medial = side if (side[0] > 0) == (s == "L") else -side
        add("foot" + s, Ellipsoid(A + f * 0.27 * Lf - up * 0.2 * Lf, np.array([0.18, 0.12, 0.33]) * Lf, F), 0.025)
        # forefoot: wide and thin, sloping down to the toes
        add("footBall" + s, Ellipsoid(A + f * 0.48 * Lf - up * 0.25 * Lf - medial * 0.02 * Lf, np.array([0.21, 0.085, 0.15]) * Lf, F), 0.02)
        # the heel carries the foot's frame, so it turns with the foot when the leg moves
        add("heel" + s, Ellipsoid(A - f * 0.02 * Lf - up * 0.24 * Lf, np.array([0.1, 0.095, 0.105]) * Lf, F), 0.025)
        # Toes: short and plump, side by side on the sole, the big toe wider with a small gap
        # after it, the others getting shorter along an arc and curling slightly down
        toe_len = (0.19, 0.17, 0.155, 0.14, 0.125)
        toe_rad = (0.058, 0.04, 0.037, 0.034, 0.031)
        toe_off = (0.118, 0.01, -0.067, -0.138, -0.203)
        toe_back = (0.0, 0.015, 0.04, 0.07, 0.105)
        splay = (0.0, 0.02, 0.05, 0.09, 0.13)
        sole = 0.33 * Lf
        for i in range(5):
            r = toe_rad[i] * Lf * (0.75 + 0.25 * feat)
            base = A + f * (0.52 - toe_back[i]) * Lf + up * (r - sole) + medial * toe_off[i] * Lf
            d = unit(f - up * 0.08 - medial * splay[i])
            tip = base + d * toe_len[i] * Lf * (0.5 + 0.5 * feat)
            add(f"toe{i}{s}", RoundCone(base, tip, r, 0.92 * r), 0.006 + 0.012 * (1 - feat))

    # --- head and face ------------------------------------------------------
    S = j["skull"]
    X = vec(1, 0, 0)
    UP = vec(0, math.cos(pitch), math.sin(pitch))
    FWD = vec(0, -math.sin(pitch), math.cos(pitch))
    HR = np.stack([X, UP, FWD], axis=1)

    def at(lx, ly, lz):
        return S + X * lx * R + UP * ly * R + FWD * lz * R

    head_top = at(0, -0.62, -0.12)
    ops.junctions = {
        "head": (head_top, 0.09, 0.06),
        "armL": (j["shL"], 0.07, 0.04), "armR": (j["shR"], 0.07, 0.04),
        "legL": (j["hipL"], 0.09, 0.05), "legR": (j["hipR"], 0.09, 0.05),
    }
    add("neck", RoundCone(j["neck"], head_top, table(NECK_R, g), table(NECK_R, g) * 0.95), 0.06)
    add("cranium", Ellipsoid(at(0, 0.04, -0.06), (0.9 * R, 0.97 * R, 1.06 * R), HR), 0.05)
    add("face", Ellipsoid(at(0, -0.42, 0.4), (0.66 * R, 0.52 * R, 0.52 * R), HR), 0.22 * R)
    lid_lines = add_face(add, sub, at, HR, R, feat, fat)
    ear_anchor = add_ears(add, sub, at, X, UP, FWD, R, g, feat)
    anchors = {
        "head": at(-0.25, 0.45, 0.6),
        "ear": ear_anchor,
        "heart": j["chest"] + vec(-0.035, 0.02, table3(CHEST, g)[2] * 0.6),
        "hand": j["wrL"] + unit(j["tipL"] - j["wrL"]) * 0.3 * hand_len,
        "knee": j["knL"],
        "navel": navel + navel_dir * 0.018,
        "footL": j["anL"] + unit(j["toeL"] - j["anL"]) * 0.45 * foot_len,
        "footR": j["anR"] + unit(j["toeR"] - j["anR"]) * 0.45 * foot_len,
    }
    gframe = genital_frame(ops, j)
    anchors["groin"] = gframe[0]
    if sex:
        add_genitals(add, sub, gframe, g, sex, grow, fat)
    ops.gframe = gframe
    ops.lid_lines = lid_lines if feat > 0.5 else []
    return ops, anchors


# --------------------------------------------------------------------------
# Grid evaluation + marching cubes
# --------------------------------------------------------------------------


def _box_slices(a, b, lo, h, dims):
    out = []
    for i in range(3):
        i0 = max(int(math.floor((a[i] - lo[i]) / h)), 0)
        i1 = min(int(math.ceil((b[i] - lo[i]) / h)) + 1, dims[i])
        out.append(slice(i0, i1))
    return out


def sculpt_grid(sculpt, h):
    live = [op for op in sculpt if op.shape.size() > 1e-4]
    boxes = [op.shape.aabb() for op in live if op.mode == "add"]
    lo = np.min([b[0] for b in boxes], axis=0) - 0.03
    hi = np.max([b[1] for b in boxes], axis=0) + 0.03
    dims = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[i] + h * np.arange(dims[i]) for i in range(3)]
    total = np.full(dims, 1.0, dtype=np.float32)
    for group in Sculpt.GROUPS:
        ops = [op for op in live if op.group == group]
        adds = [op.shape.aabb() for op in ops if op.mode == "add"]
        if not adds:
            continue
        ga = np.min([b[0] for b in adds], axis=0) - 0.04
        gb = np.max([b[1] for b in adds], axis=0) + 0.04
        gs = _box_slices(ga, gb, lo, h, dims)
        sub_lo = np.array([axes[i][gs[i].start] for i in range(3)])
        sub_dims = [gs[i].stop - gs[i].start for i in range(3)]
        sub_axes = [axes[i][gs[i]] for i in range(3)]
        field = np.full(sub_dims, 1.0)
        for op in ops:
            a, b = op.shape.aabb()
            sl = _box_slices(a - op.k - 2 * h, b + op.k + 2 * h, sub_lo, h, sub_dims)
            gx, gy, gz = np.meshgrid(sub_axes[0][sl[0]], sub_axes[1][sl[1]], sub_axes[2][sl[2]], indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
            sd = op.shape.sdf(pts).reshape(gx.shape)
            blk = field[sl[0], sl[1], sl[2]]
            field[sl[0], sl[1], sl[2]] = smin(blk, sd, op.k) if op.mode == "add" else ssub(blk, sd, op.k)
        region = total[gs[0], gs[1], gs[2]].astype(np.float64)
        if group == "torso":
            region = np.minimum(region, field)
        else:
            gx, gy, gz = np.meshgrid(*sub_axes, indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
            k = sculpt.junction_k(group, pts).reshape(field.shape)
            region = smin(region, field, k)
        total[gs[0], gs[1], gs[2]] = region.astype(np.float32)
    return total, lo


def marching_cubes(d, lo, h):
    from skimage import measure
    verts, faces, _, _ = measure.marching_cubes(d, level=0.0, spacing=(h, h, h))
    return verts + lo, faces


# --------------------------------------------------------------------------
# Morphing helpers
# --------------------------------------------------------------------------


def gradient(ops, p, eps=0.0012):
    g = np.zeros_like(p)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        g[:, i] = (eval_sdf(ops, p + e) - eval_sdf(ops, p - e)) / (2 * eps)
    return g


def project(ops, p, iterations=6, max_step=0.02):
    p = p.copy()
    for _ in range(iterations):
        d = eval_sdf(ops, p)
        g = gradient(ops, p)
        gl2 = np.maximum(np.einsum("ij,ij->i", g, g), 1e-8)
        step = -(d / gl2)[:, None] * g
        ln = np.linalg.norm(step, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(ln, 1e-12))
        p += step * scale[:, None] * 0.9
    return p


def carry(base_ops, target_ops, p, sigma=0.008):
    """Move vertices from one sculpt onto another by blending primitive frames."""
    target = {op.name: op for op in target_ops}
    pairs = [(op, target[op.name]) for op in base_ops
             if op.mode == "add" and op.shape.size() > 1e-3 and op.name in target and target[op.name].shape.size() > 1e-4]
    dist = np.stack([np.maximum(a.shape.sdf(p), 0.0) for a, _ in pairs], axis=1)
    groups = [groups_of(a.name) for a, _ in pairs]
    compatible = np.array([[bool(gi & gj) for gj in groups] for gi in groups])
    owner = dist.argmin(axis=1)
    w = np.exp(-(dist - dist.min(axis=1, keepdims=True)) / sigma) * compatible[owner]
    w /= w.sum(axis=1, keepdims=True)
    out = np.zeros_like(p)
    for i, (a, b) in enumerate(pairs):
        out += w[:, i:i + 1] * a.shape.map_to(b.shape, p)
    return out


def laplacian_smooth_delta(delta, edges, n, iterations=2, lam=0.5):
    import scipy.sparse as sp
    i, j = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix((np.ones(len(i) * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iterations):
        avg = (A @ delta) / deg[:, None]
        delta = delta + lam * (avg - delta)
    return delta


def relax(ops, p, edges, iterations=4, lam=0.5):
    """Even out vertex spacing along the surface, then snap back onto it."""
    import scipy.sparse as sp
    n = len(p)
    i, j = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix((np.ones(len(i) * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
    for _ in range(iterations):
        lap = (A @ p) / deg[:, None] - p
        g = gradient(ops, p)
        nrm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        lap -= np.einsum("ij,ij->i", lap, nrm)[:, None] * nrm
        p = project(ops, p + lam * lap, iterations=2, max_step=0.01)
    return p


def local_thickness(ops, p, nrm, max_t=0.3, iterations=48):
    """Distance through the body along the inward normal (sphere tracing inside the SDF).
    Thin parts (ears, fingers, toes, nose) come out small; the trunk and head large."""
    t = np.full(len(p), 0.002)
    done = np.zeros(len(p), dtype=bool)
    for _ in range(iterations):
        d = eval_sdf(ops, p - nrm * t[:, None])
        done |= d > 0
        t = np.where(done, t, np.minimum(t + np.maximum(-d, 0.002), max_t))
    return t


def skin_thickness(ops, p, nrm, A, deg, tilt=0.5, smooth=10):
    """Thickness for the translucency: measured on the body without its small creases (a ray
    from inside the lash line or the mouth would cross only the crease and glow red), averaged
    over a cone of rays around the inward normal and smoothed over the surface, so the glow
    follows the real shape (ears, fingers, toes) instead of the mesh's noise."""
    solid = Sculpt()
    solid.junctions = ops.junctions
    solid.extend(op for op in ops if op.mode == "add")
    a = np.where(np.abs(nrm[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    u = np.cross(nrm, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(nrm, u)
    total = local_thickness(solid, p, nrm)
    for k in range(6):
        phi = k * math.pi / 3
        ray = nrm + tilt * (math.cos(phi) * u + math.sin(phi) * v)
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        total = total + local_thickness(solid, p, ray)
    t = total / 7
    for _ in range(smooth):
        t = 0.5 * t + 0.5 * (A @ t) / deg
    return t


def ambient_occlusion(ops, p, nrm, steps=5, dist=0.012):
    occ = np.zeros(len(p))
    weight = 1.0
    for i in range(1, steps + 1):
        hh = dist * i
        occ += (hh - eval_sdf(ops, p + nrm * hh)) * weight
        weight *= 0.6
    return np.clip(1.0 - 4.5 * occ / dist / steps, 0.0, 1.0)


def local_ops(ops, centre, radius):
    """Only the primitives near centre, for fast evaluation in a small region."""
    near = Sculpt()
    near.junctions = ops.junctions
    for op in ops:
        a, b = op.shape.aabb()
        if np.all(a - op.k < centre + radius) and np.all(b + op.k > centre - radius):
            near.append(op)
    return near


def march_along(ops, p, n, max_t=0.1, iterations=48):
    """Signed distance along n from p to the surface: outwards when p is inside, inwards when outside."""
    d0 = eval_sdf(ops, p)
    sign = np.where(d0 < 0, 1.0, -1.0)
    t = np.zeros(len(p))
    active = np.abs(d0) > 1e-5
    for _ in range(iterations):
        if not active.any():
            break
        d = eval_sdf(ops, p[active] + n[active] * t[active, None])
        step = np.maximum(np.abs(d), 2e-5)
        crossed = np.sign(d) != -sign[active]
        t_act = t[active] + np.where(crossed, 0.0, sign[active] * step)
        t[active] = np.clip(t_act, -max_t, max_t)
        idx = np.flatnonzero(active)
        active[idx[crossed | (np.abs(t_act) >= max_t)]] = False
    return t


def adjacency(edges, n):
    import scipy.sparse as sp
    i, j = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix((np.ones(len(i) * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A, np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)


def relax_masked(ops, p, A, deg, idx, weight, iterations=8, lam=0.5):
    """Even out vertex spacing along the surface for the vertices idx only (weighted)."""
    rows = A[idx]
    for _ in range(iterations):
        q = p[idx]
        lap = (rows @ p) / deg[idx, None] - q
        g = gradient(ops, q)
        nrm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        lap -= np.einsum("ij,ij->i", lap, nrm)[:, None] * nrm
        p[idx] = project(ops, q + lam * weight[:, None] * lap, iterations=2, max_step=0.006)
    return p


def grow_genitals(g, sex, start, idx, weight, A, deg, steps=(0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)):
    """Grow the genital shapes for week g out of the neutral skin in a few steps: each step
    pushes the region's vertices along their normals onto the new surface and relaxes them,
    so vertices spread up the new shapes instead of piling up at their tips."""
    neutral = build(g)[0]
    centre = neutral.gframe[0]
    prev = local_ops(neutral, centre, 0.16)
    p = start.copy()
    for s in steps:
        ops_s = local_ops(build(g, sex=sex, grow=s)[0], centre, 0.16)
        q = p[idx]
        nrm = gradient(prev, q)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        p[idx] = q + nrm * march_along(ops_s, q, nrm)[:, None]
        p = relax_masked(ops_s, p, A, deg, idx, weight, iterations=10)
        prev = ops_s
    p[idx] = project(prev, p[idx], iterations=3, max_step=0.004)
    return p


def vertex_normals(p, tris):
    fn = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
# Blender
# --------------------------------------------------------------------------


def to_blender(p):
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def from_blender(b):
    return np.stack([b[:, 0], b[:, 2], -b[:, 1]], axis=1)


def apply_modifiers(obj):
    import bpy
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    new_mesh = bpy.data.meshes.new_from_object(evaluated)
    old = obj.data
    obj.modifiers.clear()
    obj.data = new_mesh
    bpy.data.meshes.remove(old)


def _decimate(obj, ratio, weights=None):
    """Collapse-decimate; with weights, only vertices of weight 1 may collapse."""
    if weights is not None:
        group = obj.vertex_groups.new(name="Decimate")
        group.add([int(i) for i in np.flatnonzero(weights)], 1.0, "REPLACE")
    dec = obj.modifiers.new("Decimate", "DECIMATE")
    dec.ratio = min(1.0, ratio)
    dec.use_collapse_triangulate = True
    if weights is not None:
        dec.vertex_group = "Decimate"
    apply_modifiers(obj)
    obj.vertex_groups.clear()


def _face_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def blender_mesh(verts, faces, target_tris, details=()):
    """Mesh from marching cubes: weld, smooth lightly and decimate to target_tris.
    details: (inside(points) -> bool mask, keep) regions that are decimated less, so small
    anatomy (ears, the groin where the genital shapes grow) keeps enough vertices."""
    import bpy
    import bmesh  # only importable once bpy is loaded
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("Fetus")
    mesh.from_pydata(to_blender(verts).tolist(), [], faces.tolist())
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Fetus", mesh)
    bpy.context.scene.collection.objects.link(obj)
    smooth = obj.modifiers.new("Smooth", "SMOOTH")
    smooth.factor = 0.5
    smooth.iterations = 2
    apply_modifiers(obj)
    if not details:
        _decimate(obj, target_tris / _face_count(obj))
    else:
        def masks():
            co, tri, _ = read_mesh(obj)
            return [inside(co) for inside, _ in details], tri
        ms, tri = masks()
        protected = np.any(ms, axis=0)
        detail_tris = int(np.all(protected[tri], axis=1).sum())
        total = len(tri)
        # 1. everything outside the detail regions, down to the body budget
        _decimate(obj, (target_tris + detail_tris) / total, ~protected)
        # 2. each detail region on its own, keeping the given share of its triangles
        for i, (_, keep) in enumerate(details):
            if keep >= 1.0:
                continue
            ms, tri = masks()
            region_tris = int(np.all(ms[i][tri], axis=1).sum())
            total = len(tri)
            _decimate(obj, (total - (1 - keep) * region_tris) / total, ms[i])
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def read_mesh(obj):
    me = obj.data
    n = len(me.vertices)
    co = np.zeros(n * 3)
    me.vertices.foreach_get("co", co)
    me.calc_loop_triangles()
    tri = np.zeros(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    nr = np.zeros(n * 3)
    try:
        me.vertex_normals.foreach_get("vector", nr)
    except AttributeError:
        me.vertices.foreach_get("normal", nr)
    return from_blender(co.reshape(-1, 3)), tri.reshape(-1, 3), from_blender(nr.reshape(-1, 3))


def write_positions(obj, p):
    obj.data.vertices.foreach_set("co", to_blender(p).ravel())
    obj.data.update()


def add_shape_key(obj, name, p):
    sk = obj.shape_key_add(name=name, from_mix=False)
    sk.data.foreach_set("co", to_blender(p).ravel())
    sk.value = 0.0


def set_colors(obj, rgb, alpha):
    """Vertex colour: RGB = skin shading detail, A = normalised local thickness (for translucency)."""
    me = obj.data
    attr = me.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
    attr.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = attr


# --------------------------------------------------------------------------
# GLB post-processing (after gltfpack)
# --------------------------------------------------------------------------

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
N_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path):
    import struct
    with open(path, "rb") as fh:
        data = fh.read()
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    off = 20 + json_len
    bin_len = struct.unpack_from("<I", data, off)[0]
    return gltf, data[off + 8:off + 8 + bin_len]


def accessor_array(gltf, binary, index):
    acc = gltf["accessors"][index]
    dtype = np.dtype(COMPONENT[acc["componentType"]])
    nc = N_COMPONENTS[acc["type"]]
    count = acc["count"]
    el = dtype.itemsize * nc
    if "bufferView" not in acc:
        return np.zeros((count, nc), dtype)
    bv = gltf["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", el)
    raw = np.frombuffer(binary, np.uint8, count=(count - 1) * stride + el, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, el), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(count, nc)


class GlbWriter:
    def __init__(self):
        self.chunks, self.views, self.accessors, self.size = [], [], [], 0

    def view(self, data, stride=None, target=None):
        pad = -self.size % 4
        if pad:
            self.chunks.append(b"\0" * pad)
            self.size += pad
        bv = {"buffer": 0, "byteOffset": self.size, "byteLength": len(data)}
        if stride:
            bv["byteStride"] = stride
        if target:
            bv["target"] = target
        self.chunks.append(data)
        self.size += len(data)
        self.views.append(bv)
        return len(self.views) - 1

    def dense(self, arr, meta, indices=False):
        """Vertex attributes padded to 4-byte elements (glTF alignment rules)."""
        n, nc = arr.shape
        el = arr.dtype.itemsize * nc
        if indices:
            bv = self.view(arr.tobytes(), target=34963)
        else:
            stride = (el + 3) // 4 * 4
            rows = np.zeros((n, stride), np.uint8)
            rows[:, :el] = arr.view(np.uint8).reshape(n, el)
            bv = self.view(rows.tobytes(), stride=stride if stride != el else None, target=34962)
        self.accessors.append(dict(meta, bufferView=bv))
        return len(self.accessors) - 1

    def sparse(self, arr, meta, where, index_view):
        acc = dict(meta)
        acc.pop("bufferView", None)
        acc["sparse"] = {"count": int(len(where)), "indices": dict(index_view),
                         "values": {"bufferView": self.view(np.ascontiguousarray(arr[where]).tobytes())}}
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def save(self, gltf, path):
        import struct
        binary = b"".join(self.chunks)
        binary += b"\0" * (-len(binary) % 4)
        gltf["buffers"] = [{"byteLength": len(binary)}]
        gltf["bufferViews"] = self.views
        gltf["accessors"] = self.accessors
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(binary)))
            fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def finalize_glb(src, dst, basis, tris, extra_targets, normal_fix=None):
    """Repack a gltfpack-ed GLB: morph targets that only move part of the body (kick, wave)
    become sparse, and extra_targets [(name, neutral_positions, delta, normals)] are added as
    sparse targets; their normal deltas take the shaped surface's normals relative to the
    neutral mesh normals. normal_fix {target name: (normals, weight)} blends better normals
    into existing targets where weight > 0. basis, tris and all per-vertex arrays are in
    build order; the GLB's vertices are matched to them by position."""
    from scipy.spatial import cKDTree
    gltf, binary = read_glb(src)
    mesh = gltf["meshes"][0]
    prim = mesh["primitives"][0]
    node = next(n for n in gltf["nodes"] if n.get("mesh") == 0)
    scale = np.array(node.get("scale", [1, 1, 1]))
    shift = np.array(node.get("translation", [0, 0, 0]))
    meta = [{k: v for k, v in a.items() if k not in ("bufferView", "byteOffset")} for a in gltf["accessors"]]

    positions = accessor_array(gltf, binary, prim["attributes"]["POSITION"]).astype(np.float64) * scale + shift
    dist, order = cKDTree(basis).query(positions)
    if dist.max() > 2e-3:
        raise RuntimeError(f"GLB vertices don't match the basis (max distance {dist.max():.4f})")

    base_n = accessor_array(gltf, binary, prim["attributes"]["NORMAL"]).astype(np.float64) / 127
    out = GlbWriter()
    attributes = {name: out.dense(accessor_array(gltf, binary, i), meta[i]) for name, i in prim["attributes"].items()}
    indices = out.dense(accessor_array(gltf, binary, prim["indices"]).ravel()[:, None], meta[prim["indices"]], indices=True)

    def add_target(arrays, metas):
        moved = np.zeros(len(positions), bool)
        for arr in arrays.values():
            moved |= np.any(arr != 0, axis=1)
        if moved.mean() > 0.5:
            return {k: out.dense(arrays[k], metas[k]) for k in arrays}
        where = np.flatnonzero(moved)
        kind = 5123 if len(positions) < 65536 else 5125
        index_view = {"bufferView": out.view(where.astype(COMPONENT[kind]).tobytes()), "componentType": kind}
        return {k: out.sparse(arrays[k], metas[k], where, index_view) for k in arrays}

    targets, names = [], list(mesh.get("extras", {}).get("targetNames", []))
    for name, t in zip(list(names), prim.get("targets", [])):
        arrays = {k: accessor_array(gltf, binary, i) for k, i in t.items()}
        if normal_fix and name in normal_fix and "NORMAL" in arrays:
            n_fix, w = normal_fix[name][0][order], normal_fix[name][1][order]
            n_old = base_n + arrays["NORMAL"] / 127
            n_new = w[:, None] * n_fix + (1 - w[:, None]) * n_old
            n_new /= np.maximum(np.linalg.norm(n_new, axis=1, keepdims=True), 1e-9)
            sel = w > 0
            arrays["NORMAL"][sel] = np.clip(np.round((n_new[sel] - base_n[sel]) * 127), -127, 127).astype(np.int8)
        targets.append(add_target(arrays, {k: meta[i] for k, i in t.items()}))

    pos_meta = meta[prim["targets"][0]["POSITION"]]
    nrm_meta = meta[prim["targets"][0]["NORMAL"]]
    for name, neutral, delta, normals in extra_targets:
        dq = np.round(delta[order] / scale).astype(np.int16)
        dn = (normals - vertex_normals(neutral, tris))[order]
        dnq = np.clip(np.round(dn * 127), -127, 127).astype(np.int8)
        pm = dict(pos_meta, min=dq.min(axis=0).tolist(), max=dq.max(axis=0).tolist())
        targets.append(add_target({"POSITION": dq, "NORMAL": dnq}, {"POSITION": pm, "NORMAL": nrm_meta}))
        names.append(name)

    prim["attributes"] = attributes
    prim["indices"] = indices
    prim["targets"] = targets
    mesh.setdefault("extras", {})["targetNames"] = names
    if "weights" in mesh:
        mesh["weights"] = mesh["weights"] + [0.0] * (len(targets) - len(mesh["weights"]))
    out.save(gltf, dst)
    return len(positions)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

BASE_WEEK = 24
GROWTH_WEEKS = (8, 12, 16, 20, 28, 32, 36, 40)
RIG_WEEKS = (8, 12, 16, 20, 24, 28, 32, 36, 40)


def axis_rotation(axis, angle):
    """Rotation matrix about a unit axis (Rodrigues)."""
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def foot_up(j, s, f):
    """The foot's up direction: from the pose when it sets one (a kick), else from the world."""
    u = j.get("footUp" + s, vec(0, 1, 0))
    return unit(u - f * np.dot(u, f))


# Joint angles of a full kick, degrees: hip extension, knee extension, ankle pointing
KICK_ANGLES = {"L": (16, 55, 18), "R": (10, 62, 18)}


def kick_pose(side, amount=1.0):
    """A kick from the hip and knee: the thigh swings down a little, the knee straightens and
    the foot moves with the shin, pointing slightly. Built from joint rotations, so the shin
    keeps its length and the ankle doesn't kink."""
    hip_deg, knee_deg, ankle_deg = (math.radians(a) * amount for a in KICK_ANGLES[side])

    def pose(j):
        hip, kn, an, toe = j["hip" + side], j["kn" + side], j["an" + side], j["toe" + side]
        thigh, shin, foot = kn - hip, an - kn, toe - an
        up = foot_up(j, side, unit(foot))
        hinge = unit(np.cross(thigh, shin))            # knee axis, normal to the leg's plane
        R_hip = axis_rotation(vec(1, 0, 0), hip_deg)    # knee swings down and forward
        R_knee = axis_rotation(R_hip @ hinge, -knee_deg) @ R_hip      # shin opens away from the thigh
        R_ankle = axis_rotation(R_knee @ hinge, ankle_deg) @ R_knee   # toes point a little
        knee = hip + R_hip @ thigh
        ankle = knee + R_knee @ shin
        return {"kn" + side: knee, "an" + side: ankle, "toe" + side: ankle + R_ankle @ foot,
                "footUp" + side: R_ankle @ up}
    return pose


def wave_pose():
    """Left hand lifts away from the face, palm towards the viewer (who sees the baby's left side)."""
    def pose(j):
        wrist = j["wrL"] + vec(-0.08, 0.08, 0.07)
        return {
            "elL": j["elL"] + vec(-0.05, 0.03, 0.03),
            "wrL": wrist,
            "tipL": wrist + unit(vec(-0.35, 1.0, 0.25)) * 0.12,
            "palmL": vec(0.35, 0.0, 1.0),
            "curlL": vec(0.25, 0.3, 0),
        }
    return pose


# Sex keys: the indifferent tubercle at 8 and 12 weeks (shared), then girl and boy shapes at
# each growth week. They are deltas on top of that week's neutral body, kept out of the GPU
# morph texture: the page blends them on the CPU (they only move a few hundred vertices).
SEX_KEYS = [("SEX_8", 8, "X"), ("SEX_12", 12, "X")] + [(f"{s}{w}", w, s) for s in "FM" for w in (16, 20, 24, 28, 32, 36, 40)]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="final GLB")
    ap.add_argument("--rig", required=True)
    ap.add_argument("--raw", help="Blender export before gltfpack (default: next to --out, .raw.glb)")
    ap.add_argument("--gltfpack", default="npx --yes gltfpack", help="gltfpack command")
    ap.add_argument("--h", type=float, default=0.0026, help="grid spacing for marching cubes")
    ap.add_argument("--tris", type=int, default=26000, help="triangle budget outside the ears, face and groin")
    args = ap.parse_args(argv)
    raw = args.raw or args.out.replace(".glb", ".raw.glb")
    t0 = time.time()

    base_ops, base_anchors = build(BASE_WEEK)
    d, lo = sculpt_grid(base_ops, args.h)
    print(f"grid {d.shape} in {time.time() - t0:.1f}s")
    verts, faces = marching_cubes(d, lo, args.h)
    print(f"marching cubes: {len(verts)} verts, {len(faces)} tris")

    # Ears and the groin keep more vertices: the ears for their folds, the groin so the
    # genital shapes (sex keys) have vertices to grow from
    groin = base_anchors["groin"]
    ear_ops = [op for op in base_ops if op.name.startswith("ear")]

    def near_ears(p):
        return np.min([op.shape.sdf(p) for op in ear_ops], axis=0) < 0.008

    def groin_core(p):
        return np.linalg.norm(p - groin, axis=1) < 0.045

    def groin_ring(p):
        r = np.linalg.norm(p - groin, axis=1)
        return (r >= 0.045) & (r < 0.085)

    face_ops = [op for op in base_ops if op.name.startswith(("eye", "nose", "lip", "mouth", "nostril", "chin"))]

    def near_face(p):
        return (np.min([op.shape.sdf(p) for op in face_ops], axis=0) < 0.006) & ~near_ears(p)

    obj = blender_mesh(verts, faces, args.tris,
                       details=((near_ears, 0.4), (near_face, 0.6), (groin_core, 1.0), (groin_ring, 0.5)))
    basis, tris, normals = read_mesh(obj)
    basis = project(base_ops, basis, iterations=2, max_step=0.004)
    write_positions(obj, basis)
    print(f"basis: {len(basis)} verts, {len(tris)} tris ({time.time() - t0:.1f}s)")

    center = (basis.min(axis=0) + basis.max(axis=0)) / 2
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])

    def solve(source_ops, source, target_ops, label, relax_steps):
        moved = carry(source_ops, target_ops, source)
        moved = project(target_ops, moved, iterations=6)
        delta = laplacian_smooth_delta(moved - source, edges, len(basis), iterations=3)
        moved = project(target_ops, source + delta, iterations=3, max_step=0.01)
        moved = relax(target_ops, moved, edges, iterations=relax_steps)
        err = np.abs(eval_sdf(target_ops, moved))
        print(f"  {label}: mean |sdf| {err.mean():.5f}, max {err.max():.4f}")
        return moved

    A, deg = adjacency(edges, len(basis))
    # Around the ears, face and neck: in the small early weeks these detailed areas shrink and
    # their vertices crowd together, so they get extra relaxation along the surface and
    # smoothed normals
    neck_ops = [op for op in base_ops if op.name == "neck"]
    early_dist = np.min([op.shape.sdf(basis) for op in ear_ops + face_ops + neck_ops], axis=0)
    ear_idx = np.flatnonzero(early_dist < 0.05)
    ear_w = 1 - np.array([smoothstep(0.03, 0.05, x) for x in early_dist[ear_idx]])
    normal_fix = {}

    obj.shape_key_add(name="Basis", from_mix=False)
    rig = {"weeks": list(RIG_WEEKS), "anchors": {}, "center": center.tolist()}
    built = {g: build(g) for g in RIG_WEEKS}
    for g in RIG_WEEKS:
        for key, val in built[g][1].items():
            rig["anchors"].setdefault(key, []).append((val - center).round(5).tolist())
    # Walk outwards from the basis week so each step is a small change
    neutral = {BASE_WEEK: basis}
    for chain in ([w for w in RIG_WEEKS if w < BASE_WEEK][::-1], [w for w in RIG_WEEKS if w > BASE_WEEK]):
        prev_ops, prev = base_ops, basis
        for g in chain:
            ops_g = built[g][0]
            prev = solve(prev_ops, prev, ops_g, f"W{g}", relax_steps=14 if g < 16 else 4)
            if g < 16:
                # enough iterations for the surface Laplacian to unfold the crowded early ear
                prev = relax_masked(ops_g, prev, A, deg, ear_idx, ear_w, iterations=100 if g < 10 else 40)
                n_mesh = vertex_normals(prev, tris)
                n_sdf = gradient(ops_g, prev[ear_idx], eps=0.002)
                n_sdf /= np.maximum(np.linalg.norm(n_sdf, axis=1, keepdims=True), 1e-9)
                n_fix = n_mesh.copy()
                blend = 0.5 * n_sdf + 0.5 * n_mesh[ear_idx]
                n_fix[ear_idx] = blend / np.maximum(np.linalg.norm(blend, axis=1, keepdims=True), 1e-9)
                weight_full = np.zeros(len(basis))
                weight_full[ear_idx] = ear_w
                normal_fix[f"W{g}"] = (n_fix, weight_full)
            prev_ops = ops_g
            neutral[g] = prev
            add_shape_key(obj, f"W{g}", prev - center)

    def solve_local(source_ops, source, target_ops, label, relax_steps=6):
        """Like solve, but only the moving limb is relaxed and everything else stays exactly
        where it was, so the morph target is sparse."""
        moved = carry(source_ops, target_ops, source)
        moved = project(target_ops, moved, iterations=6)
        delta = laplacian_smooth_delta(moved - source, edges, len(basis), iterations=3)
        moved = project(target_ops, source + delta, iterations=3, max_step=0.01)
        core = np.linalg.norm(moved - source, axis=1) > 2e-4
        ring = np.where(core, 0, 99)
        frontier = core.copy()
        for r in range(1, 5):     # rings around the moving part, for a soft edge
            frontier = (A @ frontier.astype(float) > 0) & (ring == 99)
            ring[frontier] = r
        region = ring < 99
        moved[~region] = source[~region]
        idx = np.flatnonzero(region)
        weight = np.clip(1 - (ring[idx] - 1) / 3, 0, 1)
        moved = relax_masked(target_ops, moved, A, deg, idx, weight, iterations=relax_steps)
        err = np.abs(eval_sdf(target_ops, moved[idx]))
        print(f"  {label}: {len(idx)} verts, mean |sdf| {err.mean():.5f}, max {err.max():.4f}")
        return moved

    # Kicks have a half-way key too, so the leg follows its arc instead of a straight line
    actions = (("KICK_L_MID", kick_pose("L", 0.5)), ("KICK_L", kick_pose("L")),
               ("KICK_R_MID", kick_pose("R", 0.5)), ("KICK_R", kick_pose("R")), ("WAVE_L", wave_pose()))
    for name, overrides in actions:
        ops_a, anchors_a = build(BASE_WEEK, overrides)
        add_shape_key(obj, name, solve_local(base_ops, basis, ops_a, name) - center)
        if name in ("KICK_L", "KICK_R"):
            rig[name] = (anchors_a["foot" + name[-1]] - center).round(5).tolist()

    # Sex keys, grown onto each week's neutral body inside the groin region
    r = np.linalg.norm(basis - groin, axis=1)
    region = np.flatnonzero(r < 0.085)
    weight = 1 - np.array([smoothstep(0.06, 0.085, x) for x in r[region]])
    sex_targets = []
    for name, g, sex in SEX_KEYS:
        start = neutral[g]
        o = built[g][0].gframe[0]
        outside = np.setdiff1d(np.arange(len(basis)), region)
        margin = np.linalg.norm(start[outside] - o, axis=1).min()
        shaped = grow_genitals(g, sex, start, region, weight, A, deg)
        delta = shaped - start
        normals = vertex_normals(shaped, tris)
        ops_s = local_ops(build(g, sex=sex)[0], o, 0.16)
        # half mesh, half SDF normals (sampled wide, so narrow creases don't flicker): smooth on
        # small round shapes where the mesh is coarse, steady inside the clefts
        n_sdf = gradient(ops_s, shaped[region], eps=0.0025)
        n_sdf /= np.maximum(np.linalg.norm(n_sdf, axis=1, keepdims=True), 1e-9)
        blend = 0.5 * n_sdf * weight[:, None] + normals[region] * (1 - 0.5 * weight[:, None])
        normals[region] = blend / np.maximum(np.linalg.norm(blend, axis=1, keepdims=True), 1e-9)
        print(f"  {name}: {int((np.linalg.norm(delta, axis=1) > 1e-5).sum())} verts moved, "
              f"max {np.linalg.norm(delta, axis=1).max():.4f}, region margin {margin:.3f}")
        sex_targets.append((name, start - center, delta, normals))

    # Basis last: shape keys above were relative to the uncentred basis
    write_positions(obj, basis - center)
    obj.data.shape_keys.key_blocks["Basis"].data.foreach_set("co", to_blender(basis - center).ravel())

    # Skin colour detail: occlusion in creases, a little warmth on cheeks, lips, fingertips and knees
    _, _, normals = read_mesh(obj)
    ao = ambient_occlusion(base_ops, basis, normals)
    for _ in range(4):   # soften: on a dense mesh raw occlusion speckles at sharp junctions (shoulders)
        ao = 0.5 * ao + 0.5 * (A @ ao) / deg
    warm = np.zeros(len(basis))
    by_name = {op.name: op for op in base_ops}
    for name, amount in (("cheekL", 0.55), ("cheekR", 0.55), ("lipUpperC", 0.8), ("lipUpperL", 0.8), ("lipUpperR", 0.8),
                         ("lipLower", 0.8), ("noseTip", 0.35), ("toe0L", 0.4), ("toe0R", 0.4)):
        if name in by_name:
            warm = np.maximum(warm, amount * np.exp(-np.maximum(by_name[name].shape.sdf(basis), 0) / 0.006))
    for s in "LR":
        for i in range(4):
            op = by_name[f"finger{i}{s}b"]
            warm = np.maximum(warm, 0.45 * np.exp(-np.maximum(op.shape.sdf(basis), 0) / 0.004))
    lid = np.zeros(len(basis))
    for line in base_ops.lid_lines:
        lid = np.maximum(lid, np.exp(-np.maximum(line.sdf(basis), 0) / 0.0018))
    shade = (0.55 + 0.45 * ao) * (1 - 0.32 * lid)
    rgb = np.stack([shade, shade * (1 - 0.1 * warm), shade * (1 - 0.13 * warm)], axis=1)
    rgb = rgb * np.array([1.0, 0.97, 0.96]) ** (1 - ao)[:, None]
    thickness = skin_thickness(base_ops, basis, normals, A, deg)
    print(f"thickness: min {thickness.min():.4f}, median {np.median(thickness):.4f}, max {thickness.max():.4f}")
    # only the thin parts glow; limbs and trunk are fully opaque
    alpha = np.array([smoothstep(0.008, 0.07, x) for x in thickness])
    set_colors(obj, np.clip(rgb, 0, 1), alpha)

    import bpy
    bpy.ops.export_scene.gltf(
        filepath=raw,
        export_format="GLB",
        use_selection=False,
        export_yup=True,
        export_apply=False,
        export_morph=True,
        export_morph_normal=True,
        export_materials="NONE",
        export_vertex_color="ACTIVE",
        export_animations=False,
        export_skins=False,
        export_texcoords=False,
    )
    with open(args.rig, "w") as fh:
        json.dump(rig, fh, separators=(",", ":"))

    # Quantize with gltfpack (KHR_mesh_quantization, no decoder needed), then add the sex keys
    import os
    import shlex
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        packed = os.path.join(tmp, "packed.glb")
        subprocess.run(shlex.split(args.gltfpack) + ["-i", raw, "-o", packed, "-kn", "-ke"], check=True)
        n = finalize_glb(packed, args.out, basis - center, tris, sex_targets, normal_fix)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {n} verts) and {args.rig} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Build the Luna View fetus model (glTF) from a signed-distance-field sculpt.

Pipeline
  1. A parametric skeleton for any gestational week (key poses at 8, 20 and 40
     weeks, proportions from approximate fetal biometry tables).
  2. An SDF sculpt on top of it: smooth unions of ellipsoids and round cones
     (head, face, torso, limbs, fingers, toes) and smooth subtractions for small
     creases (eyelids, mouth, nostrils, ear concha).
  3. Marching cubes on the week-24 sculpt, cleaned up and decimated in Blender.
     This is the basis mesh.
  4. Growth shape keys (W8 ... W40): the basis vertices are carried to each week
     by blending per-primitive frame mappings, then projected onto that week's
     SDF surface. All keys share one topology, so they morph smoothly.
  5. Action shape keys (KICK_L, KICK_R, WAVE_L) built the same way from posed
     skeletons at week 24.
  6. Ambient occlusion and soft skin tints computed from the SDF and stored as
     vertex colours.
  7. GLB export, plus a small JSON rig with anchor points per key week.

Coordinates: "pose space" is x = baby's right, y = up, z = front, in
crown–rump units (CRL = 1). It is written to Blender as (x, -z, y) so the
glTF export (+Y up) comes back out as pose space.

Run (needs Blender's `bpy` module, numpy and scikit-image):
  python build_fetus.py --out ../models/fetus.raw.glb --rig ../models/fetus-rig.json
then compress with gltfpack (see README).
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


def build(g, overrides=None):
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

    def sub(name, shape, k):
        ops.append(Op(name, shape, "sub", k, "head"))

    # --- torso --------------------------------------------------------------
    add("chest", Ellipsoid(j["chest"], table3(CHEST, g) + 0.014 * fat), 0.07)
    add("belly", Ellipsoid(j["belly"], table3(BELLY, g) + 0.018 * fat), 0.07)
    add("pelvis", Ellipsoid(j["pelvis"], table3(PELVIS, g) + 0.012 * fat), 0.07)
    br = table(BACK_R, g)
    add("backUpper", RoundCone(j["backTop"], j["backMid"], br, br * 1.05), 0.06)
    add("backLower", RoundCone(j["backMid"], j["backLow"], br * 1.05, br), 0.06)
    for s in "LR":
        add("glut" + s, Ellipsoid(j["glut" + s], np.full(3, table(GLUT_R, g) + 0.012 * fat)), 0.05)
    tr = table(TAIL_R, g)
    add("tail", RoundCone(j["tailA"], j["tailB"], tr, tr * 0.45), 0.04)
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
        lengths = (0.42, 0.47, 0.44, 0.36)
        offsets = (-0.16, -0.055, 0.055, 0.16)
        for i in range(4):
            base = W + d * 0.5 * L + side * offsets[i] * L
            fl = lengths[i] * L * finger_scale
            c1, c2 = j["curl" + s][0], j["curl" + s][1]
            d1 = d * math.cos(c1) + n * math.sin(c1)
            d2 = d * math.cos(c1 + c2) + n * math.sin(c1 + c2)
            mid = base + d1 * fl * 0.55
            tip = mid + d2 * fl * 0.45
            fr = 0.06 * L
            add(f"finger{i}{s}a", RoundCone(base, mid, fr, fr * 0.92), finger_k)
            add(f"finger{i}{s}b", RoundCone(mid, tip, fr * 0.92, fr * 0.82), finger_k)
        thumb_side = side if s == "L" else -side
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
        up = unit(vec(0, 1, 0) - f * f[1])
        side = np.cross(up, f)
        Lf = foot_len
        add("foot" + s, Ellipsoid(A + f * 0.3 * Lf - up * 0.2 * Lf, (0.2 * Lf, 0.15 * Lf, 0.48 * Lf), np.stack([side, up, f], axis=1)), 0.025)
        add("heel" + s, Ellipsoid(A - f * 0.08 * Lf - up * 0.2 * Lf, np.full(3, 0.17 * Lf)), 0.03)
        # big toe sits on the medial side (towards +x for the left foot)
        medial = side if (side[0] > 0) == (s == "L") else -side
        toe_r = (0.085, 0.062, 0.056, 0.05, 0.044)
        toe_off = (0.13, 0.03, -0.05, -0.12, -0.18)
        for i in range(5):
            c = A + f * (0.76 - 0.03 * i) * Lf - up * 0.2 * Lf + medial * toe_off[i] * Lf
            add(f"toe{i}{s}", Ellipsoid(c, np.full(3, toe_r[i] * Lf * (0.55 + 0.45 * feat))), 0.006 + 0.012 * (1 - feat))

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
    fs = 0.45 + 0.55 * feat
    cheek = (0.22 + 0.13 * fat) * R * fs
    for sx, s in ((-1, "L"), (1, "R")):
        add("cheek" + s, Ellipsoid(at(0.4 * sx, -0.52, 0.6), np.full(3, cheek)), 0.15 * R)
    add("chin", Ellipsoid(at(0, -0.86, 0.62), np.full(3, 0.19 * R)), 0.12 * R)
    add("nose", RoundCone(at(0, -0.2, 0.9), at(0, -0.42, 0.9 + 0.12 * fs), 0.07 * R * fs, 0.105 * R * fs), 0.07 * R)
    for sx, s in ((-1, "L"), (1, "R")):
        add("eye" + s, Ellipsoid(at(0.33 * sx, -0.18, 0.84), (0.17 * R, 0.1 * R * fs, 0.08 * R * fs), HR), 0.06 * R)
        ear = 0.45 + 0.55 * feat
        add("ear" + s, Ellipsoid(at(0.96 * sx, -0.12 - 0.3 * (1 - feat), -0.1), (0.07 * R * ear, 0.27 * R * ear, 0.18 * R * ear), HR), 0.05 * R)
    add("lipUpper", Ellipsoid(at(0, -0.64, 0.86), (0.17 * R, 0.06 * R * fs, 0.07 * R * fs), HR), 0.06 * R)
    add("lipLower", Ellipsoid(at(0, -0.75, 0.83), (0.13 * R, 0.07 * R * fs, 0.07 * R * fs), HR), 0.06 * R)

    crease = max(0.024 * R, 0.0038) * feat
    for sx, s in ((-1, "L"), (1, "R")):
        sub("nostril" + s, Ellipsoid(at(0.065 * sx, -0.48, 1.0 + 0.1 * fs), np.full(3, max(0.04 * R, 0.004) * feat)), 0.02 * R)
        ear = 0.45 + 0.55 * feat
        sub("concha" + s, Ellipsoid(at(1.03 * sx, -0.14 - 0.3 * (1 - feat), -0.08), (0.06 * R * ear, 0.15 * R * ear, 0.1 * R * ear), HR), 0.02 * R)
    sub("mouth", RoundCone(at(-0.12, -0.695, 0.9), at(0.12, -0.695, 0.9), crease * 0.9, crease * 0.9), 0.02 * R)

    lid_lines = [RoundCone(at(0.21 * sx, -0.215, 0.925), at(0.45 * sx, -0.185, 0.835), 0.004, 0.004) for sx in (-1, 1)]
    anchors = {
        "head": at(-0.25, 0.45, 0.6),
        "ear": at(-0.99, -0.12, -0.1),
        "heart": j["chest"] + vec(-0.035, 0.02, table3(CHEST, g)[2] * 0.6),
        "hand": j["wrL"] + unit(j["tipL"] - j["wrL"]) * 0.3 * hand_len,
        "knee": j["knL"],
        "navel": navel + navel_dir * 0.018,
        "footL": j["anL"] + unit(j["toeL"] - j["anL"]) * 0.45 * foot_len,
        "footR": j["anR"] + unit(j["toeR"] - j["anR"]) * 0.45 * foot_len,
    }
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


def ambient_occlusion(ops, p, nrm, steps=5, dist=0.012):
    occ = np.zeros(len(p))
    weight = 1.0
    for i in range(1, steps + 1):
        hh = dist * i
        occ += (hh - eval_sdf(ops, p + nrm * hh)) * weight
        weight *= 0.6
    return np.clip(1.0 - 4.5 * occ / dist / steps, 0.0, 1.0)


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


def blender_mesh(verts, faces, target_tris):
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
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    dec = obj.modifiers.new("Decimate", "DECIMATE")
    dec.ratio = min(1.0, target_tris / tris)
    dec.use_collapse_triangulate = True
    apply_modifiers(obj)
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


def set_colors(obj, rgb):
    me = obj.data
    attr = me.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    attr.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = attr


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

BASE_WEEK = 24
GROWTH_WEEKS = (8, 12, 16, 20, 28, 32, 36, 40)
RIG_WEEKS = (8, 12, 16, 20, 24, 28, 32, 36, 40)


def kick_pose(side):
    """Shin straightens forward, foot points."""
    def pose(j):
        kn, an, toe = j["kn" + side], j["an" + side], j["toe" + side]
        shin = np.linalg.norm(an - kn)
        foot = np.linalg.norm(toe - an)
        knee = kn + vec(0, -0.02, 0.03)
        direction = vec(-0.45, -0.55, 0.75) if side == "L" else vec(0.3, -0.35, 1.0)
        ankle = knee + unit(direction) * shin
        return {"kn" + side: knee, "an" + side: ankle, "toe" + side: ankle + unit(vec(0, -0.55, 1.0)) * foot}
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


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--h", type=float, default=0.0034, help="grid spacing for marching cubes")
    ap.add_argument("--tris", type=int, default=30000)
    args = ap.parse_args(argv)
    t0 = time.time()

    base_ops, base_anchors = build(BASE_WEEK)
    d, lo = sculpt_grid(base_ops, args.h)
    print(f"grid {d.shape} in {time.time() - t0:.1f}s")
    verts, faces = marching_cubes(d, lo, args.h)
    print(f"marching cubes: {len(verts)} verts, {len(faces)} tris")

    obj = blender_mesh(verts, faces, args.tris)
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

    obj.shape_key_add(name="Basis", from_mix=False)
    rig = {"weeks": list(RIG_WEEKS), "anchors": {}, "center": center.tolist()}
    built = {g: build(g) for g in RIG_WEEKS}
    for g in RIG_WEEKS:
        for key, val in built[g][1].items():
            rig["anchors"].setdefault(key, []).append((val - center).round(5).tolist())
    # Walk outwards from the basis week so each step is a small change
    for chain in ([w for w in RIG_WEEKS if w < BASE_WEEK][::-1], [w for w in RIG_WEEKS if w > BASE_WEEK]):
        prev_ops, prev = base_ops, basis
        for g in chain:
            ops_g = built[g][0]
            prev = solve(prev_ops, prev, ops_g, f"W{g}", relax_steps=8 if g < 16 else 4)
            prev_ops = ops_g
            add_shape_key(obj, f"W{g}", prev - center)

    for name, overrides in (("KICK_L", kick_pose("L")), ("KICK_R", kick_pose("R")), ("WAVE_L", wave_pose())):
        ops_a, anchors_a = build(BASE_WEEK, overrides)
        add_shape_key(obj, name, solve(base_ops, basis, ops_a, name, relax_steps=3) - center)
        if name.startswith("KICK"):
            rig[name] = (anchors_a["foot" + name[-1]] - center).round(5).tolist()

    # Basis last: shape keys above were relative to the uncentred basis
    write_positions(obj, basis - center)
    obj.data.shape_keys.key_blocks["Basis"].data.foreach_set("co", to_blender(basis - center).ravel())

    # Skin colour detail: occlusion in creases, a little warmth on cheeks, lips, fingertips and knees
    _, _, normals = read_mesh(obj)
    ao = ambient_occlusion(base_ops, basis, normals)
    warm = np.zeros(len(basis))
    by_name = {op.name: op for op in base_ops}
    for name, amount in (("cheekL", 0.55), ("cheekR", 0.55), ("lipUpper", 0.8), ("lipLower", 0.8), ("nose", 0.35),
                         ("toe0L", 0.4), ("toe0R", 0.4)):
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
    set_colors(obj, np.clip(rgb, 0, 1))

    import bpy
    bpy.ops.export_scene.gltf(
        filepath=args.out,
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
    print(f"wrote {args.out} and {args.rig} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

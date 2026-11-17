"""Build a Luna View embryo model for one early week (glTF), from a signed-distance-field sculpt.

The fetus model (build_fetus.py) morphs one mesh through all the weeks. An embryo looks too
different for that: its own sculpt per week reads better. Each week here is a static mesh plus
the two dark eyes (none from week 10, when the lids are shut), with per-vertex data for the skin
shader:

  COLOR_0   rgb: soft occlusion and tints, a: local thickness (for the light shining through)
  _VESSEL   where the fine surface vessels of the head may show (the shader draws them)
  _PALE     how much paler the skin is: the hand and foot plates and, less, the limb buds have
            little blood in them yet and look almost white-pink next to the red body
  _BONE     (from week 9) x: the small bones of the fingers and toes starting to harden, drawn as dark
            red dots; y: how much of the organs' shadow to leave out (none on the root of the cord)
  _MOVE     x: which part a vertex belongs to (1 left arm, 2 right arm, 3 left leg, 4 right leg,
            5 head), y: how much it follows that part when it moves (0 at the joint, 1 beyond);
            the page swings a part round its joint (in the rig) when it is touched. The eyes carry
            it too, so they nod with the head

The organs that block the light coming through the body (heart, liver, spinal cord, vertebrae and
ribs) are not painted on the skin: they are baked into a small 3D texture that goes in the rig, and
the shader looks into it along the line of sight, so they sit deep inside and shift with the view.
From week 9 the texture has a second channel with the long bones of the arms and legs, which only
the limbs look into.

Coordinates are those of build_fetus.py: pose space, x = the embryo's right, y = up, z = front,
in crown-rump units (CRL = 1), written out centred on the bounding box. A small JSON rig holds
the anchor points the page needs (face, heart, hands, feet, the cord root).

Anatomy follows the Carnegie stages: week 8 of pregnancy (counted from the last period) is about
6 weeks after conception, Carnegie stage 18 to 19, crown-rump length about 14-17 mm. The head is
nearly half the body and bent onto the chest over the heart and liver bulges; the eyes are dark
with pigment; the ear is a low ring of small hillocks behind the jaw; the hands and feet are
plates with the rays of the fingers and toes and notches between them; the gut still loops into
the base of the cord (physiological herniation); a short tail is left at the bottom.

Week 9 (Week9) is Carnegie stage 20 to 21, about 7 weeks after conception, crown-rump length about
18-23 mm: an upright, egg-shaped body with a rounder head held higher, a nose, lips and almond eyes
with a grey iris, a small outer ear, arms bent at the elbow with the hands flat in front of the chest,
thick legs with no knee showing yet, broad hands and feet with short, thick, separate fingers and
toes, and only a small point left of the tail.
Week 8's output is unchanged by it (the per-week values are class attributes).

Week 10 (Week10) is Carnegie stage 23, the end of the embryonic period, about 8 weeks after
conception, crown-rump length about 27-35 mm: a very large, round head sitting low between the
shoulders, its face small and low; the eyes shut under fused lids (no eye meshes: a large, smooth lid
with the line where it has closed); a small flat nose, lips, a small chin; a C-shaped outer ear far
back and low on the head; a narrow neck and sloping shoulders; arms of even length hanging beside the
chest to elbows low and back, hands with a palm, a thumb and four jointed fingers (one resting under
the chin, the other in front of the chest); legs drawn up, the thighs lying against the lower belly,
the knees high, the shins down to feet with a heel and five toes low in front of the bottom; a full,
low bottom and no tail. The outline of the head and body and where the limbs are were measured on
the clip's own turn (its camera fitted to the silhouettes all round), in the exact front, side and
back views. The trunk is one body of smooth sections (Loft), narrower than the whole silhouette
where the arms and thighs lie beside it, so the limbs read as limbs. Its rig carries its own look
for the page (rig.look).

Run (needs Blender's `bpy` module, numpy, scipy and scikit-image):
  python build_embryo.py --week 8 --out ../models/embryo-w08.glb --rig ../models/embryo-w08.json
  python build_embryo.py --week 9 --out ../models/embryo-w09.glb --rig ../models/embryo-w09.json
  python build_embryo.py --week 10 --out ../models/embryo-w10.glb --rig ../models/embryo-w10.json
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


class Loft:
    """A body of elliptical sections stacked along y: its back and its front (z) and its half width
    (x) at each height are smooth curves through the given rows, and it closes to a rounded end at
    the first and the last row (which give only the height and the middle of the section there).
    So the outline from the side and from the front is set row by row, and the surface in between
    has no bumps where separate shapes would meet."""
    kind = "loft"

    def __init__(self, rows, cap_bottom=0.15, cap_top=0.2, n=4000):
        from scipy.interpolate import CubicSpline
        rows = sorted(rows)   # (y, back z, front z, half width)
        y = np.array([r[0] for r in rows], float)
        self.y0, self.y1 = float(y[0]), float(y[-1])
        self.caps = (cap_bottom, cap_top)
        hw = np.array([r[3] for r in rows], float)
        hd = np.array([(r[2] - r[1]) / 2 for r in rows], float)
        hw[0] = hw[-1] = hd[0] = hd[-1] = 0.0
        # the squares of the half width and half depth are the smooth curves: they run through
        # zero at the ends, so the ends are round (not pointed) and need no separate cap
        ys = np.linspace(self.y0, self.y1, n + 1)
        ga = np.maximum(CubicSpline(y, hw * hw)(ys), 0.0)
        gb = np.maximum(CubicSpline(y, hd * hd)(ys), 0.0)
        cz = CubicSpline(y, [(r[1] + r[2]) / 2 for r in rows], bc_type="clamped")(ys)
        # for the distance each end is treated as half an ellipsoid, a tall one (see sdf): the
        # sections are kept as the ellipsoid's own, times a factor that is smooth all the way
        ub = np.clip((self.y0 + cap_bottom - ys) / cap_bottom, 0.0, 1.0)
        ut = np.clip((ys - (self.y1 - cap_top)) / cap_top, 0.0, 1.0)
        f2 = 1.0 - ub * ub - ut * ut
        f2[0], f2[-1] = f2[1], f2[-2]
        A, B = np.sqrt(ga / f2), np.sqrt(gb / f2)
        A[0], A[-1], B[0], B[-1] = 2 * A[1] - A[2], 2 * A[-2] - A[-3], 2 * B[1] - B[2], 2 * B[-2] - B[-3]
        self.ys, self.h = ys, float(ys[1] - ys[0])
        self.tab = [(v, np.gradient(v, self.h)) for v in (A, B, cz)]
        self.lo = np.array([-hw.max(), self.y0, min(r[1] for r in rows)]) - 0.02
        self.hi = np.array([hw.max(), self.y1, max(r[2] for r in rows)]) + 0.02
        self.r_min = float(min(hw[1:-1].min(), hd[1:-1].min()))

    def _at(self, y):
        """The sections' half width, half depth and middle at the heights y, and their slopes;
        beyond the ends they carry on a little way and level off (smoothly, so the distance has
        no step anywhere)."""
        yc = np.clip(y, self.y0, self.y1)
        t = (y - yc) / 0.05
        ease, slope = 0.05 * np.tanh(t), 1.0 / np.cosh(np.clip(t, -20.0, 20.0)) ** 2
        out = []
        for v, dv in self.tab:
            d = np.interp(yc, self.ys, dv)
            out.append((np.interp(yc, self.ys, v) + d * ease, d * slope))
        return out

    def sdf(self, p):
        """As for an ellipsoid (the implicit value over the length of its gradient), with the
        sections' own change along y in the gradient; each end closes like half an ellipsoid."""
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        (a, da), (b, db), (c, dc) = self._at(y)
        a, b = np.maximum(a, 1e-3), np.maximum(b, 1e-3)
        hb, ht = self.caps
        ub = np.maximum((self.y0 + hb - y) / hb, 0.0)
        ut = np.maximum((y - (self.y1 - ht)) / ht, 0.0)
        qx, qz = x / a, (z - c) / b
        k0 = np.sqrt(qx * qx + qz * qz + ub * ub + ut * ut)
        gy = qx * (-x * da / (a * a)) + qz * (-dc / b - (z - c) * db / (b * b)) - ub / hb + ut / ht
        k1 = np.sqrt((qx / a) ** 2 + (qz / b) ** 2 + gy * gy)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

    def aabb(self):
        return self.lo, self.hi

    def size(self):
        return self.r_min


def bend(d, m, angle):
    """Turn the direction d (and its own 'up', m) by the angle towards -m."""
    c, s = math.cos(angle), math.sin(angle)
    return d * c - m * s, m * c + d * s


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
    EYE_SQUASH = 0.8
    # the parts kept finer when the mesh is decimated: (op name prefixes, distance, share kept)
    DETAILS = ((("ear",), 0.01, 0.6), (("hand", "foot"), 0.01, 0.7),
               (("snout", "nasalPit", "mouth", "eyeBulge", "jaw"), 0.01, 0.6))
    # a little more red round the mouth and nose
    WARM = ("snout", "jaw", "midface")
    # the skull, where the surface vessels run, above the height where they start
    HEAD_OPS = ("cranium", "forebrain", "midbrain", "hindbrain")
    VESSEL_Y = (0.02, 0.12)
    # small hollows (op name prefixes) whose skin takes the thickness round them, so they don't glow
    CREASES = ()
    # the face's own, wider soft shadows (under the nose, round the mouth and the eyes), by op name
    FACE_AO = ()
    # darker tints in small hollows: (op name prefixes, rgb multiplier, falloff)
    LIPS = ()
    # the eye ball's mesh: rings round its own axis, so the iris and pupil are round
    EYE_ROUND = False
    # the root of the cord loses its baked shading towards its end, as the page's cord tube has none
    CORD_TONE = False
    # thin see-through upper lids over the eyes (a third mesh, "Lids")
    LIDS = False
    # how far the hands and feet pale towards the tips in the page (None: the page's own, nearly white)
    PALE = None
    # how far towards their own thinness the tips of the fingers and toes go (the rest of the way from
    # the thickness where the limb leaves the body): how much light comes through them
    TIP_THIN = 0.85
    # how much the organs' 3D texture is blurred (in texels): the softer, the more they merge
    ORGAN_BLUR = 1.0
    # how far apart the page's 12 samples into the organs are along the line of sight (None: its own
    # 0.026, a third of the embryo deep)
    ORGAN_STEP = None
    # how strongly the head's surface vessels show (the _VESSEL mask is scaled by it)
    VESSEL_GAIN = 1.0
    # the week's own look in the page (None: the page's own): the key light, the rim light and the
    # light from all round (multipliers), the light from all round on the skin itself, how fast the
    # light through the body dies away per colour, and a tint on the skin colour
    LOOK = None

    def eye_dir(self, s):
        """The way the eye looks out: sideways and a little forwards."""
        return unit(vec(s, 0.0, 0.45))

    def place_eyes(self):
        for s in (1, -1):
            y, z = self.EYE
            # set well into the head, so only a flat, almond-shaped cap shows (it doesn't bulge out)
            r = 0.032
            sx = self.surface_x(y, z)
            self.eyes.append((vec((sx - 0.72 * r) * s, y, z), r))
            # a soft fold of skin above the eye: the eyelids are just starting
            self.lid_lines.append(Ellipsoid(vec((sx - 0.012) * s, y + 0.033, z), vec(0.03, 0.006, 0.03)))

    def eye_colours(self, normals, outward):
        return eye_colours(normals, outward)

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

    def surface_z(self, x, y, hi=0.45):
        """How far forward along +z the skin is at side x and height y (the sculpt so far)."""
        lo = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if eval_sdf(self.ops, np.array([[x, y, mid]]))[0] < 0:
                lo = mid
            else:
                hi = mid
        return lo

    def build(self):
        add, sub = self.add, self.sub
        ops = self.ops
        self.dots = []
        self.bones = []
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
        # (the page's cord tube carries on from its end at the same thickness)
        self.cord_r = 0.056

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
        # leg is a short bud from the rump forwards under the cord, slimming to the ankle; the foot is a
        # broad, flat paddle of the same kind, nearly as big as the hand, pointing a little down from the
        # line of the leg with its sole turned in: from the side it shows whole, from the front as a
        # narrow plate (as in the clip). Each limb can swing a little at its root when touched (the
        # joints below go in the rig)
        self.joints = {}
        for s, side in ((1, "R"), (-1, "L")):
            arm = f"arm{side}"
            sh = vec(0.125 * s, -0.12, -0.13)
            reach = unit(vec(0.22 * s, -0.1, 0.97))
            wr = sh + reach * 0.15
            add(f"armBud{side}", Chain([RoundCone(sh, wr, 0.052, 0.042)]), 0.03, arm)
            self.limb_plate(add, f"hand{side}", arm, root=wr, axis=reach, normal=vec(1.0 * s, 0.0, 0.0),
                            length=0.15, width=0.074, thick=0.021, root_r=0.042, lobes=5, spread=2.3)
            hand_c = wr + reach * 0.09
            leg = f"leg{side}"
            hip = vec(0.08 * s, -0.49, 0.07)
            an = vec(0.11 * s, -0.495, 0.2)
            add(f"legBud{side}", Chain([RoundCone(hip, an, 0.056, 0.038)]), 0.03, leg)
            # (the foot points a little down from the line of the leg, as in the clip)
            axis = unit(unit(an - hip) + vec(0, -0.4, 0))
            self.limb_plate(add, f"foot{side}", leg, root=an, axis=axis, normal=vec(1.0 * s, 0.0, 0.0),
                            length=0.15, width=0.072, thick=0.021, root_r=0.038, lobes=5, spread=1.7)
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
        # the heart as the page shows it: only a faint, soft shade under the chest at this age, beating
        # (centre, radii, how strongly it shows)
        self.heart = (vec(0, -0.15, 0.13), vec(0.075, 0.05, 0.045), 0.4)

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
        ankle). The round limb flattens and widens gradually into a broad, softly rounded plate (not a
        thin blade: it keeps a fair thickness, fullest in the middle) (there is no
        narrowing at the wrist), whose far edge is made of soft, flat, overlapping lobes where the
        fingers or toes are starting, so the outline is gently scalloped and nothing stands out as a
        finger yet. root_r is the limb's radius where the plate begins."""
        n = np.asarray(normal, float)
        n = unit(n - axis * (n @ axis))
        R = frame(n, -axis)
        u, v = R[:, 0], R[:, 1]
        # the flattening: from the limb's own round section to the plate's
        for i, (t, w, th) in enumerate(((0.0, 0.0, 1.0), (0.16, 0.35, 0.7), (0.34, 0.7, 0.42), (0.52, 0.92, 0.2))):
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
            add(f"{name}Lobe{i}", Ellipsoid(c, vec(lobe * r, lobe * r, thick * 0.8), R), 0.012, group)



class Week9(Week8):
    """Carnegie stage 20-21: about 7 weeks after conception, crown-rump length about 18-23 mm."""

    # the eye: height and depth on the side of the face. Measured from the clip's side view against
    # its forehead (which our silhouette matches): the eye sits well back from the face, so the nose
    # and the mouth stand out in front of it like a short muzzle
    EYE = (0.09, 0.225)
    EYE_SQUASH = 1.0
    DETAILS = ((("ear",), 0.01, 0.6), (("hand", "foot"), 0.01, 0.85),
               (("nose", "nostril", "mouth", "lip", "chin", "eyeSocket"), 0.012, 0.75))
    # (only the lips: in the clip the nose and the skin round the mouth are as pale as the rest)
    WARM = ("lip", "lipLower")
    HEAD_OPS = ("cranium", "forebrain", "hindbrain")
    VESSEL_Y = (0.1, 0.2)
    CREASES = ("mouth", "nostril", "earBowl", "eyeSocket")
    LIPS = ((("mouth",), (0.96, 0.74, 0.76), 0.007), (("mouth",), (0.62, 0.36, 0.4), 0.0025),
            (("nostril",), (0.86, 0.76, 0.76), 0.0025), (("eyeSocket",), (0.76, 0.66, 0.64), 0.003),
            # (and a soft greyish-brown shade on the lids all round the eye, as in the clip)
            (("eyeSocket",), (0.72, 0.62, 0.62), 0.02))
    EYE_ROUND = True
    FACE_AO = ("nose", "lip", "chin", "cheek", "midface", "eyeSocket")
    CORD_TONE = True
    LIDS = True
    PALE = 0.25
    TIP_THIN = 0.5
    # (sharper, so the ribs stay separate bars in the light through the back, as in the clip)
    ORGAN_BLUR = 0.6

    def eye_dir(self, s):
        """The way the face looks there (the skin round the eye faces out, forwards and down, the face
        being tucked under the forehead), turned a little further forwards: so the eye lies flush in
        the skin, and from the side only a narrow almond shows (as in the clip)."""
        return self.eye_out[s]

    def place_eyes(self):
        for s in (1, -1):
            # its front just under the skin round the hollow (a little deeper than the hollow alone
            # needs: the skin under and behind the eye falls back, and the ball must stay under it)
            r = 0.034
            self.eyes.append((self.eye_at[s] - self.eye_dir(s) * (r + 0.006), r))

    def eye_colours(self, normals, outward):
        """As in the clip: a broad, light grey iris, a big black pupil that darkens into it softly, a
        dark warm grey all round the iris (not black), and a small bright catchlight up from the pupil."""
        def ramp(x, e0, e1):
            t = np.clip((x - e0) / (e1 - e0), 0, 1)
            return t * t * (3 - 2 * t)
        c = normals @ outward
        iris = ramp(c, 0.862, 0.878)
        pupil = ramp(c, 0.935, 0.97)
        outer = np.array([0.15, 0.125, 0.105])
        grey = np.array([0.5, 0.47, 0.44])
        rgb = outer + (grey - outer) * iris[:, None]
        rgb = rgb + (0.012 - rgb) * pupil[:, None]
        up = unit(vec(0, 1, 0) - outward * outward[1])
        glint = ramp(normals @ unit(outward + up * 0.22), 0.9935, 0.9965)
        rgb = rgb + (0.92 - rgb) * glint[:, None]
        return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def lid(self, c, r, outward):
        """The upper lid as the clip shows it: a thin, see-through fold of skin coming down over the
        top of the eye, just above the pupil, its edge arched (highest over the pupil, lower towards
        the corners) and soft, so the iris shows through it and it thickens into the skin above. A
        cap of a sphere a little bigger than the eye; returns positions, normals, triangles and the
        opacity at each vertex. Its far part lies under the skin round the opening, where it is hidden."""
        def ramp(x, e0, e1):
            t = np.clip((x - e0) / (e1 - e0), 0, 1)
            return t * t * (3 - 2 * t)
        v_, n_, t_ = sphere(c, r + 0.0012, rings=72, segs=64, axis=outward)
        up = unit(vec(0, 1, 0) - outward * outward[1])
        fwd = unit(vec(0, 0, 1) - outward * outward[2])
        e, f = n_ @ up, n_ @ fwd
        edge = 0.15 - 0.5 * f * f
        # (mostly over the front of the eye, towards the nose: behind, the opening keeps a clean edge)
        alpha = 0.85 * ramp(e, edge - 0.07, edge + 0.22) * (0.2 + 0.8 * ramp(f, -0.6, 0.3)) * (n_ @ outward > 0.3)
        keep = alpha[t_].max(axis=1) > 0.004
        used = np.unique(t_[keep])
        remap = -np.ones(len(v_), int)
        remap[used] = np.arange(len(used))
        return v_[used], n_[used], remap[t_[keep]], alpha[used]

    def build(self):
        add, sub = self.add, self.sub
        ops = self.ops
        # (the head joins the trunk softly at the back and the sides of the neck, as before, but less
        # so in front: under the chin and the jaw the two meet in a crease, as in the clip)
        ops.junctions = {
            "head": (vec(0, -0.03, -0.08), 0.17, 0.07),
            "armL": (vec(-0.15, -0.07, -0.01), 0.07, 0.04),
            "armR": (vec(0.15, -0.07, -0.01), 0.07, 0.04),
            "legL": (vec(-0.09, -0.45, 0.05), 0.07, 0.045),
            "legR": (vec(0.09, -0.45, 0.05), 0.07, 0.045),
        }
        self.dots = []      # the bones of the fingers and toes: (centre, radius, plate normal)
        self.bones = []     # the long bones of the limbs, for the light through them

        # ---- head: still nearly half the embryo, but rounder and held higher than a week before.
        # The forehead bulges forward over a small face that now has a nose, lips and a chin; the
        # back of the head runs straight on into the back. Proportions follow the side view of the
        # reference clip (measured in model units)
        # (the dome sits high, so that low down on the face the sides of the head fall back on either
        # side of the nose; the forehead in front is the forebrain's)
        add("cranium", Ellipsoid(vec(0, 0.22, 0.07), vec(0.185, 0.23, 0.25)), 0.1, "head")
        add("hindbrain", Ellipsoid(vec(0, 0.09, -0.08), vec(0.16, 0.18, 0.15)), 0.1, "head")
        add("forebrain", Ellipsoid(vec(0, 0.15, 0.185), vec(0.14, 0.12, 0.148)), 0.08, "head")
        # the face, small and low under the forehead, as the close-ups of the clip show it: a short
        # muzzle of nose and mouth in the middle of the lower face, the cheeks set back on either side
        # of it. The nose is its biggest feature: a bridge running down from between the eyes into a
        # broad, round tip a little above the mouth, with the nostrils under the tip. Right under it a short upper lip, a small mouth (no wider
        # than the nose) with a fuller lower lip, and a small, round chin that sits back under it;
        # below the chin a crease runs back under the jaw into the neck (the head joins the trunk
        # there with a smaller blend, see junctions)
        add("midface", Ellipsoid(vec(0, 0.03, 0.13), vec(0.085, 0.066, 0.085)), 0.03, "head")
        add("lowerFace", Ellipsoid(vec(0, 0.0, 0.11), vec(0.08, 0.058, 0.1)), 0.05, "head")
        add("noseBridge", Ellipsoid(vec(0, 0.06, 0.24), vec(0.03, 0.032, 0.04)), 0.02, "head")
        add("nose", Ellipsoid(vec(0, 0.012, 0.19), vec(0.058, 0.036, 0.051)), 0.01, "head")
        add("lip", Ellipsoid(vec(0, -0.029, 0.18), vec(0.04, 0.013, 0.03)), 0.014, "head")
        add("lipLower", Ellipsoid(vec(0, -0.048, 0.167), vec(0.028, 0.011, 0.03)), 0.01, "head")
        add("chin", Ellipsoid(vec(0, -0.068, 0.142), vec(0.04, 0.022, 0.028)), 0.03, "head")
        for s, side in ((1, "R"), (-1, "L")):
            add(f"cheek{side}", Ellipsoid(vec(0.075 * s, 0.035, 0.145), vec(0.04, 0.04, 0.045)), 0.035, "head")
        # the mouth and the nostrils cut just under the skin where it is, so they stay shallow
        sub("mouth", Ellipsoid(vec(0, -0.039, self.surface_z(0.0, -0.039) - 0.004), vec(0.022, 0.0025, 0.007)), 0.004, "head")
        nz = self.surface_z(0.022, -0.014)
        for s, side in ((1, "R"), (-1, "L")):
            sub(f"nostril{side}", Ellipsoid(vec(0.022 * s, -0.014, nz - 0.002), vec(0.005, 0.0028, 0.0035)), 0.003, "head")

        # ---- trunk: an upright, egg-shaped body, a smooth curve down the back; the liver makes the
        # belly bulge under the arms and the gut still loops into the base of the cord
        add("upperBack", Ellipsoid(vec(0, -0.06, -0.07), vec(0.165, 0.17, 0.155)), 0.08, "torso")
        add("chest", Ellipsoid(vec(0, -0.15, 0.06), vec(0.165, 0.1, 0.14)), 0.08, "torso")
        add("midBack", Ellipsoid(vec(0, -0.22, -0.045), vec(0.165, 0.15, 0.15)), 0.08, "torso")
        add("liver", Ellipsoid(vec(0, -0.26, 0.07), vec(0.17, 0.13, 0.18)), 0.08, "torso")
        add("lowerBack", Ellipsoid(vec(0, -0.36, 0.02), vec(0.15, 0.11, 0.11)), 0.08, "torso")
        add("lowerBelly", Ellipsoid(vec(0, -0.41, 0.07), vec(0.15, 0.1, 0.16)), 0.08, "torso")
        add("rump", Ellipsoid(vec(0, -0.45, 0.1), vec(0.14, 0.095, 0.115)), 0.07, "torso")
        # all that is left of the tail: a small point between the legs
        add("tail", RoundCone(vec(0, -0.515, 0.1), vec(0, -0.552, 0.135), 0.022, 0.008), 0.02, "torso")
        cord_a, cord_b = vec(0, -0.3, 0.17), vec(0, -0.31, 0.3)
        add("cordRoot", RoundCone(cord_a, cord_b, 0.07, 0.052), 0.05, "torso")
        self.cord_r = 0.052
        # the neck crease: a soft groove in the trunk just under the jaw, along the line where the
        # head meets it (found from the two surfaces), fading out below the ears; the jaw overhangs it
        # a little, as in the clip. Under the chin itself it is only a shallow fold, so the chin
        # runs smoothly into the neck and does not stand out as a ring
        seam = [(0.0, -0.09, 0.145, 0.003), (0.032, -0.073, 0.128, 0.005), (0.054, -0.047, 0.095, 0.0078),
                (0.073, -0.03, 0.072, 0.0072), (0.09, -0.014, 0.05, 0.0062), (0.109, -0.003, 0.028, 0.005),
                (0.128, 0.006, 0.001, 0.003)]
        for s in (1, -1):
            pts = [(vec(x * s, y, z), r) for x, y, z, r in seam]
            sub(f"neckCrease{'R' if s > 0 else 'L'}", Chain([RoundCone(a[0], b[0], a[1], b[1]) for a, b in zip(pts[:-1], pts[1:])]), 0.01, "torso")

        # ---- the eyes: set into the side of the face, looking out sideways and forwards, each in an
        # almond-shaped opening (the eye itself is placed in place_eyes())
        y, z = self.EYE
        # (both found before either hollow is cut: surface_x looks along +x only)
        sx = self.surface_x(y, z)
        self.eye_at = {1: vec(sx, y, z), -1: vec(-sx, y, z)}
        n = unit(gradient(ops, self.eye_at[1][None], eps=0.002)[0])
        # (turned well forwards and a little up from the skin there: from the front of the face the
        # eyes are seen as round, as in the clip, not as slits at its sides)
        n = unit(n + vec(0, 0.22, 0.65))
        self.eye_out = {1: n, -1: n * vec(-1, 1, 1)}
        for s, side in ((1, "R"), (-1, "L")):
            out = self.eye_dir(s)
            skin = self.eye_at[s]
            # an almond-shaped hollow, wider than tall: the eye ball fills it, and its soft rim is
            # where the lids are starting
            R = frame(out, vec(0, 1, 0))
            # (tilted a little, the outer corner lower)
            a = 0.18 * s
            u_, v_ = R[:, 0].copy(), R[:, 1].copy()
            R[:, 0], R[:, 1] = math.cos(a) * u_ - math.sin(a) * v_, math.sin(a) * u_ + math.cos(a) * v_
            # (its middle a little back from the eye's axis, towards the ear: the eye looks forwards in
            # it, so more of the dark shows behind the iris than in front, as in the clip)
            back = unit(vec(0, 0, -1) - out * (-out[2]))
            sub(f"eyeSocket{side}", Ellipsoid(skin + back * 0.004, vec(0.034, 0.016, 0.017), R), 0.007, "head")

        # ---- the outer ear: a small oval rim, taller than it is wide and leaning back a little at the
        # top, level with the eye and well behind it. Its hollow is set forwards, so the rim is thick
        # and raised behind and over it and thin in front, where it opens (as in the clip)
        ear_y, ear_z = 0.125, 0.035
        ear_x = self.surface_x(ear_y, ear_z)
        lean = -0.17
        Rx = np.array([[1, 0, 0], [0, math.cos(lean), -math.sin(lean)], [0, math.sin(lean), math.cos(lean)]])
        for s, side in ((1, "R"), (-1, "L")):
            c = vec((ear_x - 0.002) * s, ear_y, ear_z)
            add(f"ear{side}", Ellipsoid(c, vec(0.013, 0.039, 0.022), Rx), 0.008, "head")
            sub(f"earBowl{side}", Ellipsoid(c + vec(0.011 * s, -0.002, 0.007), vec(0.008, 0.027, 0.0125), Rx), 0.004, "head")
        self.anchors["ear"] = vec(ear_x + 0.02, ear_y, ear_z)
        # (the vessels of the head fan out from a point well above the ear and a little in front of it)
        vy, vz = 0.27, 0.07
        self.anchors["vessels"] = vec(self.surface_x(vy, vz), vy, vz)
        self.anchors["skull"] = vec(0, 0.2, 0.08)

        def back_z(x, y):
            lo_z, hi_z = -0.45, 0.0
            for _ in range(30):
                mid = 0.5 * (lo_z + hi_z)
                if eval_sdf(ops, np.array([[x, y, mid]]))[0] < 0:
                    hi_z = mid
                else:
                    lo_z = mid
            return hi_z
        back = [(float(y), back_z(0.0, y)) for y in np.arange(0.0, -0.5, -0.05)]

        # ---- limbs: the arm now bends at the elbow. It hangs from the shoulder along the side of the
        # chest, the forearm comes forwards and in, and the hand lies flat in front of the chest, palm
        # down, the fingers pointing in towards the other hand. The leg shows no knee yet (as in the
        # clip): one thick, softly curved limb from the rump forwards under the belly, carried on into a
        # broad foot with its sole turned in. Hands and feet are broad and thick, the fingers and toes
        # short, thick and separate now, webbed only at their roots. The arms are slender beside the
        # head and the trunk (as in the side view of the clip)
        self.joints = {}
        up = vec(0, 1, 0)
        for s, side in ((1, "R"), (-1, "L")):
            arm = f"arm{side}"
            sh = vec(0.15 * s, -0.07, -0.01)
            el = vec(0.21 * s, -0.16, 0.1)
            wr = vec(0.11 * s, -0.115, 0.215)
            add(f"armBud{side}", Chain([RoundCone(sh, el, 0.044, 0.035), RoundCone(el, wr, 0.035, 0.029)]), 0.02, arm)
            axis = unit(vec(-0.5 * s, -0.12, 0.86))
            back_of_hand = vec(0.7 * s, 1.0, 0.1)
            tips = self.digits(add, f"hand{side}", arm, wr, axis, back_of_hand, first=vec(-s, 0, -0.4), hand=True,
                               palm=(0.042, 0.029, 0.019), reach=0.04, web=0.018,
                               lengths=(0.019, 0.041, 0.044, 0.042, 0.037), r=(0.0106, 0.0066), spread=0.55)
            self.bones += [(RoundCone(sh + (el - sh) * 0.2, el - (el - sh) * 0.12, 0.009, 0.008), 40.0)]
            fa = unit(np.cross(wr - el, up)) * 0.007
            self.bones += [(RoundCone(el + (wr - el) * 0.12 + d, wr - (wr - el) * 0.1 + d, 0.006, 0.0055), 40.0) for d in (fa, -fa)]
            leg = f"leg{side}"
            hip = vec(0.095 * s, -0.44, 0.05)
            kn = vec(0.135 * s, -0.455, 0.175)
            an = vec(0.12 * s, -0.475, 0.29)
            add(f"legBud{side}", Chain([RoundCone(hip, kn, 0.06, 0.05), RoundCone(kn, an, 0.05, 0.038)]), 0.03, leg)
            faxis = unit(vec(-0.4 * s, 0.05, 0.92))
            dorsum = vec(0.85 * s, 0.25, -0.35)
            dorsum = unit(dorsum - faxis * (dorsum @ faxis))
            ftips = self.digits(add, f"foot{side}", leg, an, faxis, dorsum, first=vec(0, 1, 0), hand=False,
                                palm=(0.048, 0.036, 0.021), reach=0.06, web=0.009,
                                lengths=(0.024, 0.021, 0.02, 0.019, 0.017), r=(0.0086, 0.0092), spread=1.25, rows=2,
                                blend=0.013)
            self.bones += [(RoundCone(hip + (kn - hip) * 0.25, kn + (an - kn) * 0.1, 0.01, 0.009), 40.0)]
            fl = unit(np.cross(an - kn, up)) * 0.008
            self.bones += [(RoundCone(kn + (an - kn) * 0.3 + d, an - (an - kn) * 0.1 + d, 0.0065, 0.006), 40.0) for d in (fl, -fl * 0.6)]
            self.anchors[f"hand{side}"] = wr + axis * 0.04
            self.anchors[f"foot{side}"] = an + faxis * 0.04
            self.anchors[f"knee{side}"] = kn + vec(0.03 * s, 0.0, 0.01)
            hand_dir = unit(tips - sh)
            foot_dir = unit(ftips - hip)
            self.joints[arm] = {"pivot": sh, "axis": unit(np.cross(unit(wr - sh), up)), "dir": hand_dir, "plate": wr}
            self.joints[leg] = {"pivot": hip, "axis": unit(np.cross(unit(an - hip), up)), "dir": foot_dir, "plate": an}

        # ---- inside: heart and liver, the brain, the spinal cord and four ribs low on the back (as
        # in week 8); the long bones of the arms and legs go in a second channel, which only the limbs
        # look into
        self.organs = [
            (Ellipsoid(vec(0, -0.14, 0.07), vec(0.08, 0.07, 0.08)), 3.0),     # heart
            (Ellipsoid(vec(0, -0.27, 0.08), vec(0.13, 0.1, 0.12)), 2.5),      # liver
            (RoundCone(vec(0, 0.1, -0.13), vec(0, -0.02, -0.17), 0.04, 0.03), 14.0),   # medulla
        ]
        for c in (vec(0.07, 0.22, 0.16), vec(-0.07, 0.22, 0.16)):
            for k, absorb in ((1.0, 2.5), (0.75, 3.0), (0.5, 3.5)):
                self.organs.append((Ellipsoid(c, vec(0.09, 0.12, 0.13) * k), absorb))
        self.organs.append((Ellipsoid(vec(0, 0.33, 0.04), vec(0.09, 0.08, 0.1)), 2.5))
        self.organs.append((Ellipsoid(vec(0, 0.15, -0.1), vec(0.08, 0.09, 0.08)), 3.0))
        cord_pts = [vec(0, -0.02, -0.17)] + [vec(0, y, z + 0.06) for y, z in back[1:]]
        for a_, b_ in zip(cord_pts[:-1], cord_pts[1:]):
            self.organs.append((RoundCone(a_, b_, 0.025, 0.025), 30.0))

        def skin_along(c, d):
            lo_t, hi_t = 0.0, 0.45
            for _ in range(30):
                mid = 0.5 * (lo_t + hi_t)
                if eval_sdf(ops, (c + d * mid)[None])[0] < 0:
                    lo_t = mid
                else:
                    hi_t = mid
            return lo_t
        for y in (-0.25, -0.295, -0.34, -0.385):
            c = vec(0, y, 0.0)
            for s_ in (1, -1):
                for th in np.linspace(0.3, 1.75, 9):
                    d = vec(s_ * math.sin(th), 0, -math.cos(th))
                    # (narrow bars with clear gaps between them)
                    self.organs.append((Ellipsoid(c + d * (skin_along(c, d) - 0.04), vec(0.018, 0.0085, 0.018)), 55.0))

        self.joints["head"] = {"pivot": vec(0, -0.05, -0.1), "axis": vec(1, 0, 0), "dir": unit(vec(0, 0.25, 0.2))}
        # the heart as the page shows it: a soft, wide, brownish shade under the thin skin of the upper
        # chest, just under the chin and between the hands, beating (as in the clip)
        self.heart = (vec(0, -0.155, 0.15), vec(0.095, 0.042, 0.04), 1.0)

        self.anchors.update({
            "head": vec(0.1, 0.42, 0.12),
            "face": vec(0, 0.03, 0.24),
            "faceFront": vec(0, -0.05, 0.55),
            "heart": vec(0.05, -0.13, 0.2),
            "navel": cord_b + vec(0, 0, 0.03),
            "cordBase": cord_a,
            "groin": vec(0, -0.48, 0.17),
            "core": vec(0, -0.08, 0.02),
        })
        self.anchors["hand"] = self.anchors["handL"]
        self.anchors["knee"] = self.anchors["kneeL"]
        self.anchors["footL"] = self.anchors.pop("footL")
        self.anchors["footR"] = self.anchors.pop("footR")
        return self

    def digits(self, add, name, group, root, axis, back, first, hand, palm, reach, lengths, r, spread=1.0, rows=1, web=0.012, blend=0.006):
        """A hand or foot: a flat palm (or sole) on the end of the limb and five short, separate
        digits fanned out from its far edge, webbed together only at their roots. back points out
        of the back of the hand or the top of the foot, first towards the side the thumb or the big
        toe is on. The thumb sets off from the side of the palm near the wrist; the big toe is the
        first of the row. r is a digit's half width and half thickness: as in the clip the digits are
        flat, soft and a little tapered to a rounded tip (each one a flattened ellipsoid), not round
        rods with knobs on the end. blend is how softly they grow out of the palm or sole (the smooth
        union's radius): wide for the toes, which would otherwise look stuck on. Notes where the small
        bones are (the shader dots them in, as short dashes along the digit) and returns the tip of the
        middle digit."""
        back = np.asarray(back, float)
        R = frame(back - axis * (back @ axis), -axis)
        if R[:, 0] @ first < 0:
            R[:, 0] *= -1
        u, a, n = R[:, 0], -R[:, 1], R[:, 2]
        pu, pa, pn = palm
        add(f"{name}Palm", Ellipsoid(root + a * pa * 0.85, vec(pu, pa, pn), R), 0.015, group)
        far = root + a * reach
        # (for the toes no thicker than they are, so they carry on from it without a step)
        web_t = pn * 0.75 if hand else min(pn * 0.75, r[1] * 1.3)
        add(f"{name}Web", Ellipsoid(far, vec(pu * 0.95, web, web_t), R), max(0.006, blend * 0.8), group)
        rw, rt = r

        def digit(dname, base, d, L, w, t):
            # (curled a little towards the palm or sole) a flat body of nearly even width, and a
            # rounded, hardly narrower tip
            d = unit(d - n * 0.12)
            side = unit(np.cross(n, d))
            Rd = np.stack([side, d, unit(np.cross(side, d))], axis=1)
            add(dname, Ellipsoid(base + d * (L * 0.45), vec(w, L * 0.5, t), Rd), blend, group)
            add(f"{dname}Tip", Ellipsoid(base + d * (L - w * 0.85), vec(w * 0.88, w * 0.88, t * 0.92), Rd), 0.004, group)
            return base + d * L, d

        if hand:
            row = lengths[1:]
            spacing = (2 * pu - 2 * rw) / 3
            offs = [1.5 * spacing, 0.5 * spacing, -0.5 * spacing, -1.5 * spacing]
            angles = [0.3 * spread, 0.1 * spread, -0.1 * spread, -0.3 * spread]
        else:
            row = lengths
            spacing = (2 * pu - 2 * rw) / 4
            offs = [2 * spacing, spacing, 0.0, -spacing, -2 * spacing]
            angles = [0.18 * spread, 0.08 * spread, 0.0, -0.08 * spread, -0.16 * spread]
        tips = []
        for i, (L, off, ang) in enumerate(zip(row, offs, angles)):
            base = far + u * off - a * 0.004
            big = 1.08 if (not hand and i == 0) else 1.0
            tip, d = digit(f"{name}Digit{i}", base, math.cos(ang) * a + math.sin(ang) * u, L, rw * big, rt * big)
            tips.append(tip)
            for t in ((0.3, 0.68) if hand else (0.45,)):
                self.dots.append((base + (tip - base) * t, 0.0042, n, d))
            for k in range(rows):
                self.dots.append((base - a * (0.017 + 0.017 * k) - u * off * (0.15 + 0.1 * k), 0.0046, n, d))
        if hand:
            L = lengths[0]
            tb = root + a * 0.014 + u * (pu * 0.85)
            # (short, along the side of the hand, as in the clip)
            tt, d = digit(f"{name}Thumb", tb, a * 0.9 + u * 0.35 - n * 0.3, L, rw * 1.2, rt * 1.15)
            for t in (0.4, 0.78):
                self.dots.append((tb + (tt - tb) * t, 0.0042, n, d))
        return tips[len(tips) // 2]

class Week10(Week9):
    """Carnegie stage 23, the last week of the embryonic period: about 8 weeks after conception,
    crown-rump length about 27-35 mm."""

    # the eyes are shut now (the lids have grown over them and fused): no eye meshes, the lids are
    # part of the skin
    EYE = None
    DETAILS = ((("ear",), 0.01, 0.7), (("hand", "foot"), 0.01, 0.9),
               (("nose", "nostril", "mouth", "lip", "chin", "lidLine", "eyeLid"), 0.012, 0.8))
    WARM = ("lip", "lipLower")
    HEAD_OPS = ("cranium", "crown", "occiput")
    VESSEL_Y = (0.11, 0.2)
    CREASES = ("mouth", "nostril", "earBowl", "lidLine")
    LIPS = ((("mouth",), (0.95, 0.76, 0.77), 0.007), (("mouth",), (0.66, 0.42, 0.45), 0.0025),
            (("nostril",), (0.78, 0.62, 0.62), 0.0028), (("lidLine",), (0.62, 0.48, 0.48), 0.0018),
            (("earBowl",), (0.9, 0.8, 0.8), 0.004))
    FACE_AO = ("nose", "lip")
    LIDS = False
    PALE = 0.3
    TIP_THIN = 0.5
    ORGAN_BLUR = 0.6
    # (not as deep: with the chin down on the chest, the spine behind would show through the face)
    ORGAN_STEP = 0.02
    # (in the clip only a few fine vessels show on the head, mostly in front)
    VESSEL_GAIN = 0.35
    # as the clip lights it: a strong key light (on the embryo's left, the side of the hand in front
    # of the chest) and little light from all round, so the lit side is a pale, warm peach and the
    # side away from the light glows a deep orange red through the body; a matte skin
    LOOK = {"key": 1.9, "hemi": 0.5, "indirect": 0.45, "rim": 1.0, "through": [4.2, 12.0, 18.0], "tint": [1.07, 0.86, 0.74],
            "roughness": 0.78, "side": -1}

    # the trunk row by row: height, back (z), front (z), half width (see Loft: the first and the
    # last row are where it ends)
    TRUNK = ((0.045, -0.3, -0.18, 0.0), (0.0, -0.36, -0.125, 0.12), (-0.05, -0.39, -0.083, 0.153), (-0.1, -0.394, -0.06, 0.164), (-0.15, -0.39, -0.035, 0.16),
             (-0.2, -0.372, 0.0, 0.16), (-0.25, -0.35, 0.038, 0.165), (-0.3, -0.332, 0.052, 0.168),
             (-0.35, -0.315, 0.048, 0.162), (-0.4, -0.288, 0.045, 0.148), (-0.45, -0.256, 0.04, 0.132),
             (-0.5, -0.215, 0.018, 0.115), (-0.548, -0.16, -0.04, 0.0))
    # the lengths of the upper arm and the forearm
    UPPER_ARM, FOREARM = 0.19, 0.185

    def place_eyes(self):
        pass

    @staticmethod
    def elbow(sh, wr, s, out, upper, fore):
        """Where the elbow is: upper from the shoulder and fore from the wrist, hanging down and
        held out to its own side (s) by the angle out, in degrees."""
        ax = wr - sh
        reach = float(np.linalg.norm(ax))
        ax = ax / reach
        along = (upper * upper - fore * fore + reach * reach) / (2 * reach)
        r = math.sqrt(max(upper * upper - along * along, 1e-8))
        down = unit(vec(0, -1, 0) - ax * (vec(0, -1, 0) @ ax))
        side = unit(np.cross(ax, down))
        if side[0] * s < 0:
            side = -side
        a = math.radians(out)
        return sh + ax * along + (down * math.cos(a) + side * math.sin(a)) * r

    def arm(self, add, side, sh, el, wr):
        """An arm from the shoulder to the wrist: a round shoulder, an upper arm that slims to the
        elbow, and a forearm fullest just below the elbow that tapers to the wrist."""
        arm = f"arm{side}"
        u, f = el - sh, wr - el
        add(f"shoulder{side}", Ellipsoid(sh + u * 0.06, vec(0.04, 0.043, 0.043)), 0.02, arm)
        pts = [sh, sh + u * 0.4, sh + u * 0.8, el, el + f * 0.22, el + f * 0.6, wr]
        radii = [0.036, 0.0335, 0.029, 0.0275, 0.0285, 0.0235, 0.0185]
        add(f"armBud{side}", Chain([RoundCone(a, b, ra, rb) for a, b, ra, rb in zip(pts[:-1], pts[1:], radii[:-1], radii[1:])]), 0.012, arm)

    def hand(self, add, name, group, wrist, axis, back, first, curl=1.0, spread=1.0, size=1.0):
        """A hand: a broad palm with the pads of the thumb and the little finger, four fingers of
        three joints each, a little curled and fanned, and a short thumb set off from the side of
        the palm. back points out of the back of the hand, first towards the thumb's side. Returns
        the tip of the middle finger."""
        a = unit(np.asarray(axis, float))
        n = np.asarray(back, float)
        n = unit(n - a * (n @ a))
        u = unit(np.cross(n, a))
        if u @ first < 0:
            u = -u
        R = np.stack([u, a, n], axis=1)

        def P(su, sa, sn):
            return wrist + (u * su + a * sa + n * sn) * size

        def E(c, r):
            return Ellipsoid(c, vec(*r) * size, R)
        add(f"{name}Wrist", E(P(0, 0.012, 0), (0.021, 0.022, 0.0135)), 0.01, group)
        add(f"{name}Palm", E(P(-0.001, 0.036, 0.0005), (0.0305, 0.027, 0.0118)), 0.012, group)
        add(f"{name}Thenar", E(P(0.019, 0.022, -0.005), (0.013, 0.018, 0.0095)), 0.008, group)
        add(f"{name}Hypothenar", E(P(-0.021, 0.027, -0.004), (0.0105, 0.021, 0.0085)), 0.008, group)
        # (side, where it leaves the palm, length, thickness, fan in degrees)
        fingers = ((0.0215, 0.056, 0.046, 0.0080, 7.0), (0.0073, 0.06, 0.052, 0.0082, 1.5),
                   (-0.0068, 0.058, 0.048, 0.0078, -4.0), (-0.0198, 0.052, 0.038, 0.0069, -11.0))
        tips = []
        for i, (su, sa, L, r, fan) in enumerate(fingers):
            fan = math.radians(fan * spread)
            d, m = math.cos(fan) * a + math.sin(fan) * u, n
            p = P(su, sa - 0.006, -0.001)
            cones = []
            # (the three joints: each bends a little more than the last, the little finger most)
            for k, (part, ang, taper) in enumerate(((0.45, 12.0, 0.95), (0.31, 26.0, 0.88), (0.24, 16.0, 0.8))):
                d, m = bend(d, m, math.radians(ang * curl * (1.0 + 0.12 * i)))
                q = p + d * (L * part + (0.006 if k == 0 else 0.0)) * size
                r0 = r * (1.0 if k == 0 else (0.95, 0.88)[k - 1])
                cones.append(RoundCone(p, q, r0 * size, r * taper * size))
                p = q
            add(f"{name}Digit{i}", Chain(cones), 0.0045, group)
            tips.append(p)
        d, m = unit(0.6 * u + 0.76 * a - 0.25 * n), unit(n + 0.3 * u)
        m = unit(m - d * (m @ d))
        p = P(0.024, 0.014, -0.004)
        cones = []
        for part, ang, r0, r1 in ((0.02, 6.0, 0.0092, 0.0085), (0.016, 18.0, 0.0085, 0.0074)):
            d, m = bend(d, m, math.radians(ang * curl))
            q = p + d * part * size
            cones.append(RoundCone(p, q, r0 * size, r1 * size))
            p = q
        add(f"{name}Thumb", Chain(cones), 0.006, group)
        return tips[1]

    def leg(self, add, side, hip, kn, an):
        """A leg from the hip to the ankle: a full thigh that slims to the knee, a calf fullest
        below the knee on the back of the shin, and a slim ankle."""
        leg = f"leg{side}"
        t, sh = kn - hip, an - kn
        calf = unit(np.cross(np.cross(sh, t), sh))
        pts = [hip, hip + t * 0.35, hip + t * 0.75, kn, kn + sh * 0.3 + calf * 0.004, kn + sh * 0.7 + calf * 0.002, an]
        radii = [0.055, 0.053, 0.0445, 0.038, 0.0355, 0.0275, 0.0225]
        add(f"legBud{side}", Chain([RoundCone(a, b, ra, rb) for a, b, ra, rb in zip(pts[:-1], pts[1:], radii[:-1], radii[1:])]), 0.012, leg)

    def foot(self, add, name, group, ankle, axis, dorsum, first, size=0.86):
        """A foot: a round heel behind and below the ankle, the instep sloping down from the ankle
        to the ball of the foot, and five toes in an arc, the big toe the thickest. dorsum points
        out of the top of the foot, first towards the big toe's side. Returns the tip of the
        middle toe."""
        a = unit(np.asarray(axis, float))
        n = np.asarray(dorsum, float)
        n = unit(n - a * (n @ a))
        u = unit(np.cross(n, a))
        if u @ first < 0:
            u = -u
        R = np.stack([u, a, n], axis=1)

        def P(su, sa, sn):
            return ankle + (u * su + a * sa + n * sn) * size
        add(f"{name}Heel", Ellipsoid(P(0, -0.012, -0.013), vec(0.0165, 0.019, 0.0165) * size, R), 0.012, group)
        add(f"{name}Instep", RoundCone(P(0, 0.0, 0.0), P(0.001, 0.042, -0.008), 0.0215 * size, 0.0165 * size), 0.012, group)
        add(f"{name}Sole", Ellipsoid(P(0.001, 0.046, -0.0125), vec(0.0265, 0.031, 0.0105) * size, R), 0.012, group)
        # (side, where it leaves the foot, length, thickness, fan in degrees)
        toes = ((0.0178, 0.07, 0.019, 0.0088, 4.0), (0.0052, 0.075, 0.016, 0.0068, 1.0), (-0.0052, 0.074, 0.0145, 0.0063, -3.0),
                (-0.0146, 0.07, 0.013, 0.0059, -8.0), (-0.0228, 0.063, 0.0115, 0.0055, -14.0))
        tips = []
        for i, (su, sa, L, r, fan) in enumerate(toes):
            fan = math.radians(fan)
            d, m = bend(math.cos(fan) * a + math.sin(fan) * u, n, math.radians(14.0))
            p = P(su, sa - 0.006, -0.0125)
            q = p + d * (L + 0.006) * size
            add(f"{name}Digit{i}", RoundCone(p, q, r * size, r * 0.92 * size), 0.0045, group)
            tips.append(q)
        return tips[2]

    def skin_along(self, c, d, hi=0.45):
        """How far from c along the direction d the skin is (the sculpt so far; c inside it)."""
        lo = 0.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if eval_sdf(self.ops, (c + d * mid)[None])[0] < 0:
                lo = mid
            else:
                hi = mid
        return lo

    def build(self):
        add, sub = self.add, self.sub
        ops = self.ops
        # Measured on the clip's own turn: its camera was fitted to the silhouettes all round (it
        # turns at about 15.3 degrees a second and is nearly level), so the exact views are known: the
        # front at 4.4 s, the right side at 10.3 s, the back at 16.2 s and the left side at 21.8 s. The
        # silhouettes of those views (and a visual hull from all of them) were put into model units
        # and the sculpt matched to them row by row; the face, ears and limbs were read off the same
        # frames on a model-unit grid
        ops.junctions = {
            "head": (vec(0, 0.0, -0.33), 0.2, 0.085),
            "armL": (vec(-0.16, -0.06, -0.28), 0.07, 0.035),
            "armR": (vec(0.16, -0.06, -0.29), 0.07, 0.035),
            "legL": (vec(-0.08, -0.43, -0.1), 0.075, 0.04),
            "legR": (vec(0.08, -0.43, -0.1), 0.075, 0.04),
        }
        self.dots = []
        self.bones = []

        # ---- head: very large and round, nearly half the body, sitting low between the shoulders
        # (the chin rests on the chest). A tall dome: the forehead and the crown make up about three
        # quarters of its height; the face is small and low, under a high, rounded forehead; the
        # back of the head runs straight down into the back
        add("cranium", Ellipsoid(vec(0, 0.25, -0.125), vec(0.17, 0.22, 0.233)), 0.08, "head")
        # (the crown is broad and fairly flat, not pointed)
        add("crown", Ellipsoid(vec(0, 0.365, -0.12), vec(0.12, 0.09, 0.15)), 0.06, "head")
        add("occiput", Ellipsoid(vec(0, 0.11, -0.2), vec(0.155, 0.16, 0.15)), 0.08, "head")
        # the face, low and small: eyes at a fifth of the head's height above the chin, a short,
        # narrow lower face (no full cheeks), a small nose, a small mouth and a little chin that sits
        # back; under it the jaw runs back nearly level into the neck
        add("midface", Ellipsoid(vec(0, 0.05, -0.06), vec(0.085, 0.07, 0.09)), 0.055, "head")
        add("lowerFace", Ellipsoid(vec(0, 0.002, -0.065), vec(0.06, 0.035, 0.07)), 0.05, "head")
        for s, side in ((1, "R"), (-1, "L")):
            add(f"cheek{side}", Ellipsoid(vec(0.055 * s, 0.045, -0.04), vec(0.036, 0.036, 0.04)), 0.05, "head")
        # (a broad, low, flat little nose with a round tip)
        add("noseBridge", Ellipsoid(vec(0, 0.066, 0.006), vec(0.018, 0.026, 0.02)), 0.022, "head")
        add("nose", Ellipsoid(vec(0, 0.037, 0.014), vec(0.026, 0.016, 0.019)), 0.012, "head")
        for s, side in ((1, "R"), (-1, "L")):
            add(f"noseWing{side}", Ellipsoid(vec(0.02 * s, 0.029, 0.004), vec(0.013, 0.01, 0.012)), 0.01, "head")
        add("lip", Ellipsoid(vec(0, 0.011, -0.008), vec(0.028, 0.009, 0.015)), 0.008, "head")
        add("lipLower", Ellipsoid(vec(0, -0.007, -0.012), vec(0.023, 0.009, 0.014)), 0.007, "head")
        add("chin", Ellipsoid(vec(0, -0.02, -0.045), vec(0.03, 0.016, 0.025)), 0.022, "head")
        sub("mouth", Ellipsoid(vec(0, 0.002, self.surface_z(0.0, 0.002) - 0.003), vec(0.024, 0.0018, 0.006)), 0.004, "head")
        nz = self.surface_z(0.011, 0.026)
        for s, side in ((1, "R"), (-1, "L")):
            sub(f"nostril{side}", Ellipsoid(vec(0.011 * s, 0.026, nz - 0.0004), vec(0.003, 0.0018, 0.002)), 0.003, "head")

        # ---- the shut eyes: large, smooth, rounded lids, wide apart, low on the face (just above
        # the nose), and the line where the lids have closed across their lower part
        for s, side in ((1, "R"), (-1, "L")):
            ex, ey = 0.07 * s, 0.095
            out = unit(vec(0.38 * s, -0.08, 1.0))
            c = vec(ex, ey, self.surface_z(ex, ey) - 0.02)
            R = frame(out, vec(0, 1, 0))
            add(f"eyeLid{side}", Ellipsoid(c, vec(0.044, 0.029, 0.025), R), 0.022, "head")
            u, v = R[:, 0] * (1 if R[:, 0][0] * s > 0 else -1), R[:, 1]
            pts = []
            for t in np.linspace(-1, 1, 9):
                d = unit(out + u * (1.05 * t) + v * (-0.22 - 0.1 * (1 - t * t)))
                p = c + d * self.skin_along(c, d)
                pts.append((p - d * (0.001 + 0.0005 * t * t), 0.0016))
            sub(f"lidLine{side}", Chain([RoundCone(a[0], b[0], a[1], b[1]) for a, b in zip(pts[:-1], pts[1:])]), 0.003, "head")

        # ---- the outer ear: far back on the side of the head (nearer the back than the face), low,
        # behind the jaw: a thick C-shaped rim from the top round the back and down to the lobe, open
        # towards the face, round a hollow, with a small bump in front of it
        ear_y, ear_z = 0.115, -0.215
        ear_x = self.surface_x(ear_y, ear_z)
        rim = [(0.029, 0.01, 0.0065), (0.032, -0.003, 0.0082), (0.024, -0.015, 0.0096), (0.007, -0.02, 0.0102),
               (-0.012, -0.0185, 0.0102), (-0.026, -0.011, 0.0098), (-0.034, 0.0, 0.009)]
        # (the rim as a smooth curve through those points, not a line with corners)
        from scipy.interpolate import CubicSpline
        t = np.linspace(0, 1, len(rim))
        fine = CubicSpline(t, np.array(rim))(np.linspace(0, 1, 25))
        for s, side in ((1, "R"), (-1, "L")):
            c = vec((ear_x - 0.002) * s, ear_y, ear_z)
            add(f"earBase{side}", Ellipsoid(c + vec(-0.004 * s, -0.002, -0.002), vec(0.012, 0.038, 0.026)), 0.012, "head")
            pts = [(c + vec(0.005 * s, dy, dz), r) for dy, dz, r in fine]
            add(f"ear{side}", Chain([RoundCone(a[0], b[0], a[1], b[1]) for a, b in zip(pts[:-1], pts[1:])]), 0.007, "head")
            sub(f"earBowl{side}", Ellipsoid(c + vec(0.007 * s, 0.0, 0.002), vec(0.007, 0.019, 0.01)), 0.004, "head")
            add(f"earTragus{side}", Ellipsoid(c + vec(0.002 * s, -0.01, 0.015), vec(0.006, 0.008, 0.005)), 0.004, "head")
        self.anchors["ear"] = vec(ear_x + 0.02, ear_y, ear_z)
        vy, vz = 0.24, -0.06
        self.anchors["vessels"] = vec(self.surface_x(vy, vz), vy, vz)
        self.anchors["skull"] = vec(0, 0.25, -0.125)

        # ---- trunk: a long egg under the head, broad at the shoulders (the neck shows only from
        # behind, narrower than the head), its back a smooth, nearly straight line that rounds off
        # into a low, full bottom with the buttocks
        # (one body of smooth sections, row by row: height, back, front, half width. The outline is
        # the one measured on the clip; the trunk itself is narrower than the whole silhouette
        # where the arms hang beside it and the thighs lie against it, so they read as limbs on a
        # body and not as tubes growing out of a sack)
        add("trunk", Loft(self.TRUNK), 0.05, "torso")
        # (the neck carries on up inside the head, slimming: the back of the head runs straight
        # down into the back, and the join shows no line)
        add("neck", RoundCone(vec(0, -0.03, -0.25), vec(0, 0.16, -0.21), 0.128, 0.035), 0.05, "torso")
        # (the shoulders slope down from the neck to the arms: round, not a straight bar)
        for s, side in ((1, "R"), (-1, "L")):
            add(f"shoulderSlope{side}", Ellipsoid(vec(0.118 * s, -0.07, -0.25), vec(0.066, 0.052, 0.06)), 0.04, "torso")
            add(f"buttock{side}", Ellipsoid(vec(0.055 * s, -0.495, -0.105), vec(0.078, 0.07, 0.088)), 0.05, "torso")
        sub("cleft", Ellipsoid(vec(0, -0.55, -0.16), vec(0.006, 0.05, 0.065)), 0.012, "torso")
        cord_a, cord_b = vec(0, -0.27, -0.03), vec(0.02, -0.285, 0.07)
        add("cordRoot", RoundCone(cord_a, cord_b, 0.048, 0.038), 0.04, "torso")
        self.cord_r = 0.038
        self.cord_tube = 0.035

        back = [(float(y), self.back_z(0.0, y)) for y in np.arange(0.0, -0.52, -0.05)]

        # ---- limbs, where the clip has them. The shoulders are set well back; each upper arm hangs
        # down the side of the chest to an elbow low and back, and the forearm comes forwards and
        # up. The right hand rests under the chin (fingertips on it, the back of the hand
        # forwards), the left one is in front of the chest, fingers up. The legs are drawn up: the
        # thighs lie against the lower belly from hips at its sides, the knees high and forwards,
        # the shins going down to the feet low in front of the bottom, crossed at the ankles
        self.joints = {}
        up = vec(0, 1, 0)
        # (shoulder, how far out the elbow is held (degrees from hanging straight down), wrist, the
        # way the hand points, the back of the hand, the thumb's side, how far the fingers curl and
        # fan). Both arms are the same length, upper arm and forearm nearly equal: the elbow is
        # found from the shoulder and the wrist, held just far enough out for the arm to lie on
        # the side of the chest and not in it
        arms = {1: (vec(0.16, -0.06, -0.29), 18.0, vec(0.126, -0.101, -0.045),
                    unit(vec(-0.77, 0.544, 0.333)), vec(0.4, 0.1, 1.6), vec(0, 1, 0), 0.45, 0.6),
                -1: (vec(-0.16, -0.06, -0.28), 18.0, vec(-0.165, -0.13, 0.0),
                     unit(vec(0.045, 0.09, 0.085)), vec(-0.5, 0.0, 1.0), vec(-1, 0, -0.3), 0.9, 1.0)}
        # (hip, knee, ankle, the way the foot points, the top of the foot)
        legs = {1: (vec(0.1, -0.435, -0.08), vec(0.1, -0.3, 0.13), vec(0.07, -0.5, 0.15),
                    unit(vec(-0.35, -0.4, 0.8)), vec(0.3, 0.9, 0.2)),
                -1: (vec(-0.1, -0.435, -0.08), vec(-0.075, -0.28, 0.08), vec(-0.01, -0.48, 0.2),
                     unit(vec(0.2, -0.3, 0.9)), vec(-0.3, 0.9, 0.2))}
        for s, side in ((1, "R"), (-1, "L")):
            arm = f"arm{side}"
            sh, out, wr, axis, back_of_hand, first, curl, spread = arms[s]
            el = self.elbow(sh, wr, s, out, self.UPPER_ARM, self.FOREARM)
            self.arm(add, side, sh, el, wr)
            tips = self.hand(add, f"hand{side}", arm, wr, axis, back_of_hand, first, curl=curl, spread=spread)
            self.bones += [(RoundCone(sh + (el - sh) * 0.15, el - (el - sh) * 0.1, 0.0075, 0.007), 40.0)]
            fa = unit(np.cross(wr - el, up)) * 0.006
            self.bones += [(RoundCone(el + (wr - el) * 0.1 + d, wr - (wr - el) * 0.08 + d, 0.005, 0.0045), 40.0) for d in (fa, -fa)]
            leg = f"leg{side}"
            hip, kn, an, faxis, dorsum = legs[s]
            self.leg(add, side, hip, kn, an)
            ftips = self.foot(add, f"foot{side}", leg, an, faxis, dorsum, first=vec(-s, 0, 0))
            self.bones += [(RoundCone(hip + (kn - hip) * 0.2, kn - (kn - hip) * 0.08, 0.009, 0.008), 40.0)]
            fl = unit(np.cross(an - kn, up)) * 0.006
            self.bones += [(RoundCone(kn + (an - kn) * 0.12 + d, an - (an - kn) * 0.08 + d, 0.0055, 0.005), 40.0) for d in (fl, -fl * 0.6)]
            self.anchors[f"hand{side}"] = wr + axis * 0.06
            self.anchors[f"foot{side}"] = an + faxis * 0.04
            self.anchors[f"knee{side}"] = kn + vec(0.0, 0.01, 0.03)
            # (a touch lifts the arm towards the face, but the hand that is already at the mouth
            # moves down and away from it instead, so it never goes through the face)
            lift = unit(np.cross(unit(wr - sh), up))
            self.joints[arm] = {"pivot": sh, "axis": -lift if s > 0 else lift, "dir": unit(tips - sh), "plate": wr}
            self.joints[leg] = {"pivot": hip, "axis": unit(np.cross(unit(an - hip), up)), "dir": unit(ftips - hip), "plate": an}

        # ---- inside, for the light through the body: the brain fills most of the head; the brain
        # stem and the spinal cord run down the back with the first vertebrae beside it, two darker
        # patches high on the back either side of the spine (the shoulder blades, starting to
        # harden), six ribs below them; the heart and the liver in front, faint
        self.organs = [
            (Ellipsoid(vec(0, -0.16, -0.12), vec(0.065, 0.055, 0.06)), 1.5),     # heart
            (Ellipsoid(vec(0, -0.27, -0.09), vec(0.12, 0.08, 0.09)), 1.5),      # liver
            (RoundCone(vec(0, 0.12, -0.26), vec(0, -0.02, -0.3), 0.04, 0.028), 8.0),   # brain stem
        ]
        for c in (vec(0.07, 0.27, -0.12), vec(-0.07, 0.27, -0.12)):
            for k, absorb in ((1.0, 1.6), (0.75, 2.0), (0.5, 2.4)):
                self.organs.append((Ellipsoid(c, vec(0.09, 0.13, 0.15) * k), absorb))
        self.organs.append((Ellipsoid(vec(0, 0.15, -0.26), vec(0.085, 0.08, 0.07)), 3.0))
        for s in (1, -1):
            self.organs.append((Ellipsoid(vec(0.085 * s, -0.12, -0.33), vec(0.037, 0.035, 0.024)), 14.0))
        cord_pts = [vec(0, -0.02, -0.3)] + [vec(0, y, z + 0.06) for y, z in back[1:]]
        for a_, b_ in zip(cord_pts[:-1], cord_pts[1:]):
            self.organs.append((RoundCone(a_, b_, 0.022, 0.022), 6.0))
        for (ya, za), (yb, zb) in zip(back[1:-1], back[2:]):
            for t in (0.0, 0.5):
                y_, z_ = ya + (yb - ya) * t, za + (zb - za) * t
                for s in (1, -1):
                    self.organs.append((Ellipsoid(vec(0.022 * s, y_, z_ + 0.05), vec(0.008, 0.008, 0.008)), 5.0))
        for y in np.linspace(-0.19, -0.35, 6):
            c = vec(0, y, -0.18)
            for s_ in (1, -1):
                for th in np.linspace(0.35, 1.75, 9):
                    d = vec(s_ * math.sin(th), 0, -math.cos(th))
                    self.organs.append((Ellipsoid(c + d * (self.skin_along(c, d) - 0.035), vec(0.015, 0.008, 0.015)), 11.0))

        # (the nod lifts the head a little: nodding down, the face would run into the hand at the mouth)
        self.joints["head"] = {"pivot": vec(0, -0.03, -0.25), "axis": vec(-1, 0, 0), "dir": unit(vec(0, 0.6, 0.3))}
        # (the heart shows only faintly through the chest now)
        self.heart = (vec(0, -0.15, -0.08), vec(0.07, 0.035, 0.03), 0.35)
        self.dots = []

        self.anchors.update({
            "head": vec(0.1, 0.43, -0.1),
            "face": vec(0, 0.05, -0.02),
            "faceFront": vec(0, -0.05, 0.3),
            "heart": vec(0.05, -0.15, -0.02),
            "navel": cord_b + vec(0, 0, 0.03),
            "cordBase": cord_a,
            # (the page's feet view looks from this point out past the foot: here between the crossed
            # shins, so the feet are seen from the side, toes and heel, not end on)
            "groin": vec(-0.08, -0.47, 0.17),
            "core": vec(0, -0.12, -0.12),
        })
        self.anchors["hand"] = self.anchors["handL"]
        self.anchors["knee"] = self.anchors["kneeL"]
        self.anchors["footL"] = self.anchors.pop("footL")
        self.anchors["footR"] = self.anchors.pop("footR")
        return self

    def back_z(self, x, y):
        lo_z, hi_z = -0.6, -0.15
        for _ in range(30):
            mid = 0.5 * (lo_z + hi_z)
            if eval_sdf(self.ops, np.array([[x, y, mid]]))[0] < 0:
                hi_z = mid
            else:
                lo_z = mid
        return hi_z


def organ_volume(organs, h=0.012, blur=1.0, second=None):
    """The organs' light absorbance as a small 3D texture: soft-edged and blurred a little more, so
    they read as shapes inside the body. Returns the texture's lower corner and size (covering whole
    texels), the uint8 volume with x varying fastest, and the absorbance per unit length at 255.
    With a second list of shapes the volume has two channels (interleaved, one scale for both)."""
    from scipy.ndimage import gaussian_filter
    if second:
        return _organ_volume2(organs, second, h, blur)
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


def _organ_volume2(first, second, h, blur):
    from scipy.ndimage import gaussian_filter
    boxes = [shape.aabb() for shape, _ in first + second]
    lo = np.min([b[0] for b in boxes], axis=0) - 0.04
    hi = np.max([b[1] for b in boxes], axis=0) + 0.04
    dims = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[i] + h * np.arange(dims[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    chans = []
    for organs in (first, second):
        dens = np.zeros(len(pts))
        for shape, absorb in organs:
            dens += absorb * np.clip(-shape.sdf(pts) / 0.01 + 0.5, 0.0, 1.0)
        chans.append(gaussian_filter(dens.reshape(dims), blur))
    scale = float(max(c.max() for c in chans))
    vol = np.round(np.stack(chans, axis=-1) / scale * 255).astype(np.uint8).transpose(2, 1, 0, 3)
    return lo - 0.5 * h, dims * h, np.ascontiguousarray(vol), scale


def sphere(c, r, rings=32, segs=48, squash=1.0, axis=None):
    """A UV sphere, optionally flattened in y (an ellipsoid), with its poles along y or along axis."""
    verts, normals = [], []
    k = np.array([1.0, squash, 1.0])
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segs + 1):
            ph = 2 * math.pi * j / segs
            n = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
            verts.append(r * n * k)
            normals.append(unit(n / k))
    tris = []
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j
            b = a + segs + 1
            tris += [(a, a + 1, b), (a + 1, b + 1, b)]   # counter-clockwise seen from outside
    verts, normals = np.array(verts), np.array(normals)
    if axis is not None:
        R = rotation_between(vec(0, 1, 0), unit(np.asarray(axis, float)))
        verts, normals = verts @ R.T, normals @ R.T
    return c + verts, normals, np.array(tris)


def eye_colours(normals, outward):
    """Black pigment with a soft grey ring round the front, where the lens shows through."""
    cosang = normals @ outward
    # (the eye is set deep, so only the cap round the outward axis shows: keep the ring small in it)
    ring = np.exp(-((cosang - 0.95) / 0.03) ** 2)
    shade = 0.03 + 0.32 * ring
    return np.round(np.clip(np.stack([shade, shade * 0.97, shade], axis=1), 0, 1) * 255).astype(np.uint8)


def write_glb(path, skin, eyes, lids=None):
    """skin: dict(position, normal, color (uint8 RGBA), vessel (float), tris); eyes (None when they
    are shut) the same with an RGB colour and no vessel mask; lids (optional) with an RGBA colour
    whose alpha is the opacity.
    Plain floats: the file is small enough without quantization."""
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
            if "bone" in m:
                attrs["_BONE"] = out.dense(m["bone"].astype(np.float32), {"componentType": 5126, "type": "VEC2", "count": n})
        if "move" in m:
            attrs["_MOVE"] = out.dense(m["move"].astype(np.float32), {"componentType": 5126, "type": "VEC2", "count": n})
        kind, dtype = (5123, np.uint16) if n < 65536 else (5125, np.uint32)
        idx = out.dense(m["tris"].astype(dtype).ravel()[:, None], {"componentType": kind, "type": "SCALAR",
                                                                   "count": int(m["tris"].size)}, indices=True)
        return {"attributes": attrs, "indices": idx}

    gltf = {
        "asset": {"version": "2.0", "generator": "Lunabump build_embryo.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Skin", "mesh": 0}],
        "meshes": [{"name": "Skin", "primitives": [prim(skin)]}],
    }
    # (from week 10 the eyes are shut: no eye meshes)
    for name, m in (("Eyes", eyes), ("Lids", lids)):
        if m is not None:
            gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]))
            gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"])})
            gltf["meshes"].append({"name": name, "primitives": [prim(m, False)]})
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
    weeks = {8: Week8, 9: Week9, 10: Week10}
    if args.week not in weeks:
        raise SystemExit(f"weeks so far: {sorted(weeks)}")
    emb = weeks[args.week]().build()
    ops = emb.ops

    d, lo = sculpt_grid(ops, args.h)
    verts, faces = marching_cubes(d, lo, args.h)
    print(f"grid {d.shape}, marching cubes {len(verts)} verts ({time.time() - t0:.1f}s)")

    def near(prefixes, dist):
        sel = [op for op in ops if op.name.startswith(prefixes)]
        return lambda p: np.min([op.shape.sdf(p) for op in sel], axis=0) < dist

    emb.place_eyes()
    obj = blender_mesh(verts, faces, args.tris, details=tuple((near(p, d), k) for p, d, k in emb.DETAILS))
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
    for name in emb.WARM:
        warm = np.maximum(warm, 0.5 * np.exp(-np.maximum(by[name].shape.sdf(pos), 0) / 0.01))
    shade = (0.6 + 0.4 * ao) * (1 - 0.25 * lid)
    rgb = np.stack([shade, shade * (1 - 0.08 * warm), shade * (1 - 0.1 * warm)], axis=1)
    if emb.FACE_AO:
        face = near(emb.FACE_AO, 0.03)(pos)
        ao2 = np.ones(len(pos))
        ao2[face] = ambient_occlusion(ops, pos[face], nrm[face], dist=0.02)
        for _ in range(4):
            ao2 = 0.5 * ao2 + 0.5 * (A @ ao2) / deg
        rgb *= (0.72 + 0.28 * ao2)[:, None]
    if emb.LIPS:
        # the mouth: a darker, pinker line between the lips, and the nostrils a little darker
        for names, tint, dist in emb.LIPS:
            sel = [op for op in ops if op.name.startswith(names)]
            w = np.exp(-np.maximum(np.min([op.shape.sdf(pos) for op in sel], axis=0), 0) / dist)
            rgb *= 1 - w[:, None] * (1 - np.asarray(tint))

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
    on_cord = w
    if emb.CORD_TONE:
        # (and the colour: without this the shading baked on the root ends in a light band where the
        # tube carries on from it)
        rgb = rgb * (1 - w[:, None]) + w[:, None]
    if emb.CREASES:
        from scipy.spatial import cKDTree
        crease = near(emb.CREASES, 0.008)(pos)
        tree = cKDTree(pos[~crease])
        _, idx = tree.query(pos[crease], k=24)
        alpha[crease] = np.maximum(alpha[crease], np.median(alpha[~crease][idx], axis=1))

    # ---- where the head's surface vessels run
    head_ops = [by[n] for n in emb.HEAD_OPS]
    on_head = np.exp(-np.maximum(np.min([o.shape.sdf(pos) for o in head_ops], axis=0), 0) / 0.01)
    away_from_face = np.array([smoothstep(*emb.VESSEL_Y, y) for y in pos[:, 1]])
    vessel = on_head * away_from_face * emb.VESSEL_GAIN
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
    # run along each limb on one long, soft curve: dark where the limb leaves the body (the body's own
    # tone and thickness), lighter and lighter down the limb, and lightest at the very ends of the
    # fingers and toes. There is no step at the shoulder, the wrist or anywhere in between
    pale = np.zeros(len(pos))
    limb = np.zeros(len(pos), bool)
    from scipy.spatial import cKDTree
    for gi, g in enumerate(groups[1:], start=1):
        if g == "head":
            continue
        j = emb.joints[g]
        sel = owner == gi
        limb |= sel
        along = (pos[sel] - j["pivot"]) @ j["dir"]
        start = np.percentile(along[other[sel]], 97) if other[sel].any() else along.min()
        end = np.percentile(along, 99.5)   # the tips of the fingers or toes
        t = np.clip((along - start) / (end - start), 0, 1)
        t = t * t * (3 - 2 * t)
        pale[sel] = t
        # the limb starts with the thickness of the body next to it (a thin bud would otherwise glow
        # paler than the trunk it grows from) and thins out towards the tips, which let a fair amount
        # of light through, though less than the bare plate would
        rim = pos[sel & other]
        body_rim = (owner == 0) & other
        near = cKDTree(rim).query(pos[body_rim], distance_upper_bound=0.04)[0] < np.inf if len(rim) else np.zeros(int(body_rim.sum()), bool)
        a_root = np.median(alpha[body_rim][near]) if near.any() else alpha[sel].max()
        tip = along > end - 0.04
        a_tip = np.median(alpha[sel][tip]) if tip.any() else alpha[sel].min()
        a_tip = a_root + (a_tip - a_root) * emb.TIP_THIN
        alpha[sel] = a_root + (a_tip - a_root) * t
    # soften both over the surface, the thickness also a little way into the body round each limb
    near_limb = limb.astype(float)
    for _ in range(6):
        near_limb = np.maximum(near_limb, (A @ near_limb) / deg)
    near_limb = near_limb > 0.01
    for _ in range(40):
        pale = 0.5 * pale + 0.5 * (A @ pale) / deg
        alpha = np.where(near_limb, 0.5 * alpha + 0.5 * (A @ alpha) / deg, alpha)

    # ---- the small bones of the fingers and toes: dots seen on the back and the palm side alike
    bone = np.zeros(len(pos))
    for dot in emb.dots:
        c, r, n = dot[:3]
        d = pos - c
        dn = d @ n
        if len(dot) > 3:
            # a short, soft-cornered dash along the digit (as the clip draws them)
            along = unit(dot[3] - n * (dot[3] @ n))
            across = np.cross(n, along)
            x, y = np.abs(d @ along) / (r * 1.45), np.abs(d @ across) / (r * 0.95)
            bone = np.maximum(bone, np.exp(-(x ** 3 + y ** 3)) * np.exp(-(dn / 0.025) ** 2))
        else:
            dp = np.linalg.norm(d - dn[:, None] * n, axis=1)
            bone = np.maximum(bone, np.exp(-(dp / r) ** 2) * np.exp(-(dn / 0.025) ** 2))
    bone = np.clip(bone / 0.8, 0, 1)

    # ---- the organs inside, as a 3D texture (and the limbs' long bones in a second channel)
    vol_lo, vol_size, vol, vol_scale = organ_volume(emb.organs, blur=emb.ORGAN_BLUR, second=emb.bones or None)
    print(f"organ volume {vol.shape[:3][::-1]} ({vol.size / 1e3:.0f} kB), absorbance up to {vol_scale:.0f}")

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
    if emb.dots or emb.bones:
        # (with how much of the organs' shadow to leave out: none of it on the root of the cord)
        skin["bone"] = np.stack([np.clip(bone, 0, 1), on_cord], axis=1).round(3)
    ev, en, et, ec = [], [], [], []
    for c, r in emb.eyes:
        if emb.EYE_ROUND:
            v_, n_, t_ = sphere(c - center, r, rings=72, segs=64, squash=emb.EYE_SQUASH, axis=emb.eye_dir(np.sign(c[0])))
        else:
            v_, n_, t_ = sphere(c - center, r, squash=emb.EYE_SQUASH)
        et.append(t_ + sum(len(x) for x in ev))
        ev.append(v_)
        en.append(n_)
        # the eye looks out sideways and a little forwards
        ec.append(emb.eye_colours(n_, emb.eye_dir(np.sign(c[0]))))
    eyes = None
    if emb.eyes:
        eyes = {"position": np.concatenate(ev), "normal": np.concatenate(en), "tris": np.concatenate(et),
                "color": np.concatenate(ec)}
        eyes["move"] = np.tile([5.0, 1.0], (len(eyes["position"]), 1))
    lids = None
    if emb.LIDS:
        lv, ln, lt, la = [], [], [], []
        for c, r in emb.eyes:
            v_, n_, t_, a_ = emb.lid(c - center, r, emb.eye_dir(np.sign(c[0])))
            lt.append(t_ + sum(len(x) for x in lv))
            lv.append(v_)
            ln.append(n_)
            la.append(a_)
        alpha = np.concatenate(la)
        lids = {"position": np.concatenate(lv), "normal": np.concatenate(ln), "tris": np.concatenate(lt),
                "color": np.concatenate([np.full((len(alpha), 3), 255, np.uint8), np.round(alpha * 255).astype(np.uint8)[:, None]], axis=1)}
        lids["move"] = np.tile([5.0, 1.0], (len(alpha), 1))
    write_glb(args.out, skin, eyes, lids)

    import base64
    rig = {"week": args.week, "center": center.round(5).tolist(),
           "anchors": {k: (v - center).round(5).tolist() for k, v in emb.anchors.items()},
           # what each part swings round when touched: the joint, and the axis it turns about
           "joints": {g: {"part": parts[g], "pivot": (j["pivot"] - center).round(5).tolist(),
                          "axis": np.asarray(j["axis"]).round(4).tolist()} for g, j in emb.joints.items()},
           # x varies fastest, then y, then z; "scale" is the absorbance per CRL at a texel of 255
           "organs": {"min": (vol_lo - center).round(5).tolist(), "size": vol_size.round(5).tolist(),
                      "dims": list(vol.shape[:3][::-1]), "scale": round(vol_scale, 2),
                      "data": base64.b64encode(vol.tobytes()).decode("ascii")}}
    if vol.ndim == 4:
        # (the second channel, interleaved: the long bones of the limbs)
        rig["organs"]["channels"] = 2
    if emb.ORGAN_STEP:
        rig["organs"]["step"] = emb.ORGAN_STEP
    # the cord tube's thickness where it leaves the root (it sinks into the root at that thickness, so
    # the two surfaces meet only at the root's tip, at a slight angle, and not along a ragged line)
    rig["cordRadius"] = emb.cord_r
    if getattr(emb, "cord_tube", None):
        # (and how thick the tube itself is further out, where it is not the page's own default)
        rig["cordTube"] = emb.cord_tube
    if emb.PALE is not None:
        rig["pale"] = emb.PALE
    if emb.LOOK:
        rig["look"] = emb.LOOK
    if getattr(emb, "heart", None):
        rig["heart"] = {"center": (emb.heart[0] - center).round(5).tolist(), "radii": np.asarray(emb.heart[1]).round(4).tolist(),
                        "strength": emb.heart[2]}
    with open(args.rig, "w") as fh:
        json.dump(rig, fh, separators=(",", ":"))
    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) and {args.rig} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

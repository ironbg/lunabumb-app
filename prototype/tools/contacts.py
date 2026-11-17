"""Where the limbs run into the body or each other: for the middle line of every limb shape, how
far the other parts' surface is from it, against its own radius there (depth > 0: it sinks in).
  python contacts.py [share of the radius to report from, default 0.3]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import build_embryo as be
from build_fetus import eval_group, Sculpt
emb = be.Week10().build()
ops = emb.ops
groups = {g: [o for o in ops if o.group == g] for g in Sculpt.GROUPS}
def pts_of(shape):
    if shape.kind == 'ell': return [(shape.c, float(shape.r.min()))]
    if shape.kind == 'cone': return [(shape.a + (shape.b - shape.a) * t, shape.ra + (shape.rb - shape.ra) * t) for t in (0, 0.5, 1)]
    if shape.kind == 'chain': return [x for c in shape.cones for x in pts_of(c)]
    return []
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
for g in ('armR', 'armL', 'legR', 'legL'):
    for o in groups[g]:
        if o.mode != 'add': continue
        worst = None
        for c, r in pts_of(o.shape):
            for h in Sculpt.GROUPS:
                if h == g: continue
                d = eval_group(groups[h], np.asarray(c)[None])[0]
                depth = (r - d) / r     # share of the radius sunk into the other part
                if worst is None or depth > worst[0]: worst = (depth, h, d, r)
        if worst and worst[0] > thr:
            print(f"{g:5s} {o.name:22s} sinks {worst[0]*100:5.0f}% of its radius into {worst[1]} (axis {worst[2]:+.4f} from its skin, r {worst[3]:.4f})")
print('elbows', [ (o.name, [c.a.round(3).tolist() for c in o.shape.cones][3]) for o in ops if o.name.startswith('armBud')])

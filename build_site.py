"""Procedural construction-site diorama for the in-class hazard hunt.
Site units = metres, Z up, X east, Y north.  Base plate 60 x 60 m -> exported at 1:100 (0.6 m table-top).
Grid: columns A-F west->east (10 m each), rows 1-6 north->south (10 m each).
"""
import math, json, os, sys
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as RM, translation_matrix as TM
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ----------------------------------------------------------------- materials
MATS = {
    'concrete': ((168, 170, 172), .92, 0), 'concrete_dk': ((120, 122, 126), .92, 0),
    'steel': ((112, 124, 140), .45, .6), 'steel_lt': ((170, 176, 184), .5, .5),
    'yellow': ((250, 196, 20), .6, 0), 'orange': ((246, 110, 20), .6, 0),
    'red': ((214, 40, 34), .6, 0), 'white': ((238, 238, 236), .6, 0),
    'green': ((26, 150, 80), .6, 0), 'blue': ((38, 92, 178), .6, 0),
    'brick': ((176, 78, 52), .9, 0), 'wood': ((190, 142, 86), .9, 0), 'wood_dk': ((120, 86, 52), .9, 0),
    'soil': ((122, 88, 58), 1, 0), 'soil_dk': ((92, 66, 44), 1, 0), 'black': ((24, 24, 27), .8, 0),
    'skin': ((232, 184, 150), .8, 0), 'hair': ((50, 32, 22), .9, 0), 'pants': ((40, 52, 86), .9, 0),
    'shirt': ((104, 120, 138), .9, 0), 'vest': ((255, 120, 20), .7, 0), 'helmet': ((250, 250, 245), .5, 0),
    'helmet_y': ((255, 214, 0), .5, 0), 'glass': ((140, 190, 220), .2, 0), 'copper': ((205, 116, 60), .4, .6),
    'cable': ((235, 190, 30), .7, 0), 'rust': ((150, 84, 56), .8, .2), 'water': ((70, 120, 180), .1, .1),
    'cont_blue': ((46, 100, 160), .7, .1), 'cont_green': ((60, 128, 94), .7, .1), 'rubber': ((30, 30, 32), .95, 0),
    'ox_blue': ((40, 80, 190), .5, .3), 'ac_red': ((200, 40, 40), .5, .3), 'spark': ((255, 170, 40), .3, 0),
    'ply': ((214, 170, 110), .9, 0), 'pole': ((110, 80, 56), .95, 0), 'signgreen': ((20, 130, 70), .6, 0),
}
PARTS = defaultdict(list)
PLAN = []                      # (polygon, ztop, rgb)
STACK = [np.eye(4)]


def push(M): STACK.append(STACK[-1] @ M)
def pop(): STACK.pop()


def record(m, mat):
    b = m.bounds
    if (b[1][0] - b[0][0]) < .3 and (b[1][1] - b[0][1]) < .3: return
    if b[1][2] <= .05: return
    pts = m.vertices[:, :2]
    try:
        h = ConvexHull(pts); poly = pts[h.vertices]
    except Exception:
        return
    PLAN.append((poly, float(b[1][2]), MATS[mat][0]))


def add(mat, mesh, plan=True):
    m = mesh.copy(); m.apply_transform(STACK[-1]); PARTS[mat].append(m)
    if plan: record(m, mat)


def box(mat, c, s, rz=0, plan=True):
    m = trimesh.creation.box(extents=s); m.apply_transform(TM(c) @ RM(rz, [0, 0, 1])); add(mat, m, plan)


def rod(mat, p0, p1, r=.05, sec=8, plan=False):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    if np.linalg.norm(p1 - p0) < 1e-6: return
    m = trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=sec); add(mat, m, plan)


def cyl(mat, c, r, h, axis='z', sec=16, plan=True):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sec)
    if axis == 'x': m.apply_transform(RM(math.pi / 2, [0, 1, 0]))
    if axis == 'y': m.apply_transform(RM(-math.pi / 2, [1, 0, 0]))
    m.apply_transform(TM(c)); add(mat, m, plan)


def sph(mat, c, r, sub=2, plan=False, sz=1.0):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    if sz != 1: m.apply_transform(np.diag([1, 1, sz, 1]))
    m.apply_transform(TM(c)); add(mat, m, plan)


def hemi(mat, c, r, plan=False):
    m = trimesh.creation.icosphere(subdivisions=2, radius=r)
    m = m.slice_plane([0, 0, -r * .05], [0, 0, 1], cap=False)
    m.apply_transform(TM(c)); add(mat, m, plan)


def cone(mat, c, r, h, sec=28, sx=1, sy=1, plan=True):
    m = trimesh.creation.cone(radius=r, height=h, sections=sec)
    m.apply_transform(np.diag([sx, sy, 1, 1])); m.apply_transform(TM(c)); add(mat, m, plan)


# ----------------------------------------------------------------- helpers
def cell(x, y):
    col = 'ABCDEF'[min(5, max(0, int(x // 10)))]
    row = 6 - min(5, max(0, int(y // 10)))
    return f'{col}{row}'


def person(x, y, z, face=0, helmet='white', vest=True, harness=False, arm_up=False, k=1.5, shirt='shirt'):
    push(TM([x, y, z]) @ RM(face, [0, 0, 1]) @ np.diag([k, k, k, 1]))
    box('pants', [-.1, 0, .43], [.17, .2, .86]); box('pants', [.1, 0, .43], [.17, .2, .86])
    box('vest' if vest else shirt, [0, 0, 1.17], [.42, .25, .62])
    if vest: box('yellow', [0, .13, 1.17], [.43, .02, .07], plan=False)
    box(shirt, [-.3, 0, 1.15], [.12, .14, .56])
    if arm_up:
        rod('shirt', [.27, 0, 1.4], [.45, 0, 2.0], .06); box('orange', [.5, 0, 2.1], [.34, .03, .26], plan=False)
    else:
        box(shirt, [.3, 0, 1.15], [.12, .14, .56])
    sph('skin', [0, 0, 1.58], .12, sub=1)
    if helmet: hemi('helmet' if helmet == 'white' else 'helmet_y', [0, 0, 1.6], .16)
    else: hemi('hair', [0, 0, 1.6], .13)
    if harness:
        box('red', [0, .0, 1.2], [.44, .27, .05], plan=False); box('red', [0, .0, 1.0], [.44, .27, .05], plan=False)
    pop()


def pallet(x, y, z=0, rz=0, s=(1.2, .8)):
    box('wood', [x, y, z + .075], [s[0], s[1], .15], rz)


def guardrail(p0, p1, z, sk=None, post_gap=2.4):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    L = np.linalg.norm(p1 - p0); n = max(1, int(round(L / post_gap)))
    d = (p1 - p0) / L
    for i in range(n + 1):
        t = i / n; pt = p0 + d * L * t
        if sk and sk(pt): continue
        rod('yellow', [*pt, z], [*pt, z + 1.1], .06)
    rod('yellow', [*p0, z + 1.05], [*p1, z + 1.05], .04)
    rod('yellow', [*p0, z + .55], [*p1, z + .55], .04)
    mid = (p0 + p1) / 2; ang = math.atan2(d[1], d[0])
    box('yellow', [*mid, z + .08], [L, .04, .16], ang, plan=False)


def barrier(p0, p1, z=0):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    L = np.linalg.norm(p1 - p0); n = max(1, int(round(L / 2.0))); d = (p1 - p0) / L; ang = math.atan2(d[1], d[0])
    for i in range(n + 1):
        pt = p0 + d * L * i / n; rod('steel_lt', [*pt, z], [*pt, z + 1.1], .06)
    seg = int(L / .6)
    for i in range(seg):
        a = p0 + d * L * i / seg; b = p0 + d * L * (i + 1) / seg; mid = (a + b) / 2
        box('red' if i % 2 == 0 else 'white', [*mid, z + 1.0], [L / seg, .06, .16], ang, plan=False)
        box('white' if i % 2 == 0 else 'red', [*mid, z + .6], [L / seg, .06, .12], ang, plan=False)


def slab(zt, x0, y0, x1, y1, hole=None, th=.3):
    zc = zt - th / 2
    if not hole:
        box('concrete', [(x0 + x1) / 2, (y0 + y1) / 2, zc], [x1 - x0, y1 - y0, th]); return
    hx0, hy0, hx1, hy1 = hole
    box('concrete', [(x0 + x1) / 2, (y0 + hy0) / 2, zc], [x1 - x0, hy0 - y0, th])
    box('concrete', [(x0 + x1) / 2, (hy1 + y1) / 2, zc], [x1 - x0, y1 - hy1, th])
    box('concrete', [(x0 + hx0) / 2, (hy0 + hy1) / 2, zc], [hx0 - x0, hy1 - hy0, th])
    box('concrete', [(hx1 + x1) / 2, (hy0 + hy1) / 2, zc], [x1 - hx1, hy1 - hy0, th])


# ----------------------------------------------------------------- ground
SIZE = 60.0
PIT = (3.0, 28.0, 17.0, 44.0)


def make_texture(px=40):
    N = int(SIZE * px)
    base = np.full((N, N, 3), (190, 184, 168), np.uint8)
    noise = rng.integers(-7, 8, (N, N, 1)); base = np.clip(base + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(base, 'RGB'); d = ImageDraw.Draw(im, 'RGBA')
    X = lambda x: x * px; Y = lambda y: (SIZE - y) * px
    def rect(x0, y0, x1, y1, fill, outline=None, w=0):
        d.rectangle([X(x0), Y(y1), X(x1), Y(y0)], fill=fill, outline=outline, width=w)
    rect(0, 0, 60, 9, (74, 76, 80, 255))                         # road
    for xx in np.arange(3, 58, 5): rect(xx, 5.35, xx + 2.4, 5.65, (240, 230, 120, 255))
    rect(18, 9, 20.5, 28.5, (62, 150, 96, 255)); rect(17.85, 9, 18.05, 28.5, (255, 255, 255, 255)); rect(20.45, 9, 20.65, 28.5, (255, 255, 255, 255))
    rect(20.5, 27.2, 22.2, 28.5, (62, 150, 96, 255))
    for yy in np.arange(11, 28, 3.2):
        d.polygon([(X(19.25), Y(yy + 1.6)), (X(18.4), Y(yy)), (X(20.1), Y(yy))], fill=(255, 255, 255, 235))
    rect(21, 29, 47, 45, (150, 150, 152, 255))                     # building pad
    rect(40, 10, 52, 15.5, (166, 160, 146, 255), (235, 190, 30, 255), 6)   # laydown
    rect(46, 30, 54, 44, (172, 166, 152, 255))
    # mud patch around excavation
    d.ellipse([X(0.5), Y(48), X(20), Y(24)], fill=(128, 98, 70, 90))
    # grid + labels
    f_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
    f_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 78)
    for i in range(1, 6):
        d.line([X(i * 10), 0, X(i * 10), N], fill=(20, 30, 50, 120), width=4)
        d.line([0, Y(i * 10), N, Y(i * 10)], fill=(20, 30, 50, 120), width=4)
    for c in range(6):
        for r in range(6):
            d.text((X(c * 10 + .55), Y((5 - r) * 10 + 10) + 8), f'{"ABCDEF"[c]}{r + 1}', font=f_small, fill=(20, 30, 50, 150))
    band = (30, 41, 59, 255); bw = 2.2
    rect(0, 0, 60, bw, band); rect(0, 60 - bw, 60, 60, band); rect(0, 0, bw, 60, band); rect(60 - bw, 0, 60, 60, band)
    for c in range(6):
        for yy in (bw / 2, 60 - bw / 2):
            t = 'ABCDEF'[c]; w = d.textlength(t, font=f_big)
            d.text((X(c * 10 + 5) - w / 2, Y(yy) - 46), t, font=f_big, fill=(255, 255, 255, 255))
    for r in range(6):
        for xx in (bw / 2, 60 - bw / 2):
            t = str(r + 1); w = d.textlength(t, font=f_big)
            d.text((X(xx) - w / 2, Y(60 - (r * 10 + 5)) - 46), t, font=f_big, fill=(255, 255, 255, 255))
    # north arrow in NE corner
    cx, cy = X(55), Y(54.5)
    d.polygon([(cx, cy - 70), (cx - 34, cy + 40), (cx, cy + 18), (cx + 34, cy + 40)], fill=(255, 255, 255, 255), outline=(30, 41, 59, 255))
    d.text((cx - 22, cy + 46), 'N', font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 60), fill=(30, 41, 59, 255))
    path = os.path.join(OUT, 'ground.jpg'); im.save(path, quality=90)
    return im, path


def build_ground():
    im, path = make_texture()
    # soil block
    box('soil_dk', [30, 30, -7.5], [60, 60, 3], plan=False)
    box('soil', [30, 30, -4.5], [60, 60, 3], plan=False)
    x0, y0, x1, y1 = PIT
    for (a, b, c, d_) in [(0, 0, x0, 60), (x1, 0, 60, 60), (x0, 0, x1, y0), (x0, y1, x1, 60)]:
        box('soil', [(a + c) / 2, (b + d_) / 2, -1.5], [c - a, d_ - b, 3], plan=False)
    # textured top quads
    verts, faces, uvs = [], [], []
    for (a, b, c, d_) in [(0, 0, x0, 60), (x1, 0, 60, 60), (x0, 0, x1, y0), (x0, y1, x1, 60)]:
        n = len(verts)
        for (px_, py_) in [(a, b), (c, b), (c, d_), (a, d_)]:
            verts.append([px_, py_, .002]); uvs.append([px_ / SIZE, py_ / SIZE])
        faces += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
    img = Image.open(path)
    tex = trimesh.visual.TextureVisuals(uv=np.array(uvs), material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=img, roughnessFactor=1.0, metallicFactor=0.0))
    gm = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), visual=tex, process=False)
    return gm


# ----------------------------------------------------------------- building
BX = [22, 28, 34, 40, 46]; BY = [30, 37, 44]
ZS = [3.5, 7.0, 10.5]
HOLE = (29.2, 33.6, 31.7, 36.1)      # uncovered opening, level 2
COVER = (40.2, 38.2, 42.8, 40.8)     # covered opening, level 2 (decoy)


def build_building():
    slab(0, 22, 30, 46, 44)
    for z in ZS:
        slab(z, 22, 30, 46, 44, hole=HOLE if z == 7.0 else None)
    for x in BX:
        for y in BY: box('concrete_dk', [x, y, 5.25], [.7, .7, 10.5])
    # core walls
    for (c, s) in [([35, 35, 6.8], [4.2, .3, 13.6]), ([35, 39, 6.8], [4.2, .3, 13.6]), ([33, 37, 6.8], [.3, 4.2, 13.6]), ([37, 37, 6.8], [.3, 4.2, 13.6])]:
        box('concrete_dk', c, s)
    # covered opening (decoy) on level 2
    cx, cy = (COVER[0] + COVER[2]) / 2, (COVER[1] + COVER[3]) / 2
    box('ply', [cx, cy, 7.06], [2.8, 2.8, .1]); box('red', [cx, cy, 7.13], [2.8, .12, .04], plan=False); box('red', [cx, cy, 7.13], [.12, 2.8, .04], plan=False)
    for ox in (-1.2, 1.2):
        for oy in (-1.2, 1.2): cyl('yellow', [cx + ox, cy + oy, 7.2], .12, .3, plan=False, sec=10)
    # edge protection: levels 1,2 fully; level 3 all except the east edge
    for z in (3.5, 7.0):
        for (a, b) in [((22.1, 30.1), (45.9, 30.1)), ((45.9, 30.1), (45.9, 43.9)), ((45.9, 43.9), (22.1, 43.9)), ((22.1, 43.9), (22.1, 30.1))]:
            guardrail(a, b, z)
    top = 10.5
    guardrail((22.1, 30.1), (45.9, 30.1), top); guardrail((45.9, 43.9), (22.1, 43.9), top); guardrail((22.1, 43.9), (22.1, 30.1), top)
    # column stubs with rebar (hazard: uncapped at x=22,28 on north row; decoy: capped x=34,40)
    for x in BX[:4]:
        box('concrete_dk', [x, 44, top + .35], [.7, .7, .7])
        for k in range(8):
            a = k * math.pi / 4; px_, py_ = x + .25 * math.cos(a), 44 + .25 * math.sin(a)
            rod('rust', [px_, py_, top], [px_, py_, top + 2.4], .035)
            if x >= 34: sph('yellow', [px_, py_, top + 2.43], .075, sub=1)
    # brick pallet at east slab edge (hazard) + worker without guardrail
    pallet(45.0, 39.6, top, 0, (1.3, 1.0)); box('brick', [45.0, 39.6, top + .6], [1.2, .9, .9])
    box('steel_lt', [45.0, 39.6, top + .75], [1.25, .03, .03], plan=False)
    person(45.3, 34.3, top, face=0, helmet='white', vest=True)             # no harness, open edge
    # decoy: worker clipped on behind a railed edge
    box('steel_lt', [24.6, 32.6, top + .5], [.2, .2, 1.0]); cyl('yellow', [24.6, 32.6, top + 1.0], .22, .08, plan=False)
    person(24.0, 37.5, top, face=-1.2, helmet='white', vest=True, harness=True)
    rod('red', [24.15, 37.4, top + 2.0], [24.6, 32.7, top + 1.0], .03)
    # formwork material on the slab
    box('ply', [30, 41.3, top + .25], [2.4, 1.2, .5]); box('ply', [31.2, 40.9, top + .75], [2.0, 1.1, .45])


def build_scaffold():
    y0 = 28.4; y1 = 29.6; xs = np.arange(22, 46.1, 3.0)
    zl = [2.0, 4.5, 7.0]
    for x in xs:
        for y in (y0, y1): rod('steel_lt', [x, y, 0], [x, y, 9.4], .06)
    for z in zl:
        rod('steel_lt', [22, y0, z - .1], [46, y0, z - .1], .05); rod('steel_lt', [22, y1, z - .1], [46, y1, z - .1], .05)
        for i in range(len(xs) - 1):
            bay = i
            miss = (z == 7.0 and bay == 4)       # hazard: gap and no edge protection on lift 3, bay 5 (x 34-37)
            if miss:
                box('wood', [xs[i] + .75, (y0 + y1) / 2, z], [1.5, 1.2, .05])      # half the boards missing
                box('wood_dk', [xs[i] + 2.3, y0 - .35, z + .03], [1.1, .22, .03], rz=.18, plan=False)   # loose board hanging off
            else:
                box('wood', [(xs[i] + xs[i + 1]) / 2, (y0 + y1) / 2, z], [3.0, 1.2, .05])
                rod('steel_lt', [xs[i], y0, z + 1.0], [xs[i + 1], y0, z + 1.0], .035)
                rod('steel_lt', [xs[i], y0, z + .5], [xs[i + 1], y0, z + .5], .035)
                box('wood_dk', [(xs[i] + xs[i + 1]) / 2, y0, z + .12], [3.0, .04, .2], plan=False)
        for i in range(len(xs) - 1):
            if i % 2 == 0: rod('steel_lt', [xs[i], y0, z], [xs[i + 1], y0, z + 2.5], .03)
    # ladder up the scaffold (decoy-neutral)
    # unsecured short ladder on the north face (hazard): too steep/short, base on a pallet, not tied
    pallet(40.0, 47.0, 0, 0, (1.2, .8))
    for sx in (-.25, .25):
        rod('steel_lt', [40 + sx, 47.2, .15], [40 + sx, 44.2, 3.0], .045)
    for k in range(10):
        t = .06 + k * .09; rod('steel_lt', [39.75, 47.2 - 3.0 * t, .15 + 2.85 * t], [40.25, 47.2 - 3.0 * t, .15 + 2.85 * t], .03)


# ----------------------------------------------------------------- crane
def build_crane():
    cx, cy = 30.0, 18.0; H = 26.0
    for sx in (-.9, .9):
        for sy in (-.9, .9): rod('yellow', [cx + sx, cy + sy, 0], [cx + sx, cy + sy, H], .13, 6)
    for z in np.arange(1.5, H, 2.6):
        for (a, b, c, d_) in [(-.9, -.9, .9, -.9), (.9, -.9, .9, .9), (.9, .9, -.9, .9), (-.9, .9, -.9, -.9)]:
            rod('yellow', [cx + a, cy + b, z], [cx + c, cy + d_, z], .06, 6)
        rod('yellow', [cx - .9, cy - .9, z], [cx + .9, cy - .9, z + 2.6], .045, 6); rod('yellow', [cx + .9, cy + .9, z], [cx - .9, cy + .9, z + 2.6], .045, 6)
    box('concrete_dk', [cx, cy, .6], [4.6, 4.6, 1.2])
    jz = H + .4
    for sy in (-.9, .9): rod('yellow', [cx - 10, cy + sy, jz], [cx + 28, cy + sy, jz], .1, 6)
    rod('yellow', [cx - 10, cy, jz + 1.5], [cx + 28, cy, jz + 1.5], .1, 6)
    for xx in np.arange(cx - 10, cx + 27.5, 2.4):
        rod('yellow', [xx, cy - .9, jz], [xx + 1.2, cy, jz + 1.5], .04, 6); rod('yellow', [xx + 1.2, cy, jz + 1.5], [xx + 2.4, cy + .9, jz], .04, 6)
    apex = [cx, cy, H + 7.0]
    rod('steel', [cx, cy, jz + 1.5], apex, .12, 6)
    rod('steel', apex, [cx + 24, cy, jz + 1.5], .05); rod('steel', apex, [cx - 8.5, cy, jz + 1.5], .05)
    for k in range(3): box('concrete_dk', [cx - 8.5 + k * 1.0, cy, jz + .5], [.9, 2.0, 1.3])
    box('white', [cx + 1.8, cy + 1.9, jz - 1.4], [1.8, 1.8, 2.0]); box('glass', [cx + 2.75, cy + 1.9, jz - 1.2], [.05, 1.4, 1.1], plan=False)
    tx = cx + 17.0
    box('steel', [tx, cy, jz - .45], [1.6, 1.4, .6])
    hz = 11.8
    rod('black', [tx - .3, cy, jz - .7], [tx - .3, cy, hz + .6], .03); rod('black', [tx + .3, cy, jz - .7], [tx + .3, cy, hz + .6], .03)
    box('yellow', [tx, cy, hz], [.9, .9, 1.0])
    # load: 3 steel beams slung from the hook, hanging over the workers
    lz = 9.0
    for k, oz in enumerate([0, .55, 1.1]): box('steel', [tx, cy, lz + oz - .5], [9.0, .5, .45])
    rod('black', [tx, cy, hz - .5], [tx - 4.2, cy, lz + 1.1], .03); rod('black', [tx, cy, hz - .5], [tx + 4.2, cy, lz + 1.1], .03)
    person(tx + .4, cy - 1.3, 0, face=2.2); person(tx - 1.4, cy + .5, 0, face=-1.0, helmet='yellow')
    # decoy: barricaded and signed exclusion zone on a quiet laydown pad
    barrier((41, 10.5), (51, 10.5)); barrier((51, 10.5), (51, 15)); barrier((41, 15), (51, 15)); barrier((41, 10.5), (41, 15))
    for k in range(3): box('rust', [44 + k * 2.2, 12.8, .45], [1.8, 3.2, .35]); cyl('rust', [44 + k * 2.2, 12.8, .75], .22, 3.4, 'y', 10, plan=False)


# ----------------------------------------------------------------- excavation
def build_pit():
    x0, y0, x1, y1 = PIT
    # shored north half (decoy)
    for (c, s) in [([x0 + .08, 40, -1.5], [.16, 8, 3.0]), ([x1 - .08, 40, -1.5], [.16, 8, 3.0]), ([10, y1 - .08, -1.5], [14, .16, 3.0])]:
        box('steel', c, s, plan=False)
    for yy in (38.5, 42.0):
        rod('yellow', [x0 + .16, yy, -1.1], [x1 - .16, yy, -1.1], .12, 8)
    for sx in (-.22, .22): rod('steel_lt', [16.3 + sx, 40, -3.0], [16.3 + sx, 40, 1.1], .04)
    for k in range(12): rod('steel_lt', [16.08, 40, -2.8 + k * .32], [16.52, 40, -2.8 + k * .32], .025)
    barrier((x0 - .8, 35.5), (x0 - .8, y1 + .8)); barrier((x0 - .8, y1 + .8), (x1 + .8, y1 + .8)); barrier((x1 + .8, y1 + .8), (x1 + .8, 35.5))
    box('red', [x1 + .9, 37.0, 1.9], [.06, .8, .6], plan=False)
    # unshored south half (hazard): worker inside, spoil heap piled on the edge, no barrier, no ladder
    person(8.0, 32.0, -3.0, face=1.4, helmet='white', vest=True, k=1.2)
    cone('soil', [10.0, 25.4, 0], 3.8, 2.3, sx=1.7, sy=1.0)
    cone('soil', [6.4, 26.2, 0], 2.2, 1.3, sx=1.4, sy=1.0); cone('soil', [14.6, 26.0, 0], 2.0, 1.1, sx=1.4, sy=1.0)
    # idle mini-excavator beyond the heap
    push(TM([15.8, 21.0, 0]) @ RM(.5, [0, 0, 1]))
    box('yellow', [0, 0, .75], [3.2, 1.9, 1.0]); box('black', [0, 0, .25], [3.6, 2.0, .5]); box('yellow', [-.3, 0, 1.55], [1.5, 1.4, 1.2]); box('glass', [.4, 0, 1.6], [.05, 1.2, .8], plan=False)
    rod('yellow', [1.2, 0, 1.3], [3.2, 0, 2.0], .13); rod('yellow', [3.2, 0, 2.0], [4.3, 0, .6], .1); box('steel', [4.4, 0, .45], [.7, .9, .55], plan=False)
    pop()


# ----------------------------------------------------------------- electrics, gas, plant, misc
def build_misc():
    # overhead power line along x=54 (hazard: within crane/plant reach, nothing marks it)
    for y in (3.0, 57.0):
        rod('pole', [54, y, 0], [54, y, 15.5], .22, 10); box('pole', [54, y, 14.6], [.2, 3.4, .2], plan=False)
        for dy in (-1.5, 0, 1.5): cyl('white', [54, y + dy, 14.9], .1, .3, plan=False, sec=8)
    for dy in (-1.5, 0, 1.5):
        rod('black', [54, 3 + dy, 15.0], [54, 57 + dy, 15.0], .045)
    # distribution board + damaged cable across the working area, lying in a puddle (hazard)
    box('steel', [38.0, 26.2, .9], [.9, .35, 1.8]); box('red', [37.9, 26.0, 1.0], [.7, .04, .4], plan=False)
    cable_pts = [(37.6, 26.0), (36.2, 25.4), (34.8, 26.4), (33.2, 25.9), (31.8, 26.8), (30.4, 26.3)]
    for a, b in zip(cable_pts[:-1], cable_pts[1:]): rod('cable', [*a, .06], [*b, .06], .06)
    cyl('copper', [30.2, 26.3, .1], .1, .2, 'x', 8, plan=False); rod('copper', [30.4, 26.3, .1], [29.5, 26.5, .15], .03)
    m = trimesh.creation.cylinder(radius=1.7, height=.03, sections=28); m.apply_transform(np.diag([1.5, 1, 1, 1]) @ TM([33.4, 26.0, .02]) if False else TM([33.4, 26.0, .02]))
    add('water', m, plan=False)
    # welding / gas cylinders lying down beside pallets, no extinguisher (hazard)
    box('steel', [8.0, 16.0, .45], [1.3, .9, .9]); person(6.3, 16.6, 0, face=.6, helmet='white', vest=True)
    cyl('ox_blue', [9.8, 14.3, .28], .26, 1.6, 'x', 14); cyl('ac_red', [9.8, 13.6, .22], .2, 1.4, 'x', 14)
    rod('black', [9.0, 14.0, .3], [8.6, 15.2, .95], .03)
    pallet(11.5, 17.2, 0, .3); pallet(11.5, 17.2, .15, .3); box('ply', [11.5, 17.2, .55], [1.1, .8, .5], .3)
    sph('spark', [8.1, 16.0, 1.05], .06, 1); sph('spark', [8.4, 15.6, .55], .05, 1); sph('spark', [7.7, 15.9, .35], .05, 1)
    # debris blocking the pedestrian walkway (hazard)
    pallet(18.6, 18.4, 0, .4); pallet(19.6, 20.2, 0, -.3); box('ply', [19.2, 19.2, .45], [1.6, .9, .1], .8)
    for k in range(4): cyl('steel_lt', [18.4 + k * .33, 19.3, .15 + (k % 2) * .05], .12, 2.6, 'y', 8, plan=False)
    for k in range(5): box('brick', [17.6 + k * .55, 17.2 + (k % 2) * .5, .1], [.4, .2, .2], k, plan=False)
    # wheel loader reversing; worker standing in the blind spot behind it, no banksman (hazard)
    push(TM([29.0, 6.8, 0]) @ RM(0, [0, 0, 1]))
    box('yellow', [0, 0, 1.25], [4.0, 2.4, 1.5]); box('yellow', [1.6, 0, 2.6], [1.8, 2.0, 1.4]); box('glass', [1.65, 0, 2.7], [1.5, 2.0, 1.0], plan=False)
    box('black', [-.2, 0, .6], [3.4, 2.0, .4], plan=False); box('red', [-2.02, 0, 1.4], [.04, 1.6, .22], plan=False)
    for sx in (-1.2, 1.4):
        for sy in (-1.3, 1.3): cyl('rubber', [sx, sy, .85], .85, .7, 'y', 14, plan=False)
    rod('yellow', [2.0, -.9, 1.7], [4.7, -.9, 1.0], .12); rod('yellow', [2.0, .9, 1.7], [4.7, .9, 1.0], .12); box('steel', [5.0, 0, .75], [1.2, 2.8, .9])
    pop()
    person(25.4, 6.4, 0, face=0, helmet='white', vest=True)
    # decoy: dump truck reversing with a banksman and barriers
    push(TM([46.0, 5.0, 0]) @ RM(math.pi / 2, [0, 0, 1]))
    box('white', [1.5, 0, 1.6], [2.0, 2.4, 2.0]); box('glass', [2.55, 0, 1.9], [.05, 2.0, 1.1], plan=False); box('steel_lt', [-1.4, 0, 1.7], [4.8, 2.5, 1.6]); box('black', [0, 0, .6], [6.4, 2.0, .6], plan=False)
    for sx in (-2.8, .4, 2.2):
        for sy in (-1.2, 1.2): cyl('rubber', [sx, sy, .65], .65, .5, 'y', 12, plan=False)
    pop()
    person(46.0, 9.4, 0, face=-math.pi / 2, helmet='yellow', vest=True, arm_up=True)
    barrier((42.5, 7.5), (42.5, 3.0)); barrier((49.5, 7.5), (49.5, 3.0))
    # person without helmet or hi-vis walking on the road (hazard)
    person(8.0, 4.4, 0, face=.4, helmet=None, vest=False, shirt='shirt')
    # site cabin + decoy: caged, chained gas cylinders and an extinguisher by the cabin
    box('cont_blue', [6, 56.2, 1.3], [6.0, 2.4, 2.6]); box('cont_green', [6, 52.8, 1.3], [6.0, 2.4, 2.6]); box('glass', [3.2, 52.8, 1.5], [.05, 1.0, .8], plan=False)
    box('white', [6, 52.8, 2.7], [6.2, 2.5, .1], plan=False); box('white', [6, 56.2, 2.7], [6.2, 2.5, .1], plan=False)
    for dx in (-1.1, 1.1):
        for dy in (-1.1, 1.1): rod('steel_lt', [13.5 + dx, 50.5 + dy, 0], [13.5 + dx, 50.5 + dy, 1.9], .05)
    box('steel_lt', [13.5, 50.5, 1.95], [2.4, 2.4, .06])
    for k in range(3):
        cyl('ox_blue', [12.9 + k * .6, 50.5, .85], .2, 1.6, 'z', 12); rod('black', [12.2, 49.4, 1.0], [14.8, 49.4, 1.0], .02)
    cyl('red', [15.8, 50.5, .45], .13, .9, 'z', 10); box('wood', [15.8, 50.5, .02], [.5, .5, .04], plan=False)
    box('signgreen', [17.5, 56.0, 1.8], [2.4, .12, 1.6]); rod('steel_lt', [16.6, 56.0, 0], [16.6, 56.0, 1.6], .05); rod('steel_lt', [18.4, 56.0, 0], [18.4, 56.0, 1.6], .05)
    # rebar racks & formwork stacks (neutral laydown)
    for k in range(3): box('wood_dk', [51 + k * .6, 38, .1], [.2, 7, .2], plan=False)
    for k in range(5): cyl('rust', [50.9 + (k % 3) * .6, 38 - 1.3 * (k // 3), .55 + (k // 3) * .3], .08, 6.5, 'y', 8, plan=False)
    box('ply', [50.5, 31.5, .4], [3.2, 1.6, .8]); box('ply', [50.5, 31.5, 1.0], [3.0, 1.5, .4])
    # a few neutral workers around the site
    person(26.0, 24.5, 0, face=1.0, helmet='white', vest=True)
    person(38.6, 29.0, 0, face=-1.5, helmet='yellow', vest=True)


HAZ = [
    dict(id=1, cat='Fall', title='Open slab edge (no guardrail)', pos=(46, 37, 10.5),
         what='Top slab: the east edge has no guardrail or toe-board, while the other three edges are protected.',
         fix='Install top rail, mid rail and toe-board before anyone works near the edge.'),
    dict(id=2, cat='Fall', title='Worker at the edge, not clipped on', pos=(45.3, 34.3, 12.4),
         what='Worker stands right at the unprotected east edge with no harness or lanyard.',
         fix='Fit edge protection first; if work at the edge is unavoidable, harness tied to an anchor point.'),
    dict(id=3, cat='Fall', title='Uncovered floor opening', pos=(30.45, 34.85, 7.0),
         what='Hole in the level-2 slab with no cover, rail or warning. (The orange panel on the same level is a properly covered opening.)',
         fix='Cover with a secured, marked cover (or fit guardrail around it).'),
    dict(id=4, cat='Fall', title='Scaffold platform gap, no edge protection', pos=(35.5, 29.0, 7.4),
         what='Top scaffold lift, bay 5: half the boards are missing, a board hangs loose, and guardrails and toe-board are missing.',
         fix='Complete boards, guardrails and toe-boards; tag the scaffold "do not use" until fixed.'),
    dict(id=5, cat='Fall', title='Unsecured, badly placed ladder', pos=(40, 45.6, 1.8),
         what='Ladder on the north face is too flat, stands on a pallet, is not tied, and does not extend above the landing.',
         fix='Ladder at 4:1, on firm level ground, tied off, extending 1 m above the landing.'),
    dict(id=6, cat='Struck-by', title='Material stacked on the slab edge', pos=(45.0, 39.6, 11.5),
         what='Pallet of bricks on the open slab edge: it can fall onto people below.',
         fix='Keep stacks away from edges; use toe-boards; keep the area below barricaded.'),
    dict(id=7, cat='Struck-by', title='Crane load over workers', pos=(47, 16.7, 8.8),
         what='Suspended steel load is slewed directly over two workers; there is no exclusion zone, tag line or signaller.',
         fix='Never work under a load; barricade the lift zone and use a signaller.'),
    dict(id=8, cat='Struck-by', title='Reversing loader, person in blind spot', pos=(26.6, 6.6, 1.8),
         what='Wheel loader is reversing with a worker directly behind it and no banksman. (The dump truck at E6 has a banksman and barriers.)',
         fix='Separate people and plant; use a banksman; never stand in a machine blind spot.'),
    dict(id=9, cat='Caught-in/between', title='Unshored excavation with a worker inside', pos=(8, 32, -1.2),
         what='South half of the pit has vertical walls with no shoring, no barrier and no ladder, and a worker is inside. (The north half is shored, barricaded and has a ladder.)',
         fix='Shore or batter the sides, provide a ladder, barricade the edge.'),
    dict(id=10, cat='Caught-in/between', title='Spoil heap at the excavation edge', pos=(10, 25.4, 2.4),
         what='Excavated soil is piled on the edge of the pit, adding load and risking a collapse onto the worker.',
         fix='Keep spoil at least the depth of the excavation (or more) away from the edge.'),
    dict(id=11, cat='Electrical', title='Live overhead line, no warning or guard', pos=(54, 24, 15.0),
         what='Overhead power line runs through the crane and plant working area with no marker, goal-posts or isolation.',
         fix='Ask the utility to isolate or divert; set exclusion distance, goal-posts and warning signs.'),
    dict(id=12, cat='Electrical', title='Damaged cable in a puddle, open board', pos=(33.0, 26.3, .6),
         what='Cable lies across the walkway in a puddle, one end is stripped, and the distribution board door is open.',
         fix='Replace the cable, lift it off the ground, close/lock the board, use RCD protection.'),
    dict(id=13, cat='Fire/Explosion', title='Loose gas cylinders near welding', pos=(9.8, 14.0, 1.0),
         what='Oxygen and acetylene cylinders lie on the ground, not secured, next to welding sparks and a wooden pallet stack, with no extinguisher. (The caged, chained cylinders near the cabin are correct.)',
         fix='Store upright, chained, apart; keep combustibles away from hot work; extinguisher at hand.'),
    dict(id=14, cat='Housekeeping/Other', title='Uncapped rebar ends', pos=(25, 44, 11.8),
         what='Bare rebar sticking up from the column stubs at the north-west corner of the top slab. (The yellow-capped bars at the next columns are protected.)',
         fix='Fit rebar caps (or bend over / cover) so a fall or trip cannot impale a worker.'),
    dict(id=15, cat='PPE', title='No helmet or hi-vis on a live road', pos=(8, 4.4, 2.8),
         what='Worker walks along the site road without a helmet or high-visibility clothing.',
         fix='Mandatory PPE for everyone on site; enforce at the gate.'),
    dict(id=16, cat='Housekeeping/Other', title='Debris blocking the walkway', pos=(19.2, 19.2, 1.3),
         what='Pallets, pipes and offcuts block the marked pedestrian route and the access to the building.',
         fix='Keep walkways clear; designated storage; regular housekeeping checks.'),
]
DECOYS = [
    ('Top-slab edges with guardrail', 'West, south and north edges of the top level are fully protected.'),
    ('Worker clipped on behind a railed edge', 'Harness and lanyard tied to an anchor post: correct practice.'),
    ('Shored, barricaded excavation (north half)', 'Sheeting, struts, ladder and barrier: correct practice.'),
    ('Caged, chained gas cylinders at the cabin', 'Upright, secured, extinguisher beside them.'),
    ('Dump truck with banksman and barriers', 'Signaller guides the reversing truck; area is barricaded.'),
    ('Covered, marked floor opening', 'Orange cover with fixings on the level-2 slab.'),
]


def main():
    gm = build_ground()
    build_building(); build_scaffold(); build_crane(); build_pit(); build_misc()
    scene = trimesh.Scene()
    # transform: centre, Z-up -> Y-up, scale 1:100
    M = np.diag([.01, .01, .01, 1]) @ RM(-math.pi / 2, [1, 0, 0]) @ TM([-30, -30, 0])
    for mat, meshes in PARTS.items():
        merged = trimesh.util.concatenate(meshes)
        c, rough, met = MATS[mat]
        merged.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[c[0], c[1], c[2], 255], roughnessFactor=rough, metallicFactor=met, name=mat))
        merged.apply_transform(M)
        scene.add_geometry(merged, node_name=mat, geom_name=mat)
    gm.apply_transform(M); scene.add_geometry(gm, node_name='ground', geom_name='ground')
    glb = scene.export(file_type='glb')
    p = os.path.join(OUT, 'site.glb'); open(p, 'wb').write(glb)
    print('GLB', len(glb) / 1e6, 'MB; triangles', sum(len(m.faces) for m in scene.geometry.values()))
    # hazards json with cell + final coords (metres, Y-up)
    for h in HAZ:
        x, y, z = h['pos']; h['cell'] = cell(x, y)
        h['viewer'] = [round(v, 4) for v in (M @ np.array([x, y, z, 1]))[:3]]
        h['points'] = 10
    json.dump(dict(hazards=HAZ, decoys=DECOYS), open(os.path.join(OUT, 'hazards.json'), 'w'), indent=1)
    for h in HAZ: print(h['id'], h['cell'], h['cat'], '-', h['title'])
    # plan image
    render_plan()


def render_plan(px=22):
    N = int(SIZE * px)
    im = Image.open(os.path.join(OUT, 'ground.jpg')).resize((N, N), Image.LANCZOS).convert('RGB')
    d = ImageDraw.Draw(im, 'RGBA'); X = lambda x: x * px; Y = lambda y: (SIZE - y) * px
    x0, y0, x1, y1 = PIT
    d.rectangle([X(x0), Y(y1), X(x1), Y(y0)], fill=(104, 78, 52, 255), outline=(60, 40, 25, 255), width=3)
    for poly, zt, rgb in sorted(PLAN, key=lambda t: t[1]):
        sh = .78 + .22 * min(zt / 14.0, 1.0)
        col = tuple(int(min(255, c * sh)) for c in rgb) + (255,)
        d.polygon([(X(p[0]), Y(p[1])) for p in poly], fill=col, outline=(40, 40, 46, 255))
    im.save(os.path.join(OUT, 'site_plan.png'))
    print('plan', im.size)


if __name__ == '__main__':
    main()

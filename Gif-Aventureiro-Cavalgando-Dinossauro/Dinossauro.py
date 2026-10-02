# -*- coding: utf-8 -*-
"""
Dinossauro correndo com um cavaleiro em cima - Pixel Art animada
=================================================================
Mesma base do código do coração: uma matriz de índices (pixel art) é desenhada
com plt.imshow(..., cmap=cmap, interpolation='none') e plt.axis('off').
A diferença é que agora cada quadro (frame) é uma matriz 192x108 gerada
proceduralmente, e o FuncAnimation troca os quadros em loop.

Recursos gráficos:
  * Paleta indexada própria (ListedColormap) com ~130 cores
  * Céu de pôr do sol com degradê "dithering" (matriz de Bayer), sol com brilho,
    estrelas piscando e pássaros batendo asas
  * Parallax em 6 camadas (nuvens, montanhas distantes, montanhas médias,
    pinheiros, chão, grama do primeiro plano) com loop perfeito
  * Dinossauro: sombreamento Lambert (luz vindo de cima/esquerda), barriga clara,
    listras, placas dorsais, cauda ondulando, boca abrindo, olho, dentes,
    pernas com cinemática inversa (pés colados ao chão durante a passada)
  * Cavaleiro: jaqueta, calça, botas, cachecol e rabo de cavalo ao vento,
    óculos, braços segurando as rédeas, sela com manta
  * Contorno escuro em cada peça, sombra no chão e poeira levantada pelas patas

Uso:  python dinossauro_pixel_art.py
      (abre a animação e também salva  dinossauro_pixel_art.gif)
"""
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path

# =============================================================================
# 1) PARÂMETROS
# =============================================================================
W, H = 192, 108          # resolução da pixel art
HOR = 76                 # linha do horizonte
G = 94                   # linha do chão onde as patas pisam
S = 6                    # velocidade do chão (pixels por quadro)
CYC = 12                 # quadros por ciclo completo de corrida (2 passadas)
NF = 96                  # total de quadros do loop (múltiplo de CYC)
FPS = 25
MEIA = 18                # meia amplitude da passada (2*MEIA == S * 6 quadros)
ESCALA_GIF = 6           # ampliação inteira ao salvar o GIF

# =============================================================================
# 2) PALETA  (0 = índice do ListedColormap, como no código do coração)
# =============================================================================
CORES = dict(
    # céu
    sky0='#161a45', sky1='#262a62', sky2='#3b3a80', sky3='#59478f', sky4='#8a559a',
    sky5='#c15f8e', sky6='#ea7a78', sky7='#fba26a', sky8='#ffc979', sky9='#ffe39a',
    sun_core='#fff7cf', sun_edge='#ffe58c', star='#e8e4ff', bird='#2b2147',
    # nuvens
    cl0='#9a5c9c', cl1='#d9788f', cl2='#ffae86', cl3='#ffd2a0',
    # montanhas
    mf0='#4a3a7c', mf1='#68509e', mf2='#9a78c0', hz0='#d99aa8',
    mm0='#352b66', mm1='#4b3c84', mm2='#6d54a2', hz1='#9a6f9e',
    # árvores
    t0='#14233a', t1='#1f3a52', t2='#37645f', trunk='#2a1c22',
    # chão
    g0='#2f5f2e', g1='#3f7a36', g2='#5a9a3f', g3='#8cc454',
    r0='#6a4430', r1='#8c5f3b', r2='#b2834f', r3='#cfa468', rs='#e2bf86',
    shadow='#4e3226', dust0='#e6c796', dust1='#c9a572', dust2='#a98258',
    f0='#1b3a22', f1='#2f6a33', f2='#5da644',
    rock0='#463e52', rock1='#6a6176', rock2='#9990a6',
    # dinossauro
    skin0='#0f3a2c', skin1='#1a5c3c', skin2='#2b8a4a', skin3='#4fb05a', skin4='#9ae07a',
    far0='#0b2a21', far1='#123f2e', far2='#1a5238',
    belly0='#b09a62', belly1='#d4c283', belly2='#f2e6b2',
    plate0='#7d2230', plate1='#c94a38', plate2='#ff8545', plate3='#ffc66e',
    outD='#0b211c', mouth='#6b1f2c', tooth='#fff8e6', eyeY='#ffd93b', pupil='#12100f',
    brow='#0f3328', claw='#efe6c8',
    blk0='#7e2433', blk1='#b83f4a', sad0='#4f2c1a', sad1='#85522d',
    # cavaleiro
    jk0='#8c2230', jk1='#c93a3c', jk2='#f0634a',
    pn0='#1f2a5c', pn1='#33448a', pn2='#4d63b0',
    sk0='#c7845e', sk1='#e9a97c', sk2='#ffcfa6',
    ha0='#2e1a14', ha1='#4d2c1f', ha2='#774430',
    sc0='#d9921c', sc1='#ffc928', sc2='#fff08a',
    bt0='#2c1810', bt1='#53321e', bt2='#7a4b2c',
    lens='#7fe9ff', lens2='#dffbff', strap='#1e1a22', glove='#6b4226',
    rein='#e0b97a', outR='#1a0d14', blush='#f08a7a', mouthR='#8a3b3b',
)
NOMES = list(CORES)
K = {n: i for i, n in enumerate(NOMES)}
cmap = ListedColormap([CORES[n] for n in NOMES])      # <- o "cmap" da pixel art


def ks(*nomes):
    return np.array([K[n] for n in nomes])


SKY = ks('sky0', 'sky1', 'sky2', 'sky3', 'sky4', 'sky5', 'sky6', 'sky7', 'sky8', 'sky9')
SKIN = ks('skin0', 'skin1', 'skin2', 'skin3', 'skin4')
FAR = ks('far0', 'far1', 'far2')
BELLY = ks('belly0', 'belly1', 'belly2')
PLATE = ks('plate0', 'plate1', 'plate2', 'plate3')
JK = ks('jk0', 'jk1', 'jk2')
JKD = ks('jk0', 'jk0', 'jk1')
PN = ks('pn0', 'pn1', 'pn2')
SKN = ks('sk0', 'sk1', 'sk2')
HAIR = ks('ha0', 'ha1', 'ha2')
SCARF = ks('sc0', 'sc1', 'sc2')
BOOT = ks('bt0', 'bt1', 'bt2')

# =============================================================================
# 3) FERRAMENTAS DE DESENHO (máscaras, sombreamento, contorno)
# =============================================================================
YY, XX = np.mgrid[0:H, 0:W].astype(float)


def bayer(h, w):
    b = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 + 1 / 32
    return np.tile(b, (h // 4 + 1, w // 4 + 1))[:h, :w]


BAY = bayer(H, W)

L = np.array([-0.50, -0.65, 0.57])
L = L / np.linalg.norm(L)
LX, LY, LZ = L


def tone(lam, n, dith=0.12):
    """Quantiza a iluminação em n tons, com dithering ordenado."""
    u = (lam + 0.65) / 1.65 + (BAY - 0.5) * dith
    return np.clip((u * n).astype(int), 0, n - 1)


def dilate(m):
    d = m.copy()
    d[1:] |= m[:-1]
    d[:-1] |= m[1:]
    d[:, 1:] |= m[:, :-1]
    d[:, :-1] |= m[:, 1:]
    return d


def _normais(u, v, r2, c, s):
    k = np.where(r2 > 1, 1 / np.sqrt(np.maximum(r2, 1e-9)), 1.0)
    u = u * k
    v = v * k
    return (u * c - v * s, u * s + v * c, np.sqrt(np.clip(1 - u * u - v * v, 0, 1)))


def ell(cx, cy, rx, ry, rot=0.0):
    """Elipse: devolve (máscara, nx, ny, nz) com normais de esfera achatada."""
    dx, dy = XX - cx, YY - cy
    c, s = np.cos(rot), np.sin(rot)
    u = (dx * c + dy * s) / rx
    v = (-dx * s + dy * c) / ry
    r2 = u * u + v * v
    return (r2 <= 1.0,) + _normais(u, v, r2, c, s)


def tube(pts, rad):
    """Tubo afunilado (cápsulas encadeadas) com normais cilíndricas."""
    best = np.full((H, W), 9e9)
    nx = np.zeros((H, W))
    ny = np.zeros((H, W))
    for (x0, y0), (x1, y1), r0, r1 in zip(pts[:-1], pts[1:], rad[:-1], rad[1:]):
        dx, dy = x1 - x0, y1 - y0
        l2 = dx * dx + dy * dy + 1e-9
        t = np.clip(((XX - x0) * dx + (YY - y0) * dy) / l2, 0, 1)
        r = r0 + (r1 - r0) * t
        ox = XX - (x0 + t * dx)
        oy = YY - (y0 + t * dy)
        d = np.hypot(ox, oy) / r
        up = d < best
        best = np.where(up, d, best)
        nx = np.where(up, ox / r, nx)
        ny = np.where(up, oy / r, ny)
    mask = best <= 1.0
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    return mask, nx, ny, nz


def mpoly(pts):
    p = Path(np.array(pts, float))
    return p.contains_points(np.c_[XX.ravel(), YY.ravel()]).reshape(H, W)


def put(spr, mask, col, oc='outD'):
    if oc is not None:
        ring = dilate(mask) & ~mask
        spr[ring] = K[oc]
    if np.isscalar(col):
        spr[mask] = col
    else:
        spr[mask] = np.asarray(col)[mask]


def paint(spr, shape, tones, oc='outD', belly=None, stripes=False, origin=(0, 0), dith=0.12):
    mask, nx, ny, nz = shape
    lam = nx * LX + ny * LY + nz * LZ
    n = len(tones)
    idx = tone(lam, n, dith)
    out = tones[idx]
    if stripes:
        ox, oy = origin
        st = ((((XX - ox) * 0.85 + (YY - oy) * 0.55) % 12) < 2.6) & (ny < 0.3)
        out = np.where(st, tones[np.clip(idx - 1, 0, n - 1)], out)
    if belly is not None:
        bm = ny > (0.40 + (BAY - 0.5) * 0.32)
        out = np.where(bm, belly[tone(lam, len(belly), dith)], out)
    put(spr, mask, out, oc)


def px(spr, x, y, c):
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < spr.shape[1] and 0 <= yi < spr.shape[0]:
        spr[yi, xi] = K[c] if isinstance(c, str) else c


def bezier(p0, p1, p2, p3, n):
    u = np.linspace(0, 1, n)[:, None]
    return ((1 - u) ** 3 * np.array(p0) + 3 * (1 - u) ** 2 * u * np.array(p1)
            + 3 * (1 - u) * u ** 2 * np.array(p2) + u ** 3 * np.array(p3))


# =============================================================================
# 4) CENÁRIO (camadas pré-calculadas)
# =============================================================================
def build_sky():
    n = len(SKY)
    sx, sy = 148, 57
    d = np.hypot(XX - sx, YY - sy)
    g = YY / (HOR + 4) * (n - 1)
    glow = np.clip(1 - (d - 12) / 30, 0, 1) ** 1.6
    g = np.clip(g + glow * 2.6, 0, n - 1.001)
    low = np.floor(g).astype(int)
    idx = np.clip(low + ((g - low) > BAY), 0, n - 1)
    sky = SKY[idx]
    sky[d <= 12.5] = K['sun_edge']
    sky[d <= 10.2] = K['sun_core']
    return sky


def wdx(X, cx, P):
    return (X - cx + P / 2) % P - P / 2


def build_clouds():
    P = 144
    Y, X = np.mgrid[0:H, 0:P].astype(float)
    lay = np.full((H, P), -1)
    bay = bayer(H, P)
    nuvens = [(26, 20, 1.0), (92, 32, 1.25), (60, 9, 0.7), (126, 12, 0.8)]
    puffs = [(-10, 2, 10, 4), (0, -1.5, 12, 6.5), (11, 1, 9, 4.8), (3, 3, 17, 3.4)]
    for cx, cy, s in nuvens:
        m = np.zeros((H, P), bool)
        for dx, dy, rx, ry in puffs:
            m |= (wdx(X, cx + dx * s, P) / (rx * s)) ** 2 + ((Y - (cy + dy * s)) / (ry * s)) ** 2 <= 1
        m &= Y <= cy + 4.2 * s
        u = (Y - (cy - 7 * s)) / (11 * s) + (bay - 0.5) * 0.25
        idx = np.clip((u * 4).astype(int), 0, 3)
        lay[m] = ks('cl0', 'cl1', 'cl2', 'cl3')[idx][m]
    return lay


def build_mount(P, base, harm, cols, haze, seed):
    x = np.arange(P)
    ridge = base - sum(a * np.sin(2 * np.pi * k * x / P + ph) for a, k, ph in harm)
    Y, X = np.mgrid[0:H, 0:P].astype(float)
    bay = bayer(H, P)
    depth = Y - ridge[None, :]
    mask = (depth >= 0) & (Y < HOR + 4)
    lit = (np.roll(ridge, -1) > np.roll(ridge, 1))[None, :]
    c0, c1, c2 = cols
    out = np.where(lit, np.where(depth < 3.2, K[c2], K[c1]),
                   np.where(depth < 2.2, K[c1], K[c0]))
    gully = (((X * 0.55 + depth * 0.95) % 10) < 1.3) & (depth > 5)
    out = np.where(gully & lit, K[c0] if False else K[c1], out)
    out = np.where(gully & ~lit, K[c0], out)
    # névoa perto do horizonte
    p = np.clip((Y - (HOR - 12)) / 12, 0, 1)
    out = np.where(p > bay * 1.05, K[haze], out)
    lay = np.full((H, P), -1)
    lay[mask] = out[mask]
    return lay


def build_trees(P=432, n=50, seed=7):
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:P].astype(float)
    lay = np.full((H, P), -1)
    base = HOR + 2
    xs = np.sort(rng.uniform(0, P, n))
    hs = rng.uniform(13, 26, n)
    for cx, h in zip(xs, hs):
        dx = wdx(X, cx, P)
        tronco = (np.abs(dx) <= 1.2) & (Y >= base - 4) & (Y <= base)
        lay[tronco] = K['trunk']
        for i in range(3):
            ytop = base - h + i * h * 0.27
            th = h * 0.46
            hw = h * (0.17 + 0.07 * i)
            rowhw = hw * (Y - ytop + 1) / th
            m = (Y >= ytop) & (Y <= ytop + th) & (np.abs(dx) <= rowhw)
            lay[m] = K['t0']
            lit = m & (dx > rowhw * 0.35)
            lay[lit] = K['t1']
            edge = m & (dx > rowhw * 0.82) & (((Y + X) % 3) == 0)
            lay[edge] = K['t2']
    return lay


def build_ground():
    P = 288
    rng = np.random.default_rng(11)
    Y, X = np.mgrid[0:H, 0:P].astype(float)
    bay = bayer(H, P)
    lay = np.full((H, P), -1)
    # grama
    lay[(Y >= HOR) & (Y <= HOR + 1)] = K['g3']
    lay[(Y >= HOR + 2) & (Y <= HOR + 5)] = K['g2']
    tr = (Y >= HOR + 6) & (Y <= HOR + 10)
    pr = (Y - (HOR + 6)) / 5
    lay[tr] = np.where(pr > bay, K['g0'], K['g1'])[tr]
    # estrada
    v = np.clip((Y - (HOR + 11)) / (H - HOR - 11), 0, 1)
    rt = ks('r3', 'r2', 'r1', 'r0')
    ri = np.clip((v * 3.4 + (bay - 0.5) * 0.9).astype(int), 0, 3)
    est = Y >= HOR + 11
    lay[est] = rt[ri][est]
    # texturas da grama
    for _ in range(120):
        x = int(rng.integers(0, P)); y = int(rng.integers(HOR + 2, HOR + 10))
        lay[y, x] = K[rng.choice(['g1', 'g3', 'g0'])]
    for _ in range(60):
        x = int(rng.integers(0, P))
        for k in range(3):
            lay[HOR - 1 - k + 1 - 1 + 0, x] = K['g2'] if k < 2 else lay[HOR - 1, x]
    for _ in range(46):
        x = int(rng.integers(0, P)); h = int(rng.integers(2, 5))
        for k in range(h):
            lay[HOR - 1 - k, (x + (k // 2)) % P] = K['g2'] if k else K['g1']
    # pedrinhas e pontos
    for _ in range(520):
        x = int(rng.integers(0, P)); y = int(rng.integers(HOR + 13, H))
        c = K[rng.choice(['r0', 'r3', 'rs', 'r0'])]
        lay[y, x] = c
        if rng.random() < 0.5:
            lay[y, (x + 1) % P] = c
    # marcas de roda (trilhas)
    for y in (HOR + 16, HOR + 25):
        x = int(rng.integers(0, 20))
        while x < P + 20:
            ln = int(rng.integers(8, 24))
            for k in range(ln):
                lay[y, (x + k) % P] = K['r0']
                lay[y + 1, (x + k) % P] = K['rs'] if k % 2 == 0 else K['r3']
            x += ln + int(rng.integers(5, 14))
    return lay


def build_foreground():
    P = 432
    rng = np.random.default_rng(23)
    Y, X = np.mgrid[0:H, 0:P].astype(float)
    lay = np.full((H, P), -1)
    for cx, rx, ry in [(60, 8, 4.5), (170, 6, 3.6), (290, 9, 5.0), (370, 5.5, 3.2)]:
        m = ((wdx(X, cx, P) / rx) ** 2 + ((Y - (H - 2)) / ry) ** 2) <= 1
        nx = wdx(X, cx, P) / rx
        ny = (Y - (H - 2)) / ry
        lam = nx * LX + ny * LY + np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1)) * LZ
        idx = tone(lam, 3, 0.0) if False else np.clip(((lam + 0.65) / 1.65 * 3).astype(int), 0, 2)
        lay[m] = ks('rock0', 'rock1', 'rock2')[idx][m]
        ring = dilate(m) & ~m
        lay[ring & (Y < H)] = K['f0']
    for _ in range(95):
        x = int(rng.integers(0, P)); h = int(rng.integers(4, 11))
        c = K[rng.choice(['f0', 'f1', 'f1', 'f2'])]
        for k in range(h + 1):
            xx = int(round(x - 0.35 * k - 0.02 * k * k))
            lay[H - 1 - k, xx % P] = c
            if k < h // 2:
                lay[H - 1 - k, (xx + 1) % P] = c
    return lay


SKY_IMG = build_sky()
CLOUDS = build_clouds()
MFAR = build_mount(240, HOR - 30, [(7, 2, 0.6), (5, 5, 2.1), (3, 11, 0.3), (1.6, 23, 1.2)],
                   ('mf0', 'mf1', 'mf2'), 'hz0', 1)
MMID = build_mount(336, HOR - 15, [(7, 3, 1.4), (5, 7, 0.4), (3, 15, 2.2), (1.5, 31, 0.9)],
                   ('mm0', 'mm1', 'mm2'), 'hz1', 2)
TREES = build_trees()
GROUND = build_ground()
FORE = build_foreground()

_rng = np.random.default_rng(5)
STARS = [(int(_rng.integers(0, W)), int(_rng.integers(0, 24))) for _ in range(34)]
BIRDS = [(20, 22, 0.0), (70, 14, 0.5), (110, 30, 0.2)]


def view(lay, off):
    cols = (np.arange(W) + off) % lay.shape[1]
    return lay[:, cols]


# =============================================================================
# 5) DINOSSAURO
# =============================================================================
def ik(hip, ank, L1=19.0, L2=19.0):
    hx, hy = hip
    ax, ay = ank
    dx, dy = ax - hx, ay - hy
    dd = np.hypot(dx, dy)
    d = min(dd, L1 + L2 - 0.05)
    ux, uy = dx / dd, dy / dd
    a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
    h = np.sqrt(max(L1 * L1 - a * a, 0))
    knee = (hx + ux * a + uy * h, hy + uy * a - ux * h)      # joelho para a frente
    return knee, (hx + ux * d, hy + uy * d)


def perna(spr, hx, hy, fase, tons):
    p = fase % 1.0
    if p < 0.5:                                              # apoio: pé parado no chão
        s = p / 0.5
        dx, lift = MEIA - 2 * MEIA * s, 0.0
    else:                                                    # balanço: pé no ar
        s = (p - 0.5) / 0.5
        e = (1 - np.cos(np.pi * s)) / 2
        dx, lift = -MEIA + 2 * MEIA * e, 9.5 * np.sin(np.pi * s)
    fx, fy = hx + 4 + dx, G - lift
    joelho, tornozelo = ik((hx, hy), (fx, fy - 9))
    pts = [(hx, hy), joelho, tornozelo, (fx - 0.5, fy - 3.4), (fx + 2, fy - 1.7), (fx + 9.5, fy - 0.9 + 0.35 * lift)]
    rad = [8.5, 6.0, 3.4, 3.0, 2.5, 1.3]
    paint(spr, tube(pts, rad), tons)
    px(spr, fx + 10.5, fy - 0.9 + 0.35 * lift, 'claw')
    px(spr, fx + 9.5, fy + 0.3 + 0.35 * lift, 'claw')


def placa(spr, x, ytop, h, w=5.6, lean=-2.0):
    pts = [(x - w / 2, ytop + 2.0), (x + w / 2, ytop + 2.0), (x + lean, ytop - h)]
    m = mpoly(pts)
    u = (XX - (x - w / 2)) / w
    out = np.where(u < 0.35, PLATE[3], np.where(u < 0.72, PLATE[2], PLATE[1]))
    out = np.where(YY > ytop + 0.6, PLATE[0], out)
    put(spr, m, out, 'outD')


def desenhar_dino(spr, f):
    t = (f % CYC) / CYC
    ph = 2 * np.pi * t
    by = -1.7 * np.cos(4 * np.pi * t)                        # corpo desce no meio do apoio
    cx, cy = 90.0, 52.0 + by
    orig = (cx, cy)

    # --- perna de trás (mais escura)
    perna(spr, 83, 61 + by, t + 0.5, FAR)

    # --- cauda (placas primeiro, depois o tubo)
    def tail(s):
        x = 70 - 52 * s
        y = 53 + by - 11 * s ** 1.5 + 3.4 * s * np.sin(2 * ph - 4.5 * s)
        r = 12 * (1 - s) ** 1.1 + 1.6
        return x, y, r

    ss = np.linspace(0, 1, 14)
    tp = [tail(s) for s in ss]
    for s in np.arange(0.14, 0.88, 0.12):
        x, y, r = tail(s)
        placa(spr, x, y - r * 0.9, 8.5 * (1 - s) + 3.0)
    paint(spr, tube([(a, b) for a, b, _ in tp], [r for _, _, r in tp]), SKIN,
          belly=BELLY, stripes=True, origin=orig)

    # --- placas do dorso e corpo
    for x, h in [(65, 8.5), (71, 9.5), (77, 8.0), (108, 7.0), (113, 6.0)]:
        yt = cy - 15 * np.sqrt(max(1 - ((x - cx) / 28) ** 2, 0))
        placa(spr, x, yt, h)
    paint(spr, ell(cx, cy, 28, 15, 0.03), SKIN, belly=BELLY, stripes=True, origin=orig)

    # --- pescoço
    N = bezier((104, 50 + by), (124, 50 + by), (118, 33 + by * 0.6), (127, 28 + by * 0.5), 9)
    nr = np.linspace(11, 6.2, 9)
    for i in (2, 4, 6):
        tg = N[i + 1] - N[i - 1]
        tg = tg / np.hypot(*tg)
        cand = [(tg[1], -tg[0]), (-tg[1], tg[0])]
        nn = max(cand, key=lambda q: -(q[0] + q[1]))
        bx, byy = N[i][0] + nn[0] * nr[i] * 0.85, N[i][1] + nn[1] * nr[i] * 0.85
        placa(spr, bx, byy - 1, 6.5 - i * 0.3, w=5.0, lean=-1.8)
    paint(spr, tube([tuple(p) for p in N], list(nr)), SKIN, belly=BELLY, stripes=True, origin=orig)

    # --- cabeça
    hdx, hdy = 128.0, 28.0 + by * 0.5 + 1.0 * np.sin(2 * ph + 0.7)
    ang = 0.05 + 0.04 * np.sin(2 * ph + 0.2)
    ca, sa = np.cos(ang), np.sin(ang)

    def hp(x, y):
        return (hdx + x * ca - y * sa, hdy + x * sa + y * ca)

    theta = 0.10 + 0.07 * np.sin(2 * ph + 1.0)               # abertura da mandíbula
    ct, st_ = np.cos(theta), np.sin(theta)

    def jp(x, y):
        x0, y0 = x - 2, y - 5
        return hp(2 + x0 * ct - y0 * st_, 5 + x0 * st_ + y0 * ct)

    sup = [(-6, -4), (-3, -8), (4, -10), (11, -9.5), (15, -7), (23, -5.5), (30, -3.5), (34, -1),
           (34.5, 2.5), (31, 4), (24, 4.5), (16, 5), (8, 5.5), (2, 5), (-4, 3), (-7, 0)]
    jaw = [(0, 3), (8, 5.5), (16, 6.2), (24, 5.5), (31, 4.2), (33, 5.5), (31, 7.5), (24, 8.5),
           (16, 9.5), (8, 9), (1, 8)]
    boca = mpoly([hp(3, 4.5), hp(30, 3.6), jp(31, 4.2), jp(24, 5.5), jp(8, 5.5)])
    put(spr, boca, K['mouth'], None)
    jawm = mpoly([jp(*p) for p in jaw])
    nj = ell(*hp(15, 6), 22, 7)[1:]
    paint(spr, (jawm,) + nj, BELLY)
    supm = mpoly([hp(*p) for p in sup])
    ns = ell(*hp(14, -1), 24, 11)[1:]
    paint(spr, (supm,) + ns, SKIN)
    # dentes
    for xx in (17, 20.5, 24, 27.5, 31):
        a = hp(xx, 4.2); b = hp(xx, 6.4)
        put(spr, tube([a, b], [0.75, 0.45])[0], K['tooth'], None)
    for xx in (19, 23, 27):
        a = jp(xx, 5.6); b = jp(xx, 3.4)
        put(spr, tube([a, b], [0.7, 0.4])[0], K['tooth'], None)
    # sobrancelha, olho, narina
    put(spr, tube([hp(7.5, -7.4), hp(14.8, -6.4)], [1.3, 1.0])[0], K['brow'], None)
    ex, ey = hp(11, -4.2)
    put(spr, ell(ex, ey, 2.1, 1.9)[0], K['eyeY'], 'outD')
    px(spr, ex + 0.6, ey, 'pupil')
    px(spr, ex + 0.6, ey + 1, 'pupil')
    px(spr, ex - 0.8, ey - 0.8, 'tooth')
    nx_, ny_ = hp(31.5, -1.6)
    px(spr, nx_, ny_, 'outD'); px(spr, nx_ - 1, ny_, 'outD')
    mouth = hp(26, 4.5)

    # --- bracinho do dino
    bs = 2.0 * np.sin(2 * ph + 0.5)
    paint(spr, tube([(111, 55 + by), (116.5, 59.5 + by + bs * 0.5), (121, 56.5 + by + bs)], [3.3, 2.5, 1.8]), SKIN)
    px(spr, 122.8, 56.8 + by + bs, 'claw'); px(spr, 122.2, 58.2 + by + bs, 'claw')

    # --- cavaleiro (por cima do dorso)
    cavaleiro(spr, f, by, mouth)

    # --- perna da frente (por cima do corpo)
    perna(spr, 88, 61 + by, t, SKIN)


# =============================================================================
# 6) CAVALEIRO
# =============================================================================
def cavaleiro(spr, f, by, mouth):
    t = (f % CYC) / CYC
    ph = 2 * np.pi * t
    rb = by * 0.75 + 0.9 * np.sin(2 * ph - 1.0)

    # manta + sela presas ao dino
    paint(spr, ell(92, 39.5 + by, 13, 4.2), ks('blk0', 'blk1'), oc='outR')
    paint(spr, ell(92, 37.8 + by, 8.5, 2.7), ks('sad0', 'sad1'), oc='outR')

    # braço de trás
    paint(spr, tube([(95.5, 24 + rb), (99, 30 + rb + 0.8 * np.sin(2 * ph)), (105.2, 30.8 + rb)], [2.5, 2.2, 1.8]),
          JKD, oc='outR')

    # tronco
    paint(spr, tube([(92, 35.2 + rb), (94.5, 29 + rb), (96.8, 23.6 + rb)], [4.7, 4.7, 4.3]), JK, oc='outR')
    for k in range(5):                                       # cinto
        px(spr, 90.2 + k * 0.5 + 0.8, 33.4 + rb - k * 0.3, 'bt1')

    # perna do cavaleiro
    bb = by + 0.9 * np.sin(2 * ph - 0.5)
    joelho = (100.5, 41.2 + by)
    tor = (97.4, 49.6 + bb)
    paint(spr, tube([(92.5, 35.6 + rb), joelho, tor], [4.8, 3.7, 2.8]), PN, oc='outR')
    paint(spr, tube([tor, (97.8, 52.0 + bb), (103.0, 53.0 + bb)], [2.9, 2.8, 2.0]), BOOT, oc='outR')

    # cachecol e rabo de cavalo
    sc = []
    for i in range(10):
        sc.append((95.0 - 3.5 * i, 20.2 + rb + 0.8 * i + 2.6 * (i / 9) * np.sin(2 * ph - 0.8 * i + 1.2)))
    paint(spr, tube(sc, list(np.linspace(2.2, 1.0, 10))), SCARF, oc='outR')
    pt = []
    for i in range(7):
        pt.append((94.5 - 3.2 * i, 14.5 + rb + 0.6 * i + 1.8 * (i / 6) * np.sin(2 * ph - 0.9 * i)))
    paint(spr, tube(pt, list(np.linspace(2.2, 0.8, 7))), HAIR, oc='outR')

    # cabeça
    hx, hy = 99.6, 16.0 + rb
    paint(spr, ell(hx, hy, 4.6, 4.8), SKN, oc='outR')
    paint(spr, ell(hx - 1.2, hy - 2.0, 5.0, 4.5), HAIR, oc='outR')
    put(spr, ell(hx + 1.6, hy + 0.9, 3.3, 3.7)[0], SKN[np.clip(tone(
        (XX - hx) * -0.04 + 0.3 - (YY - hy) * 0.03, 3, 0.0), 0, 2)], None)
    put(spr, tube([(94.6, 13.4 + rb), (103.6, 13.2 + rb)], [0.9, 0.9])[0], K['strap'], None)
    put(spr, ell(102.0, 13.0 + rb, 2.3, 1.9)[0], K['lens'], 'strap')
    px(spr, 101.2, 12.4 + rb, 'lens2'); px(spr, 102.0, 12.2 + rb, 'lens2')
    px(spr, 102.6, 17.0 + rb, 'pupil'); px(spr, 102.6, 16.0 + rb, 'pupil')
    px(spr, 104.5, 18.0 + rb, 'sk0')
    px(spr, 102.9, 20.0 + rb, 'mouthR'); px(spr, 103.9, 19.8 + rb, 'mouthR')
    px(spr, 100.6, 19.0 + rb, 'blush')

    # braço da frente + luva
    ba = 0.8 * np.sin(2 * ph - 0.4)
    paint(spr, tube([(97.4, 24.6 + rb), (101.8, 29.5 + rb + ba), (106.4, 29.8 + rb)], [2.8, 2.4, 1.9]), JK, oc='outR')
    put(spr, ell(107.2, 29.8 + rb, 2.2, 2.0)[0], K['glove'], 'outR')

    # rédea até a boca do dino
    p0 = np.array([108.0, 29.6 + rb])
    p1 = np.array(mouth)
    u = np.linspace(0, 1, 14)[:, None]
    ped = p0 * (1 - u) + p1 * u + np.hstack([np.zeros_like(u), 4.5 * np.sin(np.pi * u) * (1 - 0.3 * u)])
    put(spr, tube([tuple(p) for p in ped], [0.75] * 14)[0], K['rein'], None)


# =============================================================================
# 7) COMPOSIÇÃO DE CADA QUADRO
# =============================================================================
def poeira(img, f):
    for idade in range(1, 17):
        e = f - idade
        if e % 6 != 0:
            continue
        rng = np.random.default_rng(1000 + (e % NF))
        for _ in range(7):
            vx = rng.uniform(-2.8, 0.6)
            vy = rng.uniform(0.7, 1.9)
            jx = rng.uniform(-3, 3)
            if rng.random() < idade / 22:
                continue
            x = 70 + jx - S * idade + vx * idade * 0.7
            y = G + 1 - vy * idade + 0.095 * idade * idade
            y = min(y, G + 2)
            c = 'dust0' if idade < 4 else ('dust1' if idade < 9 else 'dust2')
            px(img, x, y, c)
            if idade < 9:
                px(img, x + 1, y, c); px(img, x, y + 1, c)
                if idade < 4:
                    px(img, x + 1, y + 1, c)


def passaros(img, f):
    for x0, y0, fase in BIRDS:
        bx = (x0 + f * (212 / NF)) % 212 - 10
        by = y0 + 2.2 * np.sin(2 * np.pi * (f / NF) + fase * 6)
        up = ((f // 3 + int(fase * 10)) % 2) == 0
        pts = ([(-3, -1), (-2, -1), (-1, 0), (0, 0), (1, 0), (2, -1), (3, -1)] if up else
               [(-3, 1), (-2, 1), (-1, 0), (0, 0), (1, 0), (2, 1), (3, 1)])
        for dx, dy in pts:
            px(img, bx + dx, by + dy, 'bird')


def renderizar(f):
    img = SKY_IMG.copy()
    for i, (sx, sy) in enumerate(STARS):
        if SKY_IMG[sy, sx] in (K['sky0'], K['sky1'], K['sky2']) and (f // 5 + i) % 3 != 0:
            img[sy, sx] = K['star']

    def camada(lay, vel):
        sel = view(lay, int(f * vel))
        m = sel >= 0
        img[m] = sel[m]

    camada(CLOUDS, 1.5)
    passaros(img, f)
    camada(MFAR, 2.5)
    camada(MMID, 3.5)
    camada(TREES, 4.5)
    camada(GROUND, 6.0)

    # sombra
    t = (f % CYC) / CYC
    by = -1.7 * np.cos(4 * np.pi * t)
    rx = 46 - by * 1.5
    d2 = ((XX - 88) / rx) ** 2 + ((YY - (G + 3)) / 4.2) ** 2
    img[(d2 <= 1) & ((1 - np.sqrt(d2)) > BAY * 0.8)] = K['shadow']

    poeira(img, f)

    spr = np.full((H, W), -1, dtype=int)
    desenhar_dino(spr, f)
    m = spr >= 0
    img[m] = spr[m]

    camada(FORE, 9.0)
    return img


# =============================================================================
# 8) ANIMAÇÃO (matplotlib) + exportação em GIF
# =============================================================================
if __name__ == '__main__':
    import sys
    print('Gerando quadros...', file=sys.stderr)
    QUADROS = [renderizar(f) for f in range(NF)]

    # --- exporta GIF com ampliação inteira (pixels nítidos)
    try:
        from PIL import Image
        pal = []
        for n in NOMES:
            h = CORES[n].lstrip('#')
            pal += [int(h[i:i + 2], 16) for i in (0, 2, 4)]
        pal += [0] * (768 - len(pal))
        imgs = []
        for q in QUADROS:
            big = np.kron(q, np.ones((ESCALA_GIF, ESCALA_GIF), dtype=int)).astype(np.uint8)
            im = Image.fromarray(big, 'P')
            im.putpalette(pal)
            imgs.append(im)
        imgs[0].save('dinossauro_pixel_art.gif', save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, disposal=1)
        print('GIF salvo: dinossauro_pixel_art.gif', file=sys.stderr)
    except ImportError:
        pass

    # --- mesma estrutura do código do coração: imshow + cmap + axis('off') + title
    fig = plt.figure(figsize=(12.8, 7.2), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 0.94])
    im = ax.imshow(QUADROS[0], cmap=cmap, vmin=0, vmax=len(NOMES) - 1, interpolation='none')
    plt.axis('off')
    ax.set_title('Dinossauro Correndo com Cavaleiro - Pixel Art', color='white')

    def atualizar(i):
        im.set_data(QUADROS[i])
        return (im,)

    ani = FuncAnimation(fig, atualizar, frames=NF, interval=1000 / FPS, blit=True)
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
import random

# ------------------------------------------------
# Einstellungen
# ------------------------------------------------
n_points = 15000
radius = 1.0
bins = 180
smooth_sigma = 1.2
levels_quantiles = [0.90, 0.955, 0.985]  # etwas „formfreundlicher“

# Form-Hervorhebung
shape_boost_in_density = 3      # wie stark Formen in der Dichtekarte gewichtet werden
shape_size_multiplier = (1.15, 1.6)  # Bereich für größere Punkte auf Formen
shape_alpha = 0.95              # etwas deckender
shape_jitter = 0.012            # „krümelige“ Streuung um die Form

# ------------------------------------------------
# 1) Grundverteilung + Cluster
# ------------------------------------------------
r = radius * (np.random.rand(n_points) ** 1.5)
theta = 2 * np.pi * np.random.rand(n_points)
x = r * np.cos(theta)
y = r * np.sin(theta)

n_clusters = 7
for _ in range(n_clusters):
    cx, cy = np.random.uniform(-0.5, 0.5, 2)
    size = np.random.uniform(0.05, 0.15)
    n_c = np.random.randint(80, 200)
    x = np.concatenate([x, np.random.normal(cx, size, n_c)])
    y = np.concatenate([y, np.random.normal(cy, size, n_c)])

# ------------------------------------------------
# 2) Mystische Formen (deutlicher)
# ------------------------------------------------
extra_x, extra_y = [], []
shapes = ["moon", "spiral", "text"]
for _ in range(random.randint(2, 3)):  # eher 2–3 Formen
    shape = random.choice(shapes)
    cx, cy = np.random.uniform(-0.35, 0.35, 2)

    if shape == "moon":
        angles = np.linspace(0, 2*np.pi, 420)
        r_m = 0.24
        xm = cx + r_m*np.cos(angles) + np.random.normal(0, shape_jitter, len(angles))
        ym = cy + r_m*np.sin(angles) + np.random.normal(0, shape_jitter, len(angles))
        xm2 = cx + (r_m*0.68)*np.cos(angles) + np.random.normal(0, shape_jitter, len(angles))
        ym2 = cy + (r_m*0.68)*np.sin(angles) + np.random.normal(0, shape_jitter, len(angles))
        extra_x.extend(xm);  extra_y.extend(ym)
        extra_x.extend(xm2); extra_y.extend(ym2)

    elif shape == "spiral":
        t = np.linspace(0, 4.5*np.pi, 520)
        a, b = 0.02, 0.11
        rs = a * np.exp(b * t)
        xs = cx + rs*np.cos(t) + np.random.normal(0, shape_jitter, len(t))
        ys = cy + rs*np.sin(t) + np.random.normal(0, shape_jitter, len(t))
        extra_x.extend(xs); extra_y.extend(ys)

    else:  # text (angedeutet)
        word = random.choice(["OM", "∞", "??"])
        for i, _ in enumerate(word):
            n_dots = 110  # etwas dichter
            extra_x.extend(cx + i*0.12 + np.random.normal(0, 0.018, n_dots))
            extra_y.extend(cy + np.random.normal(0, 0.018, n_dots))

extra_x = np.array(extra_x); extra_y = np.array(extra_y)

# Alle Punkte für Scatter
x_all = np.concatenate([x, extra_x])
y_all = np.concatenate([y, extra_y])

# ------------------------------------------------
# 3) Dichtekarte (Formen extra gewichten)
# ------------------------------------------------
extent = (-radius, radius, -radius, radius)

# „Phantom“-Gewichtung: Formen mehrfach in die Dichte werfen
x_dens = np.concatenate([x, *( [extra_x]*shape_boost_in_density )])
y_dens = np.concatenate([y, *( [extra_y]*shape_boost_in_density )])

H, xedges, yedges = np.histogram2d(x_dens, y_dens, bins=bins, range=[extent[:2], extent[2:]])

# Glättung
try:
    from scipy.ndimage import gaussian_filter
    Hsmooth = gaussian_filter(H, sigma=smooth_sigma)
except Exception:
    K = np.ones((3, 3), dtype=float) / 9.0
    Hpad = np.pad(H, 1, mode='edge')
    Hsmooth = (
        Hpad[:-2, :-2] + Hpad[:-2, 1:-1] + Hpad[:-2, 2:] +
        Hpad[1:-1, :-2] + Hpad[1:-1, 1:-1] + Hpad[1:-1, 2:] +
        Hpad[2:, :-2] + Hpad[2:, 1:-1] + Hpad[2:, 2:]
    ) / 9.0

vals = Hsmooth[Hsmooth > 0]
levels = [np.quantile(vals, q) for q in levels_quantiles] if len(vals) else []

Xc, Yc = np.meshgrid(
    (xedges[:-1] + xedges[1:]) / 2,
    (yedges[:-1] + yedges[1:]) / 2,
    indexing='xy'
)

# ------------------------------------------------
# 4) Plot
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Punktgrößen und Farben: Formen etwas größer & dunkler
sizes_base = np.random.uniform(3, 12, len(x))
sizes_shapes = np.random.uniform(3, 12, len(extra_x)) * np.random.uniform(*shape_size_multiplier, len(extra_x))
sizes = np.concatenate([sizes_base, sizes_shapes])

# Farb-Array (RGBA) — normale Punkte leicht transparenter, Formen dunkler
brown = np.array([139/255, 69/255, 19/255])
colors_base = np.tile(np.append(brown, 0.82), (len(x), 1))
colors_shapes = np.tile(np.append(brown*0.9, shape_alpha), (len(extra_x), 1))  # etwas dunkler
colors = np.vstack([colors_base, colors_shapes])

ax.scatter(x_all, y_all, s=sizes, c=colors)

# Flecken: gefüllte Konturen + Umriss
if levels:
    ax.contourf(Xc, Yc, Hsmooth.T, levels=levels, alpha=0.22, cmap="Greys")
    ax.contour(Xc, Yc, Hsmooth.T, levels=levels, colors="saddlebrown", linewidths=1.6)

# Tassenrand
outer = plt.Circle((0, 0), radius, color="black", fill=False, lw=3)
ax.add_artist(outer)

ax.set_aspect("equal")
ax.set_xlim(-radius*1.1, radius*1.1)
ax.set_ylim(-radius*1.1, radius*1.1)
ax.axis("off")
plt.show()

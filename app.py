# Use a non-GUI backend for server-side image rendering
import matplotlib
matplotlib.use("Agg")

from flask import Flask, make_response, render_template_string, request
import io
import numpy as np
import random

# Try to import SciPy for better smoothing; fallback if unavailable
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------- Parameters (tweak as you like) ----------
N_POINTS = 15000
RADIUS = 1.0
BINS = 180
SMOOTH_SIGMA = 1.2
LEVELS_QUANTILES = [0.90, 0.955, 0.985]  # density contour layers

# Shape emphasis
SHAPE_BOOST_IN_DENSITY = 3        # weight of shape points in density map
SHAPE_SIZE_MULTIPLIER = (1.15, 1.6)
SHAPE_ALPHA = 0.95
SHAPE_JITTER = 0.012

# ----------------------------------------------------

def generate_coffee_png() -> bytes:
    # 1) Base distribution (more points toward center)
    r = RADIUS * (np.random.rand(N_POINTS) ** 1.5)
    theta = 2 * np.pi * np.random.rand(N_POINTS)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Clusters (“stains”)
    n_clusters = 8
    for _ in range(n_clusters):
        cx, cy = np.random.uniform(-0.5, 0.5, 2)
        size = np.random.uniform(0.05, 0.15)
        n_c = np.random.randint(80, 200)
        x = np.concatenate([x, np.random.normal(cx, size, n_c)])
        y = np.concatenate([y, np.random.normal(cy, size, n_c)])

    # 2) Subtle mystical shapes (a bit stronger so they don’t get lost)
    extra_x, extra_y = [], []
    shapes = ["moon", "spiral", "text"]
    for _ in range(random.randint(2, 3)):  # usually 2–3 shapes
        shape = random.choice(shapes)
        cx, cy = np.random.uniform(-0.35, 0.35, 2)

        if shape == "moon":
            angles = np.linspace(0, 2*np.pi, 420)
            r_m = 0.24
            xm = cx + r_m*np.cos(angles) + np.random.normal(0, SHAPE_JITTER, len(angles))
            ym = cy + r_m*np.sin(angles) + np.random.normal(0, SHAPE_JITTER, len(angles))
            xm2 = cx + (r_m*0.68)*np.cos(angles) + np.random.normal(0, SHAPE_JITTER, len(angles))
            ym2 = cy + (r_m*0.68)*np.sin(angles) + np.random.normal(0, SHAPE_JITTER, len(angles))
            extra_x.extend(xm);  extra_y.extend(ym)
            extra_x.extend(xm2); extra_y.extend(ym2)

        elif shape == "spiral":
            t = np.linspace(0, 4.5*np.pi, 520)
            a, b = 0.02, 0.11
            rs = a * np.exp(b * t)
            xs = cx + rs*np.cos(t) + np.random.normal(0, SHAPE_JITTER, len(t))
            ys = cy + rs*np.sin(t) + np.random.normal(0, SHAPE_JITTER, len(t))
            extra_x.extend(xs); extra_y.extend(ys)

        else:  # text (hinted)
            word = random.choice(["OM", "∞", "??"])
            for i, _ in enumerate(word):
                n_dots = 110
                extra_x.extend(cx + i*0.12 + np.random.normal(0, 0.018, n_dots))
                extra_y.extend(cy + np.random.normal(0, 0.018, n_dots))

    extra_x = np.array(extra_x); extra_y = np.array(extra_y)
    x_all = np.concatenate([x, extra_x])
    y_all = np.concatenate([y, extra_y])

    # 3) Density map (shapes get extra weight)
    extent = (-RADIUS, RADIUS, -RADIUS, RADIUS)
    x_dens = np.concatenate([x, *([extra_x] * SHAPE_BOOST_IN_DENSITY)])
    y_dens = np.concatenate([y, *([extra_y] * SHAPE_BOOST_IN_DENSITY)])

    H, xedges, yedges = np.histogram2d(x_dens, y_dens, bins=BINS,
                                       range=[extent[:2], extent[2:]])

    if HAVE_SCIPY:
        Hsmooth = gaussian_filter(H, sigma=SMOOTH_SIGMA)
    else:
        # simple 3x3 box blur fallback
        Hpad = np.pad(H, 1, mode='edge')
        Hsmooth = (
            Hpad[:-2, :-2] + Hpad[:-2, 1:-1] + Hpad[:-2, 2:] +
            Hpad[1:-1, :-2] + Hpad[1:-1, 1:-1] + Hpad[1:-1, 2:] +
            Hpad[2:, :-2] + Hpad[2:, 1:-1] + Hpad[2:, 2:]
        ) / 9.0

    vals = Hsmooth[Hsmooth > 0]
    levels = [np.quantile(vals, q) for q in LEVELS_QUANTILES] if len(vals) else []

    Xc, Yc = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2,
        indexing='xy'
    )

    # 4) Render figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)

    # point sizes and colors
    sizes_base = np.random.uniform(3, 12, len(x))
    sizes_shapes = np.random.uniform(3, 12, len(extra_x)) * np.random.uniform(*SHAPE_SIZE_MULTIPLIER, len(extra_x))
    sizes = np.concatenate([sizes_base, sizes_shapes])

    brown = np.array([139/255, 69/255, 19/255])
    colors_base = np.tile(np.append(brown, 0.82), (len(x), 1))
    colors_shapes = np.tile(np.append(brown*0.9, SHAPE_ALPHA), (len(extra_x), 1))
    colors = np.vstack([colors_base, colors_shapes])

    ax.scatter(x_all, y_all, s=sizes, c=colors)

    # filled “stains” + outlines
    if levels:
        ax.contourf(Xc, Yc, Hsmooth.T, levels=levels, alpha=0.22, cmap="Greys")
        ax.contour(Xc, Yc, Hsmooth.T, levels=levels, colors="saddlebrown", linewidths=1.6)

    # cup rim
    outer = plt.Circle((0, 0), RADIUS, color="black", fill=False, lw=3)
    ax.add_artist(outer)

    ax.set_aspect("equal")
    ax.set_xlim(-RADIUS*1.1, RADIUS*1.1)
    ax.set_ylim(-RADIUS*1.1, RADIUS*1.1)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Mystic Coffee Grounds</title>
  <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
  <style>
    body { display:flex; flex-direction:column; align-items:center; gap:1rem;
           font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; 
           background:#f6f3ee; margin:0; padding:2rem; }
    header { font-weight:700; font-size:1.25rem; color:#3b2f2f; }
    img { max-width: min(90vw, 700px); height:auto; border-radius:12px; box-shadow: 0 6px 22px rgba(0,0,0,0.15); }
    button { padding:.6rem 1rem; border-radius:10px; border:1px solid #c4b7a6; background:white; cursor:pointer; }
    button:hover { background:#f0ece6; }
    .hint { color:#6b5b4d; font-size:.95rem; }
  </style>
</head>
<body>
  <header>🔮 Mystic Coffee Grounds — new reading on every visit/refresh</header>
  <img id="coffee" alt="coffee grounds" />
  <div>
    <button id="refresh">New reading</button>
  </div>
  <div class="hint">Tip: Every page load fetches a brand-new image. Click “New reading” to force refresh.</div>
  <script>
    const img = document.getElementById('coffee');
    function loadNew() {
      img.src = '/coffee.png?ts=' + Date.now(); // cache-busting param
    }
    document.getElementById('refresh').addEventListener('click', loadNew);
    loadNew(); // load on first visit
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    # Simple HTML page that embeds the generated PNG with cache-busting
    return render_template_string(INDEX_HTML)

@app.route("/coffee.png")
def coffee_png():
    png_bytes = generate_coffee_png()
    resp = make_response(png_bytes)
    resp.headers["Content-Type"] = "image/png"
    # ensure browsers don't cache it (we also add ?ts=… in HTML)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

if __name__ == "__main__":
    # host='0.0.0.0' to reach it from other devices on your LAN
    app.run(host="0.0.0.0", port=5000, debug=False)

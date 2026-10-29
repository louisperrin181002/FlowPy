"""Render the FlowPy app icon (PNG, macOS .icns, Windows .ico) into assets/.

Run from the project root:  python packaging/make_icon.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def render_png(path: Path, size: int = 1024):
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_alpha(0)
    # Rounded tile with a blue gradient
    tile = FancyBboxPatch((0.06, 0.06), 0.88, 0.88,
                          boxstyle="round,pad=0,rounding_size=0.2",
                          fc="#2563eb", ec="none")
    ax.add_patch(tile)
    grad = np.linspace(0, 1, 256)[:, None]
    im = ax.imshow(grad, extent=(0.06, 0.94, 0.06, 0.94), origin="lower",
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       "g", ["#1e40af", "#3b82f6"]), zorder=1)
    im.set_clip_path(tile)
    # Two populations of "events"
    rng = np.random.default_rng(3)
    for cx, cy, sx, sy, n in ((0.36, 0.36, 0.07, 0.06, 500),
                              (0.63, 0.64, 0.06, 0.07, 500)):
        x, y = rng.normal(cx, sx, n), rng.normal(cy, sy, n)
        ax.scatter(x, y, s=10, c="white", alpha=0.75, lw=0, zorder=2,
                   clip_path=tile)
    # A gate around one of them
    ax.add_patch(Ellipse((0.63, 0.64), 0.34, 0.38, angle=-20, fill=False,
                         ec="#fb923c", lw=size / 70, zorder=3))
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    png = ASSETS / "flowpy.png"
    render_png(png)
    from PIL import Image
    img = Image.open(png)
    img.save(ASSETS / "flowpy.ico",
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    if sys.platform == "darwin":
        iconset = ASSETS / "flowpy.iconset"
        iconset.mkdir(exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            img.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            img.resize((2 * s, 2 * s), Image.LANCZOS).save(
                iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(ASSETS / "flowpy.icns")], check=True)
        for f in iconset.iterdir():
            f.unlink()
        iconset.rmdir()


if __name__ == "__main__":
    main()

"""Shared 'Aurora Glass' chart style (no purple): canvas, glass tiles, fonts, Persian text helper, safe save."""
from __future__ import annotations

import colorsys
import re
import struct
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Patch

try:                                    # python-bidi >= 0.5 exposes get_display at top level
    from bidi import get_display
except ImportError:                     # older python-bidi
    from bidi.algorithm import get_display

# ---------------- constants ----------------
FIG_SIZE = (16, 9)                      # inches
DPI = 200                               # -> 3200 x 1800 px
EXPECTED_PIXELS = (FIG_SIZE[0] * DPI, FIG_SIZE[1] * DPI)
MARGIN_X = 0.04                         # left/right margin (figure fraction)
CONTENT_TOP = 0.84                      # tiles start below title + subtitle
CONTENT_BOTTOM = 0.065                  # tiles end above footer
TITLE_MAX_CHARS, SUBTITLE_MAX_CHARS, FOOTER_MAX_CHARS = 60, 110, 160   # guards against text overflow

BG = "#0B1020"
TEXT = "#E6E9F5"
MUTED = "#9AA4C7"
TURQUOISE = "#2DD4BF"
BLUE = "#38BDF8"
AMBER = "#FBBF24"
CORAL = "#FB7185"
SLATE_LIGHT = "#94A3B8"
SLATE_DARK = "#64748B"

CLASS_COLORS = {                        # recommended/high = turquoise, not_recommended/low = coral, no_idea/medium = amber
    "recommended": TURQUOISE, "high": TURQUOISE,
    "not_recommended": CORAL, "low": CORAL,
    "no_idea": AMBER, "medium": AMBER,
}
MODEL_COLORS = {"dummy": SLATE_DARK, "A": SLATE_LIGHT, "B": BLUE}   # A = raw labels, B = step-1 corrected labels

# (x, y_from_top, sigma_x, sigma_y, color, strength): soft aurora blobs on the dark base
BLOBS = [
    (0.08, 0.10, 0.30, 0.35, TURQUOISE, 0.30),
    (0.92, 0.12, 0.30, 0.35, BLUE, 0.28),
    (0.85, 0.95, 0.35, 0.35, CORAL, 0.20),
    (0.10, 0.92, 0.30, 0.30, AMBER, 0.14),
]
PERSIAN_FONT = "Vazirmatn"              # use fontfamily=PERSIAN_FONT for every Persian text object


def _assert_no_purple(colors):
    """Raise if any colour has a purple/magenta hue (250-330 degrees) and is clearly saturated."""
    for color in colors:
        hue, saturation, value = colorsys.rgb_to_hsv(*to_rgb(color))
        if saturation > 0.25 and 250 <= hue * 360 <= 330:
            raise ValueError(f"purple-range colour is not allowed: {color}")


_assert_no_purple([BG, TEXT, MUTED, TURQUOISE, BLUE, AMBER, CORAL, SLATE_LIGHT, SLATE_DARK,
                   *[b[4] for b in BLOBS]])


# ---------------- fonts and global style ----------------
def init_style(font_dir):
    """Register Vazirmatn (and Poppins if present), set rcParams. Call once per notebook session."""
    registered = []
    for ttf_path in sorted(Path(font_dir).glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf_path))
        registered.append(ttf_path.name)
    available = {font.name for font in font_manager.fontManager.ttflist}
    if PERSIAN_FONT not in available:
        raise RuntimeError(f"{PERSIAN_FONT} not found in {font_dir}; registered files: {registered}")
    english_font = "Poppins" if "Poppins" in available else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": [english_font, "DejaVu Sans"],
        "font.size": 13,
        "text.color": TEXT, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.unicode_minus": False,            # plain hyphen avoids a missing-glyph warning
    })
    return {"registered_files": registered, "english_font": english_font, "persian_font": PERSIAN_FONT}


def fa(text):
    """Persian text for matplotlib. Diagnosis (matplotlib 3.11.2, Vazirmatn) showed raw text is drawn
    correctly (joined letters, right-to-left, ZWNJ), so no reshape/bidi step is applied."""
    return str(text)

# ---------------- canvas ----------------
def make_background(width=1600, height=900, seed=2026, grain=0.02):
    """Dark base + blurred aurora blobs + ~2% monochrome grain, as float RGB array in [0, 1]."""
    rows, cols = np.mgrid[0:height, 0:width]
    x_norm = cols / (width - 1)
    y_norm = rows / (height - 1)                 # row 0 = top of the image
    canvas = np.empty((height, width, 3), dtype=np.float64)
    canvas[:] = np.array(to_rgb(BG))
    for blob_x, blob_y, sigma_x, sigma_y, color, strength in BLOBS:
        weight = strength * np.exp(-(((x_norm - blob_x) / sigma_x) ** 2 + ((y_norm - blob_y) / sigma_y) ** 2))
        canvas += weight[..., None] * (np.array(to_rgb(color)) - canvas)    # blend toward the blob colour
    noise = np.random.RandomState(seed).normal(0.0, grain, size=(height, width, 1))
    return np.clip(canvas + noise, 0.0, 1.0)


def new_canvas(title, subtitle, footer):
    """Figure with background, title (30 pt bold), subtitle (16 pt) and footer (11 pt)."""
    if len(title) > TITLE_MAX_CHARS or len(subtitle) > SUBTITLE_MAX_CHARS or len(footer) > FOOTER_MAX_CHARS:
        raise ValueError(f"text too long: title {len(title)}/{TITLE_MAX_CHARS}, "
                         f"subtitle {len(subtitle)}/{SUBTITLE_MAX_CHARS}, footer {len(footer)}/{FOOTER_MAX_CHARS}")
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=BG)
    background_axes = fig.add_axes([0, 0, 1, 1], zorder=0)
    background_axes.imshow(make_background(), extent=(0, 1, 0, 1), aspect="auto", interpolation="bilinear")
    background_axes.axis("off")
    fig.text(MARGIN_X, 0.955, title, fontsize=30, fontweight="bold", color=TEXT, ha="left", va="top", zorder=5)
    fig.text(MARGIN_X, 0.890, subtitle, fontsize=16, color=MUTED, ha="left", va="top", zorder=5)
    fig.text(MARGIN_X, 0.020, footer, fontsize=11, color=MUTED, ha="left", va="bottom", zorder=5)
    return fig


def split_row(n_columns, weights=None, gap=0.015, top=CONTENT_TOP, bottom=CONTENT_BOTTOM):
    """Tile rectangles (left, bottom, width, height) for one row of tiles, in figure fractions."""
    weights = list(weights) if weights is not None else [1.0] * n_columns
    if len(weights) != n_columns:
        raise ValueError("weights length must equal n_columns")
    usable_width = (1 - 2 * MARGIN_X) - gap * (n_columns - 1)
    rects, left = [], MARGIN_X
    for weight in weights:
        width = usable_width * weight / sum(weights)
        rects.append((left, bottom, width, top - bottom))
        left += width + gap
    return rects


def style_axes(ax):
    """Transparent axes, light spines, faint horizontal grid."""
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color((1, 1, 1, 0.30))
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.grid(axis="y", color=(1, 1, 1, 0.12), linewidth=0.8)
    ax.set_axisbelow(True)


def add_tile(fig, left, bottom, width, height, tile_title=None,
             pad_left=0.05, pad_right=0.025, pad_bottom=0.09, pad_top=None):
    """Glass tile (white alpha 0.09, border alpha 0.28) and a transparent axes inside it."""
    if pad_top is None:
        pad_top = 0.085 if tile_title else 0.03
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_bottom - pad_top
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError("tile paddings leave no room for the axes")
    fig.add_artist(FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=FIG_SIZE[0] / FIG_SIZE[1],   # round (not oval) corners
        transform=fig.transFigure, facecolor=(1, 1, 1, 0.09), edgecolor=(1, 1, 1, 0.28),
        linewidth=1.2, zorder=1))
    if tile_title:
        fig.text(left + 0.02, bottom + height - 0.022, tile_title, fontsize=15, fontweight="bold",
                 color=TEXT, ha="left", va="top", zorder=4)
    ax = fig.add_axes([left + pad_left, bottom + pad_bottom, inner_width, inner_height], zorder=3)
    style_axes(ax)
    return ax


def glass_legend(ax, handles=None, labels=None, **legend_kwargs):
    """Legend with a dark translucent frame; place it outside the data area via bbox_to_anchor."""
    if handles is not None:
        legend_kwargs["handles"] = handles
    if labels is not None:
        legend_kwargs["labels"] = labels
    legend = ax.legend(frameon=True, fontsize=12, **legend_kwargs)
    frame = legend.get_frame()
    frame.set_facecolor((*to_rgb(BG), 0.65))
    frame.set_edgecolor((1, 1, 1, 0.28))
    frame.set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return legend


def save_figure(fig, path):
    """Save PNG at the exact expected pixel size; refuse a second file with the same number prefix."""
    path = Path(path)
    if not re.fullmatch(r"\d{2}_[a-z0-9_]+\.png", path.name):
        raise ValueError(f"figure name must look like 01_name.png, got {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rivals = [p.name for p in path.parent.glob(f"{path.name[:2]}_*.png") if p.name != path.name]
    if rivals:
        raise FileExistsError(f"another figure already uses number {path.name[:2]}: {rivals}")
    fig.savefig(path, dpi=DPI, facecolor=BG, format="png")      # no bbox_inches="tight": it would change the size
    plt.close(fig)
    with open(path, "rb") as handle:
        header = handle.read(24)
    width_px, height_px = struct.unpack(">II", header[16:24])   # PNG IHDR width/height
    if (width_px, height_px) != EXPECTED_PIXELS:
        raise ValueError(f"unexpected image size {(width_px, height_px)} vs {EXPECTED_PIXELS}")
    return {"path": str(path), "pixels": (width_px, height_px), "size_mb": round(path.stat().st_size / 1e6, 2)}


# ---------------- smoke test (shown inline, never saved) ----------------
def style_smoke_figure():
    """Sample figure that exercises colours, fonts, Persian text (with ZWNJ), tiles and legends."""
    fig = new_canvas("Style test: Aurora Glass",
                     "Sample values only; checks colors, fonts, Persian text and layout",
                     "Style smoke test, not a report figure")
    left_rect, right_rect = split_row(2)

    ax_left = add_tile(fig, *left_rect, tile_title="Class colors (sample data)", pad_bottom=0.17)
    class_names = ["recommended", "not_recommended", "no_idea"]
    sample_values = [0.80, 0.10, 0.07]
    ax_left.bar(class_names, sample_values, color=[CLASS_COLORS[n] for n in class_names], width=0.55)
    for index, value in enumerate(sample_values):
        ax_left.text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom", color=TEXT, fontsize=13)
    ax_left.set_ylim(0, 1.0)
    ax_left.set_ylabel("share of rows (sample)")
    glass_legend(ax_left, handles=[Patch(color=MODEL_COLORS[k], label=label) for k, label in
                                   (("dummy", "dummy"), ("A", "variant A (raw)"), ("B", "variant B (corrected)"))],
                 loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, title="Model colors")

    ax_right = add_tile(fig, *right_rect, tile_title="Persian labels (sample data)", pad_left=0.13)
    terms = ["عالی", "بد نیست", "نمی\u200cخرم", "راضی نیستم", "تقلبی"]       # includes a ZWNJ word
    weights = [5.2, 3.9, 3.1, 2.6, 2.2]
    positions = np.arange(len(terms))[::-1]
    ax_right.barh(positions, weights, color=TURQUOISE, height=0.55)
    ax_right.set_yticks(positions)
    ax_right.set_yticklabels([fa(term) for term in terms], fontfamily=PERSIAN_FONT, fontsize=16)
    ax_right.set_xlabel("coefficient (sample)")
    ax_right.grid(axis="y", visible=False)
    ax_right.grid(axis="x", color=(1, 1, 1, 0.12), linewidth=0.8)
    return fig
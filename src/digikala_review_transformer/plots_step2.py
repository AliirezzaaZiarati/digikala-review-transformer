"""Step 2 (label-noise weighting) charts in the shared 'Aurora Glass' style."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import spearmanr

from .plot_style import (
    AMBER,
    BG,
    BLUE,
    CLASS_COLORS,
    CORAL,
    MUTED,
    SLATE_LIGHT,
    TEXT,
    TURQUOISE,
    add_tile,
    glass_legend,
    new_canvas,
    save_figure,
    split_row,
)

HEAD_TITLES = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
HEAD_COLORS = {"rec": TURQUOISE, "band": BLUE}


# ── Chart 01: distribution of the label-noise weights ──
def weight_histogram_stats(weights, w_min, low_cutoff, half_cutoff, n_bins):
    """Histogram of the rows with w < 1 plus the summary numbers printed on the chart."""
    w = np.asarray(weights, dtype="float64")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    assert np.isfinite(w).all(), "non-finite weights"
    assert w.min() >= w_min - 1e-6 and w.max() <= 1.0 + 1e-6, "weights outside [w_min, 1]"

    below_one = w < 1.0
    edges = np.linspace(w_min, 1.0, n_bins + 1)
    counts, _ = np.histogram(w[below_one], bins=edges)
    assert int(counts.sum()) == int(below_one.sum()), "histogram lost rows"
    return {
        "n": len(w), "n_below_one": int(below_one.sum()),
        "mean_w": float(w.mean()), "median_w": float(np.median(w)),
        "share_w_le_low": float((w <= low_cutoff).mean()),
        "share_w_lt_half": float((w < half_cutoff).mean()),
        "share_w_eq_one": float(1.0 - below_one.mean()),
        "counts": counts, "edges": edges,
    }


def chart_01_weight_distribution(views_by_head, figures_dir, heads, w_min, low_cutoff, half_cutoff, n_bins):
    """One tile per head: log-scale histogram of w for rows with w < 1, cutoff lines, stats box."""
    unknown_heads = [head for head in heads if head not in HEAD_TITLES]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")

    # ---------- data and checks ----------
    stats = {head: weight_histogram_stats(views_by_head[head]["w"].to_numpy(), w_min,
                                          low_cutoff, half_cutoff, n_bins)
             for head in heads}
    y_top = max(int(s["counts"].max()) for s in stats.values()) * 4     # shared y range for both tiles

    # ---------- canvas ----------
    subtitle = (f"Share of rows with w < {half_cutoff}: "
                + " and ".join(f"{stats[h]['share_w_lt_half'] * 100:.2f}% ({h})" for h in heads) + ".")
    fig = new_canvas(
        title="Distribution of label-noise weights",
        subtitle=subtitle,
        footer=("Histogram: rows with w < 1, log y-axis. All labeled train_supervised rows; "
                "zero-feature rows have w=1. Weights estimate label noise, not truth."))

    # ---------- one tile per head ----------
    for tile_rect, head in zip(split_row(len(heads)), heads):
        s, color = stats[head], HEAD_COLORS[head]
        ax = add_tile(fig, *tile_rect, tile_title=f"{HEAD_TITLES[head]}: {s['n']:,} labeled rows",
                      pad_left=0.095, pad_bottom=0.19)

        ax.set_yscale("log")
        ax.bar(s["edges"][:-1], s["counts"], width=np.diff(s["edges"]), align="edge",
               color=color, edgecolor=BG, linewidth=0.4)
        ax.axvline(low_cutoff, color=CORAL, linestyle="--", linewidth=2.0)
        ax.axvline(half_cutoff, color=AMBER, linestyle="--", linewidth=2.0)

        ax.set_xlim(w_min - 0.02, 1.0)
        ax.set_xticks(np.arange(0.1, 1.01, 0.1))
        ax.set_ylim(0.7, y_top)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else ""))
        ax.set_xlabel("weight w (rows with w < 1)")
        ax.set_ylabel("rows per bin (log scale)")

        # stats box, computed from the plotted data (monospace keeps the columns aligned)
        rows = [("rows", f"{s['n']:,}"), ("mean w", f"{s['mean_w']:.4f}"), ("median w", f"{s['median_w']:.4f}"),
                (f"w <= {low_cutoff}", f"{s['share_w_le_low'] * 100:.2f}%"),
                (f"w < {half_cutoff}", f"{s['share_w_lt_half'] * 100:.2f}%"),
                ("w = 1", f"{s['share_w_eq_one'] * 100:.1f}%")]
        ax.text(0.02, 0.97, "\n".join(f"{label:<9}{value:>11}" for label, value in rows),
                transform=ax.transAxes, ha="left", va="top", fontsize=11, family="DejaVu Sans Mono",
                color=TEXT, bbox={"boxstyle": "round,pad=0.5", "facecolor": (*to_rgb(BG), 0.65),
                                      "edgecolor": (1, 1, 1, 0.28)})

        glass_legend(ax,
                     handles=[Patch(facecolor=color, label="rows with w < 1"),
                              Line2D([], [], color=CORAL, linestyle="--", linewidth=2, label=f"w = {low_cutoff}"),
                              Line2D([], [], color=AMBER, linestyle="--", linewidth=2, label=f"w = {half_cutoff}")],
                     loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    info = save_figure(fig, Path(figures_dir) / "01_weight_distribution.png")
    info["heads"] = {head: {"n": s["n"], "n_below_one": s["n_below_one"],
                            **{key: round(s[key], 6) for key in ("mean_w", "median_w", "share_w_le_low",
                                                                 "share_w_lt_half", "share_w_eq_one")}}
                     for head, s in stats.items()}
    return info

# ── Chart 02: tail of the weight reduction (1 - w) per class, survival curves ──
def tail_group_stats(weights, half_cutoff, x_grid):
    """Survival curve of d = 1 - w on x_grid (share of rows with d >= x) plus summary statistics."""
    w = np.asarray(weights, dtype="float64")
    assert np.isfinite(w).all(), "non-finite weights"
    assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-6, "weights outside [0, 1]"
    n = len(w)
    sorted_reduction = np.sort(np.clip(1.0 - w, 0.0, None))
    survival = 1.0 - np.searchsorted(sorted_reduction, x_grid, side="left") / n
    assert np.all(np.diff(survival) <= 1e-12), "survival curve is not monotone"
    return {
        "n": int(n), "mean_w": float(w.mean()),
        "ci95": float(1.96 * w.std(ddof=1) / np.sqrt(n)),          # naive: rows treated as independent
        "median_w": float(np.median(w)), "p01_w": float(np.quantile(w, 0.01)),
        "share_w_lt_half": float((w < half_cutoff).mean()),
        "survival": np.where(survival > 0, survival, np.nan),      # zeros cannot be drawn on a log axis
    }


def chart_02_class_weight_tails(views_by_head, class_names_by_head, ambiguous_class_by_head,
                                figures_dir, heads, half_cutoff, x_min, n_points):
    """One tile per head: log-log survival curves of 1 - w per class, w=0.5 marker, stats table below."""
    unknown_heads = [head for head in heads if head not in HEAD_TITLES]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")
    x_grid = np.logspace(np.log10(x_min), np.log10(0.89), n_points)
    x_mark = 1.0 - half_cutoff                     # w = half_cutoff  <=>  d = 1 - half_cutoff

    # ---------- data and checks ----------
    stats, gaps = {}, {}
    for head in heads:
        view = views_by_head[head]
        w_all = view["w"].to_numpy()
        groups = {"all rows": tail_group_stats(w_all, half_cutoff, x_grid)}
        for name in class_names_by_head[head]:
            subset = view.loc[view["class_name"] == name, "w"].to_numpy()
            assert len(subset) > 0, f"{head}/{name}: no rows"
            groups[name] = tail_group_stats(subset, half_cutoff, x_grid)
        is_ambiguous = (view["class_name"] == ambiguous_class_by_head[head]).to_numpy()
        gaps[head] = float(w_all[is_ambiguous].mean() - w_all[~is_ambiguous].mean())
        stats[head] = groups

    # ---------- canvas ----------
    subtitle = ("Ambiguous-class mean w gap vs other classes: "
                + " and ".join(f"{gaps[h]:+.3f} ({h}, {ambiguous_class_by_head[h]})" for h in heads) + ".")
    fig = new_canvas(
        title="Weight reduction tail by class",
        subtitle=subtitle,
        footer=("Curves: share of rows with weight reduction (1 - w) at or above x; w=1 rows count in "
                "the base. CI95 is naive. Weights estimate label noise."))

    # ---------- one tile per head ----------
    for tile_rect, head in zip(split_row(len(heads)), heads):
        groups = stats[head]
        colors = {"all rows": SLATE_LIGHT, **{name: CLASS_COLORS[name] for name in class_names_by_head[head]}}
        ax = add_tile(fig, *tile_rect, tile_title=f"{HEAD_TITLES[head]}: how heavy is the tail",
                      pad_left=0.095, pad_bottom=0.24)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # curves: all rows first (thick, behind), then the classes
        for label, s in groups.items():
            is_all = label == "all rows"
            ax.plot(x_grid, s["survival"], color=colors[label], linewidth=3.8 if is_all else 2.6,
                    alpha=0.9 if is_all else 1.0, zorder=2 if is_all else 3)

        # w = half_cutoff marker line, intersection points and their percentages
        ax.axvline(x_mark, color=TEXT, linestyle=":", linewidth=1.6, alpha=0.8, zorder=1)
        ax.text(x_mark * 0.93, 0.9, f"w = {half_cutoff}", ha="right", va="top", color=TEXT, fontsize=11)
        markers = sorted(((label, s["share_w_lt_half"]) for label, s in groups.items()
                          if s["share_w_lt_half"] > 0), key=lambda item: -item[1])
        previous_log = None
        for label, share in markers:
            ax.scatter([x_mark], [share], s=70, color=colors[label], edgecolor=BG, linewidth=1.2, zorder=5)
            label_log = np.log10(share)
            if previous_log is not None:               # keep labels apart: push the lower one down
                label_log = min(label_log, previous_log - 0.34)
            previous_log = label_log
            ax.annotate(f"{share * 100:.2f}%", xy=(x_mark * 0.85, 10 ** label_log), xytext=(0, -3),
                        textcoords="offset points", ha="right", va="top", color=colors[label],
                        fontsize=11, fontweight="bold")

        # axes
        x_ticks = [10.0 ** k for k in range(round(np.log10(x_min)), 1)]
        y_ticks = [10.0 ** k for k in range(-6, 1)]
        ax.set_xlim(x_min, 1.0)
        ax.set_ylim(5e-7, 1.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{v:g}" for v in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v * 100:g}%" for v in y_ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", color=(1, 1, 1, 0.07), linewidth=0.8)
        ax.set_xlabel("weight reduction d = 1 - w (log scale)")
        ax.set_ylabel("share of rows with reduction >= d")

        # stats table under the axes, one colored line per group (also acts as the legend)
        header = (f"{'group':<16}{'n':>10}{'mean w +-CI95':>16}{'median':>9}{'p01':>8}"
                  f"{'w<' + str(half_cutoff):>8}")
        lines = [(header, MUTED)]
        for label, s in groups.items():
            mean_text = f"{s['mean_w']:.4f}±{s['ci95']:.4f}"
            lines.append(((f"{label:<16}{s['n']:>10,}{mean_text:>16}{s['median_w']:>9.4f}"
                          f"{s['p01_w']:>8.3f}{s['share_w_lt_half']:>8.2%}"), colors[label]))
        for row_index, (text, color) in enumerate(lines):
            ax.annotate(text, xy=(0, 0), xycoords="axes fraction", xytext=(-70, -60 - 15 * row_index),
                        textcoords="offset points", ha="left", va="top", fontsize=10,
                        family="DejaVu Sans Mono", color=color, annotation_clip=False)

    info = save_figure(fig, Path(figures_dir) / "02_class_weight_tails.png")
    info["heads"] = {
        head: {"ambiguous_gap": round(gaps[head], 6),
               "groups": {label: {key: (s[key] if key == "n" else round(s[key], 6))
                                  for key in ("n", "mean_w", "ci95", "median_w", "p01_w", "share_w_lt_half")}
                          for label, s in groups.items()}}
        for head, groups in stats.items()}
    return info

# ── Chart 03: Q vs C density map with iso-weight curves ──
def q_c_sample(view, sample_n, seed):
    """Bounded random sample of the rows that have Q and C (zero-feature rows have neither)."""
    has_qc = (view["q"].notna() & view["c"].notna()).to_numpy()
    usable = view.loc[has_qc, ["q", "c", "w"]]
    if len(usable) > sample_n:
        picked = np.sort(np.random.default_rng(seed).choice(len(usable), size=sample_n, replace=False))
        usable = usable.iloc[picked]
    return usable, int(has_qc.sum())


def chart_03_q_c_map(views_by_head, figures_dir, heads, w_min, low_cutoff, half_cutoff,
                     iso_weights, sample_n, seed, gridsize):
    """One tile per head: log-density hexbin of Q (x) vs C (y), iso-weight curves, stats under the axes."""
    unknown_heads = [head for head in heads if head not in HEAD_TITLES]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")
    cmap = LinearSegmentedColormap.from_list("q_c_density", ["#16324F", BLUE, TURQUOISE, TEXT])

    # ---------- data and checks ----------
    stats, samples = {}, {}
    for head in heads:
        view = views_by_head[head]
        sample, n_with_qc = q_c_sample(view, sample_n, seed)
        q_s, c_s, w_s = (sample[name].to_numpy(dtype="float64") for name in ("q", "c", "w"))
        assert ((q_s >= -1e-6) & (q_s <= 1 + 1e-6)).all(), f"{head}: Q outside [0, 1]"
        assert ((c_s >= -1e-6) & (c_s <= 1 + 1e-6)).all(), f"{head}: C outside [0, 1]"
        # the iso-weight curves are only valid if the stored weights follow the formula
        recomputed = np.maximum(w_min, 1.0 - c_s * (1.0 - q_s))
        assert np.abs(recomputed - w_s).max() < 1e-5, f"{head}: stored w differs from the formula"

        w_all = view["w"].to_numpy(dtype="float64")
        q_all = view["q"].to_numpy(dtype="float64")
        q_all = q_all[~np.isnan(q_all)]
        share_q_one_full = float((q_all >= 1.0 - 1e-9).mean())
        share_q_one_sample = float((q_s >= 1.0 - 1e-9).mean())
        tolerance = 5.0 * np.sqrt(share_q_one_full * (1.0 - share_q_one_full) / len(q_s))
        assert abs(share_q_one_sample - share_q_one_full) < tolerance, f"{head}: sample is not representative"

        stats[head] = {
            "n_labeled": len(w_all), "n_with_qc": n_with_qc, "n_sample": len(q_s),
            "share_w_lt_half": float((w_all < half_cutoff).mean()),
            "share_w_le_low": float((w_all <= low_cutoff).mean()),
            "share_q_one_full": share_q_one_full, "share_q_one_sample": share_q_one_sample,
            "spearman_q_c_sample": float(spearmanr(q_s, c_s).statistic),
        }
        samples[head] = (q_s, c_s)

    # ---------- canvas ----------
    subtitle = (f"Share of rows with w < {half_cutoff}: "
                + " and ".join(f"{stats[h]['share_w_lt_half'] * 100:.2f}% ({h})" for h in heads) + ".")
    fig = new_canvas(
        title="Where the weights come from: Q vs C",
        subtitle=subtitle,
        footer=(f"Bounded sample: {sample_n:,} rows per head, zero-feature rows excluded, log colour. "
                "Curves: w = 1 - C(1 - Q). Shares use all rows."))

    # ---------- one tile per head: density map first ----------
    drawn = []
    for tile_rect, head in zip(split_row(len(heads)), heads):
        s = stats[head]
        q_s, c_s = samples[head]
        ax = add_tile(fig, *tile_rect, tile_title=f"{HEAD_TITLES[head]}: Q vs C",
                      pad_left=0.085, pad_bottom=0.26)
        hexes = ax.hexbin(q_s, c_s, gridsize=gridsize, extent=(0, 1, 0, 1), mincnt=1, cmap=cmap,
                          norm=LogNorm(vmin=1), linewidths=0.15, edgecolors="none", zorder=2)

        # suspicious zone (w < half_cutoff): shaded region above the half_cutoff iso-curve
        q_zone = np.linspace(0.0, half_cutoff, 200)
        ax.fill_between(q_zone, half_cutoff / (1.0 - q_zone), 1.0, color=CORAL, alpha=0.20,
                        linewidth=0, zorder=1)

        # iso-weight curves: C = (1 - w) / (1 - Q), defined for Q <= w
        for w_level in iso_weights:
            q_curve = np.linspace(0.0, w_level, 300)
            if np.isclose(w_level, half_cutoff):
                color, width, alpha = AMBER, 2.2, 1.0
            elif np.isclose(w_level, low_cutoff):
                color, width, alpha = CORAL, 2.2, 1.0
            else:
                color, width, alpha = TEXT, 1.2, 0.6
            ax.plot(q_curve, (1.0 - w_level) / (1.0 - q_curve), color=color, linewidth=width,
                    alpha=alpha, linestyle="--", zorder=4)
            right_side = w_level > 0.85
            ax.text(w_level - 0.012 if right_side else w_level + 0.012, 0.965, f"w={w_level:g}",
                    ha="right" if right_side else "left", va="center", color=color, fontsize=10, zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="x", color=(1, 1, 1, 0.07), linewidth=0.8)
        ax.set_xlabel("Q = min(1, p_y / m_c)   (agreement with the label)")
        ax.set_ylabel("C = 1 - H_norm   (model confidence)")

        # stats lines under the axes (monospace keeps the columns aligned)
        lines = [("share w < " + f"{half_cutoff}", f"{s['share_w_lt_half'] * 100:.2f}%"),
                 ("share w <= " + f"{low_cutoff}", f"{s['share_w_le_low'] * 100:.2f}%"),
                 ("share Q = 1 (all rows)", f"{s['share_q_one_full'] * 100:.1f}%"),
                 ("Spearman(Q, C), sample", f"{s['spearman_q_c_sample']:+.3f}"),
                 ("rows with Q and C", f"{s['n_with_qc']:,}")]
        for row_index, (label, value) in enumerate(lines):
            ax.annotate(f"{label:<24}{value:>10}", xy=(0, 0), xycoords="axes fraction",
                        xytext=(-55, -68 - 14 * row_index), textcoords="offset points", ha="left",
                        va="top", fontsize=10, family="DejaVu Sans Mono", color=TEXT, annotation_clip=False)
        drawn.append((ax, hexes))

    # ---------- shared colour range, then one horizontal colour bar per tile ----------
    shared_max = max(float(hexes.get_array().max()) for _, hexes in drawn)
    for ax, hexes in drawn:
        hexes.set_clim(1, shared_max)
        color_axes = ax.inset_axes([0.60, -0.40, 0.40, 0.035])
        color_bar = fig.colorbar(hexes, cax=color_axes, orientation="horizontal")
        color_bar.outline.set_edgecolor((1, 1, 1, 0.28))
        color_axes.tick_params(colors=MUTED, labelsize=10)
        color_axes.set_title("rows per hexagon (sample, log)", fontsize=10, color=MUTED, pad=4)

    info = save_figure(fig, Path(figures_dir) / "03_q_c_map.png")
    info["heads"] = {head: {key: (value if isinstance(value, int) else round(value, 6))
                            for key, value in s.items()} for head, s in stats.items()}
    return info
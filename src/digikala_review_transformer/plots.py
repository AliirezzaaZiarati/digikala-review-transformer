"""Chart builders for the baseline phase. Each function reads saved tables, draws one figure, saves it
and returns info about the saved file. The notebook only calls these functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

from .plot_style import (
    AMBER,
    CLASS_COLORS,
    CONTENT_BOTTOM,
    CONTENT_TOP,
    CORAL,
    MODEL_COLORS,
    PERSIAN_FONT,
    SLATE_LIGHT,
    TEXT,
    TURQUOISE,
    add_tile,
    glass_legend,
    new_canvas,
    save_figure,
    split_row,
)

NEUTRAL = "#CBD5E1"                 # 'all rows' bar (low-saturation blue-grey)
LEGEND_ANCHOR_Y = -0.24             # legend sits below the axes, under tick labels / axis label
TILE_PAD_BOTTOM = 0.19              # figure fraction reserved under the axes for labels + legend


def _single_row(table, **conditions):
    """Return the one row matching all column==value conditions; raise if 0 or more than 1 match."""
    mask = pd.Series(True, index=table.index)
    for column, value in conditions.items():
        mask &= table[column] == value
    if int(mask.sum()) != 1:
        raise ValueError(f"expected exactly 1 row for {conditions}, found {int(mask.sum())}")
    return table[mask].iloc[0]


def chart_01_dummy_vs_models_and_overlap(tables_dir, figures_dir, model_tag, split, label_view,
                                         variant, overlap_splits):
    """Left: dummy vs model (accuracy, Macro F1) for rec and band. Right: share of rows whose exact body is in train."""
    tables_dir = Path(tables_dir)
    dummy_table = pd.read_csv(tables_dir / "dummy_baseline.csv")
    metrics_table = pd.read_csv(tables_dir / "metrics_summary.csv")
    overlap_table = pd.read_csv(tables_dir / "overlap_with_train.csv")
    if len(overlap_splits) != 2:
        raise ValueError("overlap_splits must hold exactly two split names")

    # ---------- data: left tile ----------
    metric_defs = [("rec", "accuracy", "Rec\naccuracy"), ("rec", "macro_f1", "Rec\nMacro F1"),
                   ("band", "accuracy", "Band\naccuracy"), ("band", "macro_f1", "Band\nMacro F1")]
    dummy_values, model_values = [], []
    for head, metric, _ in metric_defs:
        dummy_row = _single_row(dummy_table, model_id=f"dummy_{head}_{variant}", split=split, label_view=label_view)
        model_row = _single_row(metrics_table, model_id=f"{model_tag}_{head}_{variant}", split=split,
                                label_view=label_view)
        dummy_values.append(float(dummy_row[metric]))
        model_values.append(float(model_row[metric]))
    dummy_values, model_values = np.array(dummy_values), np.array(model_values)
    assert ((dummy_values >= 0) & (dummy_values <= 1) & (model_values >= 0) & (model_values <= 1)).all()
    gap_points = (model_values - dummy_values) * 100            # percentage points

    # ---------- data: right tile ----------
    group_defs = [("all", "all rows", NEUTRAL),
                  ("recommended", "recommended", CLASS_COLORS["recommended"]),
                  ("not_recommended", "not_recommended", CLASS_COLORS["not_recommended"]),
                  ("no_idea", "no_idea", CLASS_COLORS["no_idea"]),
                  ("label_missing", "rec label missing", SLATE_LIGHT)]
    overlap_values = {}
    for split_name in overlap_splits:
        rows = [_single_row(overlap_table, split=split_name, group=key) for key, _, _ in group_defs]
        values = np.array([float(r["pct_in_train"]) for r in rows])
        counts_pct = np.array([100.0 * r["n_in_train"] / r["n"] for r in rows])
        assert np.allclose(values, counts_pct, atol=1e-6), "pct_in_train disagrees with n_in_train / n"
        overlap_values[split_name] = values
    overall_pct = overlap_values[overlap_splits[-1]][0]          # 'all rows' of the last split

    # ---------- canvas ----------
    fig = new_canvas(
        title="Dummy baseline vs TF-IDF model, and train overlap",
        subtitle=(f"Accuracy beats the dummy by only {gap_points[0]:.1f} / {gap_points[2]:.1f} points; "
                  f"{overall_pct:.0f}% of {overlap_splits[-1]} bodies also occur in train."),
        footer=("Rec = recommendation_status, Band = rating_band. Dummy always predicts the majority class "
                "of the training labels. Gaps in percentage points."))
    left_rect, right_rect = split_row(2, weights=[1.2, 0.8])

    # ---------- left tile ----------
    ax_left = add_tile(fig, *left_rect,
                       tile_title=f"{split.capitalize()} split, {label_view} labels: dummy vs model {variant}",
                       pad_bottom=TILE_PAD_BOTTOM)
    centers = np.array([0.0, 1.0, 2.4, 3.4])                    # wider gap between the rec and band groups
    bar_width, offset = 0.36, 0.2
    ax_left.bar(centers - offset, dummy_values, width=bar_width, color=MODEL_COLORS["dummy"])
    ax_left.bar(centers + offset, model_values, width=bar_width, color=MODEL_COLORS[variant])
    for center, dummy_value, model_value, gap in zip(centers, dummy_values, model_values, gap_points):
        ax_left.text(center - offset, dummy_value + 0.012, f"{dummy_value:.3f}", ha="center", va="bottom",
                     color=TEXT, fontsize=11)
        ax_left.text(center + offset, model_value + 0.012, f"{model_value:.3f}", ha="center", va="bottom",
                     color=TEXT, fontsize=11)
        ax_left.text(center, max(dummy_value, model_value) + 0.075, f"{gap:+.1f} pts", ha="center",
                     va="bottom", color=AMBER, fontsize=12, fontweight="bold")
    ax_left.set_xticks(centers)
    ax_left.set_xticklabels([label for _, _, label in metric_defs], fontsize=12)
    ax_left.set_xlim(-0.6, 4.0)
    ax_left.set_ylim(0, 1.0)
    ax_left.set_ylabel("score (0 to 1)")
    glass_legend(ax_left,
                 handles=[Patch(color=MODEL_COLORS["dummy"], label="majority-class dummy"),
                          Patch(color=MODEL_COLORS[variant], label=f"TF-IDF + LR, model {variant}")],
                 loc="upper center", bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y), ncol=2)

    # ---------- right tile ----------
    ax_right = add_tile(fig, *right_rect, tile_title="Exact body also present in train (% of rows)",
                        pad_left=0.115, pad_bottom=TILE_PAD_BOTTOM)
    y_centers = np.arange(len(group_defs))[::-1].astype(float)   # first group on top
    bar_height, y_offset = 0.36, 0.2
    for split_index, split_name in enumerate(overlap_splits):
        values = overlap_values[split_name]
        y_positions = y_centers + (y_offset if split_index == 0 else -y_offset)
        if split_index == 0:                                      # first split: solid bars
            ax_right.barh(y_positions, values, height=bar_height, color=[c for _, _, c in group_defs])
        else:                                                     # second split: light + hatch (not colour only)
            ax_right.barh(y_positions, values, height=bar_height,
                          color=[to_rgba(c, 0.5) for _, _, c in group_defs],
                          hatch="//", edgecolor=(1, 1, 1, 0.6), linewidth=0.0)
        for y_position, value in zip(y_positions, values):
            ax_right.text(value + 0.8, y_position, f"{value:.1f}%", ha="left", va="center", color=TEXT,
                          fontsize=11)
    ax_right.set_yticks(y_centers)
    ax_right.set_yticklabels([label for _, label, _ in group_defs], fontsize=12)
    ax_right.set_xlim(0, 55)
    ax_right.set_xlabel("% of rows whose exact body is in train")
    ax_right.grid(axis="y", visible=False)
    ax_right.grid(axis="x", color=(1, 1, 1, 0.12), linewidth=0.8)
    glass_legend(ax_right,
                 handles=[Patch(facecolor=NEUTRAL, label=overlap_splits[0]),
                          Patch(facecolor=to_rgba(NEUTRAL, 0.5), edgecolor=(1, 1, 1, 0.6), hatch="//",
                                label=overlap_splits[1])],
                 loc="upper center", bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y), ncol=2)

    info = save_figure(fig, Path(figures_dir) / "01_dummy_vs_models_and_overlap.png")
    info["left_values"] = {f"{h}_{m}": {"dummy": round(float(d), 4), "model": round(float(v), 4)}
                           for (h, m, _), d, v in zip(metric_defs, dummy_values, model_values)}
    info["overlap_values"] = {s: [round(float(x), 2) for x in v] for s, v in overlap_values.items()}
    return info

def _read_curve(table, model_id):
    """Rows of one model sorted by C; 'converged' and 'selected' converted to real booleans."""
    curve = table[table["model_id"] == model_id].copy()
    if curve.empty:
        raise ValueError(f"no rows for model_id={model_id}")
    for column in ("converged", "selected"):
        curve[column] = curve[column].astype(str).str.lower().eq("true")
    curve["C"] = curve["C"].astype(float)
    curve = curve.sort_values("C").reset_index(drop=True)
    if curve["C"].duplicated().any():
        raise ValueError(f"duplicate C values for {model_id}")
    return curve


def _draw_curve(ax, curve, color, linestyle, line_width, alpha, marker_size):
    """Line + filled dots for converged fits + cross markers for fits that did not converge."""
    ax.plot(curve["C"], curve["val_macro_f1"], color=color, linestyle=linestyle,
            linewidth=line_width, alpha=alpha, zorder=3)
    converged = curve[curve["converged"]]
    not_converged = curve[~curve["converged"]]
    ax.scatter(converged["C"], converged["val_macro_f1"], s=marker_size, color=color, alpha=alpha, zorder=4)
    ax.scatter(not_converged["C"], not_converged["val_macro_f1"], s=marker_size * 2.2, marker="x",
               color=color, linewidths=2.4, alpha=alpha, zorder=5)


def chart_02_c_tuning_curves(tables_dir, figures_dir, model_tag, heads, variants, highlight_variant):
    """One tile per head: validation Macro F1 vs C. Dashed = tuning stage, solid = final fit, star = selected C."""
    tables_dir = Path(tables_dir)
    tuning_table = pd.read_csv(tables_dir / "tuning_results.csv")
    final_table = pd.read_csv(tables_dir / "final_fit.csv")
    head_titles = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
    unknown_heads = [head for head in heads if head not in head_titles]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")
    if highlight_variant not in variants:
        raise ValueError("highlight_variant must be one of variants")
    tune_sizes = tuning_table["tune_n"].unique()
    if len(tune_sizes) != 1:
        raise ValueError(f"tuning used several subset sizes: {tune_sizes}")
    tune_label = f"{int(tune_sizes[0]) // 1000}k"

    # ---------- data: curves and picks per (head, variant) ----------
    curves, picks = {}, {}
    for head in heads:
        for variant in variants:
            model_id = f"{model_tag}_{head}_{variant}"
            tuning_curve = _read_curve(tuning_table, model_id)
            final_curve = _read_curve(final_table, model_id)
            final_pick = _single_row(final_curve, selected=True)
            tuning_pick = _single_row(tuning_curve, selected=True)
            for values in (tuning_curve["val_macro_f1"], final_curve["val_macro_f1"]):
                assert ((values >= 0) & (values <= 1)).all(), f"Macro F1 outside [0, 1] for {model_id}"
            curves[(head, variant)] = (tuning_curve, final_curve)
            picks[(head, variant)] = {
                "final_C": float(final_pick["C"]),
                "final_f1": float(final_pick["val_macro_f1"]),
                "final_is_max_f1": bool(final_pick["val_macro_f1"] >= final_curve["val_macro_f1"].max()),
                "tuning_C": float(tuning_pick["C"]),
            }

    final_text = ", ".join(f"{picks[(head, highlight_variant)]['final_C']:g} {head}" for head in heads)
    tuning_text = ", ".join(f"{picks[(head, highlight_variant)]['tuning_C']:g} {head}" for head in heads)

    # ---------- canvas ----------
    fig = new_canvas(
        title="Choosing C: tuning stage vs final fit",
        subtitle=(f"Model {highlight_variant}: final peak C = {final_text}; "
                  f"tuning stage ({tune_label} rows) picked {tuning_text}."),
        footer=(f"Solid: final fit on all rows. Dashed: tuning on {tune_label} rows. Cross: not converged. "
                "Star: selected C. A = raw labels, B = corrected labels."))
    tile_rects = split_row(len(heads))

    # ---------- one tile per head ----------
    for tile_rect, head in zip(tile_rects, heads):
        ax = add_tile(fig, *tile_rect, tile_title=f"{head_titles[head]}: validation Macro F1 vs C",
                      pad_left=0.075, pad_bottom=TILE_PAD_BOTTOM)
        all_c, all_f1 = [], []
        for variant in variants:
            tuning_curve, final_curve = curves[(head, variant)]
            color = MODEL_COLORS[variant]
            _draw_curve(ax, tuning_curve, color, "--", 2.0, 0.55, 40)       # tuning: dashed, lighter
            _draw_curve(ax, final_curve, color, "-", 3.2, 1.0, 60)          # final: solid
            pick = picks[(head, variant)]
            ax.scatter([pick["final_C"]], [pick["final_f1"]], marker="*", s=520, color=TEXT,
                       edgecolors=color, linewidths=2.0, zorder=6)
            all_c += list(tuning_curve["C"]) + list(final_curve["C"])
            all_f1 += list(tuning_curve["val_macro_f1"]) + list(final_curve["val_macro_f1"])

        c_values = np.unique(np.array(all_c))
        f1_low, f1_high = min(all_f1), max(all_f1)
        f1_span = f1_high - f1_low
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(c_values)
        ax.set_xticklabels([f"{c:g}" for c in c_values], fontsize=12)
        ax.set_xlim(c_values.min() / 1.35, c_values.max() * 1.35)
        ax.set_ylim(f1_low - 0.10 * f1_span, f1_high + 0.35 * f1_span)   # headroom for the text lines
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel("C (inverse regularisation strength, log scale)")
        ax.set_ylabel("validation Macro F1")

        # selected values as text lines in the empty headroom (top-left)
        for line_index, variant in enumerate(variants):
            pick = picks[(head, variant)]
            ax.text(0.03, 0.965 - 0.075 * line_index,
                    f"Model {variant}: final C = {pick['final_C']:g}, Macro F1 = {pick['final_f1']:.4f}",
                    transform=ax.transAxes, ha="left", va="top", color=MODEL_COLORS[variant],
                    fontsize=12, fontweight="bold")

        legend_handles = [Line2D([], [], color=MODEL_COLORS[v], linewidth=3, marker="o", markersize=7,
                                 label=f"Model {v}") for v in variants]
        legend_handles.append(Line2D([], [], color=NEUTRAL, linewidth=2.2, linestyle="--", label="tuning stage"))
        legend_handles.append(Line2D([], [], color="none", marker="*", markerfacecolor=TEXT,
                                     markeredgecolor=NEUTRAL, markersize=15, label="selected C"))
        glass_legend(ax, handles=legend_handles, loc="upper center",
                     bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y), ncol=len(legend_handles))

    info = save_figure(fig, Path(figures_dir) / "02_c_tuning_curves.png")
    info["picks"] = {f"{head}_{variant}": {key: (round(value, 4) if isinstance(value, float) else value)
                                           for key, value in pick.items()}
                     for (head, variant), pick in picks.items()}
    drawn_tuning = pd.concat([curves[key][0] for key in curves])
    info["tuning_fits"] = len(drawn_tuning)
    info["tuning_not_converged"] = int((~drawn_tuning["converged"]).sum())
    return info


# ---------------------------------------------------------------------------------------------
# Chart 03: confusion matrices
# ---------------------------------------------------------------------------------------------
HEAD_CLASS_ORDER = {"rec": ["recommended", "not_recommended", "no_idea"], "band": ["low", "medium", "high"]}
# dark navy -> blue -> light turquoise (no purple, no pink, not red/green)
CONFUSION_CMAP = LinearSegmentedColormap.from_list("aurora_blue_teal", ["#10182F", "#1F6FA8", "#5EEAD4"])
CELL_DARK_TEXT = "#0B1020"       # text colour on light cells
LIGHT_CELL_LUMINANCE = 0.55      # cells brighter than this get dark text


def _confusion_arrays(confusion_table, model_id, split, label_view, classes):
    """3x3 count matrix and stored row_pct matrix (rows = true class, columns = predicted class)."""
    rows = confusion_table[(confusion_table["model_id"] == model_id) & (confusion_table["split"] == split)
                           & (confusion_table["label_view"] == label_view)]
    if len(rows) != len(classes) ** 2:
        raise ValueError(f"{model_id}/{split}/{label_view}: expected {len(classes) ** 2} rows, found {len(rows)}")
    arrays = []
    for value_column in ("count", "row_pct"):
        matrix = rows.pivot(index="true", columns="pred", values=value_column)   # raises on duplicate cells
        matrix = matrix.reindex(index=classes, columns=classes)
        if matrix.isna().any().any():
            raise ValueError(f"{model_id}: class names in confusion_long do not match {classes}")
        arrays.append(matrix.to_numpy(dtype=float))
    return arrays[0].astype(np.int64), arrays[1]


def chart_03_confusion_matrices(tables_dir, figures_dir, model_tag, split, label_view, variant, heads):
    """One tile per head: row-normalised confusion matrix (% of true class) with raw counts below each %."""
    tables_dir = Path(tables_dir)
    confusion_table = pd.read_csv(tables_dir / "confusion_long.csv")
    metrics_table = pd.read_csv(tables_dir / "metrics_summary.csv")
    per_class_table = pd.read_csv(tables_dir / "metrics_per_class.csv")
    head_titles = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
    unknown_heads = [head for head in heads if head not in head_titles]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")

    # ---------- data and checks per head ----------
    results = {}
    for head in heads:
        classes = HEAD_CLASS_ORDER[head]
        model_id = f"{model_tag}_{head}_{variant}"
        counts, stored_row_pct = _confusion_arrays(confusion_table, model_id, split, label_view, classes)
        row_totals = counts.sum(axis=1)
        assert (row_totals > 0).all(), f"{model_id}: a true class has no rows"
        row_pct = counts / row_totals[:, None] * 100
        assert np.allclose(row_pct, stored_row_pct, atol=1e-4), f"{model_id}: row_pct differs from counts"

        metrics_row = _single_row(metrics_table, model_id=model_id, split=split, label_view=label_view)
        assert int(counts.sum()) == int(metrics_row["n"]), f"{model_id}: confusion total != n in metrics_summary"
        recall_pct = np.diag(row_pct)
        for index, class_name in enumerate(classes):            # diagonal must equal per-class recall
            class_row = _single_row(per_class_table, model_id=model_id, split=split, label_view=label_view,
                                    **{"class": class_name})
            assert abs(float(class_row["recall"]) * 100 - recall_pct[index]) < 1e-4, \
                f"{model_id}/{class_name}: diagonal != recall in metrics_per_class"

        off_diagonal = row_pct.copy()
        np.fill_diagonal(off_diagonal, -1.0)
        true_index, pred_index = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
        adjacent_pct = None
        if head == "band":                                       # ordinal head: share of errors next to the diagonal
            distance = np.abs(np.subtract.outer(np.arange(len(classes)), np.arange(len(classes))))
            adjacent_pct = 100.0 * counts[distance == 1].sum() / counts[distance > 0].sum()
            assert abs(adjacent_pct - float(metrics_row["neighbor_error_pct"])) < 0.01, \
                "adjacent-error share differs from neighbor_error_pct in metrics_summary"
        results[head] = {"classes": classes, "counts": counts, "row_pct": row_pct, "recall_pct": recall_pct,
                         "n": int(counts.sum()), "adjacent_pct": adjacent_pct,
                         "largest_off": (classes[true_index], classes[pred_index], float(row_pct[true_index, pred_index]))}

    # ---------- subtitle from the data ----------
    parts = []
    for head in heads:
        lowest = int(np.argmin(results[head]["recall_pct"]))
        parts.append(f"{results[head]['classes'][lowest]} {results[head]['recall_pct'][lowest]:.1f}% ({head})")
    subtitle = "Lowest recall: " + ", ".join(parts) + "."
    if results.get("band") is not None:
        subtitle += f" {results['band']['adjacent_pct']:.0f}% of band errors are adjacent classes."

    # ---------- canvas ----------
    fig = new_canvas(
        title="Confusion matrices of the reference models",
        subtitle=subtitle,
        footer=(f"{split.capitalize()} split, {label_view} labels, model {variant}. Large number: % of true-class row "
                "(outlined diagonal = recall); small number: rows. Rec: labeled rows only."))
    tile_rects = split_row(len(heads))

    # ---------- one tile per head ----------
    for tile_rect, head in zip(tile_rects, heads):
        result = results[head]
        classes, counts, row_pct = result["classes"], result["counts"], result["row_pct"]
        ax = add_tile(fig, *tile_rect, tile_title=f"{head_titles[head]}: true vs predicted (n = {result['n']:,})",
                      pad_left=0.14, pad_bottom=0.11)
        ax.imshow(row_pct, cmap=CONFUSION_CMAP, vmin=0, vmax=100, aspect="auto")
        ax.grid(False)
        for k in range(len(classes) - 1):                        # thin separators between cells
            ax.axhline(k + 0.5, color=(1, 1, 1, 0.25), linewidth=1.2)
            ax.axvline(k + 0.5, color=(1, 1, 1, 0.25), linewidth=1.2)
        for i in range(len(classes)):
            for j in range(len(classes)):
                red, green, blue, _ = CONFUSION_CMAP(row_pct[i, j] / 100)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = CELL_DARK_TEXT if luminance > LIGHT_CELL_LUMINANCE else TEXT
                ax.text(j, i - 0.1, f"{row_pct[i, j]:.1f}%", ha="center", va="center", color=text_color,
                        fontsize=22, fontweight="bold")
                ax.text(j, i + 0.22, f"{counts[i, j]:,}", ha="center", va="center", color=text_color, fontsize=13)
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=AMBER, linewidth=3, zorder=4))
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=12)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=12)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")

    info = save_figure(fig, Path(figures_dir) / "03_confusion_matrices.png")
    info["heads"] = {head: {"n": r["n"],
                            "recall_pct": [round(float(x), 2) for x in r["recall_pct"]],
                            "row_pct": [[round(float(x), 2) for x in row] for row in r["row_pct"]],
                            "largest_off_diagonal": (r["largest_off"][0], r["largest_off"][1], round(r["largest_off"][2], 2)),
                            "adjacent_error_pct": None if r["adjacent_pct"] is None else round(r["adjacent_pct"], 2)}
                     for head, r in results.items()}
    return info


# ---------------------------------------------------------------------------------------------
# Chart 04: per-class precision / recall / F1
# ---------------------------------------------------------------------------------------------
# (metric column, legend label, bar alpha, hatch): the metric is told apart by fill style, class by colour
PER_CLASS_METRIC_STYLES = [("precision", "Precision", 0.5, "//"),
                           ("recall", "Recall", 1.0, None),
                           ("f1", "F1", 0.75, "..")]
HATCH_EDGE = (1, 1, 1, 0.6)


def chart_04_per_class_metrics(tables_dir, figures_dir, model_tag, split, label_view, variant, heads):
    """One tile per head: precision, recall and F1 of every class, with true/predicted row counts under each class."""
    tables_dir = Path(tables_dir)
    per_class_table = pd.read_csv(tables_dir / "metrics_per_class.csv")
    metrics_table = pd.read_csv(tables_dir / "metrics_summary.csv")
    head_titles = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
    unknown_heads = [head for head in heads if head not in head_titles]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")

    # ---------- data and checks per head ----------
    results = {}
    for head in heads:
        classes = HEAD_CLASS_ORDER[head]
        model_id = f"{model_tag}_{head}_{variant}"
        class_rows = [_single_row(per_class_table, model_id=model_id, split=split, label_view=label_view,
                                  **{"class": class_name}) for class_name in classes]
        scores = {name: np.array([float(row[name]) for row in class_rows]) for name, _, _, _ in PER_CLASS_METRIC_STYLES}
        support = np.array([int(row["support"]) for row in class_rows])
        predicted = np.array([int(row["predicted_count"]) for row in class_rows])
        for name, values in scores.items():
            assert ((values >= 0) & (values <= 1)).all(), f"{model_id}: {name} outside [0, 1]"
        metrics_row = _single_row(metrics_table, model_id=model_id, split=split, label_view=label_view)
        assert int(support.sum()) == int(metrics_row["n"]), f"{model_id}: sum of support != n in metrics_summary"
        assert int(predicted.sum()) == int(support.sum()), f"{model_id}: predicted_count total != support total"
        assert abs(scores["f1"].mean() - float(metrics_row["macro_f1"])) < 1e-4, \
            f"{model_id}: mean of class F1 != macro_f1 in metrics_summary"
        weakest = int(np.argmin(scores["f1"]))
        results[head] = {"classes": classes, "scores": scores, "support": support, "predicted": predicted,
                         "weakest": weakest, "weakest_ratio": float(predicted[weakest] / support[weakest])}

    # ---------- subtitle from the data ----------
    weakest_text = ", ".join(f"{results[head]['classes'][results[head]['weakest']]} "
                             f"{results[head]['scores']['f1'][results[head]['weakest']]:.3f} ({head})" for head in heads)
    ratio_text = " and ".join(f"{results[head]['weakest_ratio']:.2f}x" for head in heads)
    subtitle = f"Lowest F1: {weakest_text}; predicted {ratio_text} their true count."

    # ---------- canvas ----------
    fig = new_canvas(
        title="Per-class precision, recall and F1",
        subtitle=subtitle,
        footer=(f"{split.capitalize()}, {label_view} labels, model {variant}. Precision = right among predicted; "
                "recall = found among true. Under each class: true count / predicted count."))
    tile_rects = split_row(len(heads))

    # ---------- one tile per head ----------
    for tile_rect, head in zip(tile_rects, heads):
        result = results[head]
        classes = result["classes"]
        ax = add_tile(fig, *tile_rect, tile_title=f"{head_titles[head]}: metrics per class",
                      pad_left=0.075, pad_bottom=0.18)
        centers = np.arange(len(classes), dtype=float)
        class_colors = [CLASS_COLORS[class_name] for class_name in classes]
        bar_width, spacing = 0.26, 0.28
        for metric_index, (name, _, alpha, hatch) in enumerate(PER_CLASS_METRIC_STYLES):
            x_positions = centers + (metric_index - 1) * spacing
            values = result["scores"][name]
            bar_kwargs = {"hatch": hatch, "edgecolor": HATCH_EDGE, "linewidth": 0.0} if hatch else {}
            ax.bar(x_positions, values, width=bar_width, color=[to_rgba(c, alpha) for c in class_colors], **bar_kwargs)
            for x_position, value in zip(x_positions, values):
                ax.text(x_position, value + 0.012, f"{value:.3f}", ha="center", va="bottom", color=TEXT, fontsize=10)
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{class_name}\ntrue {int(result['support'][index]):,}\npred {int(result['predicted'][index]):,}"
                            for index, class_name in enumerate(classes)], fontsize=11)
        ax.set_xlim(-0.6, len(classes) - 0.4)
        ax.set_ylim(0, 1.25)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("score (0 to 1)")
        ax.set_xlabel("class")
        legend_handles = [Patch(facecolor=to_rgba(NEUTRAL, alpha), edgecolor=HATCH_EDGE, hatch=hatch, label=label)
                          for _, label, alpha, hatch in PER_CLASS_METRIC_STYLES]
        glass_legend(ax, handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                     ncol=len(legend_handles))

    info = save_figure(fig, Path(figures_dir) / "04_per_class_metrics.png")
    info["heads"] = {head: {"classes": r["classes"],
                            **{name: [round(float(x), 4) for x in r["scores"][name]] for name in r["scores"]},
                            "support": [int(x) for x in r["support"]],
                            "predicted": [int(x) for x in r["predicted"]],
                            "weakest_class": r["classes"][r["weakest"]],
                            "weakest_pred_over_true": round(r["weakest_ratio"], 3)}
                     for head, r in results.items()}
    return info


# ---------------------------------------------------------------------------------------------
# Chart 05: same predictions scored on two label views
# ---------------------------------------------------------------------------------------------
LABEL_VIEW_METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                      ("balanced_accuracy", "Balanced\naccuracy")]


def chart_05_label_views(tables_dir, figures_dir, model_tag, split, variant, heads, label_views):
    """One tile per head: accuracy, Macro F1 and balanced accuracy on two label views (first solid, second hatched)."""
    tables_dir = Path(tables_dir)
    metrics_table = pd.read_csv(tables_dir / "metrics_summary.csv")
    head_titles = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
    unknown_heads = [head for head in heads if head not in head_titles]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")
    if len(label_views) != 2:
        raise ValueError("label_views must hold exactly two view names")
    first_view, second_view = label_views
    f1_index = [column for column, _ in LABEL_VIEW_METRICS].index("macro_f1")

    # ---------- data and checks per head ----------
    results = {}
    for head in heads:
        model_id = f"{model_tag}_{head}_{variant}"
        views = {}
        for view in label_views:
            row = _single_row(metrics_table, model_id=model_id, split=split, label_view=view)
            values = np.array([float(row[column]) for column, _ in LABEL_VIEW_METRICS])
            assert ((values >= 0) & (values <= 1)).all(), f"{model_id}/{view}: a metric is outside [0, 1]"
            assert abs(float(row["balanced_accuracy"]) - float(row["macro_recall"])) < 1e-9, \
                f"{model_id}/{view}: balanced_accuracy != macro_recall"
            views[view] = values
        results[head] = {"views": views, "gap_points": (views[second_view] - views[first_view]) * 100}

    # ---------- subtitle from the data ----------
    gap_text = " and ".join(f"{results[head]['gap_points'][f1_index]:+.2f} pts ({head})" for head in heads)
    subtitle = f"Macro F1 changes by {gap_text} on {second_view} labels."

    # ---------- canvas ----------
    fig = new_canvas(
        title="Same predictions, two label views",
        subtitle=subtitle,
        footer=(f"Raw = original labels (benchmark); corrected = step-1 labels. {split.capitalize()} split, "
                f"model {variant}; same predictions, only the reference labels differ."))
    tile_rects = split_row(len(heads))

    # ---------- one tile per head ----------
    model_color = MODEL_COLORS[variant]
    for tile_rect, head in zip(tile_rects, heads):
        result = results[head]
        ax = add_tile(fig, *tile_rect, tile_title=f"{head_titles[head]}: model {variant}, {split}",
                      pad_left=0.075, pad_bottom=0.14)
        centers = np.arange(len(LABEL_VIEW_METRICS), dtype=float)
        bar_width, offset = 0.36, 0.2
        first_values, second_values = result["views"][first_view], result["views"][second_view]
        ax.bar(centers - offset, first_values, width=bar_width, color=model_color)
        ax.bar(centers + offset, second_values, width=bar_width, color=to_rgba(model_color, 0.5),
               hatch="//", edgecolor=HATCH_EDGE, linewidth=0.0)
        for center, first_value, second_value, gap in zip(centers, first_values, second_values,
                                                          result["gap_points"]):
            ax.text(center - offset, first_value + 0.012, f"{first_value:.4f}", ha="center", va="bottom",
                    color=TEXT, fontsize=10)
            ax.text(center + offset, second_value + 0.012, f"{second_value:.4f}", ha="center", va="bottom",
                    color=TEXT, fontsize=10)
            ax.text(center, max(first_value, second_value) + 0.075, f"{gap:+.2f} pts", ha="center",
                    va="bottom", color=AMBER, fontsize=12, fontweight="bold")
        ax.set_xticks(centers)
        ax.set_xticklabels([label for _, label in LABEL_VIEW_METRICS], fontsize=12)
        ax.set_xlim(-0.6, len(LABEL_VIEW_METRICS) - 0.4)
        ax.set_ylim(0, 1.25)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("score (0 to 1)")
        ax.set_xlabel("metric")
        glass_legend(ax,
                     handles=[Patch(facecolor=model_color, label=f"{first_view} labels"),
                              Patch(facecolor=to_rgba(model_color, 0.5), edgecolor=HATCH_EDGE, hatch="//",
                                    label=f"{second_view} labels")],
                     loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2)

    info = save_figure(fig, Path(figures_dir) / "05_label_views.png")
    info["heads"] = {head: {**{view: {column: round(float(value), 4)
                                      for (column, _), value in zip(LABEL_VIEW_METRICS, r["views"][view])}
                               for view in label_views},
                            "gap_pts": {column: round(float(value), 3)
                                        for (column, _), value in zip(LABEL_VIEW_METRICS, r["gap_points"])}}
                     for head, r in results.items()}
    return info

# ---------------------------------------------------------------------------------------------
# Chart 06: model B minus model A
# ---------------------------------------------------------------------------------------------
AB_METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("balanced_accuracy", "Balanced\naccuracy")]
AB_TOLERANCE = 1e-8              # table consistency checks
AB_LIMIT_FACTOR = 1.8            # x-limit = factor * largest |difference| (room for value labels)


def chart_06_a_vs_b(tables_dir, figures_dir, model_tag, split, label_view, heads):
    """One tile per head: B minus A (percentage points) for accuracy, Macro F1, balanced accuracy.
    Same x range in all tiles; bars right of zero (B higher) are blue, bars left (A higher) are grey and hatched."""
    tables_dir = Path(tables_dir)
    ab_table = pd.read_csv(tables_dir / "ab_comparison.csv")
    metrics_table = pd.read_csv(tables_dir / "metrics_summary.csv")
    head_titles = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
    unknown_heads = [head for head in heads if head not in head_titles]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")

    # ---------- data and checks per head ----------
    results = {}
    for head in heads:
        ab_row = _single_row(ab_table, head=head, split=split, label_view=label_view)
        diff_points = []
        for column, _ in AB_METRICS:
            value_a, value_b = float(ab_row[f"{column}_A"]), float(ab_row[f"{column}_B"])
            stored_diff = float(ab_row[f"{column}_diff_B_minus_A"])
            assert 0 <= value_a <= 1 and 0 <= value_b <= 1, f"{head}/{column}: value outside [0, 1]"
            assert abs((value_b - value_a) - stored_diff) < AB_TOLERANCE, f"{head}/{column}: diff != B - A"
            for variant, value in (("A", value_a), ("B", value_b)):          # cross-check with metrics_summary
                summary_row = _single_row(metrics_table, model_id=f"{model_tag}_{head}_{variant}",
                                          split=split, label_view=label_view)
                assert abs(float(summary_row[column]) - value) < AB_TOLERANCE, \
                    f"{head}/{variant}/{column}: ab_comparison differs from metrics_summary"
            diff_points.append(stored_diff * 100)                           # percentage points
        assert abs(float(ab_row["balanced_accuracy_diff_B_minus_A"])
                   - float(ab_row["macro_recall_diff_B_minus_A"])) < AB_TOLERANCE, \
            f"{head}: balanced_accuracy diff != macro_recall diff"
        disagreement = float(ab_row["pred_disagreement_pct"])
        assert 0 <= disagreement <= 100, f"{head}: pred_disagreement_pct outside [0, 100]"
        results[head] = {"diff_points": np.array(diff_points), "disagreement_pct": disagreement}

    max_abs_points = max(float(np.abs(r["diff_points"]).max()) for r in results.values())
    if max_abs_points <= 0:
        raise ValueError("all A/B differences are zero; nothing to draw")
    axis_limit = AB_LIMIT_FACTOR * max_abs_points

    # ---------- subtitle and canvas ----------
    disagreement_text = ", ".join(f"{results[head]['disagreement_pct']:.2f}% ({head})" for head in heads)
    fig = new_canvas(
        title="Model B minus model A",
        subtitle=(f"B minus A is at most {max_abs_points:.2f} pts on any metric; "
                  f"predictions differ on {disagreement_text} of rows."),
        footer=(f"{split.capitalize()} split, {label_view} labels. Bars: B minus A in percentage points. "
                "No significance test was run, so no model is declared better."))
    tile_rects = split_row(len(heads))

    # ---------- one tile per head ----------
    color_b, color_a = MODEL_COLORS["B"], MODEL_COLORS["A"]
    for tile_rect, head in zip(tile_rects, heads):
        result = results[head]
        ax = add_tile(fig, *tile_rect, tile_title=f"{head_titles[head]}: B minus A ({split}, {label_view})",
                      pad_left=0.115, pad_bottom=0.11)
        y_positions = np.arange(len(AB_METRICS))[::-1].astype(float)        # first metric on top
        for y_position, value in zip(y_positions, result["diff_points"]):
            if value >= 0:
                ax.barh(y_position, value, height=0.5, color=color_b)
            else:
                ax.barh(y_position, value, height=0.5, color=to_rgba(color_a, 0.75), hatch="//",
                        edgecolor=HATCH_EDGE, linewidth=0.0)
            label_offset = 0.02 * axis_limit
            ax.text(value + (label_offset if value >= 0 else -label_offset), y_position, f"{value:+.3f} pts",
                    ha="left" if value >= 0 else "right", va="center", color=TEXT, fontsize=12)
        ax.axvline(0, color=TEXT, linewidth=1.5, alpha=0.8, zorder=3)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-0.6, 3.2)                                              # headroom for the text lines
        ax.set_yticks(y_positions)
        ax.set_yticklabels([label for _, label in AB_METRICS], fontsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:+.2f}"))
        ax.set_xlabel("B minus A (percentage points)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=(1, 1, 1, 0.12), linewidth=0.8)
        ax.text(0.25, 0.965, "A higher", transform=ax.transAxes, ha="center", va="top", color=color_a,
                fontsize=13, fontweight="bold")
        ax.text(0.75, 0.965, "B higher", transform=ax.transAxes, ha="center", va="top", color=color_b,
                fontsize=13, fontweight="bold")
        ax.text(0.5, 0.885, f"A and B predict different classes on {result['disagreement_pct']:.2f}% of rows",
                transform=ax.transAxes, ha="center", va="top", color=TEXT, fontsize=12)

    info = save_figure(fig, Path(figures_dir) / "06_a_vs_b.png")
    info["axis_limit_pts"] = round(axis_limit, 4)
    info["heads"] = {head: {"diff_pts": {column: round(float(value), 4)
                                         for (column, _), value in zip(AB_METRICS, r["diff_points"])},
                            "disagreement_pct": round(r["disagreement_pct"], 3)}
                     for head, r in results.items()}
    return info

# ---------------------------------------------------------------------------------------------
# Chart 07: top n-gram features per class (Persian labels)
# ---------------------------------------------------------------------------------------------
NGRAM_ROW_GAP = 0.015            # vertical gap between tile rows (figure fraction)
NGRAM_X_LIMIT_FACTOR = 1.25      # x-limit = factor * largest coefficient of the head (room for value labels)
NGRAM_LABEL_SIZE = 11            # font size of the Persian feature labels
ZWNJ = "\u200c"


def _class_features(ngram_table, model_id, feature_filter, class_name):
    """Rows of one (model, filter, class) sorted by rank; raises if the table is inconsistent."""
    rows = ngram_table[(ngram_table["model_id"] == model_id) & (ngram_table["filter"] == feature_filter)
                       & (ngram_table["class"] == class_name)]
    if rows.empty:
        raise ValueError(f"no n-gram rows for {model_id} / {feature_filter} / {class_name}")
    rows = rows.sort_values("rank").reset_index(drop=True)
    for column in ("is_bigram", "has_zwnj"):
        rows[column] = rows[column].astype(str).str.lower().eq("true")
    rows["coef"] = rows["coef"].astype(float)
    where = f"{model_id} / {feature_filter} / {class_name}"
    assert (rows["rank"].astype(int).to_numpy() == np.arange(1, len(rows) + 1)).all(), f"{where}: ranks are not 1..n"
    assert (np.diff(rows["coef"].to_numpy()) <= 1e-12).all(), f"{where}: coefficients are not descending"
    assert (rows["df_train"].astype(int) > 0).all(), f"{where}: df_train must be positive"
    spaces = rows["feature"].astype(str).str.count(" ").to_numpy()
    assert (spaces == rows["is_bigram"].to_numpy().astype(int)).all(), f"{where}: is_bigram != one space in feature"
    return rows


def chart_07_top_ngrams(tables_dir, figures_dir, model_tag, variant, heads, feature_filter, top_n):
    """One row of tiles per head, one tile per class: top_n features by coefficient (hatched = bigram)."""
    tables_dir = Path(tables_dir)
    ngram_table = pd.read_csv(tables_dir / "top_ngrams.csv", keep_default_na=False)
    unknown_heads = [head for head in heads if head not in HEAD_CLASS_ORDER]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")
    column_counts = {len(HEAD_CLASS_ORDER[head]) for head in heads}
    if len(column_counts) != 1:
        raise ValueError("all heads must have the same number of classes")
    n_columns = column_counts.pop()
    available_filters = list(ngram_table["filter"].astype(str).unique())
    if feature_filter not in available_filters:
        raise ValueError(f"filter {feature_filter!r} not in table; available: {available_filters}")
    other_filters = [name for name in available_filters if name != feature_filter]

    # ---------- data and checks ----------
    results = {}
    same_in_other = {name: [] for name in other_filters}
    zwnj_shown, zwnj_flag_mismatch = [], 0
    for head in heads:
        model_id = f"{model_tag}_{head}_{variant}"
        per_class = {}
        for class_name in HEAD_CLASS_ORDER[head]:
            rows = _class_features(ngram_table, model_id, feature_filter, class_name)
            if not 1 <= top_n <= len(rows):
                raise ValueError(f"top_n must be between 1 and {len(rows)}")
            shown = rows.head(top_n)
            per_class[class_name] = shown
            for name in other_filters:      # does the other filter change the shown features?
                other_rows = _class_features(ngram_table, model_id, name, class_name)
                same_in_other[name].append(list(other_rows["feature"].head(top_n)) == list(shown["feature"]))
            has_zwnj_in_text = shown["feature"].astype(str).str.contains(ZWNJ, regex=False).to_numpy()
            zwnj_flag_mismatch += int((has_zwnj_in_text != shown["has_zwnj"].to_numpy()).sum())
            zwnj_shown += [str(feature) for feature in shown["feature"][has_zwnj_in_text]]
        all_flags = np.concatenate([shown["is_bigram"].to_numpy() for shown in per_class.values()])
        results[head] = {"per_class": per_class,
                         "max_coef": max(float(shown["coef"].max()) for shown in per_class.values()),
                         "bigram_pct": 100.0 * float(all_flags.mean())}

    # ---------- canvas ----------
    bigram_text = ", ".join(f"{results[head]['bigram_pct']:.0f}% ({head})" for head in heads)
    fig = new_canvas(
        title=f"Top n-gram features per class, model {variant}",
        subtitle=f"Bigram share of the top {top_n} features: {bigram_text}.",
        footer=(f"Top {top_n} features per class by coefficient, model {variant} (filter: {feature_filter}). "
                "Hatched = bigram. Coefficients are not true contribution shares."))
    column_rects = split_row(n_columns)
    row_height = (CONTENT_TOP - CONTENT_BOTTOM - NGRAM_ROW_GAP * (len(heads) - 1)) / len(heads)

    # ---------- one tile per (head, class) ----------
    for row_index, head in enumerate(heads):
        row_bottom = CONTENT_TOP - (row_index + 1) * row_height - row_index * NGRAM_ROW_GAP
        x_limit = NGRAM_X_LIMIT_FACTOR * results[head]["max_coef"]       # shared by the classes of one head
        for column_rect, class_name in zip(column_rects, HEAD_CLASS_ORDER[head]):
            left, _, width, _ = column_rect
            shown = results[head]["per_class"][class_name]
            color = CLASS_COLORS[class_name]
            ax = add_tile(fig, left, row_bottom, width, row_height,
                          tile_title=f"{head}: {class_name}", pad_left=0.08, pad_bottom=0.095)
            y_positions = np.arange(top_n)[::-1].astype(float)           # rank 1 on top
            for y_position, coef, is_bigram in zip(y_positions, shown["coef"], shown["is_bigram"]):
                if is_bigram:
                    ax.barh(y_position, coef, height=0.68, color=to_rgba(color, 0.6), hatch="//",
                            edgecolor=HATCH_EDGE, linewidth=0.0)
                else:
                    ax.barh(y_position, coef, height=0.68, color=color)
                ax.text(coef + 0.015 * x_limit, y_position, f"{coef:.2f}", ha="left", va="center",
                        color=TEXT, fontsize=10)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(shown["feature"].astype(str).tolist(), fontsize=NGRAM_LABEL_SIZE,
                               fontfamily=PERSIAN_FONT)
            ax.set_xlim(0, x_limit)
            ax.set_ylim(-0.6, top_n - 0.4)
            ax.set_xlabel("LR coefficient")
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=(1, 1, 1, 0.12), linewidth=0.8)

    info = save_figure(fig, Path(figures_dir) / "07_top_ngrams.png")
    info["top_features"] = {
        f"{head}/{class_name}": [f"{row.feature} ({row.coef:.2f}{', bigram' if row.is_bigram else ''})"
                                 for row in shown.itertuples()]
        for head in heads for class_name, shown in results[head]["per_class"].items()}
    info["bigram_pct"] = {head: round(results[head]["bigram_pct"], 1) for head in heads}
    info["zwnj_shown"] = zwnj_shown
    info["zwnj_flag_mismatch"] = zwnj_flag_mismatch
    info["same_top_n_in_other_filters"] = {name: all(flags) for name, flags in same_in_other.items()}
    return info


# ---------------------------------------------------------------------------------------------
# Chart 08: confidence bins (share of rows, accuracy on two label views, mean confidence)
# ---------------------------------------------------------------------------------------------
CONF_VALUE_LIMIT = 130           # y-limit in percent; the headroom holds the legend
CONF_LABEL_OFFSET = 4.5          # height of the accuracy label above its point (percent units)
CONF_TOLERANCE = 1e-5            # table consistency checks (tables hold rounded values)


def _confidence_bins(confidence_table, model_id):
    """Confidence bins of one model in ascending order; raises if the bins are incomplete or inconsistent."""
    bins = confidence_table[confidence_table["model_id"] == model_id].sort_values("conf_from").reset_index(drop=True)
    if bins.empty:
        raise ValueError(f"no confidence rows for {model_id}")
    assert abs(float(bins["conf_from"].iloc[0])) < 1e-9 and abs(float(bins["conf_to"].iloc[-1]) - 1.0) < 1e-9, \
        f"{model_id}: bins must cover 0 to 1"
    assert np.allclose(bins["conf_to"].to_numpy()[:-1], bins["conf_from"].to_numpy()[1:]), \
        f"{model_id}: bins are not contiguous"
    assert abs(float(bins["pct_rows"].sum()) - 100.0) < CONF_TOLERANCE, f"{model_id}: pct_rows does not sum to 100"
    assert np.allclose(bins["pct_rows"], bins["n"] / bins["n"].sum() * 100, atol=CONF_TOLERANCE), \
        f"{model_id}: pct_rows differs from n / total"
    for column in ("accuracy_raw", "accuracy_corrected", "mean_conf"):
        assert ((bins[column] >= 0) & (bins[column] <= 1)).all(), f"{model_id}: {column} outside [0, 1]"
    assert ((bins["n_raw"] <= bins["n"]) & (bins["n_corrected"] <= bins["n"])).all(), \
        f"{model_id}: labeled count larger than n"
    assert ((bins["mean_conf"] >= bins["conf_from"] - 1e-9) & (bins["mean_conf"] <= bins["conf_to"] + 1e-9)).all(), \
        f"{model_id}: mean_conf outside its own bin"
    return bins


def chart_08_confidence(tables_dir, figures_dir, model_tag, split, variant, heads, high_conf_from):
    """One tile per head: share of rows per confidence bin (bars), accuracy on corrected/raw labels (lines),
    mean confidence (rings). Accuracy uses labeled rows only."""
    tables_dir = Path(tables_dir)
    confidence_table = pd.read_csv(tables_dir / "confidence_table.csv")
    metrics_table = pd.read_csv(tables_dir / "metrics_summary.csv")
    head_titles = {"rec": "Recommendation (rec)", "band": "Rating band (band)"}
    unknown_heads = [head for head in heads if head not in head_titles]
    if unknown_heads:
        raise ValueError(f"unknown heads: {unknown_heads}")

    # ---------- data and checks per head ----------
    results = {}
    for head in heads:
        model_id = f"{model_tag}_{head}_{variant}"
        bins = _confidence_bins(confidence_table, model_id)
        for view, count_column, accuracy_column in (("raw", "n_raw", "accuracy_raw"),
                                                    ("corrected", "n_corrected", "accuracy_corrected")):
            metrics_row = _single_row(metrics_table, model_id=model_id, split=split, label_view=view)
            assert int(bins[count_column].sum()) == int(metrics_row["n"]), \
                f"{model_id}/{view}: labeled rows in bins != n in metrics_summary ({split})"
            weighted_accuracy = float((bins[count_column] * bins[accuracy_column]).sum() / bins[count_column].sum())
            assert abs(weighted_accuracy - float(metrics_row["accuracy"])) < CONF_TOLERANCE, \
                f"{model_id}/{view}: weighted bin accuracy != accuracy in metrics_summary ({split})"
        assert np.isclose(bins["conf_from"], high_conf_from).any(), \
            f"{model_id}: no bin starts at {high_conf_from}"
        high = bins[bins["conf_from"] >= high_conf_from - 1e-9]
        results[head] = {
            "bins": bins,
            "high_pct_rows": float(high["pct_rows"].sum()),
            "high_accuracy_corrected": float((high["n_corrected"] * high["accuracy_corrected"]).sum()
                                             / high["n_corrected"].sum()),
            "high_accuracy_raw": float((high["n_raw"] * high["accuracy_raw"]).sum() / high["n_raw"].sum())}

    # ---------- canvas ----------
    high_text = "; ".join(f"{results[head]['high_pct_rows']:.1f}% of rows, "
                          f"{results[head]['high_accuracy_corrected'] * 100:.1f}% right ({head})" for head in heads)
    fig = new_canvas(
        title="Model confidence vs accuracy",
        subtitle=f"Confidence >= {high_conf_from:.1f}: {high_text}.",
        footer=(f"{split.capitalize()}, model {variant}. Bars: % of all rows per bin. Lines: accuracy on labeled rows, "
                "corrected (solid) vs raw (dashed). Rings: mean confidence."))
    tile_rects = split_row(len(heads))

    # ---------- one tile per head ----------
    model_color = MODEL_COLORS[variant]
    for tile_rect, head in zip(tile_rects, heads):
        bins = results[head]["bins"]
        ax = add_tile(fig, *tile_rect, tile_title=f"{head_titles[head]}: model {variant}, {split}",
                      pad_left=0.075, pad_bottom=0.12)
        x_positions = np.arange(len(bins), dtype=float)
        pct_rows = bins["pct_rows"].to_numpy(dtype=float)
        accuracy_corrected = bins["accuracy_corrected"].to_numpy(dtype=float) * 100
        accuracy_raw = bins["accuracy_raw"].to_numpy(dtype=float) * 100
        mean_confidence = bins["mean_conf"].to_numpy(dtype=float) * 100

        ax.bar(x_positions, pct_rows, width=0.62, color=to_rgba(model_color, 0.55), zorder=2)
        for x_position, value in zip(x_positions, pct_rows):
            ax.text(x_position, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=10)
        ax.plot(x_positions, accuracy_raw, color=CORAL, linestyle="--", linewidth=2.4, marker="o",
                markersize=6, zorder=3)
        ax.plot(x_positions, accuracy_corrected, color=TURQUOISE, linewidth=3.0, marker="o",
                markersize=7, zorder=4)
        ax.scatter(x_positions, mean_confidence, s=200, facecolors="none", edgecolors=AMBER, linewidths=2.4,
                   zorder=5)
        for x_position, value in zip(x_positions, accuracy_corrected):
            ax.text(x_position, value + CONF_LABEL_OFFSET, f"{value:.1f}", ha="center", va="bottom",
                    color=TURQUOISE, fontsize=10, fontweight="bold")

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{low:.1f}-{high:.1f}" for low, high in zip(bins["conf_from"], bins["conf_to"])],
                           fontsize=11)
        ax.set_xlim(-0.6, len(bins) - 0.4)
        ax.set_ylim(0, CONF_VALUE_LIMIT)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel("percent")
        ax.set_xlabel("model confidence bin")
        legend_handles = [
            Patch(facecolor=to_rgba(model_color, 0.55), label="% of rows"),
            Line2D([], [], color=TURQUOISE, linewidth=3, marker="o", markersize=7, label="accuracy, corrected"),
            Line2D([], [], color=CORAL, linewidth=2.4, linestyle="--", marker="o", markersize=6,
                   label="accuracy, raw"),
            Line2D([], [], color="none", marker="o", markerfacecolor="none", markeredgecolor=AMBER,
                   markeredgewidth=2.4, markersize=11, label="mean confidence")]
        glass_legend(ax, handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2)

    info = save_figure(fig, Path(figures_dir) / "08_confidence.png")
    info["heads"] = {
        head: {"high_pct_rows": round(r["high_pct_rows"], 2),
               "high_accuracy_corrected": round(r["high_accuracy_corrected"], 4),
               "high_accuracy_raw": round(r["high_accuracy_raw"], 4),
               "bins": [f"{row.conf_from:.1f}-{row.conf_to:.1f}: rows {row.pct_rows:.2f}%, mean_conf {row.mean_conf:.3f}, "
                        f"acc_corrected {row.accuracy_corrected:.4f}, acc_raw {row.accuracy_raw:.4f}"
                        for row in r["bins"].itertuples()]}
        for head, r in results.items()}
    return info
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd, os

from cycler import cycler
from itertools import combinations
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LinearRegression

# ── Constants ────────────────────────────────────────────────────────────────

PALETTE = ["#2E4057", "#048A81", "#7D5BA6", "#D1495B",
           "#1B2B3A", "#EDAE49", "#66A182", "#F08C78"]
SIZES = {"half": (3.5, 2.6), "full": (8.7, 2.8), "slide": (10, 6.5)}
LABEL_MAP = {
    "hendrycks_math":    "MATH",
    "drop":              "DROP",
    "quant_bits":        "Quant (bits)",
    "thinking_budget":   "Thinking budget",
    "mean_req_s":        "Time (s)",
    "mem_bandwidth_pct": "Mem BW (%)",
    "power_W":           "Power (W)",
}

# kept for backward compat
class PublicationStyle:
    ACCENT_PALETTE = PALETTE

# ── Pure helpers ─────────────────────────────────────────────────────────────

def label(key) -> str:
    return LABEL_MAP.get(str(key), str(key))

def ci_t(s) -> float:
    """95% t-interval half-width."""
    return stats.sem(s) * stats.t.ppf(0.975, max(len(s) - 1, 1))

# internal alias
_ci = ci_t

def _group_stats(df, group_col, col):
    groups = sorted(df[group_col].unique())
    slices = [df[df[group_col] == g][col] for g in groups]
    return groups, [s.mean() for s in slices], [_ci(s) for s in slices]

def _bar(ax, x, means, cis, color, xticklabels, **kw):
    ax.bar(x, means, color=color, yerr=cis, capsize=4, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    for k, v in kw.items():
        getattr(ax, f"set_{k}")(v)

# ── Theme ────────────────────────────────────────────────────────────────────

def apply_theme(mode="half", font_scale=1.0, squeeze_x=1.0, squeeze_y=1.0):
    """Style for rainclpoud plot formats, reused from past project."""
    scale = font_scale * (0.85 if mode == "half" else 1.0)
    sns.set_context("paper", font_scale=scale)
    sns.set_style("white", {"axes.grid": True})
    w, h = SIZES.get(mode, (3.5, 2.6))
    plt.rcParams.update({
        "figure.figsize": (w * squeeze_x, h * squeeze_y),
        "figure.constrained_layout.use": True,
        "axes.grid": True, "grid.color": "#E5E5E5",
        "grid.linestyle": "-", "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6 if mode == "half" else 0.8,
        "axes.edgecolor": "#888888",
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.titlesize": 10 if mode == "half" else 12,
        "axes.labelpad": 4, "axes.labelsize": 9 if mode == "half" else 10,
        "lines.linewidth": 1.5 if mode == "half" else 2.0, "lines.markersize": 6,
        "patch.linewidth": 0.8, "patch.edgecolor": "white",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)

def save_fig(ax, path, legend=True, fmt="png"):
    sns.despine(ax=ax, top=True, right=True)
    ax.set_axisbelow(True)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            w = ax.get_figure().get_size_inches()[0]
            ax.legend(handles=handles, labels=labels, frameon=False,
                      ncol=min(len(labels), max(1, int(w // 1.3))),
                      fontsize=9 if w < 5 else 10, columnspacing=1.5)
    plt.xticks(rotation=25, ha="right")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, format=fmt, transparent=False)
    print(f"Saved → {path}")

# ── Data ─────────────────────────────────────────────────────────────────────

def load_benchmark_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["start_time"])

# ── Plots ────────────────────────────────────────────────────────────────────

def plot_benchmark_metrics(df, metrics=None, group_by="thinking_budget",
                           hue_var="quant_bits", title="", ylim=(0, 1.05),
                           mode="half", save_path=None):
    """Raincloud (strip + point-CI) plot for benchmark metrics, reused from past project."""
    id_cols = [group_by, hue_var]
    df_long = (df[id_cols + metrics]
               .melt(id_vars=id_cols, value_vars=metrics,
                     var_name="metric", value_name="Score")
               .assign(Score=lambda d: pd.to_numeric(d["Score"], errors="coerce"),
                       _x=lambda d: d[group_by].astype(str) + " | "
                                  + d[hue_var].astype(str) + "b"))
    x_order = sorted(df_long["_x"].unique(),
                     key=lambda s: [int(t) for t in s.replace("b", "").split(" | ")])

    apply_theme(mode=mode)
    fig, ax = plt.subplots()
    shared_kw = dict(data=df_long, x="_x", y="Score", hue="metric",
                     order=x_order, palette=PALETTE, ax=ax)

    sns.stripplot(**shared_kw, dodge=True, alpha=0.35, jitter=True,
                  size=6 if mode == "slide" else 2, legend=False, zorder=1)
    sns.pointplot(**shared_kw, dodge=0.62, join=False, markers="_", scale=0.8,
                  errwidth=2.5 if mode == "slide" else 1.6,
                  capsize=0.2 if mode == "slide" else 0.08,
                  legend=mode != "half", zorder=2)

    ax.set(title=title, xlabel=f"{label(group_by)} | {label(hue_var)}",
           ylabel="Score", ylim=ylim)

    if save_path:
        save_fig(ax, save_path)
    else:
        handles, lbs = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=lbs, loc="lower center",
                      bbox_to_anchor=(0.5, 1.15), ncol=min(len(lbs), 4), frameon=False)
        plt.tight_layout()
    return fig, ax, df_long


def plot_bars_by_group(df, group_col, cols, labels, supxlabel,
                       xticklabels=None, score_range=True,
                       palette_start=0, figsize=(8, 3)):
    """Grouped bar chart with 95% CI, one subplot per metric."""
    fig, axs = plt.subplots(1, len(cols), figsize=figsize)
    axs = [axs] if len(cols) == 1 else list(axs)
    for ax, col, lbl, color in zip(axs, cols, labels, PALETTE[palette_start:]):
        groups, means, cis = _group_stats(df, group_col, col)
        _bar(ax, np.arange(len(groups)), means, cis, color,
             xticklabels if xticklabels is not None else groups,
             ylabel=f"{lbl} ($\\in$[0,1])" if score_range else lbl)
    fig.supxlabel(supxlabel)
    plt.tight_layout()
    return fig, axs


def plot_regression_scatter(df, predictor_cols, target_cols, target_labels,
                             title, palette_offset=0):
    """Regression scatter with ±1$\sigma$ band, one subplot per predictor x target."""
    fig, axs = plt.subplots(len(predictor_cols), len(target_cols),
                             figsize=(2.5 * len(target_cols), 2.8 * len(predictor_cols)),
                             squeeze=False)
    for r, pred in enumerate(predictor_cols):
        X = df[pred].values.reshape(-1, 1)
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        for c, (col, lbl) in enumerate(zip(target_cols, target_labels)):
            ax = axs[r][c]
            y = df[col].values
            m = LinearRegression().fit(X, y)
            y_line = m.predict(x_line)
            color = PALETTE[(c + palette_offset) % len(PALETTE)]
            ax.scatter(X, y, s=12, alpha=0.5, color=color, zorder=3)
            ax.plot(x_line, y_line, color=color, linewidth=1.6)
            ax.fill_between(x_line.ravel(),
                            y_line - np.std(y - m.predict(X)),
                            y_line + np.std(y - m.predict(X)),
                            alpha=0.18, color=color,
                            label=f"±$\\sigma$  R²={m.score(X, y):.3f}")
            ax.legend(fontsize=6, loc="best", frameon=False)
            ax.set(xlabel=label(pred), ylabel=lbl)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig

def plot_score_per_cost(df, cost_col, cost_label, title, score_cols=None, palette=PALETTE):
    """Score / cost tradeoff bar chart split by quant bits and thinking budget."""
    score_cols = score_cols or ["f1", "accuracy", "ROUGE", "METEOR"]
    q_groups = sorted(df.quant_bits.unique())
    b_groups = sorted(df.thinking_budget.unique())
    groups   = [(f"{b}-bit", df[df.quant_bits == b]) for b in q_groups] + \
               [(f"τ={t}",   df[df.thinking_budget == t]) for t in b_groups]
    x      = np.arange(len(groups))
    colors = [palette[1]] * len(q_groups) + [palette[4]] * len(b_groups)

    fig, axs = plt.subplots(1, len(score_cols), figsize=(3.2 * len(score_cols), 3.2))
    for ax, score_col in zip(axs, score_cols):
        ratios = [sub[score_col] / sub[cost_col] for _, sub in groups]
        _bar(ax, x, [r.mean() for r in ratios], [_ci(r) for r in ratios],
             colors, [g[0] for g in groups],
             ylabel=f"{score_col} / {cost_label}", title=score_col)
        ax.axvline(len(q_groups) - 0.5, color="gray", linewidth=0.8, linestyle="--")

    fig.legend(handles=[Patch(color=palette[1], label="quant bits"),
                        Patch(color=palette[4], label="thinking budget")],
               loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


def plot_scores_faceted(df, score_cols=None, score_labels=None, palette=PALETTE,
                        row_col="dataset", x_col="thinking_budget", hue_col="quant_bits"):
    """Grid of bar charts: rows=row_col, cols=metrics, x=x_col, hue=hue_col."""
    score_cols   = score_cols   or ["ROUGE", "METEOR", "f1", "accuracy"]
    score_labels = score_labels or ["ROUGE", "METEOR", "F1", "Accuracy"]
    row_vals = sorted(df[row_col].unique())
    hue_vals = sorted(df[hue_col].unique())
    x_vals   = sorted(df[x_col].unique())
    bw       = 0.8 / len(hue_vals)

    fig, axs = plt.subplots(len(row_vals), len(score_cols),
                             figsize=(3.5 * len(score_cols), 2.8 * len(row_vals)),
                             squeeze=False)
    for r, row_val in enumerate(row_vals):
        sub = df[df[row_col] == row_val]
        for c, (col, lbl) in enumerate(zip(score_cols, score_labels)):
            ax = axs[r][c]
            x  = np.arange(len(x_vals))
            for h, (hue_val, color) in enumerate(zip(hue_vals, palette)):
                grp   = sub[sub[hue_col] == hue_val]
                slices = [grp[grp[x_col] == xv][col] for xv in x_vals]
                ax.bar(x + (h - (len(hue_vals) - 1) / 2) * bw,
                       [s.mean() for s in slices], width=bw, color=color,
                       yerr=[_ci(s) for s in slices],
                       capsize=3, error_kw={"linewidth": 1.0}, label=label(hue_val))
            ax.set_xticks(x)
            ax.set_xticklabels(x_vals, rotation=45, ha="right")
            if r == 0:
                ax.set_title(lbl)
            ax.set_ylabel(f"{label(row_val)}\n{lbl}" if c == 0 else lbl)
            if r == len(row_vals) - 1:
                ax.set_xlabel(label(x_col))
            if r == 0 and c == len(score_cols) - 1:
                ax.legend(title=label(hue_col), fontsize=7, frameon=False)
    plt.tight_layout()
    return fig


def _ci_bootstrap(s, n_boot=2000, ci=0.95) -> float:
    """Bootstrap half-width CI (replaces t-interval for non-normal data)."""
    rng = np.random.default_rng(0)
    boot_means = [rng.choice(s, size=len(s), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return (hi - lo) / 2


def plot_3d_scatter(df, z_col="accuracy", color_col="dataset",
                    x_col="quant_bits", y_col="thinking_budget"):
    """3D scatter: x=quant bits, y=thinking budget, z=score, colour=dataset."""
    color_vals = sorted(df[color_col].unique())
    colors = {v: PALETTE[i] for i, v in enumerate(color_vals)}

    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")

    for val in color_vals:
        sub = df[df[color_col] == val]
        ax.scatter(sub[x_col], sub[y_col], sub[z_col],
                   color=colors[val], label=label(val), s=40, alpha=0.8)

    ax.set_xlabel(label(x_col))
    ax.set_ylabel(label(y_col))
    ax.set_zlabel(z_col)
    ax.legend(title=label(color_col), fontsize=8, frameon=False)
    plt.tight_layout()
    return fig

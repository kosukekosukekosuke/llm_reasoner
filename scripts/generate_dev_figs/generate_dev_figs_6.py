#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dev図6枚を一括生成するスクリプト（Baseline-R + RGGS候補 θ）

入力:
  - baseline_tau_sweep_metrics.csv
  - theta_candidates_metrics.csv

表記は principle.tex に合わせる（主要な記号）:
  - Coverage, SelRisk, OoM_FA, ExRate, κ, δ, ε
  - Baseline-R のしきい値: τ_conf
  - 代表運用点: θ* と τ_conf^*

出力（png; オプションでpdfも）:
  fig_dev_theta_scatter_cov_selrisk.png
  fig_dev_theta_scatter_cov_oomfa.png
  fig_dev_overlay_baseline_oomfa_curve_and_theta_points.png
  fig_dev_overlay_baseline_curve_and_theta_points.png
  fig_dev_baseline_risk_coverage_curve.png
  fig_dev_baseline_coverage_oomfa_curve.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Notation (aligned with principle.tex)
# -----------------------------
SYM_COVERAGE = r"$\mathrm{Coverage}$"
SYM_SELRISK  = r"$\mathrm{SelRisk}$"
SYM_OOMFA    = r"$\mathrm{OoM\_FA}$"
SYM_EPS      = r"$\varepsilon$"
SYM_KAPPA    = r"$\kappa$"
SYM_DELTA    = r"$\delta$"
SYM_TAU_CONF = r"$\tau_{\mathrm{conf}}$"
SYM_TAU_CONF_STAR = r"$\tau_{\mathrm{conf}}^\ast$"

# theta / theta*
def tex_theta(i: int) -> str:
    return rf"$\theta_{{{i}}}$"

def tex_theta_range(i_min: int, i_max: int) -> str:
    return rf"$\theta_{{{i_min}}}$〜$\theta_{{{i_max}}}$"

def tex_theta_star_eq(i: int) -> str:
    return rf"$\theta^\ast=\theta_{{{i}}}$"


# -----------------------------
# Utilities
# NOTE: pandas>=2.0 + older matplotlib can raise
#   ValueError: Multi-dimensional indexing (obj[:, None]) ...
# when passing a pandas Series directly to ax.plot.
# We therefore convert Series to numpy via .to_numpy() before plotting.

# -----------------------------
def _maybe_set_japanese_font():
    """JPフォントが無い環境でも落ちないように、利用可能なフォントを検出して1つだけ設定する（警告抑制）。"""
    import matplotlib.font_manager as fm

    candidates = [
        "IPAexGothic", "IPAGothic",
        "Noto Sans CJK JP", "Noto Sans JP",
        "Yu Gothic", "YuGothic",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",
        "Meiryo",
    ]

    chosen = None
    for name in candidates:
        try:
            # fallback_to_default=False で「無い場合に例外」を出し、警告を避ける
            fm.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:
            continue

    if chosen is None:
        chosen = "DejaVu Sans"  # 最低限のフォールバック

    plt.rcParams["font.family"] = chosen
def setup_matplotlib():
    _maybe_set_japanese_font()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.dpi"] = 200


def require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{name}] missing required columns: {missing}\n"
            f"available columns: {list(df.columns)}"
        )


def round_for_label(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}"


def sort_by_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """カバレッジ（x軸）で安定ソートして曲線を描きやすくする。"""
    keys = ["coverage"]
    if "tau" in df.columns:
        keys.append("tau")
    return df.sort_values(keys, kind="mergesort")


def annotate_ids(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: List[str], fontsize: int = 10) -> None:
    """点番号を付与（重なりを少し抑えるため、オフセットを周期的に変更）。"""
    offsets = [
        (6, 6), (6, -10), (-10, 6), (-10, -10),
        (10, 0), (-14, 0), (0, 10), (0, -12),
    ]
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            lab, (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center", va="center",
            fontsize=fontsize,
        )


def save_fig(fig: plt.Figure, out_png: Path, save_pdf: bool) -> None:
    fig.savefig(out_png)
    if save_pdf:
        fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


@dataclass(frozen=True)
class SelectionResult:
    # "theta_star" or "tau_conf_star"
    key: str
    # representative value (theta id or tau_conf)
    value: float
    # selected point coordinates
    coverage: float
    selrisk: float
    oom_fa: float


def select_representative(
    df: pd.DataFrame,
    *,
    kind: str,
    epsilon: float,
    kappa: float,
    delta: float,
    tie_break_cols: Optional[List[str]] = None,
) -> SelectionResult:
    """
    principle.tex（selection_rule2）に合わせた代表運用点の選択:

      1) ExRate制約（ε）: いずれかの ExRate_* > ε の候補は棄却
      2) OoM_FA制約（κ）: OoM_FA < κ を満たす候補が存在するなら、その集合に制限（存在しない場合は制限しない）
      3) SelRisk制約（δ）: SelRisk <= δ を満たす候補が存在するなら、その集合に制限（存在しない場合は制限しない）
      4) Coverage 最大
      5) タイブレーク: SelRisk 最小 → avg_trials 最小（列があれば）→ tie_break_cols の辞書順
         Baseline-Rでは最後に τ_conf が大きい（より保守）方を優先
    """
    req = ["coverage", "selrisk", "oom_fa", "exrate_total", "exrate_in", "exrate_oom"]
    require_columns(df, req, name=kind)
    work = df.copy()

    # 1) ExRate constraint
    ex_ok = (work["exrate_total"] <= epsilon) & (work["exrate_in"] <= epsilon) & (work["exrate_oom"] <= epsilon)
    work = work[ex_ok].copy()
    if work.empty:
        raise ValueError(f"[{kind}] no candidates left after ExRate constraint (epsilon={epsilon}).")

    # 2) kappa (strict < as in Eq.(kappa))
    if (work["oom_fa"] < kappa).any():
        work = work[work["oom_fa"] < kappa].copy()

    # 3) delta (<= as in selection rule)
    if (work["selrisk"] <= delta).any():
        work = work[work["selrisk"] <= delta].copy()

    # 4) maximize coverage
    max_cov = work["coverage"].max()
    tied = work[work["coverage"] == max_cov].copy()

    # 5) tie-break: selrisk min
    tied = tied.sort_values(["selrisk"], ascending=[True], kind="mergesort")
    tied = tied[tied["selrisk"] == tied["selrisk"].iloc[0]].copy()

    # avg_trials min if available
    if "avg_trials" in tied.columns and not tied["avg_trials"].isna().all():
        tied = tied.sort_values(["avg_trials"], ascending=[True], kind="mergesort")
        tied = tied[tied["avg_trials"] == tied["avg_trials"].iloc[0]].copy()

    # additional lexicographic tie-break
    if tie_break_cols:
        present = [c for c in tie_break_cols if c in tied.columns]
        if present:
            tied = tied.sort_values(present, ascending=[True] * len(present), kind="mergesort")

    # Baseline-R: prefer larger tau (more conservative)
    if kind.lower().startswith("baseline") and "tau" in tied.columns:
        tied = tied.sort_values(["tau"], ascending=[False], kind="mergesort")

    chosen = tied.iloc[0]

    if kind.lower().startswith("baseline"):
        return SelectionResult(
            key="tau_conf_star",
            value=float(chosen["tau"]),
            coverage=float(chosen["coverage"]),
            selrisk=float(chosen["selrisk"]),
            oom_fa=float(chosen["oom_fa"]),
        )
    else:
        if "theta" not in chosen.index:
            raise KeyError("[RGGS] 'theta' column is required.")
        return SelectionResult(
            key="theta_star",
            value=float(chosen["theta"]),
            coverage=float(chosen["coverage"]),
            selrisk=float(chosen["selrisk"]),
            oom_fa=float(chosen["oom_fa"]),
        )


# -----------------------------
# Figure generators (6 figures)
# -----------------------------
def fig_theta_scatter_cov_selrisk(
    theta_df: pd.DataFrame,
    theta_star: SelectionResult,
    *,
    epsilon: float,
    delta: float,
    out_png: Path,
    save_pdf: bool,
) -> None:
    setup_matplotlib()
    require_columns(theta_df, ["theta", "coverage", "selrisk", "exrate_total", "exrate_in", "exrate_oom"], "theta_df")

    ex_ok = (theta_df["exrate_total"] <= epsilon) & (theta_df["exrate_in"] <= epsilon) & (theta_df["exrate_oom"] <= epsilon)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(theta_df.loc[~ex_ok, "coverage"], theta_df.loc[~ex_ok, "selrisk"],
               marker="x", s=90, label=f"ExRate制約({SYM_EPS})を満たさない")
    ax.scatter(theta_df.loc[ex_ok, "coverage"], theta_df.loc[ex_ok, "selrisk"],
               marker="o", s=90, label=f"ExRate制約({SYM_EPS})を満たす")

    # θ*
    ax.scatter([theta_star.coverage], [theta_star.selrisk], marker="*", s=500,
               label=tex_theta_star_eq(int(theta_star.value)))

    # δ line
    ax.axhline(delta, linestyle="--", linewidth=2, label=rf"{SYM_DELTA} ({SYM_SELRISK}上限)")

    # numeric labels
    annotate_ids(
        ax,
        theta_df["coverage"].to_numpy(),
        theta_df["selrisk"].to_numpy(),
        [str(int(t)) for t in theta_df["theta"].to_numpy()],
        fontsize=10,
    )

    tmin, tmax = int(theta_df["theta"].min()), int(theta_df["theta"].max())
    # ax.set_title(rf"Dev上の運用点候補（{tex_theta_range(tmin, tmax)}）の散布図")
    ax.set_xlabel(rf"{SYM_COVERAGE}")
    # ax.set_xlabel(rf"{SYM_COVERAGE} (Dev In-Map)")
    ax.set_ylabel(rf"{SYM_SELRISK}")

    ymax = max(theta_df["selrisk"].max(), delta) * 1.05
    ymin = max(0.0, theta_df["selrisk"].min() * 0.8)
    ax.set_ylim(bottom=ymin, top=ymax)

    ax.legend(loc="center right")
    fig.tight_layout()
    save_fig(fig, out_png, save_pdf)


def fig_theta_scatter_cov_oomfa(
    theta_df: pd.DataFrame,
    theta_star: SelectionResult,
    *,
    kappa: float,
    out_png: Path,
    save_pdf: bool,
) -> None:
    setup_matplotlib()
    require_columns(theta_df, ["theta", "coverage", "oom_fa"], "theta_df")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(theta_df["coverage"], theta_df["oom_fa"], s=120, label=r"運用点候補（$\theta$）")
    ax.scatter([theta_star.coverage], [theta_star.oom_fa], marker="*", s=600, label=tex_theta_star_eq(int(theta_star.value)))
    ax.axhline(kappa, linestyle="--", linewidth=2, label=rf"{SYM_KAPPA} ({SYM_OOMFA}上限)")

    annotate_ids(
        ax,
        theta_df["coverage"].to_numpy(),
        theta_df["oom_fa"].to_numpy(),
        [str(int(t)) for t in theta_df["theta"].to_numpy()],
        fontsize=10,
    )

    # ax.set_title(rf"RGGS候補の {SYM_COVERAGE}–{SYM_OOMFA} 散布図（Dev）")
    ax.set_xlabel(rf"{SYM_COVERAGE}")
    # ax.set_xlabel(rf"{SYM_COVERAGE} (Dev In-Map)")
    ax.set_ylabel(rf"{SYM_OOMFA}")
    # ax.set_ylabel(rf"{SYM_OOMFA} (Dev OoM)")

    ymax = max(theta_df["oom_fa"].max(), kappa) * 1.05
    ax.set_ylim(bottom=0.0, top=ymax)

    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_png, save_pdf)


def fig_overlay_baseline_coverage_oomfa_and_theta(
    baseline_df: pd.DataFrame,
    theta_df: pd.DataFrame,
    theta_star: SelectionResult,
    tau_star: SelectionResult,
    *,
    kappa: float,
    out_png: Path,
    save_pdf: bool,
) -> None:
    setup_matplotlib()
    require_columns(baseline_df, ["coverage", "oom_fa", "tau"], "baseline_df")
    require_columns(theta_df, ["theta", "coverage", "oom_fa"], "theta_df")

    b = sort_by_coverage(baseline_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(b["coverage"].to_numpy(), b["oom_fa"].to_numpy(), marker="o", markersize=3, linewidth=1.5, label="Baseline-R 曲線")
    ax.scatter(theta_df["coverage"], theta_df["oom_fa"], s=120, label=r"運用点候補（$\theta$）")

    ax.scatter([theta_star.coverage], [theta_star.oom_fa], marker="*", s=600, label=rf"採用点 {tex_theta_star_eq(int(theta_star.value))}")
    tau_label = round_for_label(tau_star.value, 3)
    ax.scatter([tau_star.coverage], [tau_star.oom_fa], marker="*", s=600, label=rf"採用点 {SYM_TAU_CONF_STAR} = {tau_label}")

    ax.axhline(kappa, linestyle="--", linewidth=2, label=rf"{SYM_KAPPA}")

    annotate_ids(
        ax,
        theta_df["coverage"].to_numpy(),
        theta_df["oom_fa"].to_numpy(),
        [str(int(t)) for t in theta_df["theta"].to_numpy()],
        fontsize=10,
    )

    # ax.set_title(rf"{SYM_COVERAGE}–{SYM_OOMFA}: Baseline-R曲線とRGGS候補点（Dev）")
    ax.set_xlabel(rf"{SYM_COVERAGE}")
    # ax.set_xlabel(rf"{SYM_COVERAGE} (Dev In-Map)")
    ax.set_ylabel(rf"{SYM_OOMFA}")
    # ax.set_ylabel(rf"{SYM_OOMFA} (Dev OoM)")

    ax.set_xlim(left=0.0, right=1.02)
    ymax = max(baseline_df["oom_fa"].max(), theta_df["oom_fa"].max(), kappa) * 1.05
    ax.set_ylim(bottom=0.0, top=ymax)

    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_png, save_pdf)


def fig_overlay_baseline_risk_coverage_and_theta(
    baseline_df: pd.DataFrame,
    theta_df: pd.DataFrame,
    theta_star: SelectionResult,
    tau_star: SelectionResult,
    *,
    delta: float,
    out_png: Path,
    save_pdf: bool,
) -> None:
    setup_matplotlib()
    require_columns(baseline_df, ["coverage", "selrisk", "tau"], "baseline_df")
    require_columns(theta_df, ["theta", "coverage", "selrisk"], "theta_df")

    b = sort_by_coverage(baseline_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(b["coverage"].to_numpy(), b["selrisk"].to_numpy(), marker="o", markersize=3, linewidth=1.5, label="Baseline-R 曲線")
    ax.scatter(theta_df["coverage"], theta_df["selrisk"], s=120,
               label=rf"運用点候補（{tex_theta_range(int(theta_df['theta'].min()), int(theta_df['theta'].max()))}）")

    ax.scatter([theta_star.coverage], [theta_star.selrisk], marker="*", s=600, label=rf"採用点 {tex_theta_star_eq(int(theta_star.value))}")
    tau_label = round_for_label(tau_star.value, 3)
    ax.scatter([tau_star.coverage], [tau_star.selrisk], marker="*", s=600, label=rf"採用点 {SYM_TAU_CONF_STAR} = {tau_label}")

    ax.axhline(delta, linestyle="--", linewidth=2, label=rf"{SYM_DELTA}")

    annotate_ids(
        ax,
        theta_df["coverage"].to_numpy(),
        theta_df["selrisk"].to_numpy(),
        [str(int(t)) for t in theta_df["theta"].to_numpy()],
        fontsize=10,
    )

    # ax.set_title("Baseline-R曲線とRGGS候補点の重ね合わせ（Dev）")
    ax.set_xlabel(rf"{SYM_COVERAGE}")
    # ax.set_xlabel(rf"{SYM_COVERAGE} (Dev In-Map)")
    ax.set_ylabel(rf"{SYM_SELRISK}")

    ax.set_xlim(left=0.0, right=1.02)
    ymax = max(baseline_df["selrisk"].max(), theta_df["selrisk"].max(), delta) * 1.05
    ax.set_ylim(bottom=0.0, top=ymax)

    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_png, save_pdf)


def fig_baseline_risk_coverage(
    baseline_df: pd.DataFrame,
    tau_star: SelectionResult,
    *,
    delta: float,
    out_png: Path,
    save_pdf: bool,
) -> None:
    setup_matplotlib()
    require_columns(baseline_df, ["coverage", "selrisk", "tau"], "baseline_df")
    b = sort_by_coverage(baseline_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(b["coverage"].to_numpy(), b["selrisk"].to_numpy(), marker="o", markersize=3, linewidth=1.5, label=rf"Baseline-R（{SYM_TAU_CONF}走査）")

    tau_label = round_for_label(tau_star.value, 3)
    ax.scatter([tau_star.coverage], [tau_star.selrisk], marker="*", s=600, label=rf"採用点 {SYM_TAU_CONF_STAR} = {tau_label}")
    ax.axhline(delta, linestyle="--", linewidth=2, label=rf"{SYM_DELTA} ({SYM_SELRISK}上限)")

    # ax.set_title("Baseline-Rのリスク–カバレッジ曲線（Dev）")
    ax.set_xlabel(rf"{SYM_COVERAGE}")
    # ax.set_xlabel(rf"{SYM_COVERAGE} (Dev In-Map)")
    ax.set_ylabel(rf"{SYM_SELRISK}")

    ax.set_xlim(left=0.0, right=1.02)
    ymax = max(baseline_df["selrisk"].max(), delta) * 1.05
    ax.set_ylim(bottom=0.0, top=ymax)

    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_png, save_pdf)


def fig_baseline_coverage_oomfa(
    baseline_df: pd.DataFrame,
    tau_star: SelectionResult,
    *,
    kappa: float,
    out_png: Path,
    save_pdf: bool,
) -> None:
    setup_matplotlib()
    require_columns(baseline_df, ["coverage", "oom_fa", "tau"], "baseline_df")
    b = sort_by_coverage(baseline_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(b["coverage"].to_numpy(), b["oom_fa"].to_numpy(), marker="o", markersize=3, linewidth=1.5, label=rf"Baseline-R（{SYM_TAU_CONF}走査）")

    tau_label = round_for_label(tau_star.value, 3)
    ax.scatter([tau_star.coverage], [tau_star.oom_fa], marker="*", s=600, label=rf"採用点 {SYM_TAU_CONF_STAR} = {tau_label}")
    ax.axhline(kappa, linestyle="--", linewidth=2, label=rf"{SYM_KAPPA} ({SYM_OOMFA}上限)")

    # ax.set_title(rf"Baseline-Rの{SYM_COVERAGE}–{SYM_OOMFA}曲線（Dev）")
    ax.set_xlabel(rf"{SYM_COVERAGE}")
    # ax.set_xlabel(rf"{SYM_COVERAGE} (Dev In-Map)")
    ax.set_ylabel(rf"{SYM_OOMFA}")
    # ax.set_ylabel(rf"{SYM_OOMFA} (Dev OoM)")

    ax.set_xlim(left=0.0, right=1.02)
    ymax = max(baseline_df["oom_fa"].max(), kappa) * 1.05
    ax.set_ylim(bottom=0.0, top=ymax)

    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_png, save_pdf)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", type=str, default="baseline_tau_sweep_metrics.csv")
    ap.add_argument("--theta_csv", type=str, default="theta_candidates_metrics.csv")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--kappa", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.20)
    ap.add_argument("--epsilon", type=float, default=0.20)
    ap.add_argument("--save_pdf", action="store_true", help="pngに加えてpdfも保存する")
    args = ap.parse_args()

    baseline_csv = Path(args.baseline_csv)
    theta_csv = Path(args.theta_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(baseline_csv)
    theta_df = pd.read_csv(theta_csv)

    require_columns(baseline_df, ["tau", "coverage", "selrisk", "oom_fa", "exrate_total", "exrate_in", "exrate_oom"], "baseline_df")
    require_columns(theta_df, ["theta", "coverage", "selrisk", "oom_fa", "exrate_total", "exrate_in", "exrate_oom"], "theta_df")

    # Representative points
    theta_star = select_representative(
        theta_df,
        kind="RGGS",
        epsilon=args.epsilon,
        kappa=args.kappa,
        delta=args.delta,
        tie_break_cols=["alpha", "tau_cmb", "tau_max", "tau_m", "theta"],
    )
    tau_star = select_representative(
        baseline_df,
        kind="Baseline-R",
        epsilon=args.epsilon,
        kappa=args.kappa,
        delta=args.delta,
        tie_break_cols=["tau"],
    )

    print("[Selection]")
    print(f"  theta* = theta_{int(theta_star.value)}  (coverage={theta_star.coverage:.6f}, selrisk={theta_star.selrisk:.6f}, oom_fa={theta_star.oom_fa:.6f})")
    print(f"  tau_conf* = {tau_star.value:.6f}        (coverage={tau_star.coverage:.6f}, selrisk={tau_star.selrisk:.6f}, oom_fa={tau_star.oom_fa:.6f})")

    # Output paths (match the provided filenames)
    p1 = out_dir / "fig_dev_theta_scatter_cov_selrisk.png"
    p2 = out_dir / "fig_dev_theta_scatter_cov_oomfa.png"
    p3 = out_dir / "fig_dev_overlay_baseline_oomfa_curve_and_theta_points.png"
    p4 = out_dir / "fig_dev_overlay_baseline_curve_and_theta_points.png"
    p5 = out_dir / "fig_dev_baseline_risk_coverage_curve.png"
    p6 = out_dir / "fig_dev_baseline_coverage_oomfa_curve.png"

    fig_theta_scatter_cov_selrisk(theta_df, theta_star, epsilon=args.epsilon, delta=args.delta, out_png=p1, save_pdf=args.save_pdf)
    fig_theta_scatter_cov_oomfa(theta_df, theta_star, kappa=args.kappa, out_png=p2, save_pdf=args.save_pdf)
    fig_overlay_baseline_coverage_oomfa_and_theta(baseline_df, theta_df, theta_star, tau_star, kappa=args.kappa, out_png=p3, save_pdf=args.save_pdf)
    fig_overlay_baseline_risk_coverage_and_theta(baseline_df, theta_df, theta_star, tau_star, delta=args.delta, out_png=p4, save_pdf=args.save_pdf)
    fig_baseline_risk_coverage(baseline_df, tau_star, delta=args.delta, out_png=p5, save_pdf=args.save_pdf)
    fig_baseline_coverage_oomfa(baseline_df, tau_star, kappa=args.kappa, out_png=p6, save_pdf=args.save_pdf)

    print("[Done] Figures saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()

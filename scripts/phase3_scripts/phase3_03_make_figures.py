#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase3_03_make_figures.py

Phase 3: 図表生成（risk–coverage, OoM bar）

入力
- metrics_overall.json（phase3_02_aggregate_metrics.py の出力）
- （任意）risk_coverage_curve.csv（Dev のみ。Test ロックボックスでは生成しない）

出力
- {prefix}fig_risk_coverage.png  （任意：risk_curve_csv を渡した場合のみ）
- {prefix}fig_oom_bar.png        （必須）

注意（仕様・運用）
- Test では tau 掃引を行わない（risk_coverage_curve.csv を作らない）。
  したがって Test の図は OoM bar のみを生成する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def normalize_prefix(prefix: str) -> str:
    """File name prefix normalizer.

    - ""  -> ""
    - "dev" -> "dev_"
    - "dev_" -> "dev_"
    """
    p = (prefix or "").strip()
    if not p:
        return ""
    return p if p.endswith("_") else (p + "_")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON: {path} ({e})")
    if not isinstance(obj, dict):
        raise SystemExit(f"metrics_overall.json must be an object: {path}")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_json", required=True, help="{prefix}metrics_overall.json")
    ap.add_argument(
        "--risk_curve_csv",
        default="",
        help="（任意）{prefix}risk_coverage_curve.csv（Dev のみ）。未指定なら risk–coverage 図は作らない",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    metrics_path = Path(args.metrics_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = normalize_prefix(args.prefix)

    metrics = _read_json(metrics_path)

    # -------------------------
    # 1) risk–coverage (optional)
    # -------------------------
    out1: Optional[Path] = None
    if args.risk_curve_csv:
        curve_path = Path(args.risk_curve_csv)
        if not curve_path.exists():
            raise SystemExit(f"risk_curve_csv not found: {curve_path}")

        curve = pd.read_csv(curve_path)
        # Expected columns from phase3_02: tau, coverage, selective_risk, n_ok, n_accept
        # Backward-compat: accept 'risk' if 'selective_risk' is absent.
        if "selective_risk" not in curve.columns and "risk" in curve.columns:
            curve = curve.rename(columns={"risk": "selective_risk"})

        required_cols = ["coverage", "selective_risk"]
        missing = [c for c in required_cols if c not in curve.columns]
        if missing:
            raise SystemExit(f"risk_coverage_curve.csv missing columns: {missing}")

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(curve["coverage"].to_numpy(), curve["selective_risk"].to_numpy())
        ax1.set_xlabel("Coverage")
        ax1.set_ylabel("Selective Risk")
        ax1.set_ylim(0, 1)
        ax1.set_title("Risk–Coverage (In-map, status==ok, y!=-1)")
        fig1.tight_layout()

        out1 = out_dir / f"{prefix}fig_risk_coverage.png"
        fig1.savefig(out1, dpi=200)
        plt.close(fig1)

    # -------------------------
    # 2) OoM bar (always)
    # -------------------------
    # phase3_02 writes OoM operating point into metrics["operating_point"] with keys:
    #   oom_true_reject_rate / oom_false_accept_rate
    # Backward-compat: also accept the older nested style if present.
    op = metrics.get("operating_point", {})
    tr = op.get("oom_true_reject_rate", float("nan"))
    fa = op.get("oom_false_accept_rate", float("nan"))

    if (tr != tr) and isinstance(metrics.get("oom_operating_point"), dict):  # NaN check
        oom = metrics.get("oom_operating_point", {})
        tr = oom.get("true_reject_rate", tr)
        fa = oom.get("false_accept_rate", fa)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.bar(["TrueReject", "FalseAccept"], [tr, fa])
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("OoM (y=-1) Operating Point")
    fig2.tight_layout()

    out2 = out_dir / f"{prefix}fig_oom_bar.png"
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)

    print("Saved:")
    if out1 is not None:
        print(" -", out1)
    print(" -", out2)


if __name__ == "__main__":
    main()

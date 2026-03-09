#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phase3_02_aggregate_metrics.py

Phase3: 集計スクリプト（ExRate / risk–coverage / OoM / tag・scenario 内訳）

入力
----
- phase3_01_run_eval_ros.py が出力した JSONL（1クエリ1行）

出力（例）
----------
- metrics_overall.json
- exrate_by_status.csv
- tag_breakdown.csv / scenario_breakdown.csv
- （Dev のみ任意）risk_coverage_curve.csv / oom_curve.csv

定義（Phase2/3 の固定）
----------------------
- D_ok: status == "ok"
- D_fail: status != "ok"（timeout / parse_error / out_of_set / exception など）
- OoM: y == -1
- in-map: y != -1

注意（ロックボックス運用）
--------------------------
- prefix が "test" で始まる場合、tau 掃引の出力は原則生成しない。
  事故防止のため、この場合 --no_sweep が無いと停止します。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# -----------------
# helpers
# -----------------

def normalize_prefix(prefix: str) -> str:
    """"dev"/"dev_" のどちらでも同じ結果になるように正規化する。"""
    p = (prefix or "").strip()
    if not p:
        return ""
    return p if p.endswith("_") else (p + "_")


def read_jsonl(path: Path) -> pd.DataFrame:
    """JSONL を DataFrame に読み込み、最低限の妥当性検証を行う。"""
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"JSONL parse error at line {ln}: {e}")

    df = pd.DataFrame(rows)

    if "id" not in df.columns:
        raise ValueError("Missing column 'id'")
    if df["id"].duplicated().any():
        dups = df[df["id"].duplicated()]["id"].tolist()[:20]
        raise ValueError(f"Duplicate ids found (showing up to 20): {dups}")

    # runner が出力する想定の必須列（欠損すると silent な集計崩れ=嘘に見える）
    required = [
        "id",
        "y",
        "y_hat",
        "status",
        "scenario",
        "tags",
        "chosen_id",
        "conf",
        "cos",
        "accept_score",
        "dataset_n_total",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # completeness check (prevents partial runs from being aggregated as if complete)
    # runner writes dataset_n_total to every row; enforce len(df) == dataset_n_total.
    nvals = pd.to_numeric(df['dataset_n_total'], errors='coerce').dropna().astype(int).unique().tolist()
    if len(nvals) != 1:
        raise ValueError(f"dataset_n_total must be a single integer value in the JSONL, got: {sorted(set(nvals))}")
    dataset_n_total = int(nvals[0])
    if dataset_n_total <= 0:
        raise ValueError(f"dataset_n_total must be > 0, got: {dataset_n_total}")
    if len(df) != dataset_n_total:
        raise ValueError(
            f"Incomplete JSONL: expected {dataset_n_total} records (dataset_n_total) but found {len(df)}. "
            "This usually means the run was interrupted. Re-run phase3_01 to complete the JSONL."
        )

    # tags の型を正規化（list 以外は空扱い）
    df = df.copy()
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

    return df


def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")



def assert_singleton_fields(df: pd.DataFrame, fields: list[str]) -> None:
    """Fail fast if a single JSONL contains mixed experimental conditions.

    Mixing different run_id/method/config into one JSONL is a common source of
    'data tampering' suspicion, so we treat it as a hard error.
    """
    for f in fields:
        if f not in df.columns:
            continue
        s = df[f]
        vals = []
        for v in s.dropna().tolist():
            sv = str(v).strip()
            if sv == "":
                continue
            vals.append(sv)
        uniq = sorted(set(vals))
        if len(uniq) > 1:
            raise SystemExit(
                f"Mixed values detected in column '{f}'. Refusing to aggregate.\n"
                f"  uniques (showing up to 10): {uniq[:10]}\n"
                "This likely means you appended different conditions into the same JSONL."
            )


def explode_tags(df_ok: pd.DataFrame) -> pd.DataFrame:
    d = df_ok.copy()
    if "tags" not in d.columns:
        d["tags"] = [[] for _ in range(len(d))]
    d["tags"] = d["tags"].apply(lambda x: x if isinstance(x, list) else [])
    e = d.explode("tags").rename(columns={"tags": "tag"})
    e["tag"] = e["tag"].fillna("").astype(str)
    return e


# -----------------
# metrics
# -----------------

@dataclass
class ExRateSummary:
    n_total: int
    n_fail: int
    exrate_total: float
    n_in: int
    n_fail_in: int
    exrate_in: float
    n_oom: int
    n_fail_oom: int
    exrate_oom: float


_KNOWN_FAILURE_STATUSES = ["timeout", "parse_error", "out_of_set", "exception"]


def exrate_breakdown_by_status(df: pd.DataFrame) -> Dict[str, Any]:
    """ExRate の内訳（status別件数/率）を overall / in_map / oom で返す。

    - status は {ok, timeout, parse_error, out_of_set, exception} を想定。
    - 想定外の status があれば other にまとめる。
    """
    d = df.copy()
    d["y"] = as_num(d["y"])

    def one(sub: pd.DataFrame) -> Dict[str, Any]:
        n_total = int(len(sub))
        st = sub.get("status", "ok").astype(str).fillna("missing")
        vc = st.value_counts(dropna=False)
        n_ok = int(vc.get("ok", 0))

        out: Dict[str, Any] = {
            "n_total": n_total,
            "n_ok": n_ok,
            "exrate": float((n_total - n_ok) / n_total) if n_total > 0 else float("nan"),
            "by_status": {},
        }
        if n_total == 0:
            return out

        other = 0
        for s in ["ok"] + _KNOWN_FAILURE_STATUSES:
            c = int(vc.get(s, 0))
            out["by_status"][s] = {"count": c, "rate": float(c / n_total)}

        for s, c in vc.items():
            if s in ("ok", *_KNOWN_FAILURE_STATUSES):
                continue
            other += int(c)
        if other > 0:
            out["by_status"]["other"] = {"count": int(other), "rate": float(other / n_total)}

        return out

    overall = one(d)
    in_map = one(d[d["y"] != -1])
    oom = one(d[d["y"] == -1])
    return {"overall": overall, "in_map": in_map, "oom": oom}


def compute_exrate(df: pd.DataFrame) -> ExRateSummary:
    status = df.get("status", pd.Series(["ok"] * len(df)))
    fail = status.astype(str) != "ok"
    y = as_num(df["y"])
    is_oom = y == -1
    is_in = ~is_oom

    n_total = int(len(df))
    n_fail = int(fail.sum())

    n_in = int(is_in.sum())
    n_fail_in = int((fail & is_in).sum())

    n_oom = int(is_oom.sum())
    n_fail_oom = int((fail & is_oom).sum())

    ex_total = (n_fail / n_total) if n_total else float("nan")
    ex_in = (n_fail_in / n_in) if n_in else float("nan")
    ex_oom = (n_fail_oom / n_oom) if n_oom else float("nan")

    return ExRateSummary(
        n_total=n_total,
        n_fail=n_fail,
        exrate_total=ex_total,
        n_in=n_in,
        n_fail_in=n_fail_in,
        exrate_in=ex_in,
        n_oom=n_oom,
        n_fail_oom=n_fail_oom,
        exrate_oom=ex_oom,
    )


def compute_accept_by_tau(df_ok: pd.DataFrame, tau: float) -> pd.Series:
    """accept_score>=tau かつ chosen_id が有効な場合を accept とみなす。

    JSONLの欠損・NaNに頑健にする（NaN != -1 が True になってしまう事故を防ぐ）。
    """
    score = as_num(df_ok.get("accept_score", df_ok.get("combined", df_ok.get("conf", df_ok.get("confidence_topic")))))
    chosen = as_num(df_ok.get("chosen_id", df_ok.get("chosen_node_id_topic")))
    # has_choice: chosen_id が数値かつ -1 ではない
    has_choice = chosen.notna() & (chosen != -1.0)
    return has_choice & score.notna() & (score >= float(tau))

def risk_coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    """In-map (y!=-1) の risk–coverage を accept_score のしきい値 tau で掃引して作る。"""
    d = df.copy()
    d["y"] = as_num(d["y"])
    d["chosen_id"] = as_num(d.get("chosen_id"))
    d["accept_score"] = as_num(d.get("accept_score"))

    df_ok_in = d[(d["status"].astype(str) == "ok") & (d["y"] != -1)]
    if len(df_ok_in) == 0:
        return pd.DataFrame(columns=["tau", "coverage", "selective_risk", "n_ok", "n_accept"])

    scores = df_ok_in["accept_score"].dropna().values
    if len(scores) == 0:
        taus = np.array([0.0])
    else:
        qs = np.linspace(0, 1, 101)
        taus = np.unique(np.quantile(scores, qs))
        taus = np.unique(
            np.concatenate(
                [
                    [float(np.min(scores) - 1e-12)],
                    taus,
                    [float(np.max(scores) + 1e-12)],
                ]
            )
        )

    n_ok = len(df_ok_in)
    rows = []
    for tau in taus:
        acc = compute_accept_by_tau(df_ok_in, float(tau))
        n_acc = int(acc.sum())
        if n_acc == 0:
            rows.append({"tau": float(tau), "coverage": 0.0, "selective_risk": float("nan"), "n_ok": n_ok, "n_accept": 0})
            continue

        y_true = df_ok_in.loc[acc, "y"].astype(int).values
        y_pred = df_ok_in.loc[acc, "chosen_id"].astype(int).values
        wrong = int((y_pred != y_true).sum())
        risk = wrong / n_acc
        cov = n_acc / n_ok
        rows.append({"tau": float(tau), "coverage": float(cov), "selective_risk": float(risk), "n_ok": n_ok, "n_accept": n_acc})

    return pd.DataFrame(rows).sort_values(["coverage", "tau"]).reset_index(drop=True)


def oom_curve(df: pd.DataFrame) -> pd.DataFrame:
    """OoM (y==-1) の TrueReject / FalseAccept を accept_score のしきい値 tau で掃引して作る。"""
    d = df.copy()
    d["y"] = as_num(d["y"])
    d["chosen_id"] = as_num(d.get("chosen_id"))
    d["accept_score"] = as_num(d.get("accept_score"))

    df_ok_oom = d[(d["status"].astype(str) == "ok") & (d["y"] == -1)]
    if len(df_ok_oom) == 0:
        return pd.DataFrame(columns=["tau", "true_reject_rate", "false_accept_rate", "n_ok", "n_accept"])

    scores = df_ok_oom["accept_score"].dropna().values
    if len(scores) == 0:
        taus = np.array([0.0])
    else:
        qs = np.linspace(0, 1, 101)
        taus = np.unique(np.quantile(scores, qs))
        taus = np.unique(
            np.concatenate(
                [
                    [float(np.min(scores) - 1e-12)],
                    taus,
                    [float(np.max(scores) + 1e-12)],
                ]
            )
        )

    n_ok = len(df_ok_oom)
    rows = []
    for tau in taus:
        acc = compute_accept_by_tau(df_ok_oom, float(tau))
        n_acc = int(acc.sum())
        far = n_acc / n_ok
        trr = 1.0 - far
        rows.append({"tau": float(tau), "true_reject_rate": float(trr), "false_accept_rate": float(far), "n_ok": n_ok, "n_accept": n_acc})

    return pd.DataFrame(rows).sort_values(["false_accept_rate", "tau"]).reset_index(drop=True)


def operating_point_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """実運用点（y_hat）での指標（in-map coverage/risk, OoM TRR/FAR）。"""
    d = df.copy()
    d["y"] = as_num(d["y"])
    d["y_hat"] = as_num(d.get("y_hat")).fillna(-1)
    ok = d["status"].astype(str) == "ok"

    in_ok = d[ok & (d["y"] != -1)]
    oom_ok = d[ok & (d["y"] == -1)]

    out: Dict[str, float] = {}

    # in-map
    if len(in_ok) > 0:
        accept = in_ok["y_hat"] != -1
        n_acc = int(accept.sum())
        out["in_n_ok"] = int(len(in_ok))
        out["in_coverage"] = float(n_acc / len(in_ok))
        if n_acc > 0:
            wrong = int((in_ok.loc[accept, "y_hat"].astype(int).values != in_ok.loc[accept, "y"].astype(int).values).sum())
            out["in_selective_risk"] = float(wrong / n_acc)
        else:
            out["in_selective_risk"] = float("nan")
    else:
        out["in_n_ok"] = 0
        out["in_coverage"] = float("nan")
        out["in_selective_risk"] = float("nan")

    # OoM
    if len(oom_ok) > 0:
        accept = oom_ok["y_hat"] != -1
        n_acc = int(accept.sum())
        out["oom_n_ok"] = int(len(oom_ok))
        out["oom_false_accept_rate"] = float(n_acc / len(oom_ok))
        out["oom_true_reject_rate"] = float(1.0 - out["oom_false_accept_rate"])
    else:
        out["oom_n_ok"] = 0
        out["oom_false_accept_rate"] = float("nan")
        out["oom_true_reject_rate"] = float("nan")

    return out


def tag_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """tag ごとの内訳（D_ok のみ）。"""
    d = df.copy()
    d["y"] = as_num(d["y"])
    d["y_hat"] = as_num(d.get("y_hat")).fillna(-1)
    d_ok = d[d["status"].astype(str) == "ok"].copy()

    e = explode_tags(d_ok)

    rows = []
    for tag, g in e.groupby("tag", dropna=False):
        in_g = g[g["y"] != -1]
        oom_g = g[g["y"] == -1]

        rec: Dict[str, Any] = {"tag": tag, "n_ok_total": int(len(g))}

        # in-map
        rec["n_ok_in"] = int(len(in_g))
        if len(in_g) > 0:
            acc = in_g["y_hat"] != -1
            n_acc = int(acc.sum())
            rec["in_coverage"] = float(n_acc / len(in_g))
            if n_acc > 0:
                wrong = int((in_g.loc[acc, "y_hat"].astype(int).values != in_g.loc[acc, "y"].astype(int).values).sum())
                rec["in_selective_risk"] = float(wrong / n_acc)
            else:
                rec["in_selective_risk"] = float("nan")
        else:
            rec["in_coverage"] = float("nan")
            rec["in_selective_risk"] = float("nan")

        # OoM
        rec["n_ok_oom"] = int(len(oom_g))
        if len(oom_g) > 0:
            acc = oom_g["y_hat"] != -1
            n_acc = int(acc.sum())
            rec["oom_false_accept_rate"] = float(n_acc / len(oom_g))
            rec["oom_true_reject_rate"] = float(1.0 - rec["oom_false_accept_rate"])
        else:
            rec["oom_false_accept_rate"] = float("nan")
            rec["oom_true_reject_rate"] = float("nan")

        rows.append(rec)

    return pd.DataFrame(rows).sort_values(["n_ok_total", "tag"], ascending=[False, True]).reset_index(drop=True)


def scenario_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """scenario ごとの内訳（D_ok のみ）。"""
    d = df.copy()
    d["y"] = as_num(d["y"])
    d["y_hat"] = as_num(d.get("y_hat")).fillna(-1)
    d_ok = d[d["status"].astype(str) == "ok"].copy()
    if "scenario" not in d_ok.columns:
        d_ok["scenario"] = ""

    rows = []
    for sc, g in d_ok.groupby("scenario", dropna=False):
        in_g = g[g["y"] != -1]
        oom_g = g[g["y"] == -1]

        rec: Dict[str, Any] = {"scenario": sc, "n_ok_total": int(len(g))}

        # in-map
        rec["n_ok_in"] = int(len(in_g))
        if len(in_g) > 0:
            acc = in_g["y_hat"] != -1
            n_acc = int(acc.sum())
            rec["in_coverage"] = float(n_acc / len(in_g))
            if n_acc > 0:
                wrong = int((in_g.loc[acc, "y_hat"].astype(int).values != in_g.loc[acc, "y"].astype(int).values).sum())
                rec["in_selective_risk"] = float(wrong / n_acc)
            else:
                rec["in_selective_risk"] = float("nan")
        else:
            rec["in_coverage"] = float("nan")
            rec["in_selective_risk"] = float("nan")

        # OoM
        rec["n_ok_oom"] = int(len(oom_g))
        if len(oom_g) > 0:
            acc = oom_g["y_hat"] != -1
            n_acc = int(acc.sum())
            rec["oom_false_accept_rate"] = float(n_acc / len(oom_g))
            rec["oom_true_reject_rate"] = float(1.0 - rec["oom_false_accept_rate"])
        else:
            rec["oom_false_accept_rate"] = float("nan")
            rec["oom_true_reject_rate"] = float("nan")

        rows.append(rec)

    return pd.DataFrame(rows).sort_values(["scenario"]).reset_index(drop=True)


# -----------------
# CLI
# -----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument(
        "--no_sweep",
        action="store_true",
        help="tau掃引（risk_coverage_curve/oom_curve）を生成しない（Testロックボックス運用向け）",
    )
    args = ap.parse_args()

    prefix = normalize_prefix(args.prefix)
    in_path = Path(args.in_jsonl)

    # ロックボックス運用の事故防止：Test での tau 掃引を拒否する
    # （prefix だけに依存すると事故るので、入力ファイル名のヒントも使う）
    if (prefix.lower().startswith("test") or in_path.name.lower().startswith("test")) and not args.no_sweep:
        raise SystemExit(
            "Refusing to generate tau-sweep outputs for Test-like inputs. "
            "Use --no_sweep for Test."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(in_path)

    # 混在事故（=嘘扱いされ得る）を集計側でも強制的に止める
    assert_singleton_fields(
        df,
        [
            'dataset_sha256',
            'dataset_name',
            'run_id',
            'method',
            'map_sha256',
            'code_sha256',
            'run_mode',
            'decision_mode',
            'map_info_variant',
        ],
    )


    # dataset_name でも Test を検知して、tau 掃引を二重にブロックする
    ds_name = str(df["dataset_name"].dropna().iloc[0]) if len(df) > 0 else ""
    if ds_name.lower().startswith("test") and not args.no_sweep:
        raise SystemExit(
            "Refusing to generate tau-sweep outputs for dataset_name starting with 'test'. "
            "Use --no_sweep for Test."
        )

    # 数値化（後段で使う）
    for col in ['y', 'y_hat', 'chosen_id', 'conf', 'cos', 'combined', 'accept_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # runner の仕様上、y_hat 欠損は abstain(-1) として扱う
    if 'y_hat' in df.columns:
        df['y_hat'] = df['y_hat'].fillna(-1)

    ex = compute_exrate(df)
    ex_bd = exrate_breakdown_by_status(df)
    op = operating_point_metrics(df)

    rc = None
    oc = None
    if not args.no_sweep:
        rc = risk_coverage_curve(df)
        oc = oom_curve(df)

    tb = tag_breakdown(df)
    sb = scenario_breakdown(df)

    # 保存（JSON）
    (out_dir / f"{prefix}metrics_overall.json").write_text(
        json.dumps({"exrate": asdict(ex), "exrate_breakdown": ex_bd, "operating_point": op}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ExRate の内訳を CSV でも出力（レビュー対応・本文への貼り付け用）
    rows = []
    for subset_name in ["overall", "in_map", "oom"]:
        sub = ex_bd.get(subset_name, {})
        n_total = sub.get("n_total", 0)
        by = sub.get("by_status", {})
        for st, v in by.items():
            rows.append(
                {
                    "subset": subset_name,
                    "status": st,
                    "count": int(v.get("count", 0)),
                    "rate_of_subset": float(v.get("rate", float("nan"))),
                    "n_total_subset": int(n_total),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}exrate_by_status.csv", index=False)

    if rc is not None:
        rc.to_csv(out_dir / f"{prefix}risk_coverage_curve.csv", index=False)
    if oc is not None:
        oc.to_csv(out_dir / f"{prefix}oom_curve.csv", index=False)

    tb.to_csv(out_dir / f"{prefix}tag_breakdown.csv", index=False)
    sb.to_csv(out_dir / f"{prefix}scenario_breakdown.csv", index=False)

    print("Saved:")
    saved = [
        out_dir / f"{prefix}metrics_overall.json",
        out_dir / f"{prefix}exrate_by_status.csv",
        out_dir / f"{prefix}tag_breakdown.csv",
        out_dir / f"{prefix}scenario_breakdown.csv",
    ]
    if rc is not None:
        saved.insert(1, out_dir / f"{prefix}risk_coverage_curve.csv")
    if oc is not None:
        idx = 2 if rc is not None else 1
        saved.insert(idx, out_dir / f"{prefix}oom_curve.csv")

    for p in saved:
        print(" -", p)


if __name__ == "__main__":
    main()

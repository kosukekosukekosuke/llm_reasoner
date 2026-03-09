#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase3_01_run_eval_ros.py

Phase 3: Dev/Test 共通の「実験実行」スクリプト（ROS版）

目的
- /llm_reasoner/query に自然言語クエリを 1 件ずつ publish
- /llm_reasoner/chosen_node_id, /confidence, /cosine, /meta を購読
- 1 クエリ = 1 行 (JSONL) として "一意" に保存する（追記・再開可能）
- timeout / parse_error / out_of_set / exception 等の実行失敗は status!=ok として記録し、
  指標の母集団 D_ok (= status==ok) から分離できるようにする

重要な安全策（研究が「嘘」に見えないため）
- 既存 out.jsonl への追記時に「別条件の混在」を防ぐ（dataset_sha256 を必ず一致チェック）
- ROS 遅延により「前クエリの meta が後から届く」混入を、timestamp_wall で可能な限り排除
- 例外・タイムアウトでも 1 行は必ず書き、後段の集計で失敗率(ExRate)として明示できる

入力（dataset JSON）
- JSON 配列: [{"id": str, "q": str, "y": int, "scenario": str, "tags": [str], ...}, ...]
  ※ 本スクリプトは `id/q/y` を必須とする。その他はそのままログに転記する。

出力（JSONL; 1 行 1 レコード）
- 主要列（Phase3 で「必須」とみなす列）:
    id, q, y, y_hat, status, scenario, tags, chosen_id, conf, cos, accept_score, dataset_n_total
  そのほか再現性のための補助列も付与する（run_id/method/dataset_sha256/...）。

使い方（例）
  python3 phase3_01_run_eval_ros.py \
    --dataset /path/to/dev_core.json \
    --out /path/to/out/dev_run.jsonl \
    --run_id 2026-01-14T12-00+0900 \
    --method llmreasoner_old_10

前提
- roscore が起動している
- llm_reasoner.launch 等で llm_reasoner.py ノードが起動している
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ROS imports (runtime dependency)
import rospy
from std_msgs.msg import Float32, Int32, String


# -------------------------
# Utilities
# -------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)) and not (isinstance(x, float) and (x != x)):  # not NaN
            return int(x)
        if isinstance(x, str) and x.strip() != "":
            return int(float(x))
        return None
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            v = float(x)
            if v != v:  # NaN
                return None
            return v
        if isinstance(x, str) and x.strip() != "":
            v = float(x)
            if v != v:
                return None
            return v
        return None
    except Exception:
        return None


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load dataset JSON (array) and perform lightweight schema validation.

    This runner is used to produce thesis-evidence logs, so we validate early to avoid
    silent omissions (e.g., duplicate ids being skipped on resume).
    """
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read dataset JSON: {dataset_path} ({e})")

    if not isinstance(data, list):
        raise SystemExit(f"Dataset must be a JSON array: {dataset_path}")

    ids: set[str] = set()
    for i, r in enumerate(data):
        if not isinstance(r, dict):
            raise SystemExit(f"Dataset element #{i} is not an object")

        if "id" not in r or "q" not in r or "y" not in r:
            raise SystemExit(f"Dataset element #{i} missing required keys: id/q/y")

        qid = str(r.get("id", "")).strip()
        if qid == "":
            raise SystemExit(f"Dataset element #{i} has empty id")

        if qid in ids:
            raise SystemExit(f"Dataset contains duplicate id: {qid}")
        ids.add(qid)

        q = r.get("q", "")
        if not isinstance(q, str) or q.strip() == "":
            raise SystemExit(f"Dataset element #{i} has empty/non-string q (id={qid})")

        y = _safe_int(r.get("y"))
        if y is None:
            raise SystemExit(f"Dataset element #{i} has non-integer y (id={qid})")

        scenario = r.get("scenario", None)
        if scenario is not None and not isinstance(scenario, str):
            raise SystemExit(f"Dataset element #{i} has non-string scenario (id={qid})")

        tags = r.get("tags", None)
        if tags is not None:
            if not isinstance(tags, list) or any((not isinstance(t, str)) for t in tags):
                raise SystemExit(f"Dataset element #{i} has invalid tags (must be list[str]) (id={qid})")

    return data


def _parse_meta(meta_json: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse llm_reasoner meta JSON.

    Returns:
      (meta_dict, None) on success
      (None, error_string) on failure
    """
    try:
        m = json.loads(meta_json)
        if isinstance(m, dict):
            return m, None
        return None, "meta_not_object"
    except Exception as e:
        return None, f"meta_json_parse_error: {e}"


def _read_first_record(path: Path) -> Optional[Dict[str, Any]]:
    """Read the first non-empty JSONL record (dict) from a file."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _guard_against_mixed_runs(out_path: Path, *, dataset_sha256: str, run_id: str, method: str) -> None:
    """Prevent appending results of a different condition into the same JSONL.

    Rationale: accidental mixing can look like "data tampering" even if unintentional.
    """
    first = _read_first_record(out_path)
    if first is None:
        return

    prev_ds = str(first.get("dataset_sha256", ""))
    if prev_ds and prev_ds != dataset_sha256:
        raise SystemExit(
            "Refusing to append: dataset_sha256 differs from existing JSONL.\n"
            f"  existing: {prev_ds}\n"
            f"  current : {dataset_sha256}\n"
            "Use a different --out path for another dataset."
        )

    # Soft checks: only compare if both sides have non-empty values.
    prev_run_id = str(first.get("run_id", "")).strip()
    prev_method = str(first.get("method", "")).strip()

    # If existing JSONL already encodes run_id/method, require the CLI to specify them too.
    # This prevents accidental mixed append when resuming with missing flags.
    if prev_run_id and not run_id:
        raise SystemExit(
            'Refusing to append: existing JSONL has run_id but --run_id is empty.\n'
            f'  existing: {prev_run_id}\n'
            'Please pass the same --run_id when resuming (or write to a new --out).\n'
        )

    if prev_method and not method:
        raise SystemExit(
            'Refusing to append: existing JSONL has method but --method is empty.\n'
            f'  existing: {prev_method}\n'
            'Please pass the same --method when resuming (or write to a new --out).\n'
        )

    if prev_run_id and run_id and (prev_run_id != run_id):
        raise SystemExit(
            "Refusing to append: run_id differs from existing JSONL.\n"
            f"  existing: {prev_run_id}\n"
            f"  current : {run_id}\n"
            "Use a different --out path (or keep run_id consistent when resuming)."
        )

    if prev_method and method and (prev_method != method):
        raise SystemExit(
            "Refusing to append: method differs from existing JSONL.\n"
            f"  existing: {prev_method}\n"
            f"  current : {method}\n"
            "Use a different --out path (or keep method consistent when resuming)."
        )

def _scan_existing_jsonl(out_path: Path, *, dataset_sha256: str, run_id: str, method: str) -> Tuple[set[str], Optional[str], Optional[str]]:
    """Scan an existing JSONL to support resume and to guard against mixed appends.

    - Enforces: dataset_sha256 consistency across all records
    - Enforces: run_id/method singleton if present (and matches CLI if provided)
    - Enforces: no duplicate ids already in the JSONL (duplicates look like data issues)
    """
    seen: set[str] = set()
    if not out_path.exists():
        return seen, None, None

    run_ids: set[str] = set()
    methods: set[str] = set()
    map_shas: set[str] = set()
    code_shas: set[str] = set()

    for ln, line in enumerate(out_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            raise SystemExit(f"Existing JSONL parse error at line {ln}: {out_path}")
        if not isinstance(rec, dict):
            raise SystemExit(f"Existing JSONL record is not an object at line {ln}: {out_path}")

        prev_ds = str(rec.get("dataset_sha256", "")).strip()
        if prev_ds and prev_ds != dataset_sha256:
            raise SystemExit(
                "Refusing to append: dataset_sha256 differs inside existing JSONL.\n"
                f"  line     : {ln}\n"
                f"  existing : {prev_ds}\n"
                f"  current  : {dataset_sha256}\n"
                "Use a different --out path for another dataset."
            )

        rid = str(rec.get("run_id", "")).strip()
        if rid:
            run_ids.add(rid)
        mth = str(rec.get("method", "")).strip()
        if mth:
            methods.add(mth)

        msh = str(rec.get("map_sha256", "")).strip()
        if msh:
            map_shas.add(msh)
        csh = str(rec.get("code_sha256", "")).strip()
        if csh:
            code_shas.add(csh)

        if "id" in rec:
            qid = str(rec["id"]).strip()
            if qid in seen:
                raise SystemExit(f"Existing JSONL has duplicated id at line {ln}: {qid}")
            if qid:
                seen.add(qid)

    if len(run_ids) > 1:
        raise SystemExit(f"Existing JSONL has multiple run_id values: {sorted(run_ids)}")
    if len(methods) > 1:
        raise SystemExit(f"Existing JSONL has multiple method values: {sorted(methods)}")

    if run_ids:
        prev = next(iter(run_ids))
        if not run_id:
            raise SystemExit(
                'Refusing to append: existing JSONL has run_id but --run_id is empty.\n'
                f'  existing: {prev}\n'
                'Please pass the same --run_id when resuming (or write to a new --out).\n'
            )
        if prev != run_id:
            raise SystemExit(
                "Refusing to append: run_id differs from existing JSONL.\n"
                f"  existing: {prev}\n"
                f"  current : {run_id}\n"
                "Use a different --out path (or keep run_id consistent when resuming)."
            )

    if methods:
        prev = next(iter(methods))
        if not method:
            raise SystemExit(
                'Refusing to append: existing JSONL has method but --method is empty.\n'
                f'  existing: {prev}\n'
                'Please pass the same --method when resuming (or write to a new --out).\n'
            )
        if prev != method:
            raise SystemExit(
                "Refusing to append: method differs from existing JSONL.\n"
                f"  existing: {prev}\n"
                f"  current : {method}\n"
                "Use a different --out path (or keep method consistent when resuming)."
            )

    if len(map_shas) > 1:
        raise SystemExit(
            "Existing JSONL appears to contain multiple map_sha256 values (mixed-map run).\n"
            f"  values: {sorted(map_shas)}\n"
            "Please split runs by map and regenerate JSONL."
        )
    if len(code_shas) > 1:
        raise SystemExit(
            "Existing JSONL appears to contain multiple code_sha256 values (mixed-code run).\n"
            f"  values: {sorted(code_shas)}\n"
            "Please split runs by code revision and regenerate JSONL."
        )

    expected_map_sha256 = next(iter(map_shas)) if map_shas else None
    expected_code_sha256 = next(iter(code_shas)) if code_shas else None

    return seen, expected_map_sha256, expected_code_sha256



# -------------------------
# ROS subscriber state
# -------------------------
@dataclass
class Latest:
    """Holds the latest messages (value + reception time)."""

    chosen_node_id: Optional[int] = None
    confidence: Optional[float] = None
    cosine: Optional[float] = None
    meta_json: Optional[str] = None

    t_chosen: Optional[float] = None
    t_conf: Optional[float] = None
    t_cos: Optional[float] = None
    t_meta: Optional[float] = None

    def clear(self) -> None:
        self.chosen_node_id = None
        self.confidence = None
        self.cosine = None
        self.meta_json = None
        self.t_chosen = None
        self.t_conf = None
        self.t_cos = None
        self.t_meta = None

    def ready(self, *, t0: float, sync_window_sec: Optional[float]) -> bool:
        """Return True when all 4 outputs have been received after publish time (t0).

        We also require the 4 receptions to be close enough (sync window) to reduce cross-query mix-in.
        """
        required_times = [self.t_chosen, self.t_conf, self.t_cos, self.t_meta]
        if any(t is None for t in required_times):
            return False

        # All must be received *after* publish time.
        if any(float(t) < float(t0) for t in required_times):
            return False

        if sync_window_sec is not None:
            tmin = min(float(self.t_chosen), float(self.t_conf), float(self.t_cos), float(self.t_meta))
            tmax = max(float(self.t_chosen), float(self.t_conf), float(self.t_cos), float(self.t_meta))
            if (tmax - tmin) > float(sync_window_sec):
                return False

        return True


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="test_*.json など（JSON配列）")
    ap.add_argument("--out", required=True, help="出力JSONLパス（追記モード。再開可）")
    ap.add_argument("--run_id", default="", help="任意：実行ID（例: 2026-01-14T12-00+0900）")
    ap.add_argument("--method", default="", help="任意：手法名（例: llmreasoner_old_10）")

    ap.add_argument("--timeout_sec", type=float, default=60.0, help="1クエリの待ち時間（秒）")
    ap.add_argument("--sleep_after_pub", type=float, default=0.05, help="publish直後の短い待ち（秒）")
    ap.add_argument("--flush_before_pub", type=float, default=0.05, help="publish前に古い出力が流れ切るのを待つ（秒）")
    ap.add_argument(
        "--sync_window_sec",
        type=float,
        default=0.30,
        help="4出力が揃う許容時間幅（秒）。遅延混入を避けるための簡易同期",
    )

    ap.add_argument("--topic_query", default="/llm_reasoner/query")
    ap.add_argument("--topic_id", default="/llm_reasoner/chosen_node_id")
    ap.add_argument("--topic_conf", default="/llm_reasoner/confidence")
    ap.add_argument("--topic_cos", default="/llm_reasoner/cosine")
    ap.add_argument("--topic_meta", default="/llm_reasoner/meta")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_sha256 = _sha256_file(dataset_path)
    dataset_name = dataset_path.name

    # Prevent accidental mixing when appending.
    _guard_against_mixed_runs(out_path, dataset_sha256=dataset_sha256, run_id=str(args.run_id).strip(), method=str(args.method).strip())

    data = _load_dataset(dataset_path)

    # Load existing ids (resume-safe) + strict consistency checks
    seen, expected_map_sha256, expected_code_sha256 = _scan_existing_jsonl(
        out_path,
        dataset_sha256=dataset_sha256,
        run_id=str(args.run_id).strip(),
        method=str(args.method).strip(),
    )

    # ROS init
    rospy.init_node("phase3_eval_runner", anonymous=True, disable_signals=True)

    pub = rospy.Publisher(args.topic_query, String, queue_size=1)

    latest = Latest()

    def _cb_id(msg: Int32) -> None:
        latest.chosen_node_id = int(msg.data)
        latest.t_chosen = time.time()

    def _cb_conf(msg: Float32) -> None:
        latest.confidence = float(msg.data)
        latest.t_conf = time.time()

    def _cb_cos(msg: Float32) -> None:
        latest.cosine = float(msg.data)
        latest.t_cos = time.time()

    def _cb_meta(msg: String) -> None:
        latest.meta_json = str(msg.data)
        latest.t_meta = time.time()

    rospy.Subscriber(args.topic_id, Int32, _cb_id, queue_size=1)
    rospy.Subscriber(args.topic_conf, Float32, _cb_conf, queue_size=1)
    rospy.Subscriber(args.topic_cos, Float32, _cb_cos, queue_size=1)
    rospy.Subscriber(args.topic_meta, String, _cb_meta, queue_size=1)

    # Give subscribers a moment to connect
    time.sleep(0.2)

    n_total = len(data)
    n_skip = sum(1 for r in data if str(r["id"]) in seen)
    print(f"Dataset: {dataset_name} | total={n_total} | already_done={n_skip} | out={out_path}")
    if args.run_id or args.method:
        print(f"run_id={args.run_id!r} | method={args.method!r}")

    with out_path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(data):
            qid = str(r["id"])
            q = str(r["q"])

            if qid in seen:
                continue

            # Clear old state and flush possible in-flight messages
            latest.clear()
            time.sleep(float(args.flush_before_pub))

            t0 = time.time()
            pub.publish(String(data=q))
            time.sleep(float(args.sleep_after_pub))

            status: Optional[str] = None
            status_detail: Optional[str] = None
            meta: Optional[Dict[str, Any]] = None
            meta_err: Optional[str] = None

            # Wait for outputs
            while (time.time() - t0) < float(args.timeout_sec) and not rospy.is_shutdown():
                if latest.ready(t0=t0, sync_window_sec=args.sync_window_sec):
                    meta, meta_err = _parse_meta(latest.meta_json or "")
                    if meta_err is None and isinstance(meta, dict):
                        # Additional stale-meta defense:
                        # llm_reasoner.meta.timestamp_wall is "processing start" time;
                        # if it's clearly older than publish time, treat as previous query and retry.
                        try:
                            tw = meta.get("timestamp_wall")
                            if isinstance(tw, (int, float)) and float(tw) < (t0 - 1e-3):
                                latest.clear()
                                meta = None
                                meta_err = None
                                time.sleep(0.005)
                                continue
                        except Exception:
                            pass

                        status = str(meta.get("status", "ok"))
                        if "gen_status" in meta:
                            status_detail = f"gen_status={meta.get('gen_status')}"
                        elif "detail" in meta:
                            status_detail = str(meta.get("detail"))
                        else:
                            status_detail = None
                    else:
                        status = "parse_error"
                        status_detail = meta_err
                        meta = None
                    break

                time.sleep(0.005)

            if status is None:
                status = "timeout"
                status_detail = "runner_timeout"

            # Pull raw topics (might be None on failure)
            chosen_topic = latest.chosen_node_id
            conf_topic = latest.confidence
            cos_topic = latest.cosine
            meta_json = latest.meta_json

            # Parse meta-derived values
            chosen_id = _safe_int(meta.get("chosen_id")) if isinstance(meta, dict) else None
            abstain_node_id = _safe_int(meta.get("abstain_node_id")) if isinstance(meta, dict) else None
            conf_m = _safe_float(meta.get("conf")) if isinstance(meta, dict) else None
            cos_m = _safe_float(meta.get("cos")) if isinstance(meta, dict) else None
            combined = _safe_float(meta.get("combined")) if isinstance(meta, dict) else None

            accept: Optional[bool] = None
            if isinstance(meta, dict) and "accept" in meta:
                v = meta.get("accept")
                if isinstance(v, bool):
                    accept = v
                elif isinstance(v, (int, float)) and v in (0, 1):
                    accept = bool(v)

            accept_reason = str(meta.get("accept_reason")) if isinstance(meta, dict) and "accept_reason" in meta else None

            # CL13 整合: status!=ok は "execution_failure" 扱い（meta 側で出ていない場合は runner 側で補完）
            if status != "ok" and not accept_reason:
                accept_reason = "execution_failure"


            # Effective scores for analysis (prefer meta if available; fallback to topics)
            conf_eff = conf_m if conf_m is not None else conf_topic
            cos_eff = cos_m if cos_m is not None else cos_topic

            # Final decision y_hat (research definition): node_id or -1 (abstain)
            # - status!=ok => -1
            # - status==ok and accept==False => abstain_node_id (fallback -1)
            # - status==ok and accept==True  => chosen_topic (fallback chosen_id/meta)
            # - accept missing => fallback chosen_topic
            if status != "ok":
                y_hat = -1
            else:
                if accept is False:
                    y_hat = int(abstain_node_id) if abstain_node_id is not None else -1
                elif accept is True:
                    if chosen_topic is not None:
                        y_hat = int(chosen_topic)
                    elif chosen_id is not None:
                        y_hat = int(chosen_id)
                    else:
                        y_hat = -1
                else:
                    y_hat = int(chosen_topic) if chosen_topic is not None else -1

            # Internal chosen id (for sweeping / debugging). Prefer meta.chosen_id when available.
            # For OoM gate / meta-missing cases, fall back to the published chosen_node_id_topic.
            chosen_id_final = chosen_id
            if chosen_id_final is None and chosen_topic is not None:
                chosen_id_final = int(chosen_topic)

            # Risk–coverage sweep score (model-side accept score)
            accept_score = combined if combined is not None else conf_eff

            # Extra reproducibility helpers (easy columns)
            map_sha256 = str(meta.get("map_sha256")) if isinstance(meta, dict) and "map_sha256" in meta else None
            code_sha256 = str(meta.get("code_sha256")) if isinstance(meta, dict) and "code_sha256" in meta else None
            run_mode = str(meta.get("run_mode")) if isinstance(meta, dict) and "run_mode" in meta else None
            decision_mode = str(meta.get("decision_mode")) if isinstance(meta, dict) and "decision_mode" in meta else None
            map_info_variant = str(meta.get("map_info_variant")) if isinstance(meta, dict) and "map_info_variant" in meta else None


            # --- Run-condition consistency checks (only when appending) ---
            if out_path.exists() and out_path.stat().st_size > 0:
                # If the existing JSONL has a defined map/code hash, enforce match.
                if expected_map_sha256 is not None and map_sha256 is not None and map_sha256 != expected_map_sha256:
                    raise SystemExit(
                        f"""Refusing to append: map_sha256 differs from existing JSONL.
              existing: {expected_map_sha256}
              current : {map_sha256}
            This would mix different maps in one run file."""
                    )
                if expected_code_sha256 is not None and code_sha256 is not None and code_sha256 != expected_code_sha256:
                    raise SystemExit(
                        f"""Refusing to append: code_sha256 differs from existing JSONL.
              existing: {expected_code_sha256}
              current : {code_sha256}
            This would mix different code revisions in one run file."""
                    )

                # If the existing JSONL didn't record map/code yet (e.g., early timeouts),
                # lock them in at the first time we observe non-empty values.
                if expected_map_sha256 is None and map_sha256 is not None:
                    expected_map_sha256 = map_sha256
                if expected_code_sha256 is None and code_sha256 is not None:
                    expected_code_sha256 = code_sha256
            # --- end checks ---

            out_rec: Dict[str, Any] = {
                # dataset fields
                "id": qid,
                "q": q,
                "y": r["y"],
                "scenario": r.get("scenario"),
                "tags": (r.get("tags") or []),
                "lang": r.get("lang"),

                # run metadata
                "ts_utc": _utc_now_iso(),
                "run_id": str(args.run_id),
                "method": str(args.method),
                "dataset_name": dataset_name,
                "dataset_sha256": dataset_sha256,
                "dataset_n_total": int(n_total),

                # raw topics (debug / verification)
                "chosen_node_id_topic": chosen_topic,
                "confidence_topic": conf_topic,
                "cosine_topic": cos_topic,

                # final output for evaluation
                "y_hat": int(y_hat),

                # meta (minimal columns used by aggregation)
                "status": status,
                "status_detail": status_detail,
                "accept": accept,
                "accept_reason": accept_reason,
                "chosen_id": chosen_id_final,
                "conf": conf_eff,
                "cos": cos_eff,
                "combined": combined,
                "accept_score": accept_score,

                # handy config fingerprints (optional)
                "map_sha256": map_sha256,
                "code_sha256": code_sha256,
                "run_mode": run_mode,
                "decision_mode": decision_mode,
                "map_info_variant": map_info_variant,
            }

            if meta_err is not None:
                out_rec["meta_error"] = meta_err
            if meta_json is not None:
                # 解析再現性のため raw を残す（必要なら後で削除してよい）
                out_rec["meta"] = meta_json
                out_rec["meta_json"] = meta_json

            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()
            seen.add(qid)

            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{n_total}] wrote: id={qid} status={status} y_hat={y_hat}")

    print(f"Done. Wrote JSONL: {out_path}")


if __name__ == "__main__":
    main()

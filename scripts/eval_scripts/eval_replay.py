#!/usr/bin/env python3
"""
============================================================
eval_replay.py  （第1段階 実験用：クエリ送出 & メタ収集）
============================================================
■ 目的
  - /llm_reasoner/query にクエリを順次 publish
  - /llm_reasoner/meta を購読し、対応メタを確実に受信・保存
  - 実験の成功（期待どおりの往復が起きているか）を、その場の1行ログとJSON保存で可視化

■ 特徴
  - 往復同期：meta受信 → JSON追記 → cooldown待機 → 次のクエリ
  - --timeout 到達時は timeout=true で記録し次へ
  - --wait-for-subs で /llm_reasoner/query の購読者待ち（ノード起動待ち）
  - verbose（一行サマリ出力）、query一致チェック、保存先ディレクトリ自動作成

■ 想定トピック
  - publish: /llm_reasoner/query  (std_msgs/String)
  - subscribe: /llm_reasoner/meta (std_msgs/String)

■ 使い方（例）
  ./eval_replay.py \
    --queries ~/catkin_ws/src/llm_reasoner/config/evaluation_dataset/eval_v0/eval_v0_basic.json \
    --out     ~/catkin_ws/src/llm_reasoner/config/evaluation_log/eval_v0/results_eval_v0_basic.json \
    --timeout 60 --cooldown 5 --wait-for-subs --verbose

※ 本スクリプトは ROS (rospy) が必要です。
"""

import os
import sys
import json
import time
import argparse
from threading import Event

import rospy
from std_msgs.msg import String

EMO_OK = "✅"
EMO_INFO = "ℹ️"
EMO_WARN = "⚠️"
EMO_PUB = "▶"
EMO_SAVE = "💾"

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_queries(path):
    with open(path, "r") as f:
        data = json.load(f)
    # 正規化：最低限 "query" を持つ dict のリストにする
    out = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and "query" in item:
            out.append(item)
        elif isinstance(item, str):
            out.append({"query": item})
        else:
            rospy.logwarn(f"{EMO_WARN} skip malformed query at index {i}: {item}")
    return out

def fmt_num(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "-"

class EvalReplayer(object):
    def __init__(self, args):
        self.args = args
        self.queries = load_queries(args.queries)
        self.total = len(self.queries)
        if self.total == 0:
            rospy.logerr("No queries found.")
            sys.exit(1)

        self.pub = rospy.Publisher("/llm_reasoner/query", String, queue_size=10)
        self.meta_sub = rospy.Subscriber("/llm_reasoner/meta", String, self._on_meta)

        self.current_query = None
        self.current_index = -1
        self.pending_event = Event()
        self.last_meta = None

        ensure_dir(args.out)
        self.saved_metas = []

        if self.args.verbose:
            rospy.loginfo(f"{EMO_INFO} Loaded {self.total} queries from {self.args.queries}")
            rospy.loginfo(f"{EMO_INFO} Output will be saved to {self.args.out}")

    def _on_meta(self, msg):
        try:
            meta = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"{EMO_WARN} failed to parse meta JSON: {e}")
            return

        # クエリ整合性チェック（ズレは warn）
        mq = meta.get("query", "")
        if self.current_query is not None and mq != self.current_query:
            rospy.logwarn(f"{EMO_WARN} query mismatch: sent='{self.current_query}' got='{mq}'")

        # enrich（index, timeout=false, gold補助）
        meta["_ts"] = time.time()
        meta["index"] = self.current_index + 1
        meta["timeout"] = False

        qrec = self.queries[self.current_index] if 0 <= self.current_index < self.total else {}
        if "gold_node_id" in qrec:  meta["gold_node_id"] = qrec["gold_node_id"]
        if "gold_label"   in qrec:  meta["gold_label"]   = qrec["gold_label"]

        self.last_meta = meta
        self.pending_event.set()

        if self.args.verbose:
            rospy.loginfo(f"[eval_replay] {EMO_OK} meta received: {mq if mq else '(empty)'}")
            rospy.loginfo(self._pretty_line(self.current_index + 1, self.total, meta))

    def _pretty_line(self, idx, total, meta):
        if not meta:
            return f"{EMO_OK} [{idx}/{total}] (no meta)"

        q    = meta.get("query", self.current_query)
        nid  = meta.get("node_id", None)
        lbl  = meta.get("label", "?")
        rsn  = meta.get("accept_reason", "?")
        att  = meta.get("attempts", "?")
        mode = meta.get("decision_mode", "?")

        conf = fmt_num(meta.get("confidence"))
        cos  = fmt_num(meta.get("cosine"))
        comb = fmt_num(meta.get("combined_score"))

        node_str = f"id={nid}({lbl})" if nid is not None else "id=- (?)"

        return (f"{EMO_OK} [{idx}/{total}] '{q}' -> {node_str} "
                f"conf={conf} cos={cos} comb={comb} "
                f"reason={rsn} attempts={att} mode={mode}")

    def run(self):
        if self.args.wait_for_subs:
            # /llm_reasoner/query に購読者が付くまで待機（LLMノード起動待ち）
            rospy.loginfo(f"{EMO_INFO} waiting subscribers on /llm_reasoner/query ...")
            while not rospy.is_shutdown() and self.pub.get_num_connections() == 0:
                rospy.sleep(0.2)
            rospy.loginfo(f"{EMO_OK} subscriber detected.")

        try:
            for i, qrec in enumerate(self.queries):
                if rospy.is_shutdown():
                    break

                self.current_index = i
                self.current_query = qrec.get("query", "")
                self.last_meta = None
                self.pending_event.clear()

                # publish
                self.pub.publish(String(self.current_query))
                if self.args.verbose:
                    rospy.loginfo(f"{EMO_INFO} [{i+1}/{self.total}] {EMO_PUB} publish: {self.current_query}")

                # wait meta (or timeout)
                got = self.pending_event.wait(timeout=self.args.timeout)
                if not got:
                    # timeout 記録
                    meta = {
                        "query": self.current_query,
                        "node_id": None,
                        "label": "",
                        "confidence": None,
                        "cosine": None,
                        "combined_score": None,
                        "accept_reason": "timeout",
                        "attempts": None,
                        "decision_mode": "",
                        "fallback_used": None,
                        "index": i+1,
                        "timeout": True,
                        "_ts": time.time()
                    }
                    if "gold_node_id" in qrec: meta["gold_node_id"] = qrec["gold_node_id"]
                    if "gold_label"   in qrec: meta["gold_label"]   = qrec["gold_label"]
                    self.saved_metas.append(meta)
                    if self.args.verbose:
                        rospy.logwarn(f"{EMO_WARN} timeout: {self.current_query}")
                else:
                    # 受信メタ保存
                    self.saved_metas.append(self.last_meta)

                # 保存（毎回上書き保存で冪等化）
                with open(self.args.out, "w") as f:
                    json.dump(self.saved_metas, f, ensure_ascii=False, indent=2)
                if self.args.verbose:
                    rospy.loginfo(f"[eval_replay] {EMO_SAVE} saved metas: {self.args.out}")

                # cooldown
                rospy.sleep(self.args.cooldown)

        except KeyboardInterrupt:
            rospy.loginfo("Interrupted by user.")
        finally:
            # 最終保存
            with open(self.args.out, "w") as f:
                json.dump(self.saved_metas, f, ensure_ascii=False, indent=2)
            rospy.loginfo(f"{EMO_SAVE} final saved: {self.args.out}")

def main():
    parser = argparse.ArgumentParser(description="Replay queries to /llm_reasoner/query and collect /llm_reasoner/meta.")
    parser.add_argument("--queries", required=True, help="JSON file of queries")
    parser.add_argument("--out", required=True, help="Output JSON file to save collected metas")
    parser.add_argument("--timeout", type=float, default=60.0, help="Max seconds to wait meta per query")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds to sleep after receiving meta before next query")
    parser.add_argument("--wait-for-subs", action="store_true", help="Wait until /llm_reasoner/query has subscribers")
    parser.add_argument("--verbose", action="store_true", help="Print per-step logs")
    args, _ = parser.parse_known_args()

    rospy.init_node("eval_replay", anonymous=True)
    EvalReplayer(args).run()

if __name__ == "__main__":
    main()

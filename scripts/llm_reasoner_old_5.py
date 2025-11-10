#!/usr/bin/env python3
"""
LLM-based Reasoner ROS Node (ops-minimal)
- YAMLからノード情報を読み込む (id, label, semantic, description)
- 最小プロンプトで LLM にノード番号を返させる
- token logprob から信頼度を計算（未取得なら -1）
- 安全設計：数字検証・低信頼度リトライ・fallback ノード
- 運用最小出力：node_id / confidence / cosine(今はNaN) / meta(JSON)
- 人間向けの可読出力は ROS log のみ（トピックは増やさない）
"""

import os
import re
import json
import yaml
import rospy
import numpy as np
from std_msgs.msg import String, Float32, Int32

try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    _llama_import_error = e


def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None


class LLMReasonerNode:
    def __init__(self):
        rospy.init_node("llm_reasoner", anonymous=False)

        # -------- ROS Params --------
        self.map_yaml_path   = rospy.get_param("~map_yaml_path", "")
        self.model_path      = rospy.get_param("~model_path", "")
        self.n_ctx           = int(rospy.get_param("~ctx_size", 2048))
        self.n_threads       = int(rospy.get_param("~threads", 4))
        self.temperature     = float(rospy.get_param("~temperature", 0.1))
        self.top_k           = int(rospy.get_param("~top_k", 10))
        self.top_p           = float(rospy.get_param("~top_p", 0.0))
        self.max_tokens      = int(rospy.get_param("~max_tokens", 64))
        self.conf_threshold  = float(rospy.get_param("~confidence_threshold", 0.5))
        self.max_retries     = int(rospy.get_param("~max_retries", 2))
        self.fallback_node   = int(rospy.get_param("~fallback_node", -1))

        # -------- ROS Topics --------
        self.sub_query = rospy.get_param("~sub_query", "/llm_reasoner/query")
        self.pub_node  = rospy.get_param("~pub_node",  "/llm_reasoner/chosen_node_id")
        self.pub_conf  = rospy.get_param("~pub_conf",  "/llm_reasoner/confidence")
        self.pub_cos   = rospy.get_param("~pub_cos",   "/llm_reasoner/cosine")
        self.pub_meta  = rospy.get_param("~pub_meta",  "/llm_reasoner/meta")

        # -------- Load Model --------
        if Llama is None:
            rospy.logwarn("llama_cpp import failed: %s", str(_llama_import_error))
            self.llm = None
        else:
            rospy.loginfo("🧩 Loading LLM model from: %s", self.model_path)
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                logits_all=True
            )

        # -------- Load Map --------
        self.node_ids, self.node_labels, self.node_semantics, self.node_descriptions = self.load_map(self.map_yaml_path)
        rospy.loginfo("📄 Loaded %d nodes from map.", len(self.node_ids))

        if self.fallback_node < 0 and len(self.node_ids) > 0:
            self.fallback_node = int(self.node_ids[0])

        # id→label / id→semantics（人間向けログで最終決定を可読表示）
        self.id2label = {int(i): str(l) for i, l in zip(self.node_ids, self.node_labels)}
        self.id2sem   = {int(i): str(s) for i, s in zip(self.node_ids, self.node_semantics)}

        # -------- ROS I/O --------
        self.pub_node_id     = rospy.Publisher(self.pub_node, Int32,  queue_size=10)
        self.pub_confidence  = rospy.Publisher(self.pub_conf, Float32, queue_size=10)
        self.pub_cosine      = rospy.Publisher(self.pub_cos,  Float32, queue_size=10)
        self.pub_meta_json   = rospy.Publisher(self.pub_meta, String,  queue_size=10)

        rospy.Subscriber(self.sub_query, String, self.callback, queue_size=1)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("✅ LLM Reasoner Node ready.")
        rospy.spin()

    # -------- Map --------
    def load_map(self, yaml_path):
        if not yaml_path or not os.path.exists(yaml_path):
            rospy.logwarn("Map YAML not found: %s", yaml_path)
            return [], [], [], []
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        nodes = data.get("NODE", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        ids, labels, semantics, descriptions = [], [], [], []
        for n in nodes:
            ids.append(int(n.get("id")))
            labels.append(str(n.get("label", "")).strip())
            sem = n.get("semantic", [])
            if isinstance(sem, list):
                sem = ", ".join(sem)
            semantics.append(str(sem))
            descriptions.append(str(n.get("description", "")).strip())
        return ids, labels, semantics, descriptions

    # -------- Prompt --------
    def build_prompt(self, query_text):
        lines = ["Available Locations:"]
        for nid, lbl, sem in zip(self.node_ids, self.node_labels, self.node_semantics):
            lines.append(f"{nid}. {lbl} ({sem})")
        lines.append(f"User Task: {query_text}")
        lines.append("Answer with the NUMBER of the correct choice:")
        return "\n".join(lines)

    # -------- LLM --------
    def call_llm(self, prompt):
        if self.llm is None:
            return {"text": "", "token_logprobs": None}
        out = self.llm(
            prompt=prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["###", "\n", "Instruction:"],
            echo=False,
            logprobs=1
        )
        try:
            text = out["choices"][0]["text"].strip()
        except Exception:
            text = ""
        try:
            token_logprobs = out["choices"][0]["logprobs"]["token_logprobs"]
        except Exception:
            token_logprobs = None
        return {"text": text, "token_logprobs": token_logprobs}

    def compute_confidence(self, token_logprobs):
        if not token_logprobs:
            return None
        avg_logp = float(np.mean(token_logprobs))
        conf = float(np.exp(avg_logp))
        return float(np.clip(conf, 0.0, 1.0))

    def extract_valid_num(self, text):
        m = re.search(r"\b-?\d+\b", text)
        if not m:
            return None
        v = _safe_int(m.group(0))
        return v if (v is not None and v in self.node_ids) else None

    # -------- Callback --------
    def callback(self, msg: String):
        query_text = msg.data.strip()
        if not query_text:
            rospy.logwarn("⚠️ Received empty query.")
            return

        rospy.loginfo("🟢 Inference started. query='%s'", query_text)

        prompt = self.build_prompt(query_text)
        attempts = 0
        used_fallback = False
        decision_source = "model"
        last_num = None
        chosen_node = None
        chosen_conf = None

        while attempts <= self.max_retries and not rospy.is_shutdown():
            attempts += 1
            out = self.call_llm(prompt)
            raw = out.get("text", "")
            num = self.extract_valid_num(raw)
            conf = self.compute_confidence(out.get("token_logprobs"))

            if num is not None:
                last_num = num

            lbl = self.id2label.get(num, "<?>") if num is not None else "<?>"
            rospy.loginfo("  Try %d → raw='%s' num=%s(label=%s) conf=%s",
                          attempts, raw, str(num), lbl, f"{conf:.3f}" if conf is not None else "None")

            if (num is not None) and (conf is None or conf >= self.conf_threshold):
                chosen_node = num
                chosen_conf = conf
                decision_source = "model"
                break

            # リトライ（閾値未達/非数）
            if attempts <= self.max_retries:
                continue

            # ここに来るのは最後の試行が終わった後
            if last_num is not None:
                chosen_node = last_num
                chosen_conf = conf
                decision_source = "low_conf_last_num"
            else:
                chosen_node = self.fallback_node
                chosen_conf = None
                decision_source = "fallback"
                used_fallback = True

        if chosen_node is None:
            chosen_node = self.fallback_node
            chosen_conf = None
            decision_source = "fallback"
            used_fallback = True

        # 出力（cosine は後日実装，今は NaN）
        cosine_sim = float("nan")

        self.pub_node_id.publish(Int32(int(chosen_node)))
        self.pub_confidence.publish(Float32(chosen_conf if chosen_conf is not None else -1.0))
        self.pub_cosine.publish(Float32(cosine_sim))

        meta = {
            "node_id": int(chosen_node),
            "label": self.id2label.get(int(chosen_node), "<?>"),
            "label_semantics": self.id2sem.get(int(chosen_node), ""),
            "confidence": float(chosen_conf) if chosen_conf is not None else -1.0,
            "cosine": None,
            "attempts": int(attempts),
            "last_num": int(last_num) if last_num is not None else None,
            "fallback_used": 1 if used_fallback else 0,
            "decision_source": decision_source,
            "fallback_node": int(self.fallback_node),
            "valid_ids": list(map(int, self.node_ids)),
            "query": query_text,
        }
        self.pub_meta_json.publish(String(json.dumps(meta, ensure_ascii=False)))

        # 最終決定（人間向けはログのみ）
        final_label = self.id2label.get(int(chosen_node), "<?>")
        final_sem   = self.id2sem.get(int(chosen_node), "")
        rospy.loginfo("✅ Inference finished. DECISION: id=%d, label=%s, semantics=%s",
                      int(chosen_node), final_label, final_sem)
        rospy.loginfo("📦 META: %s", json.dumps(meta, ensure_ascii=False))

    def cleanup(self):
        rospy.loginfo("🧹 Shutting down LLM Reasoner Node...")
        try:
            if hasattr(self, "llm"):
                del self.llm
            rospy.loginfo("✅ Model resources released.")
        except Exception:
            rospy.logwarn("⚠️ Failed to clean up model resources.")


if __name__ == "__main__":
    try:
        LLMReasonerNode()
    except rospy.ROSInterruptException:
        pass

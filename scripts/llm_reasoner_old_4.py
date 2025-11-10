#!/usr/bin/env python3
"""
LLM-based Reasoner ROS Node
- このノードは「意味推論モジュール」として用いる
- 機能
    - YAMLからノード情報を読み込む (id, label, semantic, description)
    - 最小プロンプトでLLMにノード番号を返させる
    - token logprobから信頼度を計算
    - 安全設計: 数字検証・低信頼度リトライ・fallbackノード
"""

import os
import re
import rospy
import yaml
import numpy as np
from std_msgs.msg import String
from llama_cpp import Llama


class LLMReasonerNode:
    def __init__(self):
        # --- ROSノード初期化 ---
        rospy.init_node("llm_reasoner", anonymous=False)

        # --- ROSパラメータ ---
        self.model_path = rospy.get_param(
            "~model_path",
            os.path.join(os.path.dirname(__file__), "llama.cpp/models/meta-llama-3-8b-instruct.Q4_K_M.gguf")
        )
        self.map_yaml_path = rospy.get_param(
            "~map_yaml_path",
            "/home/amsl/catkin_ws/src/rover_navigator/map/graph/ikuta_graph.yaml"
        )
        self.n_ctx = rospy.get_param("~ctx_size", 2048)
        self.n_threads = rospy.get_param("~threads", 4)
        self.temperature = rospy.get_param("~temperature", 0.1)
        self.top_k = rospy.get_param("~top_k", 10)
        self.top_p = rospy.get_param("~top_p", 0.0)
        self.max_tokens = rospy.get_param("~max_tokens", 64)

        # --- Safety / retry parameters ---
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        self.max_retries = rospy.get_param("~max_retries", 2)
        self.fallback_node = rospy.get_param("~fallback_node", None)  # 未指定時は最初のノード

        # --- LLMモデル読み込み ---
        rospy.loginfo(f"🧩 Loading LLM model from: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
            logits_all=True
        )

        # --- ノードマップ読み込み ---
        self.node_ids, self.node_labels, self.node_semantics, self.node_descriptions = self.load_map(self.map_yaml_path)
        rospy.loginfo(f"📄 Loaded {len(self.node_ids)} nodes from map.")

        # fallbackノード未指定なら最初のノードを設定
        if self.fallback_node is None and self.node_ids:
            self.fallback_node = self.node_ids[0]

        # --- ROSトピック設定 ---
        self.sub = rospy.Subscriber("llm_prompt", String, self.callback)
        self.pub = rospy.Publisher("llm_result", String, queue_size=10)

        rospy.loginfo("✅ LLM Reasoner Node initialized.")
        rospy.on_shutdown(self.cleanup)
        rospy.spin()

    # --------------------------
    def load_map(self, filename):
        """YAMLからノード情報を取得: id, label, semantic, description"""
        if not os.path.exists(filename):
            rospy.logwarn(f"⚠️ Map file not found: {filename}")
            return [], [], [], []

        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        ids, labels, semantics, descriptions = [], [], [], []
        for node in data.get("NODE", []):
            ids.append(node.get("id"))
            labels.append(node.get("label", "").strip())
            sem = node.get("semantic", [])
            semantics.append(", ".join(sem) if sem else "")
            descriptions.append(node.get("description", "").strip())
        return ids, labels, semantics, descriptions

    # --------------------------
    def make_prompt(self, user_query):
        """最小限のプロンプトを作成: ノード一覧 + タスク + NUMBER指示"""
        options = "\n".join(
            f"{self.node_ids[i]}. {self.node_labels[i]} ({self.node_semantics[i]})"
            for i in range(len(self.node_ids))
        )
        prompt = (
            f"Available Locations:\n{options}\n"
            f"User Task: {user_query}\n"
            f"Answer with the NUMBER of the correct choice:"
        )
        return prompt

    # --------------------------
    def call_llm_once(self, prompt):
        """LLMを呼び出して (テキスト出力, 信頼度) を返す"""
        output = self.llm(
            prompt=prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["###", "\n", "Instruction:"],
            echo=False,
            logprobs=1
        )
        text = output["choices"][0]["text"].strip()
        token_logprobs = output["choices"][0].get("logprobs", {}).get("token_logprobs", [])
        if token_logprobs:
            avg_logprob = np.mean(token_logprobs)
            confidence = float(np.exp(avg_logprob))  # 平均logprob -> 確率近似
        else:
            confidence = None
        return text, confidence

    # --------------------------
    def validate_and_extract_number(self, text):
        """数字を抽出し、node_idsに存在するか確認"""
        match = re.search(r'\b\d+\b', text)
        if not match:
            return None
        num = int(match.group(0))
        if num in self.node_ids:
            return num
        return None

    # --------------------------
    def callback(self, msg):
        """LLM推論 → 信頼度確認 → リトライ → fallback"""
        user_query = msg.data.strip()
        if not user_query:
            rospy.logwarn("⚠️ 空のプロンプトを受信しました")
            return

        rospy.loginfo(f"🧠 User Task: {user_query}")

        prompt = self.make_prompt(user_query)
        rospy.loginfo(f"📝 Prompt to LLM:\n{prompt}")

        chosen_node = None
        chosen_conf = None

        for attempt in range(self.max_retries + 1):
            text, conf = self.call_llm_once(prompt)
            rospy.loginfo(f"🔁 LLM output (try {attempt+1}): {text}")
            rospy.loginfo(f"   -> confidence: {conf:.3f}" if conf is not None else "   -> confidence: None")

            num = self.validate_and_extract_number(text)
            if num is not None and ((conf is None) or (conf >= self.confidence_threshold)):
                chosen_node = num
                chosen_conf = conf
                rospy.loginfo(f"✅ Accept: node {num} (conf={conf})")
                break
            else:
                rospy.logwarn("⚠️ Invalid or low-confidence response, retrying...")

        # fallback処理
        if chosen_node is None:
            last_num = None
            try:
                last_num = int(re.search(r'\b\d+\b', text).group(0))
            except Exception:
                last_num = None

            if last_num in self.node_ids:
                rospy.logwarn(f"⚠️ Low confidence but using last num: {last_num}")
                chosen_node = last_num
                chosen_conf = conf
            else:
                rospy.logerr(f"❌ Fail → fallback to {self.fallback_node}")
                chosen_node = self.fallback_node
                chosen_conf = None

        self.pub.publish(str(chosen_node))
        rospy.loginfo(f"📤 Publish: {chosen_node} (conf={chosen_conf})")

    # --------------------------
    def cleanup(self):
        rospy.loginfo("🧹 Shutting down LLM Reasoner Node...")
        try:
            del self.llm
            rospy.loginfo("✅ Model resources released.")
        except Exception:
            rospy.logwarn("⚠️ Failed to clean up model resources.")


if __name__ == "__main__":
    try:
        LLMReasonerNode()
    except rospy.ROSInterruptException:
        pass

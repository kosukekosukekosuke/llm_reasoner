#!/usr/bin/env python3

import os
import re
import rospy
import yaml
from std_msgs.msg import String
from llama_cpp import Llama
import numpy as np

class LLMReasonerNode:
    def __init__(self):
        rospy.init_node("llm_reasoner", anonymous=False)

        # --- ROS パラメータ ---
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

        # --- LLM準備 ---
        rospy.loginfo(f"🧩 Loading LLM model from: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
            logits_all=True  
        )

        # --- YAML読み込み ---
        self.node_ids, self.node_labels = self.extract_node_ids_and_labels(self.map_yaml_path)
        rospy.loginfo(f"📄 Loaded {len(self.node_ids)} labeled nodes from map file.")

        # --- Subscriber / Publisher ---
        self.sub = rospy.Subscriber("llm_prompt", String, self.callback)
        self.pub = rospy.Publisher("llm_result", String, queue_size=10)

        rospy.loginfo("✅ LLM Reasoner Node initialized and ready.")
        rospy.on_shutdown(self.cleanup)
        rospy.spin()

    # ============================================================
    # YAMLからノードIDとラベルを抽出
    # ============================================================
    def extract_node_ids_and_labels(self, filename):
        if not os.path.exists(filename):
            rospy.logwarn(f"⚠️ Map file not found: {filename}")
            return [], []

        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        ids, labels = [], []
        for node in data.get("NODE", []):
            label = node.get("label", "").strip()
            if label:
                ids.append(node["id"])
                labels.append(label)
        return ids, labels

    # ============================================================
    # プロンプト生成
    # ============================================================
    def make_prompt(self, user_query):
        user_query = user_query.strip()
        if not self.node_labels:
            return f"User: {user_query}\nAssistant: Sorry, no labeled nodes found in the map."

        # ノードIDとラベルを1行ずつ結合
        options = "\n".join(f"{self.node_ids[i]}. {self.node_labels[i]}" for i in range(len(self.node_ids)))

        prompt = (
            f"Available Locations:\n{options}\n"
            f"User Task: {user_query}\n"
            f"Answer with the NUMBER of the correct choice:"
        )
        return prompt

    # ============================================================
    # コールバック
    # ============================================================
    def callback(self, msg):
        user_query = msg.data.strip()
        if not user_query:
            rospy.logwarn("⚠️ Received empty prompt.")
            return

        try:
            rospy.loginfo(f"🧠 Received prompt:\n{user_query}")

            prompt = self.make_prompt(user_query)
            rospy.loginfo(f"📝 Generated LLM Prompt:\n{prompt}")

            # --- 推論実行 ---
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

            result_text = output["choices"][0]["text"].strip()

            # --- 信頼度計算 ---
            token_logprobs = output["choices"][0].get("logprobs", {}).get("token_logprobs", [])
            if token_logprobs:
                avg_logprob = np.mean(token_logprobs)
                confidence = np.exp(avg_logprob)  # 確率に変換
                rospy.loginfo(f"🔹 Average confidence: {confidence:.3f}")
            else:
                confidence = None
                rospy.logwarn("⚠️ Logprobs not available.")

            # 正規表現で番号のみ抽出
            match = re.search(r'\b\d+\b', result_text)
            if match:
                result_number = match.group(0)
                rospy.loginfo(f"🗣️ LLM output number: {result_number}")
                self.pub.publish(result_number)
            else:
                rospy.logwarn("⚠️ No number found in LLM output.")
                self.pub.publish(result_text)  # 念のため全文も送る

        except Exception as e:
            rospy.logerr(f"❌ Error during LLM inference: {e}")

    # ============================================================
    # 終了処理
    # ============================================================
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

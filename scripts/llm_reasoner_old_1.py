#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from llama_cpp import Llama

class LLMReasonerNode:
    def __init__(self):
        rospy.init_node("llm_reasoner", anonymous=False)

        # --- ROS パラメータ ---
        self.model_path = rospy.get_param(
            "~model_path",
            os.path.join(os.path.dirname(__file__), "llama.cpp/models/meta-llama-3-8b-instruct.Q4_K_M.gguf")
        )
        self.n_ctx = rospy.get_param("~ctx_size", 2048)
        self.n_threads = rospy.get_param("~threads", 4)
        self.temperature = rospy.get_param("~temperature", 0.1)
        self.top_k = rospy.get_param("~top_k", 10)
        self.top_p = rospy.get_param("~top_p", 0.0)
        self.max_tokens = rospy.get_param("~max_tokens", 64)

        # --- LLM 準備 ---
        rospy.loginfo(f"🧩 Loading LLM model from: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )

        # --- Subscriber / Publisher ---
        self.sub = rospy.Subscriber("llm_prompt", String, self.callback)
        self.pub = rospy.Publisher("llm_result", String, queue_size=10)

        rospy.loginfo("✅ LLM Reasoner Node initialized and ready.")
        rospy.on_shutdown(self.cleanup)
        rospy.spin()

    def callback(self, msg):
        # --- プロンプトを受け取り、LLM推論を行う---
        user_prompt = msg.data.strip()
        if not user_prompt:
            rospy.logwarn("⚠️ Received empty prompt.")
            return

        try:
            rospy.loginfo(f"🧠 Received prompt:\n{user_prompt}")
            output = self.llm(
                prompt=user_prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=["###", "\n", "Instruction:"],
                echo=False
            )

            result_text = output["choices"][0]["text"].strip()
            rospy.loginfo(f"🗣️ LLM output: {result_text}")

            self.pub.publish(result_text)

        except Exception as e:
            rospy.logerr(f"❌ Error during LLM inference: {e}")

    def cleanup(self):
        # --- ノード終了時のクリーンアップ処理 ---
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

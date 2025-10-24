import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import argparse
import os

class ElderlyPPOChatbot:
    def __init__(self, sft_merged_model_path, ppo_adapter_path):
        self.device = self._get_device()
        self.sft_merged_model_path = sft_merged_model_path
        self.ppo_adapter_path = ppo_adapter_path
        self.system_prompt = "你是一个专门为老年人提供生活帮助和健康咨询的智能助手。请用温和、耐心、详细的语言回答问题，考虑到老年人可能存在的视力、听力和认知能力下降的问题。"
        self.tokenizer = None
        self.model = None

    def _get_device(self):
        """自动检测可用设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载PPO微调后的模型"""
        print("🚀 正在加载老年人关怀PPO模型...")
        print(f"--> 基础模型 (SFT合并后): {self.sft_merged_model_path}")
        print(f"--> PPO适配器: {self.ppo_adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sft_merged_model_path, use_fast=False, trust_remote_code=True
        )
        # 设置pad_token，避免与eos_token相同导致警告
        if self.tokenizer.pad_token is None:
            # 优先使用unk_token，如果没有则使用eos_token
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # 如果pad_token和eos_token相同，设置不同的pad_token_id
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.sft_merged_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(
            self.model, model_id=self.ppo_adapter_path
        )
        self.model.eval()
        print(f"✅ PPO模型加载完成，使用设备: {self.model.device}")

    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """使用聊天模板生成回复"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,  # 显式传递attention_mask
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

def test_model(chatbot: ElderlyPPOChatbot, output_file: str):
    """批量测试PPO模型效果"""
    test_questions = [
        "我最近总是忘记事情，这是老年痴呆吗？",
        "如何保持身体健康？",
        "老年人应该怎样合理饮食？",
        "我晚上总是睡不好，有什么办法吗？",
        "如何预防跌倒？",
        "老年人需要补充哪些维生素？",
        "如何保持心情愉快？",
        "老年人适合做什么运动？",
        "我的血压有点高，应该注意什么？",
        "如何使用智能手机拍照？"
    ]

    print("\n" + "="*80 + "\n🎯 老年人关怀PPO模型批量测试开始\n" + "="*80)
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 测试 {i}/{len(test_questions)}: {question}\n" + "-" * 60)
        response = chatbot.generate_response(question)
        print(f"🤖 回复: {response}")
        results.append({"question": question, "response": response})
        print("-" * 60)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ PPO测试完成！结果已保存到 {output_file}")

def interactive_chat(chatbot: ElderlyPPOChatbot):
    """PPO模型交互式对话"""
    print("\n" + "="*80 + "\n🎯 老年人关怀助手交互式对话 (PPO增强版)\n" + "="*80)
    print("💡 输入 'exit' 或 'quit' 退出。")

    while True:
        try:
            # 使用更健壮的输入方式，处理编码问题
            try:
                user_input = input("\n👴 老年人: ").strip()
            except UnicodeDecodeError:
                # 如果直接input出现问题，尝试其他方式
                import sys
                line = sys.stdin.readline()
                user_input = line.strip() if line else ""

            if user_input.lower() in ['exit', 'quit']: break
            if not user_input: continue
            print("🤖 助手: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)
        except UnicodeDecodeError as e:
            print(f"\n❌ 输入编码错误: {e}")
            print("💡 请确保输入的是有效的UTF-8编码文本")
            continue
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"\n❌ 发生未知错误: {e}")
            print("💡 请重新输入")
            continue
    print("\n👋 再见！祝您身体健康！")

def main():
    parser = argparse.ArgumentParser(description="老年人关怀PPO模型推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="SFT合并后的基础模型的路径 (例如 ./output/elderly/sft_merged_model)")
    parser.add_argument("--adapter_path", type=str, required=True, help="PPO LoRA适配器的路径 (例如 ./output/elderly/ppo_adapter)")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "test"], help="运行模式: 'interactive' (交互式) 或 'test' (批量测试)")
    parser.add_argument("--test_output_file", type=str, default="elderly_ppo_test_results.json", help="批量测试结果的输出文件路径")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌错误: 基础模型路径不存在: {args.model_path}")
        return
    if not os.path.exists(args.adapter_path):
        print(f"❌错误: 适配器路径不存在: {args.adapter_path}")
        return

    chatbot = ElderlyPPOChatbot(args.model_path, args.adapter_path)
    chatbot.load_model()

    if args.mode == 'interactive':
        interactive_chat(chatbot)
    elif args.mode == 'test':
        test_model(chatbot, args.test_output_file)

if __name__ == "__main__":
    main()

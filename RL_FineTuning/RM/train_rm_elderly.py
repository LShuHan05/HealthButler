import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import RewardConfig, RewardTrainer
import swanlab
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "SFT合并后的模型路径"})
    dataset_path: str = field(default="data/mixed_partially_shuffled.json", metadata={"help": "数据集路径"})
    rm_adapter_output_dir: str = field(default="./output/elderly/rm_adapter_elderly", metadata={"help": "RM LoRA适配器保存目录"})
    system_prompt: str = field(default="你是一个专门为老年人提供生活帮助和健康咨询的智能助手。请用温和、耐心、详细的语言回答问题。", metadata={"help": "系统提示语"})
    max_length: int = field(default=1024, metadata={"help": "输入最大长度"})
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})
    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab"})

def setup_swanlab(args: ScriptArguments):
    if not args.use_swanlab:
        return
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-rm-elderly"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    swanlab.init(
        project="qwen3-sft-rm-elderly",
        run_name="rm-training-elderly",
        config=vars(args)
    )

def load_rm_dataset(dataset_path):
    """加载老年人数据集并转换为RM格式"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌错误: 数据集文件未找到 at {dataset_path}")
        exit()
    
    processed_data = []
    for item in data:
        # 从新数据集格式转换为RM格式
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        system_text = item.get('system', '')
        
        # 构造human input
        if input_text:
            human_input = f"{instruction}\n\n{input_text}" if instruction else input_text
        else:
            human_input = instruction if instruction else system_text
            
        # 对于RM训练，我们需要构造chosen和rejected样本
        # 这里我们使用原始输出作为chosen，通过添加噪声或修改来创建rejected样本
        chosen_response = output_text
        
        # 创建一个简单的rejected版本（通过截断输出）
        if len(output_text) > 50:
            rejected_response = output_text[:len(output_text)//2]  # 取前半部分作为rejected
        else:
            rejected_response = output_text[:-10] if len(output_text) > 10 else output_text  # 去掉最后几个字符
            
        # 确保rejected和chosen不相同
        if rejected_response == chosen_response:
            rejected_response = chosen_response + " "  # 添加空格使其不同

        if human_input and chosen_response and rejected_response:
            processed_data.append({
                "input": human_input,
                "chosen": chosen_response,
                "rejected": rejected_response
            })
    return processed_data

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if not os.path.exists(args.model_path):
        print(f"❌错误: 基础模型 (SFT合并后) 在 '{args.model_path}' 未找到。请先运行SFT微调和合并脚本。")
        exit()

    print("🚀 1. 配置和初始化 SwanLab...")
    setup_swanlab(args)
    
    print("🚀 2. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        new_examples = {"input_ids_chosen": [], "attention_mask_chosen": [], "input_ids_rejected": [], "attention_mask_rejected": []}
        for human_input, chosen, rejected in zip(examples["input"], examples["chosen"], examples["rejected"]):
            text_chosen = tokenizer.apply_chat_template(
                [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": human_input}, {"role": "assistant", "content": chosen}],
                tokenize=False, add_generation_prompt=False
            )
            text_rejected = tokenizer.apply_chat_template(
                [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": human_input}, {"role": "assistant", "content": rejected}],
                tokenize=False, add_generation_prompt=False
            )
            tokenized_chosen = tokenizer(text_chosen, truncation=True, max_length=args.max_length)
            tokenized_rejected = tokenizer(text_rejected, truncation=True, max_length=args.max_length)
            
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    print("🚀 3. 加载和预处理数据集...")
    raw_data = load_rm_dataset(args.dataset_path)
    full_dataset = Dataset.from_list(raw_data)
    train_test_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train'].map(preprocess_function, batched=True)
    eval_dataset = train_test_split['test'].map(preprocess_function, batched=True)
    print(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")
    
    print("🚀 4. 加载模型并配置LoRA for RM...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False
    
    rm_lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, rm_lora_config)
    print("已为RM任务添加新的可训练LoRA适配器。")
    model.print_trainable_parameters()

    print("🚀 5. 配置训练参数...")
    training_args = RewardConfig(
        output_dir="./output/elderly/rm_model_temp_elderly",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="swanlab" if args.use_swanlab else "none",
        run_name="rm-training-elderly",
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_length=args.max_length,
    )

    print("🚀 6. 创建并启动RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    print(f"💾 7. 保存RM LoRA适配器到: {args.rm_adapter_output_dir}")
    os.makedirs(args.rm_adapter_output_dir, exist_ok=True)
    trainer.save_model(args.rm_adapter_output_dir)
    
    print("✅ 奖励模型训练完成！")
    if args.use_swanlab:
        swanlab.finish()

if __name__ == "__main__":
    main()
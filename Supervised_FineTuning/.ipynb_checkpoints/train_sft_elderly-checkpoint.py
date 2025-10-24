import json
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model
import swanlab
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    """
    SFT脚本的配置参数
    """
    model_path: str = field(metadata={"help": "模型仓库的路径"})
    dataset_path: str = field(default="data/mixed_partially_shuffled.json", metadata={"help": "数据集的路径"})
    sft_adapter_output_dir: str = field(default="./output/sft_adapter_elderly", metadata={"help": "SFT LoRA适配器的保存目录"})
    system_prompt: str = field(default="你是一个专门为老年人提供生活帮助和健康咨询的智能助手。请用温和、耐心、详细的语言回答问题，考虑到老年人可能存在的视力、听力和认知能力下降的问题。", metadata={"help": "系统提示语"})
    max_length: int = field(default=1024, metadata={"help": "输入的最大长度"})
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})
    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab记录实验"})

def setup_swanlab(args: ScriptArguments):
    """配置并初始化SwanLab"""
    if not args.use_swanlab:
        return
    
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-elderly"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    swanlab.init(
        project="qwen3-sft-elderly",
        run_name="sft-training-elderly",
        config={
            "model": args.model_path,
            "method": "SFT_with_Trainer",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "dataset": args.dataset_path,
            "system_prompt": args.system_prompt
        }
    )

def load_and_format_dataset(dataset_path, system_prompt):
    """
    加载老年人关怀JSON文件，并将其转换为SFT的 instruction, input, output 格式.
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌错误: 数据集文件未找到 at {dataset_path}")
        exit()
    
    formatted_data = []
    for item in data:
        # 新数据集格式直接包含 instruction, input, output 字段
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        system_text = item.get('system', system_prompt)  # 如果数据中有system字段则使用，否则使用默认的
        
        # 如果没有明确的instruction，使用默认的系统提示语
        if not instruction:
            instruction = system_text
            
        if input_text and output_text:
            formatted_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
        elif instruction and output_text:  # 有些数据可能没有input字段
            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": output_text
            })
    return formatted_data

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print("🚀 1. 配置和初始化 SwanLab...")
    setup_swanlab(args)

    print("🚀 2. 加载和格式化数据集...")
    sft_data = load_and_format_dataset(args.dataset_path, args.system_prompt)
    full_dataset = Dataset.from_list(sft_data)
    
    # 使用 train_test_split 划分数据集
    train_test_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"SFT训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    print("🚀 3. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        use_fast=False, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def process_func(example):
        # 根据新数据集的特点调整格式
        if example['input']:
            full_input = f"{example['instruction']}\n\n{example['input']}"
        else:
            full_input = example['instruction']
            
        # 构造聊天模板格式的输入
        chat = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
            {"role": "assistant", "content": example['output']}
        ]
        
        # 应用聊天模板
        prompt = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True)
        full_text = prompt + example['output'] + tokenizer.eos_token
        
        # Tokenize完整文本
        model_inputs = tokenizer(full_text, max_length=args.max_length, truncation=True)
        
        # 创建labels（只对assistant的回复计算损失）
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        labels = [-100] * len(prompt_ids) + model_inputs["input_ids"][len(prompt_ids):]
        
        # 确保labels不会超过最大长度
        if len(labels) > args.max_length:
            labels = labels[:args.max_length]
            
        model_inputs["labels"] = labels
        return model_inputs

    print("🚀 4. 对数据集进行Tokenization...")
    tokenized_train_ds = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
    tokenized_eval_ds = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)
    
    print("🚀 5. 加载模型并配置LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("🚀 6. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./output/elderly/sft_model_temp_elderly",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        report_to="swanlab" if args.use_swanlab else "none",
        run_name="sft-training-elderly",
    )

    print("🚀 7. 创建并启动Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    print(f"💾 8. 保存SFT LoRA适配器到: {args.sft_adapter_output_dir}")
    os.makedirs(args.sft_adapter_output_dir, exist_ok=True)
    trainer.save_model(args.sft_adapter_output_dir)
    
    print("\n✅ SFT训练完成！")
    if args.use_swanlab:
        swanlab.finish()

if __name__ == "__main__":
    main()
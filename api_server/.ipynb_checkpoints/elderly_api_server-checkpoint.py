"""
老年人关怀模型 API 服务器
兼容 OpenAI API 格式，可直接接入 Open WebUI
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
import argparse
import os

# ==================== 数据模型定义 ====================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "elderly-care"

# ==================== 模型管理类 ====================

class ElderlyModelAPI:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, model_type: str = "sft"):
        """
        初始化老年人关怀模型API
        
        Args:
            model_path: 基础模型路径或SFT合并后的模型路径
            adapter_path: LoRA适配器路径（可选）
            model_type: 模型类型，'sft' 或 'ppo'
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model_type = model_type
        self.device = self._get_device()
        self.system_prompt = "你是一个专门为老年人提供生活帮助和健康咨询的智能助手。请用温和、耐心、详细的语言回答问题，考虑到老年人可能存在的视力、听力和认知能力下降的问题。"
        
        print(f"🚀 正在加载老年人关怀模型 ({model_type.upper()})...")
        self._load_model()
        print(f"✅ 模型加载完成！使用设备: {self.device}")

    def _get_device(self):
        """自动检测可用设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """加载模型和tokenizer"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=False, trust_remote_code=True
        )
        
        # 设置pad_token，避免与eos_token相同导致警告
        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # 如果提供了adapter路径，加载LoRA适配器
        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"📦 加载LoRA适配器: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, model_id=self.adapter_path)
        
        self.model.eval()

    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        max_tokens: int = 512
    ) -> str:
        """
        生成回复
        
        Args:
            messages: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]
            temperature: 温度参数
            top_p: nucleus sampling参数
            max_tokens: 最大生成token数
            
        Returns:
            生成的回复文本
        """
        # 构建完整的消息列表（添加system prompt）
        full_messages = []
        
        # 检查是否已有system消息
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            full_messages.append({"role": "system", "content": self.system_prompt})
        
        # 添加用户消息
        full_messages.extend(messages)
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            full_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize输入
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 解码输出
        response_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response.strip()

# ==================== FastAPI应用 ====================

app = FastAPI(
    title="老年人关怀模型API",
    description="基于Qwen3微调的老年人关怀助手，兼容OpenAI API格式",
    version="1.0.0"
)

# 添加CORS中间件，允许Open WebUI访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
model_api: Optional[ElderlyModelAPI] = None

# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "老年人关怀模型API服务",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_api is not None
    }

@app.get("/v1/models")
async def list_models():
    """列出可用模型（OpenAI API兼容）"""
    model_id = f"elderly-care-{model_api.model_type}" if model_api else "elderly-care"
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "elderly-care",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天补全端点（OpenAI API兼容）
    这是Open WebUI调用的主要接口
    """
    if model_api is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 转换消息格式
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # 生成回复
        response_text = model_api.generate_response(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        # 构建响应（OpenAI格式）
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # 简化版本，不计算token
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回复时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model_api is not None,
        "device": model_api.device if model_api else "unknown"
    }

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="老年人关怀模型API服务器")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="基础模型路径或SFT合并后的模型路径"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default=None, 
        help="LoRA适配器路径（可选）"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="sft", 
        choices=["sft", "ppo"],
        help="模型类型: sft 或 ppo"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="服务器监听地址"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="服务器端口"
    )
    
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型路径不存在: {args.model_path}")
        return
    
    if args.adapter_path and not os.path.exists(args.adapter_path):
        print(f"❌ 错误: 适配器路径不存在: {args.adapter_path}")
        return
    
    # 初始化全局模型
    global model_api
    model_api = ElderlyModelAPI(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        model_type=args.model_type
    )
    
    print("\n" + "="*80)
    print("🎉 老年人关怀模型API服务器启动成功！")
    print("="*80)
    print(f"📡 API地址: http://{args.host}:{args.port}")
    print(f"📚 API文档: http://{args.host}:{args.port}/docs")
    print(f"🔗 OpenAI兼容端点: http://{args.host}:{args.port}/v1/chat/completions")
    print("\n💡 在Open WebUI中配置:")
    print(f"   - API URL: http://localhost:{args.port}/v1")
    print(f"   - API Key: 随意填写（本服务不验证）")
    print(f"   - 模型名称: elderly-care-{args.model_type}")
    print("="*80 + "\n")
    
    # 启动服务器
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

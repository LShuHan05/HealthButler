"""
API服务器测试脚本
用于验证老年人关怀模型API是否正常工作
"""

import requests
import json
import argparse
from typing import List, Dict

def test_health_check(base_url: str):
    """测试健康检查端点"""
    print("\n" + "="*80)
    print("🔍 测试1: 健康检查")
    print("="*80)
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过")
            print(f"   状态: {data.get('status')}")
            print(f"   模型已加载: {data.get('model_loaded')}")
            print(f"   设备: {data.get('device')}")
            return True
        else:
            print(f"❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查失败: {str(e)}")
        return False

def test_list_models(base_url: str):
    """测试模型列表端点"""
    print("\n" + "="*80)
    print("🔍 测试2: 获取模型列表")
    print("="*80)
    
    try:
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            print(f"✅ 获取模型列表成功")
            print(f"   可用模型数量: {len(models)}")
            for model in models:
                print(f"   - {model.get('id')}")
            return True
        else:
            print(f"❌ 获取模型列表失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取模型列表失败: {str(e)}")
        return False

def test_chat_completion(base_url: str, model_name: str):
    """测试聊天补全端点"""
    print("\n" + "="*80)
    print("🔍 测试3: 聊天补全")
    print("="*80)
    
    test_messages = [
        "你好，请介绍一下你自己",
        "如何保持身体健康？",
        "我最近总是忘记事情，这是老年痴呆吗？"
    ]
    
    for i, user_message in enumerate(test_messages, 1):
        print(f"\n📝 测试消息 {i}/{len(test_messages)}: {user_message}")
        print("-" * 60)
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 256
        }
        
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data['choices'][0]['message']['content']
                print(f"✅ 回复成功:")
                print(f"   {assistant_message[:200]}{'...' if len(assistant_message) > 200 else ''}")
            else:
                print(f"❌ 回复失败: HTTP {response.status_code}")
                print(f"   错误信息: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 回复失败: {str(e)}")
            return False
    
    print("\n✅ 所有聊天测试通过！")
    return True

def test_multi_turn_conversation(base_url: str, model_name: str):
    """测试多轮对话"""
    print("\n" + "="*80)
    print("🔍 测试4: 多轮对话")
    print("="*80)
    
    conversation = [
        "我今年70岁了，最近感觉膝盖疼",
        "那我应该怎么办？",
        "除了运动，饮食上需要注意什么？"
    ]
    
    messages = []
    
    for i, user_message in enumerate(conversation, 1):
        print(f"\n📝 第{i}轮对话")
        print(f"👴 用户: {user_message}")
        
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 256
        }
        
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data['choices'][0]['message']['content']
                print(f"🤖 助手: {assistant_message[:150]}{'...' if len(assistant_message) > 150 else ''}")
                messages.append({"role": "assistant", "content": assistant_message})
            else:
                print(f"❌ 回复失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 回复失败: {str(e)}")
            return False
    
    print("\n✅ 多轮对话测试通过！")
    return True

def test_parameter_variations(base_url: str, model_name: str):
    """测试不同参数配置"""
    print("\n" + "="*80)
    print("🔍 测试5: 参数变化测试")
    print("="*80)
    
    test_configs = [
        {"name": "低温度(保守)", "temperature": 0.3, "top_p": 0.9},
        {"name": "高温度(创造性)", "temperature": 0.9, "top_p": 0.9},
        {"name": "低Top-P", "temperature": 0.7, "top_p": 0.5},
    ]
    
    user_message = "请给我一些保持心情愉快的建议"
    
    for config in test_configs:
        print(f"\n📝 测试配置: {config['name']}")
        print(f"   Temperature: {config['temperature']}, Top-P: {config['top_p']}")
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": config['temperature'],
            "top_p": config['top_p'],
            "max_tokens": 200
        }
        
        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content']
                print(f"✅ 回复: {reply[:100]}...")
            else:
                print(f"❌ 失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 失败: {str(e)}")
            return False
    
    print("\n✅ 参数变化测试通过！")
    return True

def run_all_tests(base_url: str, model_name: str):
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 开始API服务器测试")
    print("="*80)
    print(f"📡 API地址: {base_url}")
    print(f"🤖 模型名称: {model_name}")
    
    results = []
    
    # 测试1: 健康检查
    results.append(("健康检查", test_health_check(base_url)))
    
    # 测试2: 模型列表
    results.append(("模型列表", test_list_models(base_url)))
    
    # 测试3: 聊天补全
    results.append(("聊天补全", test_chat_completion(base_url, model_name)))
    
    # 测试4: 多轮对话
    results.append(("多轮对话", test_multi_turn_conversation(base_url, model_name)))
    
    # 测试5: 参数变化
    results.append(("参数变化", test_parameter_variations(base_url, model_name)))
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*80)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！API服务器运行正常！")
        print("\n💡 下一步:")
        print("   1. 在Open WebUI中配置API: http://localhost:8000/v1")
        print("   2. 开始使用老年人关怀助手！")
    else:
        print("⚠️ 部分测试失败，请检查API服务器日志")
    
    print("="*80 + "\n")
    
    return passed == total

def main():
    parser = argparse.ArgumentParser(description="测试老年人关怀模型API服务器")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:6000",
        help="API服务器基础URL"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="elderly-care-sft",
        help="模型名称（elderly-care-sft 或 elderly-care-ppo）"
    )
    
    args = parser.parse_args()
    
    # 运行测试
    success = run_all_tests(args.base_url, args.model_name)
    
    # 返回退出码
    exit(0 if success else 1)

if __name__ == "__main__":
    main()

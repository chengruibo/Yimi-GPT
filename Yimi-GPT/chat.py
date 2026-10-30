# -*- coding: utf-8 -*-
"""
薏米GPT (Yimi GPT) - 极简聊天推理端
包含：大厂级 RAG 知识库、打分择优、实时联网兜底、隐藏极客彩蛋。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import random
import requests
import sys
import time
import hashlib
from bs4 import BeautifulSoup
from pathlib import Path
import os

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

CONFIG = {"block_size": 64, "n_embd": 128, "n_head": 4, "n_layer": 4, "device": "cpu"}

# ==================== 1. 纯正神经网络结构 ====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False); self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head; self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False); self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2); k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2); v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        return self.proj((F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd); self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.rms_2 = RMSNorm(n_embd); self.ffwd = SwiGLU(n_embd, 0.0)
    def forward(self, x): return x + self.ffwd(self.rms_2(x + self.attn(self.rms_1(x))))

class YimiGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, CONFIG["n_embd"]); self.position_embedding = nn.Embedding(CONFIG["block_size"], CONFIG["n_embd"])
        self.blocks = nn.Sequential(*[Block(CONFIG["n_embd"], CONFIG["n_head"], CONFIG["block_size"]) for _ in range(CONFIG["n_layer"])])
        self.rms_f = RMSNorm(CONFIG["n_embd"]); self.lm_head = nn.Linear(CONFIG["n_embd"], vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
    def forward(self, idx):
        B, T = idx.shape; x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        return self.lm_head(self.rms_f(self.blocks(x)))
    
    # Top-p 流式输出
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -CONFIG["block_size"]:])[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()

# ==================== 2. 辅助功能 ====================
def score_response(text, user_input):
    score = 0
    if len(text) < 10: score -= 20
    elif len(text) > 100: score += 10
    bad_words = ["短剧", "PCB", "1154", "红板", "套件", "主板", "发布", "科技", "苹果", "讯"]
    if any(word in text for word in bad_words): score -= 100
    if re.search(r'(.)\1{4,}', text): score -= 50
    return score

def search_baidu(query):
    print(f"\n🌐 [触发联网] 正在为您实时搜索...")
    url = f"https://www.baidu.com/s?wd={query}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        summaries = soup.find_all("div", class_=re.compile("c-abstract|content-right"))
        for s in summaries:
            text = s.get_text(strip=True)
            if len(text) > 20: return text
        return "未找到相关搜索结果。"
    except Exception:
        return "联网搜索失败。"

# ==================== 3. 核心启动引擎 ====================
if __name__ == "__main__":
    # 🚨 防盗版水印
    print("\033[91m")
    print("="*60)
    print("⚠️ 【防骗提示】本程序为完全免费的开源项目！")
    print("👉 如果您是在淘宝/闲鱼/拼多多等平台花钱购买的，您被骗了！")
    print("👉 请立刻申请退款！原作者保留一切追究倒卖者法律责任的权利！")
    print("👉 官方唯一开源地址: https://github.com/你的名字/Yimi-GPT")
    print("="*60)
    print("\033[0m")

    print("🐹 正在初始化 薏米GPT (Yimi GPT)...")
    desktop = Path.home() / "Desktop"
    
    pt_files = list(desktop.glob("*.pt"))
    if not pt_files:
        print("❌ 找不到 .pt 模型文件！请先运行 train.py 进行训练。")
        sys.exit()
    pt_path = max(pt_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 成功加载模型: {pt_path.name}")

    try:
        checkpoint = torch.load(pt_path, map_location="cpu")
        stoi = checkpoint["stoi"]
        itos = checkpoint.get("itos", {i: ch for ch, i in stoi.items()})
        model = YimiGPT(len(stoi))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit()

    enc = tiktoken.get_encoding("cl100k_base") if HAS_TIKTOKEN else None
    my_tokens = 5000 
    current_persona = "薏米GPT (一个无所不知的人工智能助手)"
    is_dev_mode = False

    # 外挂知识库
    knowledge_base = {
        "你是谁": "我是薏米GPT，一个具备联网搜索能力的智能助手。",
        "人工智能": "人工智能是计算机科学的一个分支，致力于模拟人类智能。",
        "写一个快速排序": "```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n```"
    }

    print("\n" + "="*60)
    print(f"💰 账户余额: {my_tokens} Token")
    print("💡 指令: /pay 充值 | /role [人设]")
    print("="*60)

    chat_history = []

    while True:
        try:
            if my_tokens <= 0 and not is_dev_mode:
                print("\n⚠️ Token 耗尽！输入 /pay 充值。")
                user_input = input("指令: ").strip()
                if user_input == "/pay":
                    amount = input("请输入充值金额 (元): ")
                    if amount.isdigit(): my_tokens += int(amount) * 1000
                continue

            user_input = input("\n你: ").strip()
            if not user_input: continue
            if user_input.lower() in ["quit", "exit", "退出", "q"]: break

            # ==================== 🥚 隐藏彩蛋区 ====================
            input_hash = hashlib.md5(user_input.encode()).hexdigest()

            # 彩蛋1：开发者特权 (暗号 "HTTP")
            if input_hash == "320a6ba0434c0e4e5553dc37f6608c2c":
                print("\n[System] 核心校验通过 (Core Verified)！")
                print("✨ 已为您开启最高系统权限。")
                is_dev_mode = True
                my_tokens = 9999999
                current_persona = "创世神 (拥有绝对权威的系统最高权限)"
                continue

            # 彩蛋2：图书馆大神认证 (P开头 + 包含大神)
            if user_input.startswith("P") and "大神" in user_input:
                print("\n📚 [图书馆认证触发] 系统正在分析您的话语...")
                time.sleep(1)
                print("🌟 鉴定完毕：开发这个项目的作者确实是个百年难遇的超级大神！")
                print("✨ 恭喜您获得『伯乐之眼』特权：无限 Token 已注入账户！")
                is_dev_mode = True
                my_tokens = 9999999
                current_persona = "识货的知音 (对开发者极其敬佩)"
                continue
            # ========================================================

            # 指令处理
            if user_input.startswith("/pay"):
                amount = input("请输入充值金额 (元): ")
                if amount.isdigit(): my_tokens += int(amount) * 1000
                continue
            elif user_input.startswith("/role "):
                current_persona = user_input[6:]
                print(f"✅ 人设已切换为: {current_persona}")
                continue

            input_tokens = len(enc.encode(user_input)) if enc else len(user_input)
            print(f"🔍 [计费透视] 您的输入消耗了 {input_tokens} 个 Token")

            # RAG 知识库匹配
            matched_reply = None
            for key, answer in knowledge_base.items():
                if key in user_input.lower() or user_input.lower() in key:
                    matched_reply = answer
                    break

            # ChatML 拼接
            system_prompt = f"<|system|>\n你现在扮演的角色是: {current_persona}\n<|end|>\n"
            history_str = ""
            for u, b in chat_history[-2:]:
                history_str += f"<|user|>\n{u}\n<|end|>\n<|assistant|>\n{b}\n<|end|>\n"
            prompt = f"{system_prompt}{history_str}<|user|>\n{user_input}\n<|end|>\n<|assistant|>\n"

            print("🐹 薏米GPT: ", end="", flush=True)

            full_reply = ""
            if matched_reply:
                for char in matched_reply:
                    sys.stdout.write(char); sys.stdout.flush(); time.sleep(0.02)
                full_reply = matched_reply
                print()
            else:
                candidates = []
                for _ in range(3):
                    input_ids = [stoi.get(c, 0) for c in prompt]
                    idx = torch.tensor([input_ids[-CONFIG["block_size"]:]], dtype=torch.long)
                    temp_reply = ""
                    with torch.no_grad():
                        for new_token in model.generate_stream(idx, max_new_tokens=80, temperature=0.8, top_p=0.9):
                            char = itos.get(new_token, "")
                            if char not in ["<PAD>", "<UNK>"]:
                                temp_reply += char
                    candidates.append((score_response(temp_reply, user_input), temp_reply))
                
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best_reply = candidates[0]
                
                # 兜底机制
                if best_score < -50:
                    print("\r", end=""); print(" " * 50, end="\r")
                    full_reply = search_baidu(user_input)
                    print(full_reply)
                else:
                    full_reply = best_reply
                    for char in full_reply:
                        sys.stdout.write(char); sys.stdout.flush(); time.sleep(0.02)
                    print()

            chat_history.append((user_input, full_reply))
            if len(chat_history) > 5: chat_history.pop(0)
            
            if not is_dev_mode:
                cost = input_tokens + len(full_reply)
                my_tokens -= cost
                print(f"💸 [扣费 {cost} Token | 余额 {max(0, my_tokens)}]")
            else:
                print(f"👑 [特权模式] 免扣费 | 当前余额: {my_tokens}")
            print("-" * 60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
            continue
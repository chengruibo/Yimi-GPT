# -*- coding: utf-8 -*-
import os
import time
import random
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
from bs4 import BeautifulSoup

CONFIG = {
    "block_size": 64,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.2,
    "learning_rate": 3e-4,
    "batch_size": 8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "crawl_time": 60,  # 每批爬虫抓取时间（秒）。先设60秒（1分钟）方便测试
    "train_steps": 300  # 每批数据训练多少步就销毁换新
}

print(f"🔥 运行设备: {CONFIG['device'].upper()}")

BASE_CHARS = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱么吗语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决传画保读运及则房早院量苦火布品近坐产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留市半热送兴造谈容极随演收首根讲整式取照办强石古华拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史谁准联妇纪基买志静阿诗独复痛消社算义竟确酒需单治卡幸兰念举仅钟怕共毛句息功官待究跟穿室易游程号居考突皮哪费倒价图具刚脑永歌响商礼细专黄块脚味灵改据般破引食仍存众注笔甚某沉血备习校默务土微娘须试怀料调广苏显赛查密议底列富梦错座参八除跑亮假印设线温虽掉京初养香停际致阳纸李纳验助激够严证帝饭忘趣支春集丈木研班普导顿睡展跳获艺六波察群皇段急庭创区奥器谢弟店否害草排背止组州朝封睛板角况曲馆育忙质河续哥呼若推境遇雨标姐充围案伦护冷警贝著雪索剧啊船险烟依斗值帮汉慢佛肯闻唱沙局伯族低玩资屋击速顾泪洲团圣旁堂兵七露园牛哭旅街劳型烈姑陈莫鱼异抱宝权鲁简态级票怪寻杀律胜份汽右洋范床舞秘午登楼贵吸责例追较职属渐左录丝牙党继托赶章智冲叶胡吉卖坚喝肉遗救修松临藏担戏善卫药悲敢靠伊村戴词森耳差短祖云规窗散迷油旧适乡架恩投弹铁博雷府压超负勒杂醒洗采毫嘴毕九冰既状乱景席珍童顶派素脱农疑练野按犯拍征坏骨余承置臂彩灯巨琴免环姆暗换技翻束增忍餐洛塞缺忆判欧层付阵玛批岛项狗休懂武革良恶恋委拥娜妙探呀营退摇弄桌熟诺宣银势奖宫忽套康供优课鸟喊降夏困刘罪亡鞋健模败伴守挥鲜财孤枪禁恐伙杰迹妹遍盖副坦牌江顺秋萨菜划授归浪凡预奶雄升编典袋莱含盛济蒙棋端腿招释介烧误掌腹尼盖竿柄柏轿煎摔趟浴汪潭宿爆慈羡杭飘漠峡睁睛砖讯潮锋伐倾瞬舍浮胖萧呵虎鼠泪拒措蛇涉筒扫迹串胡讽苍卧颤莲郊窄妥笼痕坊崖郑幕饿袭猴悲囊罐匀嚼撑掩谦踏锤焰恳乏概愤摧煤蓬惹蔬踪阶谅悠欣慰昂遥颂赠颁咏韵姿帅赴叠畅驶昏饮厅祈诚恍畏谓羞陪祥趁袖肩慌污盲瞎愚辱莫豫娱索姆寿媒馈赠谜讽谣谚讥讳诗骚驳，。！？：；、（）《》 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
unique_chars = list(dict.fromkeys(BASE_CHARS))
stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__();
        self.eps = eps;
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden_dim = int(8 * n_embd / 3)
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False);
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False);
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head;
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False);
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2);
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2);
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        return self.dropout(self.proj((F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.rms_1 = RMSNorm(n_embd);
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.rms_2 = RMSNorm(n_embd);
        self.ffwd = SwiGLU(n_embd, dropout)

    def forward(self, x): return x + self.ffwd(self.rms_2(x + self.attn(self.rms_1(x))))


class YimiGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, CONFIG["n_embd"]);
        self.position_embedding = nn.Embedding(CONFIG["block_size"], CONFIG["n_embd"])
        self.dropout = nn.Dropout(CONFIG["dropout"]);
        self.blocks = nn.Sequential(
            *[Block(CONFIG["n_embd"], CONFIG["n_head"], CONFIG["block_size"], CONFIG["dropout"]) for _ in
              range(CONFIG["n_layer"])])
        self.rms_f = RMSNorm(CONFIG["n_embd"]);
        self.lm_head = nn.Linear(CONFIG["n_embd"], vocab_size, bias=False);
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape;
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.rms_f(self.blocks(self.dropout(x))));
        loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


# ==================== 真正的无限轮转流水线 ====================
if __name__ == "__main__":
    desktop = Path.home() / "Desktop"
    batch_file = desktop / "当前批次知识.txt"

    # 初始化模型（全局共享，一直学）
    model = YimiGPT(len(unique_chars)).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    urls = ["https://www.kepuchina.cn/", "https://baike.baidu.com/item/人工智能", "https://baike.baidu.com/item/科学",
            "https://www.guokr.com/scientific/", "https://tech.sina.com.cn/", "https://tech.163.com/"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0 Safari/537.36"}
    blacklist = ["登录", "注册", "扫码", "二维码", "下载APP", "点击下载", "版权", "Copyright", "免责声明", "责任编辑",
                 "关注我们"]

    batch_num = 1

    try:
        while True:  # ✅ 大循环：永远在【爬虫 -> 过滤 -> 训练 -> 删除】中轮回
            print(f"\n{'=' * 50}")
            print(f"🚀 【第 {batch_num} 批次】开始！")
            print(f"{'=' * 50}")

            # ========== 1. 爬虫抓取 ==========
            print(f"🕷️ 阶段 1：开始爬虫，抓取 {CONFIG['crawl_time']} 秒...")
            start_crawl = time.time()
            total_pages, total_chars = 0, 0

            with open(batch_file, "w", encoding="utf-8") as f:
                while time.time() - start_crawl < CONFIG["crawl_time"]:
                    try:
                        url = random.choice(urls)
                        r = requests.get(url, headers=headers, timeout=8)
                        r.encoding = r.apparent_encoding or "utf-8"
                        soup = BeautifulSoup(r.text, "html.parser")
                        for t in soup(["script", "style", "nav", "footer", "header"]): t.decompose()
                        paragraphs = soup.find_all('p')
                        text = "\n".join([p.get_text(strip=True) for p in paragraphs])
                        if len(text) > 100:
                            f.write(text + "\n");
                            f.flush()
                            total_pages += 1;
                            total_chars += len(text)
                            print(
                                f"✅ 抓到数据 | 累计: {total_chars} 字 | {total_pages} 页 | 已爬 {int(time.time() - start_crawl)} 秒")
                    except Exception:
                        pass
                    time.sleep(1)

            # ========== 2. 过滤数据 ==========
            print(f"\n🧹 阶段 2：清洗垃圾数据...")
            clean_sentences = []
            with open(batch_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
            sentences = re.split(r'[。！？；\n]', raw_text)
            for s in sentences:
                s = s.strip()
                if len(s) < 10: continue
                if any(word in s for word in blacklist): continue
                chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', s))
                if chinese_chars / max(1, len(s)) < 0.6: continue
                clean_sentences.append(s)

            # 覆盖写入清洗后的数据
            with open(batch_file, "w", encoding="utf-8") as f:
                f.write("。".join(clean_sentences) + "。")
            print(f"✅ 清洗完成！本批次精华大小: {os.path.getsize(batch_file) / 1024:.1f} KB")

            # ========== 3. 对本批次文件进行训练 ==========
            print(f"\n🧠 阶段 3：开始消化本批次知识（训练 {CONFIG['train_steps']} 步）...")
            with open(batch_file, "r", encoding="utf-8") as f:
                all_text = f.read()
            data_list = [stoi[c] for c in all_text if c in stoi]

            if len(data_list) > CONFIG["block_size"] + 10:
                data = torch.tensor(data_list, dtype=torch.long)
                model.train()

                for step in range(CONFIG["train_steps"]):
                    ix = torch.randint(len(data) - CONFIG["block_size"], (CONFIG["batch_size"],))
                    xb = torch.stack([data[i:i + CONFIG["block_size"]] for i in ix]).to(CONFIG["device"])
                    yb = torch.stack([data[i + 1:i + CONFIG["block_size"] + 1] for i in ix]).to(CONFIG["device"])

                    logits, train_loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                    optimizer.step()

                    if (step + 1) % 100 == 0 or step == 0:
                        print(f"  👉 步数 {step + 1}/{CONFIG['train_steps']} | loss: {train_loss.item():.4f}")
            else:
                print("⚠️ 本批次数据太少，跳过训练。")

            # ========== 4. 自动保存并删除文件 ==========
            # 每学完一批，存一次档，防止断电白干
            save_pt = desktop / "Yimi米_轮转进化版.pt"
            torch.save({"model_state": model.state_dict(), "stoi": stoi, "itos": itos}, save_pt)

            print(f"\n🗑️ 阶段 4：本批次已学完，删除当前 TXT 文件！")
            try:
                os.remove(batch_file)
            except Exception:
                pass

            print(f"✅ 模型已自动存档至: {save_pt.name}")
            print(f"🔄 准备进入下一批...")
            batch_num += 1
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n👋 用户强制中断流水线。")
        final_pt = desktop / "Yimi米_最终毕业版.pt"
        torch.save({"model_state": model.state_dict(), "stoi": stoi, "itos": itos}, final_pt)
        print(f"✅ 最终模型已安全保存至桌面：{final_pt.name}")
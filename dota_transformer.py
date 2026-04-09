import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def build_hero_pool(matches: Sequence[Dict]) -> List[int]:
    pool = sorted(
        {
            int(h)
            for m in matches
            for h in (m["radiant_team"] + m["dire_team"])
        }
    )
    return pool

def load_embedding_payload(path: Path) -> Tuple[List[int], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("embedding file must be a dict payload")

    # Preferred format from embedding.py: {hero_pool, embedding}
    if "hero_pool" in payload and "embedding" in payload:
        hero_pool = [int(x) for x in payload["hero_pool"]]
        embedding = payload["embedding"]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        return hero_pool, embedding

    # Fallback checkpoint format: {hero_pool, state_dict}
    if "hero_pool" in payload and "state_dict" in payload:
        hero_pool = [int(x) for x in payload["hero_pool"]]
        state_dict = payload["state_dict"]
        if "hero_emb.weight" not in state_dict:
            raise ValueError("state_dict does not contain hero_emb.weight")
        return hero_pool, state_dict["hero_emb.weight"].detach().cpu()

    raise ValueError("unknown embedding payload format")


def load_hero_id_to_name(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_matches(data_path: Path) -> List[Dict]:
    with data_path.open("r", encoding="utf-8") as f:
        raw_matches = json.load(f)

    matches: List[Dict] = []
    dropped = 0
    for m in raw_matches:
        radiant = m.get("radiant_team", [])
        dire = m.get("dire_team", [])
        if len(radiant) == 5 and len(dire) == 5:
            matches.append(m)
        else:
            dropped += 1

    if dropped > 0:
        print(f"dropped {dropped} invalid matches (non-5v5)")
    return matches

class DotaMultiTaskDataset(Dataset):
    def __init__(self, matches, num_heroes):
        self.matches = matches # load_matches 处理好的 [winner, loser] 列表
        self.num_heroes = num_heroes
        self.mask_token_id = 0
        self.ignore_index = -100
        self.sample_multiplier = 3
    def __len__(self):
        return len(self.matches)*self.sample_multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.matches)
        m = self.matches[real_idx]
        if bool(m.get("radiant_win", False)):
            winner_team = [int(x) for x in m["radiant_team"]]
            loser_team = [int(x) for x in m["dire_team"]]
        else:
            winner_team = [int(x) for x in m["dire_team"]]
            loser_team = [int(x) for x in m["radiant_team"]]
        if random.random() > 0.5:
            ally_pool, enemy_pool, win_label = winner_team, loser_team, 1.0
        else:
            ally_pool, enemy_pool, win_label = loser_team, winner_team, 0.0

        r = random.random()
        if r < 0.2:
            total_mask_count = 0
        elif r < 0.5:
            total_mask_count = random.randint(1, 2)
        elif r < 0.8:
            total_mask_count = random.randint(3, 4)
        else:
            # 20% 概率挖 4-8 个，但为了保证至少 1v1，上限设为 8
            total_mask_count = random.randint(4, 8)
        mask_ally_count = total_mask_count // 2
        mask_enemy_count = total_mask_count - mask_ally_count
        if random.random() > 0.5:
            mask_ally_count, mask_enemy_count = mask_enemy_count, mask_ally_count

        # 4. 执行挖空
        # 采样索引：从 [0,1,2,3,4] 选 ally 的坑，从 [5,6,7,8,9] 选 enemy 的坑
        ally_indices = random.sample(range(0, 5), mask_ally_count)
        enemy_indices = random.sample(range(5, 10), mask_enemy_count)
        all_mask_indices = ally_indices + enemy_indices

        full_seq = ally_pool + enemy_pool
        masked_seq = list(full_seq)
        
        # 准备 Mask 任务的标签：只有被挖掉的位置才有真实 ID，其余为 ignore_index
        target_labels = [self.ignore_index] * 10 
        
        for i in all_mask_indices:
            target_labels[i] = full_seq[i] # 记录原英雄 ID 用于 Loss
            masked_seq[i] = self.mask_token_id # 填入 0 (Mask)

        side_ids = [0]*5 + [1]*5

        return (
            torch.tensor(masked_seq, dtype=torch.long), 
            torch.tensor(side_ids, dtype=torch.long),
            torch.tensor(target_labels, dtype=torch.long), # [10] 维标签
            torch.tensor([win_label], dtype=torch.float)
        )
    
class DotaMultiTaskTransformer(nn.Module):
    def __init__(self, num_heroes, embed_dim=64, nhead=8, num_layers=3):
        super().__init__()
        # 1. 共享 Embedding 层 (这就是你之后用来查“谁和 Puck 近”的)
        self.hero_emb = nn.Embedding(num_heroes + 1, embed_dim)
        self.side_emb = nn.Embedding(2, embed_dim) # 0: 盟友, 1: 敌人
        self.win_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 2. 共享 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 头 A: Mask 预测 (输出 130 维，看哪个英雄概率最高)
        self.mask_head = nn.Linear(embed_dim, num_heroes+1)

        # 4. 头 B: 胜率预测 (输出 1 维)
        self.win_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, hero_ids, side_ids):
        x = self.hero_emb(hero_ids) + self.side_emb(side_ids)

        batch_size = x.shape[0]
        # 把单个 win_token 扩展到当前 batch_size
        w_token = self.win_token.expand(batch_size, -1, -1)
        w_side_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=hero_ids.device)
        w_token_with_side = w_token + self.side_emb(w_side_idx)
        # 拼接到序列最前面 (位置 0)
        x = torch.cat([w_token_with_side, x], dim=1)

        features = self.transformer(x) 
        
        hero_features = features[:, 1:, :]
        mask_logits = self.mask_head(hero_features)
        
        global_features = features[:, 0, :]
        win_logits = self.win_head(global_features)
        
        return mask_logits, win_logits
    
def run_train():
    # 1. 基础配置
    try:
        matches = load_matches(Path("data.json"))
        print(f"Loaded {len(matches)} matches.")
    except FileNotFoundError:
        raise FileNotFoundError("data.json not found. Using dummy data for testing.")
    
    hero_pool = build_hero_pool(matches)
    num_heroes = max(hero_pool)
    batch_size = 512
    epochs = 50
    patience = 5
    patience_counter=0
    best_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 构造 Dataset 和 DataLoader
    dataset = DotaMultiTaskDataset(matches, num_heroes=num_heroes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    
    # 3. 初始化模型、损失函数、优化器
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    
    # reduction='none' 是关键，允许我们逐样本过滤
    criterion_mask = nn.CrossEntropyLoss(reduction='none') 
    criterion_win = nn.BCEWithLogitsLoss(reduction='none')

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_win_loss = 0
        total_mask_loss = 0
        
        for batch_idx, (masked_seq, side_ids, target_hero_labels, win_label) in enumerate(dataloader):
            # 将数据推入 GPU
            masked_seq = masked_seq.to(device)
            side_ids = side_ids.to(device)
            target_hero_labels = target_hero_labels.to(device)
            win_label = win_label.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            mask_logits, win_logits = model(masked_seq, side_ids)
            
            # 计算 Win Loss
            known_count = (masked_seq != 0).sum(dim=1).float() 
            # 线性权重：10个英雄权重1.0，1个英雄权重0.1
            win_loss_weights = known_count / 10.0
            raw_win_loss = criterion_win(win_logits.squeeze(), win_label.squeeze()) 
            loss_win = (raw_win_loss * win_loss_weights).mean()
            
            raw_mask_loss = F.cross_entropy(
                mask_logits.view(-1, num_heroes+1), 
                target_hero_labels.view(-1), 
                ignore_index=-100
            )      
            # 总损失合并
            total_loss = loss_win + 0.15 * raw_mask_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            total_win_loss += loss_win.item()
            total_mask_loss += raw_mask_loss.item()
            
        current_avg_loss = total_epoch_loss / len(dataloader)        
        print(f"--- Epoch {epoch} End | Avg Loss: {total_epoch_loss / len(dataloader):.4f} (Win: {total_win_loss / len(dataloader):.4f}, Mask: {total_mask_loss / len(dataloader):.4f}) ---")
        if current_avg_loss < best_loss:
            best_loss = current_avg_loss
            patience_counter = 0  # 只要打破记录，计数器直接清零
            torch.save(model.state_dict(), "dota_bert_best.pt")
            print("发现更低 Loss，已保存 Best Model!")
        else:
            patience_counter += 1
            print(f"Loss 未下降，Patience 积累: {patience_counter} / {patience}")
            
            if patience_counter >= patience:
                print(f"连续 {patience} 个 Epoch Loss 未创新低，触发 Early Stopping 提前结束训练。")
                break 

if __name__ == "__main__":
    run_train()
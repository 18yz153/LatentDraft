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
        self.mask_token_id = num_heroes + 1 # 用一个特殊 ID 代表 MASK，确保它不和任何英雄 ID 冲突
    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        m = self.matches[idx]
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


        # 构造长度 10 的序列
        full_seq = ally_pool + enemy_pool
        # 这里的 mask_idx 必须落在 [0, 9] 之间，确保遮住的是个英雄而不是 0
        local_mask_idx = random.randint(0, 9) 
        target_hero = full_seq[local_mask_idx]
        
        masked_seq = list(full_seq)
        masked_seq[local_mask_idx] = self.mask_token_id # 用特殊 ID 代表 MASK

        # side_ids: [0,0,0,0,0, 1,1,1,1,1] 区分敌我
        side_ids = [0]*5 + [1]*5

        return (
            torch.tensor(masked_seq, dtype=torch.long), 
            torch.tensor(side_ids, dtype=torch.long),
            torch.tensor(target_hero, dtype=torch.long), # Mask 任务的标签
            torch.tensor(local_mask_idx, dtype=torch.long),
            torch.tensor([win_label], dtype=torch.float) # 胜率任务的标签
        )
    
class DotaMultiTaskTransformer(nn.Module):
    def __init__(self, num_heroes, embed_dim=64, nhead=8, num_layers=3):
        super().__init__()
        # 1. 共享 Embedding 层 (这就是你之后用来查“谁和 Puck 近”的)
        self.hero_emb = nn.Embedding(num_heroes + 2, embed_dim, padding_idx=0)
        self.side_emb = nn.Embedding(2, embed_dim) # 0: 盟友, 1: 敌人
        self.win_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 2. 共享 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 头 A: Mask 预测 (输出 130 维，看哪个英雄概率最高)
        self.mask_head = nn.Linear(embed_dim, num_heroes + 1)

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

        win_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        pad_mask = (hero_ids == 0)
        full_mask = torch.cat([win_mask, pad_mask], dim=1)
        features = self.transformer(x, src_key_padding_mask=full_mask) 
        
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
    batch_size = 128
    epochs = 50
    patience = 5
    patience_counter=0
    best_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 构造 Dataset 和 DataLoader
    dataset = DotaMultiTaskDataset(matches, num_heroes=num_heroes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 初始化模型、损失函数、优化器
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # reduction='none' 是关键，允许我们逐样本过滤
    criterion_mask = nn.CrossEntropyLoss(reduction='none') 
    criterion_win = nn.BCEWithLogitsLoss()

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_win_loss = 0
        total_mask_loss = 0
        
        for batch_idx, (masked_seq, side_ids, target_hero, mask_pos, win_label) in enumerate(dataloader):
            # 将数据推入 GPU
            masked_seq = masked_seq.to(device)
            side_ids = side_ids.to(device)
            target_hero = target_hero.to(device)
            mask_pos = mask_pos.to(device)
            win_label = win_label.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            mask_logits, win_logits = model(masked_seq, side_ids)
            
            # 计算 Win Loss
            loss_win = criterion_win(win_logits, win_label)
            
            # 计算 Mask Loss (高级矩阵过滤)
            b_size = masked_seq.size(0)
            batch_idx_tensor = torch.arange(b_size).to(device)
            
            # 提取每一个 batch 样本中被 mask 掉的那个位置的 logits
            pos_logits = mask_logits[batch_idx_tensor, mask_pos] # [B, num_heroes + 1]
            
            # 算出所有样本的原始交叉熵
            raw_mask_loss = criterion_mask(pos_logits, target_hero) # [B]
            
            # 仅保留 win_label 为 1 的样本损失 (win_label.squeeze() 包含 1.0 或 0.0)
            mask_weights = win_label.squeeze() # 形状 [B]
            # 算总和，除以有效(赢局)的数量，clamp 防止除以 0
            filtered_mask_loss = (raw_mask_loss * mask_weights).sum() / mask_weights.sum().clamp(min=1.0)            
            # 总损失合并
            total_loss = loss_win + 0.15 * filtered_mask_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            total_win_loss += loss_win.item()
            total_mask_loss += filtered_mask_loss.item()
            
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
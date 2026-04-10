import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

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


def load_hero_static_json(path: Path) -> Dict[int, Dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_warm_start_embedding(
    hero_json: Dict[int, Dict],
    num_heroes: int,
    embed_dim: int,
    seed: int = 42,
) -> torch.Tensor:
    """Build deterministic warm-start vectors from selected hero metadata fields."""
    attrs = ["str", "agi", "int", "all"]
    atk_types = ["Melee", "Ranged"]
    roles = sorted(
        {
            role
            for h in hero_json.values()
            for role in h.get("roles", [])
            if isinstance(role, str)
        }
    )
    role_to_idx = {r: i for i, r in enumerate(roles)}

    # 只选一部分可靠字段做热启动，避免噪声太大。
    numeric_fields = [
        "move_speed",
        "attack_range",
        "base_armor",
        "base_str",
        "base_agi",
        "base_int",
        "str_gain",
        "agi_gain",
        "int_gain",
    ]

    field_stats = {}
    for field in numeric_fields:
        vals = []
        for h in hero_json.values():
            v = h.get(field)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            field_stats[field] = (min(vals), max(vals))
        else:
            field_stats[field] = (0.0, 1.0)

    feat_dim = len(attrs) + len(atk_types) + len(roles) + len(numeric_fields)
    feat = torch.zeros((num_heroes + 1, feat_dim), dtype=torch.float32)

    for hid, h in hero_json.items():
        if hid <= 0 or hid > num_heroes:
            continue

        offset = 0
        attr = h.get("primary_attr")
        if attr in attrs:
            feat[hid, offset + attrs.index(attr)] = 1.0
        offset += len(attrs)

        attack_type = h.get("attack_type")
        if attack_type in atk_types:
            feat[hid, offset + atk_types.index(attack_type)] = 1.0
        offset += len(atk_types)

        for role in h.get("roles", []):
            ridx = role_to_idx.get(role)
            if ridx is not None:
                feat[hid, offset + ridx] = 1.0
        offset += len(roles)

        for i, field in enumerate(numeric_fields):
            v = h.get(field)
            if isinstance(v, (int, float)):
                mn, mx = field_stats[field]
                feat[hid, offset + i] = float(v - mn) / float(mx - mn + 1e-6)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    proj = torch.randn((feat_dim, embed_dim), generator=generator, dtype=torch.float32)
    proj = proj / math.sqrt(max(feat_dim, 1))

    warm = feat @ proj
    warm = F.layer_norm(warm, (embed_dim,))
    return warm


def init_hero_embedding_warm_start(
    emb_layer: nn.Embedding,
    heroes_path: Path,
    num_heroes: int,
    embed_dim: int,
    alpha: float = 0.6,
    seed: int = 42,
) -> None:
    """Blend random init and metadata init: new = (1-alpha)*random + alpha*warm."""
    hero_json = load_hero_static_json(heroes_path)
    warm = build_warm_start_embedding(
        hero_json=hero_json,
        num_heroes=num_heroes,
        embed_dim=embed_dim,
        seed=seed,
    ).to(emb_layer.weight.device)

    with torch.no_grad():
        w = emb_layer.weight.data
        has_meta = warm.abs().sum(dim=1) > 0
        w[has_meta] = (1.0 - alpha) * w[has_meta] + alpha * warm[has_meta]

class DotaMultiTaskDataset(Dataset):
    def __init__(self, matches, num_heroes, is_train=True):
        self.matches = matches # load_matches 处理好的 [winner, loser] 列表
        self.num_heroes = num_heroes
        self.mask_token_id = 0
        self.ignore_index = -100
        self.is_train = is_train
        self.sample_multiplier = 2 if is_train else 1

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
        if idx%2 == 0:
            ally_pool, enemy_pool, win_label = winner_team, loser_team, 0.75
        else:
            ally_pool, enemy_pool, win_label = loser_team, winner_team, 0.25
        if self.is_train:
            rng = random  # 训练时：狂野随机，每次都不一样
        else:
            rng = random.Random(42 + idx)
        r = rng.random()
        if r < 0.15:
            total_mask_count = 0
        elif r < 0.4:
            total_mask_count = rng.randint(1, 2)
        elif r < 0.7:
            total_mask_count = rng.randint(3, 4)
        else:
            # 20% 概率挖 4-8 个，但为了保证至少 1v1，上限设为 8
            total_mask_count = rng.randint(5, 8)
        mask_ally_count = total_mask_count // 2
        mask_enemy_count = total_mask_count - mask_ally_count
        if rng.random() > 0.5:
            mask_ally_count, mask_enemy_count = mask_enemy_count, mask_ally_count

        # 4. 执行挖空
        # 采样索引：从 [0,1,2,3,4] 选 ally 的坑，从 [5,6,7,8,9] 选 enemy 的坑
        ally_indices = rng.sample(range(0, 5), mask_ally_count)
        enemy_indices = rng.sample(range(5, 10), mask_enemy_count)
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
        self.role_emb = nn.Embedding(6, embed_dim)
        self.win_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 2. 共享 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 头 A: Mask 预测 (输出 130 维，看哪个英雄概率最高)
        self.mask_head = nn.Linear(embed_dim, num_heroes+1)

        # 4. 头 B: 胜率预测 (输出 1 维)
        concat_dim = embed_dim + (10 * 5)
        self.win_head = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        # role head
        self.role_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5) # 输出 5 个位置的 logits
        )

    def forward(self, hero_ids, side_ids, role_labels=None):
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
        global_features = features[:, 0, :]
        
        role_logits = self.role_head(hero_features)
        mask_logits = self.mask_head(hero_features)

        if role_labels is None:
            # 如果没传 role_labels，就生成全 0 的张量，表示“全靠系统自动分析”
            role_labels = torch.zeros((batch_size, 10), dtype=torch.long, device=hero_ids.device)
        role_probs = F.softmax(role_logits, dim=-1) 
        
        # 2. 将输入的真实 Role (1-5) 转化为 One-Hot 向量
        hard_role_probs = torch.zeros_like(role_probs)
        # 找出哪些位置是已知的 (大于0)
        valid_mask = (role_labels > 0) # [Batch, 10]
        
        # 把 1~5 映射到索引 0~4，并用 scatter_ 填入 1.0
        safe_indices = torch.clamp(role_labels - 1, min=0)
        hard_role_probs.scatter_(2, safe_indices.unsqueeze(-1), 1.0)
        
        # 3. 完美融合 (Where): 已知的用 One-Hot，未知的 (0) 用模型预测的 Probs
        valid_mask_expanded = valid_mask.unsqueeze(-1) # [Batch, 10, 1]
        final_role_features = torch.where(valid_mask_expanded, hard_role_probs, role_probs)

        # ==========================================
        
        # 4. 降维拼接给 WinHead 
        # 现在 final_role_features 要么是绝对的 [0,1,0,0,0]，要么是猜测的 [0.1,0.7...]
        role_summary = final_role_features.view(batch_size, -1) # [Batch, 50]
        combined_features = torch.cat([global_features, role_summary], dim=-1)
        
        win_logits = self.win_head(combined_features)

        
        return mask_logits, win_logits, role_logits
    
def run_train():
    # 1. 基础配置
    random.seed(42)
    try:
        matches = load_matches(Path("data/allmatch.json"))
        print(f"Loaded {len(matches)} matches.")
    except FileNotFoundError:
        raise FileNotFoundError("data/allmatch.json not found. Using dummy data for testing.")
    
    hero_pool = build_hero_pool(matches)
    num_heroes = max(hero_pool)
    embed_dim = 64
    batch_size = 1024
    epochs = 50
    use_warm_start = True
    warm_start_alpha = 0.6
    heroes_path = Path("data/heroes.json")

    patience = 5
    patience_counter=0
    best_val_auc = 0
    best_mask_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random.shuffle(matches)
    val_size = 15000 
    val_matches = matches[:val_size]
    train_matches = matches[val_size:]
    # 2. 构造 Dataset 和 DataLoader
    train_dataset = DotaMultiTaskDataset(train_matches, num_heroes=num_heroes, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10,pin_memory=True)
    
    val_dataset = DotaMultiTaskDataset(val_matches, num_heroes=num_heroes, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10,pin_memory=True)
    # 3. 初始化模型、损失函数、优化器
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=embed_dim).to(device)

    if use_warm_start:
        if heroes_path.exists():
            init_hero_embedding_warm_start(
                emb_layer=model.hero_emb,
                heroes_path=heroes_path,
                num_heroes=num_heroes,
                embed_dim=embed_dim,
                alpha=warm_start_alpha,
                seed=42,
            )
            print(f"Warm-start enabled: init hero_emb from {heroes_path} with alpha={warm_start_alpha}")
        else:
            print(f"Warm-start skipped: {heroes_path} not found")

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    
    # reduction='none' 是关键，允许我们逐样本过滤
    criterion_win = nn.BCEWithLogitsLoss(reduction='none')

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_win_loss = 0
        total_mask_loss = 0
        
        for batch_idx, (masked_seq, side_ids, target_hero_labels, win_label) in enumerate(train_dataloader):
            # 将数据推入 GPU
            masked_seq = masked_seq.to(device)
            side_ids = side_ids.to(device)
            target_hero_labels = target_hero_labels.to(device)
            win_label = win_label.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            mask_logits, win_logits , role_logits= model(masked_seq, side_ids)
            
            # 计算 Win Loss
            known_count = (masked_seq != 0).sum(dim=1).float() 
            # 线性权重：10个英雄权重1.0，1个英雄权重0.1
            win_loss_weights = 0.3+0.7*(known_count / 10.0)
            raw_win_loss = criterion_win(win_logits.squeeze(), win_label.squeeze()) 
            loss_win = (raw_win_loss * win_loss_weights).mean()
            
            raw_mask_loss = F.cross_entropy(
                mask_logits.view(-1, num_heroes+1), 
                target_hero_labels.view(-1), 
                ignore_index=-100
            )      
            # 总损失合并

            mask_weight = 0.6 if epoch < 15 else 0.15
            win_weight = 1 if epoch < 15 else 1
            total_loss = win_weight * loss_win + mask_weight * raw_mask_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            total_win_loss += loss_win.item()
            total_mask_loss += raw_mask_loss.item()
            
        current_avg_loss = total_epoch_loss / len(train_dataloader)        
        print(f"--- Epoch {epoch} End | Avg Loss: {current_avg_loss:.4f} (Win: {total_win_loss / len(train_dataloader):.4f}, Mask: {total_mask_loss / len(train_dataloader):.4f}) ---")
        
        
        model.eval()
        val_mask_loss_sum = 0

        # 用列表收集整个 Epoch 的预测概率和真实标签
        all_win_preds = []
        all_win_labels = []

        with torch.no_grad():
            for m_seq, s_ids, t_labels, win_label in val_loader:
                m_seq, s_ids, t_labels, win_label = m_seq.to(device), s_ids.to(device), t_labels.to(device), win_label.to(device)
                mask_logits, win_logits, _ = model(m_seq, s_ids)
                
                # 1. 计算验证集 Mask Loss
                m_loss = F.cross_entropy(mask_logits.view(-1, num_heroes+1), t_labels.view(-1), ignore_index=-100)
                val_mask_loss_sum += m_loss.item()
                
                # 2. 收集胜率的概率值 (不用 > 0.5 截断)
                probs = torch.sigmoid(win_logits).squeeze().cpu().numpy()
                labels = win_label.squeeze().cpu().numpy()
                
                # 处理可能出现的标量问题 (当 batch_size=1 时)
                if probs.ndim == 0:
                    probs = [probs]
                    labels = [labels]
                    
                all_win_preds.extend(probs)
                all_win_labels.extend(labels)

        avg_val_mask_loss = val_mask_loss_sum / len(val_loader)

        # --- 新增的 AUC 计算逻辑 ---
        # 将收集到的 labels 转换回严格的 0 和 1 (如果你用了 Label Smoothing，这一步极其关键)
        binary_labels = (np.array(all_win_labels) > 0.5).astype(int)
        preds_array = np.array(all_win_preds)

        try:
            avg_val_win_auc = roc_auc_score(binary_labels, preds_array)
        except ValueError:
            # 防御性代码：极端情况下(如测试小样本)如果全是一个类别，roc_auc_score 会报错
            avg_val_win_auc = 0.5

        print(f"Val Mask Loss: {avg_val_mask_loss:.4f} | Val Win AUC: {avg_val_win_auc:.4f}")

        # --- 更新后的早停逻辑 ---
        if epoch < 20:
            # 预热期内：只记录最好指标，绝不退出
            if avg_val_mask_loss < best_mask_loss:
                best_mask_loss = avg_val_mask_loss
                # 注意：预热期可以不保存模型，或者分开保存
                torch.save(model.state_dict(), "models/dota_bert_warmup_best.pt")
            print(f"预热期内 (Remaining: {20 - epoch}), 跳过早停检查。")
            continue

        if avg_val_mask_loss < best_mask_loss:
            best_mask_loss = avg_val_mask_loss
            torch.save(model.state_dict(), "models/dota_bert_best_embedding.pt")
            print("Embedding 质量提升，保存模型！")
            patience_counter = 0 # 重置计数器
        else:
            # 2. 如果 Mask Loss 不降了，再看 Win AUC 是否在涨
            if avg_val_win_auc > best_val_auc:  # <--- 这里改成了比较 AUC
                best_val_auc = avg_val_win_auc
                # 这里建议也保存一个专注胜率的权重分支，供你的终极推荐系统使用
                torch.save(model.state_dict(), "models/dota_bert_best_winhead.pt") 
                print(f"胜率 AUC 提升至 {best_val_auc:.4f}，继续观察...")
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= patience:
            print("Mask Loss 和 Win AUC 均不再优化，触发早停。")
            break
        # if current_avg_loss < best_loss:
        #     best_loss = current_avg_loss
        #     patience_counter = 0  # 只要打破记录，计数器直接清零
        #     torch.save(model.state_dict(), "dota_bert_best.pt")
        #     print("发现更低 Loss，已保存 Best Model!")
        # else:
        #     patience_counter += 1
        #     print(f"Loss 未下降，Patience 积累: {patience_counter} / {patience}")
            
        #     if patience_counter >= patience:
        #         print(f"连续 {patience} 个 Epoch Loss 未创新低，触发 Early Stopping 提前结束训练。")
        #         break 

if __name__ == "__main__":
    run_train()

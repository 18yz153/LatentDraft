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
from src.dataset import DotaMultiTaskDataset
from src.model import DotaMultiTaskTransformer

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

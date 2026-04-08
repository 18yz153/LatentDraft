import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_hero_pool(matches: Sequence[Dict]) -> List[int]:
    pool = sorted(
        {
            int(h)
            for m in matches
            for h in (m["radiant_team"] + m["dire_team"])
        }
    )
    return pool


def load_hero_id_to_name(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}
def load_hero_static_json(path: Path) -> Dict[int, Dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def prepare_static_features(hero_json: dict, num_heroes: int):
    # 1. 定义特征维度
    attrs = ['str', 'agi', 'int', 'all']
    attack_types = ['Melee', 'Ranged']
    unique_roles = set()
    for h in hero_json.values():
        if 'roles' in h and isinstance(h['roles'], list):
            unique_roles.update(h['roles'])
    roles_lookup = sorted(list(unique_roles))
    
    # 计算总维度: 4(attr) + 2(type) + 9(roles) + 3(数值: ms, range, armor)
    static_feat_dim = len(attrs) + len(attack_types) + len(roles_lookup) + 3
    
    # 初始化全零矩阵 (padding_idx=0 留空)
    feature_matrix = np.zeros((num_heroes + 1, static_feat_dim), dtype=np.float32)
    
    # 2. 收集数值特征用于归一化 (可选，也可以用硬编码的经验值)
    all_ms = [h['move_speed'] for h in hero_json.values() if h['move_speed']]
    all_range = [h['attack_range'] for h in hero_json.values() if h['attack_range']]
    all_armor = [h['base_armor'] for h in hero_json.values() if h['base_armor'] is not None]
    
    max_ms, min_ms = max(all_ms), min(all_ms)
    max_range, min_range = max(all_range), min(all_range)
    max_armor, min_armor = max(all_armor), min(all_armor)

    for h_id_str, h in hero_json.items():
        h_id = int(h_id_str)
        if h_id > num_heroes: continue
        
        row = []
        # One-hot: Primary Attr
        attr_vec = [1 if h['primary_attr'] == a else 0 for a in attrs]
        row.extend(attr_vec)
        
        # One-hot: Attack Type
        type_vec = [1 if h['attack_type'] == t else 0 for t in attack_types]
        row.extend(type_vec)
        
        # Multi-hot: Roles
        role_vec = [1 if r in h['roles'] else 0 for r in roles_lookup]
        row.extend(role_vec)
        
        # Min-Max Normalization: 数值特征
        # 归一化很重要，否则 315 的移速会直接淹没 0/1 信号
        ms_norm = (h['move_speed'] - min_ms) / (max_ms - min_ms + 1e-6)
        range_norm = (h['attack_range'] - min_range) / (max_range - min_range + 1e-6)
        armor_norm = (h['base_armor'] - min_armor) / (max_armor - min_armor + 1e-6)
        row.extend([ms_norm, range_norm, armor_norm])
        
        feature_matrix[h_id] = np.array(row)
        
    return torch.from_numpy(feature_matrix), static_feat_dim

class LatentDraftBPR(nn.Module):
    def __init__(self, num_heroes: int, static_feat_dim: int, embed_dim: int = 64, enemy_weight: float = 0.8):
        super().__init__()
        self.hero_emb = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)
        self.static_feats = nn.Parameter(torch.zeros((num_heroes + 1, static_feat_dim)), requires_grad=False)
        self.static_proj = nn.Sequential(
            nn.Linear(static_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh() # 将静态特征输出限制在 [-1, 1]，防止数值爆炸
        )
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.enemy_weight = enemy_weight

        nn.init.xavier_uniform_(self.hero_emb.weight)

    def get_hero_rep(self, hero_ids: torch.Tensor) -> torch.Tensor:
        """获取融合了属性和实战的‘英雄最终形态’向量"""
        dyn_emb = self.hero_emb(hero_ids) # (batch, seq_len, embed_dim)
        stat_emb = self.static_proj(self.static_feats[hero_ids])
        combined = dyn_emb + 0.1 * stat_emb
        return combined
    
    def get_context_vector(self, ally_ids: torch.Tensor, enemy_ids: torch.Tensor) -> torch.Tensor:
        ally_emb = self.get_hero_rep(ally_ids).mean(dim=1)
        enemy_emb = self.get_hero_rep(enemy_ids).mean(dim=1)
        context_cat = torch.cat([ally_emb, self.enemy_weight * enemy_emb], dim=-1)
        return self.context_proj(context_cat)

    def score(self, context_vec: torch.Tensor, hero_ids: torch.Tensor) -> torch.Tensor:
        hero_vec = self.get_hero_rep(hero_ids)
        return (context_vec * hero_vec).sum(dim=-1)

    def forward(
        self,
        ally_ids: torch.Tensor,
        enemy_ids: torch.Tensor,
        pos_hero_id: torch.Tensor,
        neg_hero_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_vec = self.get_context_vector(ally_ids, enemy_ids)
        pos_score = self.score(context_vec, pos_hero_id)
        neg_score = self.score(context_vec, neg_hero_id)
        return pos_score, neg_score


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_score - neg_score).mean()


class Dota2LineupDataset(Dataset):
    def __init__(
        self,
        matches: Sequence[Dict],
        hero_pool: Sequence[int],
        mask_min: int = 1,
        mask_max: int = 1,
        samples_per_match: int = 1,
    ):
        self.matches = list(matches)
        self.hero_pool = list(hero_pool)
        self.hero_set = set(hero_pool)
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.samples_per_match = samples_per_match

    def __len__(self) -> int:
        return len(self.matches) * self.samples_per_match

    def _build_one(self, match: Dict) -> Dict[str, torch.Tensor]:
        if bool(match.get("radiant_win", False)):
            winner_team = [int(x) for x in match["radiant_team"]]
            loser_team = [int(x) for x in match["dire_team"]]
        else:
            winner_team = [int(x) for x in match["dire_team"]]
            loser_team = [int(x) for x in match["radiant_team"]]

        # 2. 我们只从【胜方】提取正样本逻辑
        # 这样模型学到的是：针对败方这5个人，胜方这5个人的协同是有效的
        ally_team = winner_team
        enemy_team = loser_team

        # 3. 经典的 Context Masking
        # 随机抠掉一个胜方英雄作为 Positive
        pos_idx = random.randint(0, 4)
        pos_hero_id = ally_team[pos_idx]
        ally_context = [h for i, h in enumerate(ally_team) if i != pos_idx]

        # 4. 核心改进：负采样策略 (Hard Negative)
        # 我们不仅要从全池抽路人，还要有概率抽【败方】的英雄作为负样本
        used_heroes = set(winner_team + loser_team)
        
        available_neg = list(self.hero_set - used_heroes)
        neg_hero_id = random.choice(available_neg)

        return {
            "ally_ids": torch.tensor(ally_context, dtype=torch.long),
            "enemy_ids": torch.tensor(enemy_team, dtype=torch.long),
            "pos_hero_id": torch.tensor(pos_hero_id, dtype=torch.long),
            "neg_hero_id": torch.tensor(neg_hero_id, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        match_idx = idx//self.samples_per_match
        match = self.matches[match_idx]
        return self._build_one(match)



def train(args: argparse.Namespace) -> None:
    set_seed(42)

    matches = load_matches(args.data)
    hero_pool = build_hero_pool(matches)
    num_heroes = max(hero_pool)
    hero_json = load_hero_static_json(args.hero_file) 
    static_feats, static_feat_dim = prepare_static_features(hero_json, num_heroes)
    dataset = Dota2LineupDataset(
        matches=matches,
        hero_pool=hero_pool,
        mask_min=args.mask_min,
        mask_max=args.mask_max,
        samples_per_match=args.samples_per_match,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        print("cuda is not available, fallback to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(requested)

    model = LatentDraftBPR(
        num_heroes=num_heroes,
        static_feat_dim=static_feat_dim,
        embed_dim=args.embed_dim,
        enemy_weight=args.enemy_weight,
    ).to(device)
    model.static_feats.data.copy_(static_feats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"device={device} matches={len(matches)} samples={len(dataset)} "
        f"heroes={len(hero_pool)} embed_dim={args.embed_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in loader:
            ally_ids = batch["ally_ids"].to(device)
            enemy_ids = batch["enemy_ids"].to(device)
            pos_id = batch["pos_hero_id"].to(device)
            neg_id = batch["neg_hero_id"].to(device)

            optimizer.zero_grad()
            pos_score, neg_score = model(ally_ids, enemy_ids, pos_id, neg_id)
            loss = bpr_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
            # 在每个 epoch 结束时打印：
        dyn_norm = model.hero_emb.weight.norm(dim=-1).mean().item()
        print(f"动态 Embedding 平均模长: {dyn_norm:.4f}")

        avg_loss = total_loss / max(steps, 1)
        print(f"epoch={epoch} avg_bpr_loss={avg_loss:.6f}")

    emb_weight = model.hero_emb.weight.detach().cpu()
    hero_id_to_name = load_hero_id_to_name(args.hero_id_to_name)

    ckpt = {
        "state_dict": model.state_dict(),
        "hero_pool": hero_pool,
        "num_heroes": num_heroes,
        "embed_dim": args.embed_dim,
        "enemy_weight": args.enemy_weight,
    }
    torch.save(ckpt, args.out_ckpt)

    model.eval()
    with torch.no_grad():
        # 生成 0 到 num_heroes 的 Tensor
        all_hero_ids = torch.arange(0, num_heroes + 1, dtype=torch.long, device=device)
        # 调用模型的 get_hero_rep，拿到融合了 JSON 和实战的终极向量！
        fused_embeddings = model.get_hero_rep(all_hero_ids)
        emb_weight_to_save = fused_embeddings.detach().cpu()

    # 然后再保存这个融合后的向量
    emb_payload = {
        "hero_pool": hero_pool,
        "embedding": emb_weight_to_save, 
    }
    torch.save(emb_payload, args.out_embedding)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPR hero embedding with context masking")
    parser.add_argument("--data", type=Path, default=Path("data.json"))
    parser.add_argument("--hero-id-to-name", type=Path, default=Path("hero_id_to_name.json"))
    parser.add_argument("--hero-file", type=Path, default=Path("heroes.json"))
    parser.add_argument("--out-ckpt", type=Path, default=Path("bpr_context_model.pt"))
    parser.add_argument("--out-embedding", type=Path, default=Path("hero_embedding.pt"))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument("--mask-min", type=int, default=1, help="min masked ally heroes per sample")
    parser.add_argument("--mask-max", type=int, default=2, help="max masked ally heroes per sample")

    parser.add_argument("--samples-per-match", type=int, default=1, help="samples drawn per match per side")

    parser.add_argument("--query-hero-id", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weight_decay = 1e-4
    args.enemy_weight = 1
    train(args)


if __name__ == "__main__":
    main()
import random
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class DotaMultiTaskDataset(Dataset):
    def __init__(self, matches, roles, is_train=True, num_heroes=None):
        self.matches = matches # load_matches 处理好的 [winner, loser] 列表
        self.roles = roles # load_roles 处理好的角色列表
        if num_heroes is None:
            num_heroes = max(
                int(h)
                for m in self.matches
                for h in (m["radiant_team"] + m["dire_team"])
            )
        self.pad_token_id = 0
        self.mask_ignore_index = -100
        self.is_train = is_train
        self.sample_multiplier = 1 if is_train else 1

    def __len__(self):
        return len(self.matches)*self.sample_multiplier

    def __getitem__(self, idx):
        real_idx = idx // self.sample_multiplier
        m = self.matches[real_idx]
        if bool(m.get("radiant_win", False)):
            winner_team = [int(x) for x in m["radiant_team"]]
            loser_team = [int(x) for x in m["dire_team"]]
            # --- 新增：提取位置（如果是 AP 局没这个字段，就默认全 0） ---
            winner_roles = m.get("rad_positions", [0]*5)
            loser_roles = m.get("dire_positions", [0]*5)
        else:
            winner_team = [int(x) for x in m["dire_team"]]
            loser_team = [int(x) for x in m["radiant_team"]]
            # --- 新增：提取位置 ---
            winner_roles = m.get("dire_positions", [0]*5)
            loser_roles = m.get("rad_positions", [0]*5)
        if idx % 2 == 0:
            ally_pool, enemy_pool = winner_team, loser_team
            ally_roles, enemy_roles = winner_roles, loser_roles # 位置跟着换
            win_label = 1
        else:
            ally_pool, enemy_pool = loser_team, winner_team
            ally_roles, enemy_roles = loser_roles, winner_roles # 位置跟着换
            win_label = 0


        if self.is_train:
            rng = random  # 训练时：狂野随机，每次都不一样
        else:
            rng = random.Random(42 + idx)
        
        r = rng.random()
        if r < 0.2:
            total_mask_count = 0  
        elif r < 0.7:
            total_mask_count = rng.randint(1, 5) 
        else:
            total_mask_count = rng.randint(6, 8)
            
        mask_ally_count = total_mask_count // 2
        mask_enemy_count = total_mask_count - mask_ally_count
        if rng.random() > 0.5:
            mask_ally_count, mask_enemy_count = mask_enemy_count, mask_ally_count

        ally_indices = rng.sample(range(0, 5), mask_ally_count)
        enemy_indices = rng.sample(range(5, 10), mask_enemy_count)
        all_mask_indices = ally_indices + enemy_indices

        # 3. 执行双重挖空 (英雄挖空 + 位置挖空)
        full_seq = ally_pool + enemy_pool
        masked_seq = list(full_seq)
        
        full_roles = ally_roles + enemy_roles
        masked_roles = list(full_roles) # 新增：准备一根能被挖空的位置轴
        
        target_labels = [self.mask_ignore_index] * 10 
        
        for i in all_mask_indices:
            target_labels[i] = full_seq[i] 
            masked_seq[i] = self.pad_token_id  # 英雄变 PAD(未知)
            masked_roles[i] = 0                 # 🌟 关键：被挖空的英雄，位置强行变 0！

        side_ids = [0]*5 + [1]*5
        actual_hero_count = 10 - len(all_mask_indices)
        # smooth_win_label = 0.5+ (win_label-0.5)*0.5*(actual_hero_count/10)**0.7
        return (
            torch.tensor(masked_seq, dtype=torch.long), 
            torch.tensor(side_ids, dtype=torch.long),
            torch.tensor(masked_roles, dtype=torch.long), 
            torch.tensor(target_labels, dtype=torch.long), 
            torch.tensor(full_seq, dtype=torch.long),
            torch.tensor([win_label], dtype=torch.float)
        )
    

def load_and_merge_s3_matches(
    match_role_path: Path = Path("data/match_role.json"),
    allmatch_path: Path = Path("data/allmatch.json"),
    prefer_match_role: bool = True,
) -> list:
    """
    读取 match_role + allmatch，合并成统一的字典列表返回。
    """
    with open(match_role_path, "r", encoding="utf-8") as f:
        role_rows = json.load(f)
    with open(allmatch_path, "r", encoding="utf-8") as f:
        all_rows = json.load(f)

    role_by_match_id = {}
    for r in role_rows:
        try:
            mid = int(r["match_id"])
            role_by_match_id[mid] = {
                "match_id": mid,
                "radiant_team": [int(x) for x in r["rad_hero_ids"]],
                "dire_team": [int(x) for x in r["dire_hero_ids"]],
                "rad_positions": [int(x) for x in r.get("rad_positions", [0, 0, 0, 0, 0])],
                "dire_positions": [int(x) for x in r.get("dire_positions", [0, 0, 0, 0, 0])],
                "radiant_win": bool(int(float(r.get("rad_winlabel", 0)))),
            }
        except (KeyError, TypeError, ValueError):
            continue

    merged_matches = []
    seen = set()

    for m in all_rows:
        try:
            mid = int(m["match_id"])
            rad = [int(x) for x in m["radiant_team"]]
            dire = [int(x) for x in m["dire_team"]]
            if len(rad) != 5 or len(dire) != 5:
                continue

            if prefer_match_role and mid in role_by_match_id:
                merged = role_by_match_id[mid]
            else:
                role_ref = role_by_match_id.get(mid)
                merged = {
                    "match_id": mid,
                    "radiant_team": rad,
                    "dire_team": dire,
                    "rad_positions": role_ref["rad_positions"] if role_ref else [0, 0, 0, 0, 0],
                    "dire_positions": role_ref["dire_positions"] if role_ref else [0, 0, 0, 0, 0],
                    "radiant_win": bool(m.get("radiant_win", False)),
                }

            merged_matches.append(merged)
            seen.add(mid)
        except (KeyError, TypeError, ValueError):
            continue

    for mid, row in role_by_match_id.items():
        if mid not in seen:
            if len(row["radiant_team"]) == 5 and len(row["dire_team"]) == 5:
                merged_matches.append(row)

    print(f"S3 merge done: allmatch={len(all_rows)}, match_role={len(role_rows)}, merged={len(merged_matches)}")
    return merged_matches


# ==========================================
# 2. 统一入口：负责切分和生成 DataLoader
# ==========================================
def build_s3_dataloader(
    match_role_path: Path = Path("data/match_role.json"),
    allmatch_path: Path = Path("data/match_rank_80.json"),
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    prefer_match_role: bool = True,
    xgb = False, # 新增参数：如果是 True 就直接返回 Dataset，不封装 DataLoader
):
    # 1. 获取清洗合并后的纯 List 数据
    all_matches = load_and_merge_s3_matches(
        match_role_path=match_role_path,
        allmatch_path=allmatch_path,
        prefer_match_role=prefer_match_role,
    )

    # 2. 按 Match 级别进行 Train/Val 划分 (防止 Data Leakage)
    total_matches = len(all_matches)
    val_match_count = int(total_matches * val_ratio)
    train_match_count = total_matches - val_match_count

    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_matches, generator=generator).tolist()
    train_idx = perm[:train_match_count]
    val_idx = perm[train_match_count:]

    train_matches = [all_matches[i] for i in train_idx]
    val_matches = [all_matches[i] for i in val_idx]

    # 3. 构造真正的 Dataset
    train_dataset = DotaMultiTaskDataset(
        matches=train_matches,
        roles=[None] * len(train_matches),
        is_train=True,
    )
    val_dataset = DotaMultiTaskDataset(
        matches=val_matches,
        roles=[None] * len(val_matches),
        is_train=False,
    )

    # 4. 封装进 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    print(
        "S3 Split (by match): "
        f"{len(train_matches)} Train | {len(val_matches)} Valid "
        f"(expanded samples: {len(train_dataset)} Train | {len(val_dataset)} Valid)"
    )
    
    # 返回了 Dataset 对象方便你之后跑 XGBoost 直接提取数据
    if xgb:
        return train_dataset, val_dataset

    return train_loader, val_loader

class DotaCMDataset(Dataset):
    """CM 阶段数据：默认读取 match_role.json，返回 role+win 联合训练所需张量。"""

    def __init__(self, data_source, is_train=True, num_heroes=None):
        if isinstance(data_source, (str, Path)):
            with open(data_source, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = data_source
        if num_heroes is None:
            num_heroes = max(
                int(h)
                for m in self.data
                for h in (m["rad_hero_ids"] + m["dire_hero_ids"])
            )
        self.pad_token_id = 0
        self.is_train = is_train
        self.variants_per_match = 13

    def __len__(self):
        return len(self.data) * self.variants_per_match

    def __getitem__(self, idx):
        match_idx = idx // self.variants_per_match
        variant_type = idx % self.variants_per_match
        item = self.data[match_idx]

        hero_ids = item["rad_hero_ids"] + item["dire_hero_ids"]
        positions = item["rad_positions"] + item["dire_positions"]
        win_label = item["rad_winlabel"]
        side_ids = [0] * 5 + [1] * 5

        if self.is_train:
            rng = random
        else:
            # 验证集固定随机性，避免指标抖动
            rng = random.Random(42 + idx)
        
        masked_hero_ids = [0] * 10
        masked_positions = [0] * 10 

        # 13 个变体对应 13 组总挖空数，覆盖轻/中/重遮盖
        if variant_type < 10:
            masked_hero_ids[variant_type] = hero_ids[variant_type]
            masked_positions[variant_type] = positions[variant_type]
        elif variant_type == 10:
            masked_hero_ids = list(hero_ids)
            masked_positions = list(positions)
        else:
            total_mask_count = rng.randint(1, 8)
            masked_hero_ids = list(hero_ids)
            masked_positions = list(positions)
            # 按阵营拆分挖空数量（与 DotaMultiTaskDataset 一致）
            mask_ally_count = total_mask_count // 2
            mask_enemy_count = total_mask_count - mask_ally_count
            if rng.random() > 0.5:
                mask_ally_count, mask_enemy_count = mask_enemy_count, mask_ally_count

            ally_indices = rng.sample(range(0, 5), mask_ally_count)
            enemy_indices = rng.sample(range(5, 10), mask_enemy_count)
            all_mask_indices = ally_indices + enemy_indices

            for i in all_mask_indices:
                masked_hero_ids[i] = self.pad_token_id
                masked_positions[i] = 0
        
        actual_hero_count = sum(1 for hid in masked_hero_ids if hid != self.pad_token_id)

        smooth_win_label = 0.5+ (win_label-0.5)*0.5*(actual_hero_count/10)**0.7
        return (
            torch.tensor(masked_hero_ids, dtype=torch.long),
            torch.tensor(side_ids, dtype=torch.long),
            torch.tensor(masked_positions, dtype=torch.long),
            torch.tensor(list(hero_ids), dtype=torch.long), 
            torch.tensor(smooth_win_label, dtype=torch.float),
        )


def build_cm_dataloader(
    data_path: Path = Path("data/match_role.json"),
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    val_ratio: float = 0.1,
) -> DataLoader:
    with open(data_path, "r", encoding="utf-8") as f:
        all_matches = json.load(f)

    total_matches = len(all_matches)
    val_match_count = int(total_matches * val_ratio)
    train_match_count = total_matches - val_match_count

    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_matches, generator=generator).tolist()
    train_idx = perm[:train_match_count]
    val_idx = perm[train_match_count:]

    train_matches = [all_matches[i] for i in train_idx]
    val_matches = [all_matches[i] for i in val_idx]

    train_dataset = DotaCMDataset(train_matches, is_train=True)
    val_dataset = DotaCMDataset(val_matches, is_train=False)
    
    # 构造 Loader：训练集打乱，验证集不需要打乱
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,persistent_workers=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True,pin_memory=True)
    
    print(f"CM Split (by match): {len(train_matches)} Train | {len(val_matches)} Valid")
    return train_loader, val_loader

class RerankDataset(Dataset):
    def __init__(self, matches):
        self.matches = matches

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        win_label = match['win_label']
        smoothed_label = 0.75 if win_label > 0.5 else 0.25
        # actual_hero_count = sum(1 for h in match['masked_seq'] if h > 0)
        # k = 1 - (actual_hero_count / 10)
        # gamma = 1.5
        # conf_weight = k ** gamma
        # # 将 0/1 标签拉向 0.5
        # smoothed_label = (1.0 - conf_weight) * win_label + conf_weight * 0.5
        
        return {
            'masked_seq': torch.tensor(match['masked_seq'], dtype=torch.long),
            'side_ids': torch.tensor(match['side_ids'], dtype=torch.long),
            'role_labels': torch.tensor(match['role_labels'], dtype=torch.long),
            'full_seq': torch.tensor(match['full_seq'], dtype=torch.long),
            'win_label': torch.tensor(smoothed_label, dtype=torch.float),
            'fill_pos': torch.tensor(match['fill_pos'], dtype=torch.long),
            'hard_negs': torch.tensor(match['hard_negs'], dtype=torch.long),
            'easy_negs': torch.tensor(match['easy_negs'], dtype=torch.long),
            'target_hero': torch.tensor(match['target_hero'], dtype=torch.long),
            'candidates': torch.tensor(
                [match['target_hero']] + match['easy_negs'][:5], 
                dtype=torch.long
            )
        }
    
def load_jsonl(filepath):
    print(f"Loading {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: # 过滤可能存在的空行
                data.append(json.loads(line))
    print(f"Loaded {len(data)} records.")
    return data

def build_rerank_dataloader(
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 4,
    xgb = False, # 新增参数：如果是 True 就直接返回 Dataset，不封装 DataLoader
) -> DataLoader:
    train_matches = load_jsonl('data/rerank_s3_train.jsonl')
    val_matches = load_jsonl('data/rerank_s3_val.jsonl')

    train_dataset = RerankDataset(
        matches=train_matches
    )
    val_dataset = RerankDataset(
        matches=val_matches
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    if xgb:
        return train_dataset, val_dataset
    return train_loader, val_loader
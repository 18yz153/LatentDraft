import random
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split


class DotaMultiTaskDataset(Dataset):
    def __init__(self, matches, roles, is_train=True):
        self.matches = matches # load_matches 处理好的 [winner, loser] 列表
        self.roles = roles # load_roles 处理好的角色列表
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
            win_label = 0.75
        else:
            ally_pool, enemy_pool = loser_team, winner_team
            ally_roles, enemy_roles = loser_roles, winner_roles # 位置跟着换
            win_label = 0.25


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
            total_mask_count = rng.randint(5, 8)
            
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
        
        target_labels = [self.ignore_index] * 10 
        
        for i in all_mask_indices:
            target_labels[i] = full_seq[i] 
            masked_seq[i] = self.mask_token_id  # 英雄变 0
            masked_roles[i] = 0                 # 🌟 关键：被挖空的英雄，位置强行变 0！

        side_ids = [0]*5 + [1]*5

        return (
            torch.tensor(masked_seq, dtype=torch.long), 
            torch.tensor(side_ids, dtype=torch.long),
            torch.tensor(masked_roles, dtype=torch.long),  # 🌟 新增：把处理好的 Role 传出去
            torch.tensor(target_labels, dtype=torch.long), 
            torch.tensor([win_label], dtype=torch.float)
        )
    


def build_s3_dataset(
    match_role_path: Path = Path("data/match_role.json"),
    allmatch_path: Path = Path("data/allmatch.json"),
    is_train: bool = True,
    prefer_match_role: bool = True,
) -> DotaMultiTaskDataset:
    """
    读取 match_role + allmatch，合并成统一样本后构造 DotaMultiTaskDataset。
    不改 DotaCMDataset，仅用于 S3 阶段快速实验。
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

    print(
        f"S3 merge done: allmatch={len(all_rows)}, match_role={len(role_rows)}, merged={len(merged_matches)}"
    )

    # roles 参数当前在 Dataset 内部未使用，传占位列表保证接口稳定。
    roles_placeholder = [None] * len(merged_matches)
    return DotaMultiTaskDataset(
        matches=merged_matches,
        roles=roles_placeholder,
        is_train=is_train,
    )


def build_s3_dataloader(
    match_role_path: Path = Path("data/match_role.json"),
    allmatch_path: Path = Path("data/allmatch.json"),
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    prefer_match_role: bool = True,
):
    dataset = build_s3_dataset(
        match_role_path=match_role_path,
        allmatch_path=allmatch_path,
        is_train=True,
        prefer_match_role=prefer_match_role,
    )

    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

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

    print(f"S3 Dataset Split: {train_size} Train | {val_size} Valid")
    return train_loader, val_loader

class DotaCMDataset(Dataset):
    """CM 阶段数据：默认读取 match_role.json，返回 role+win 联合训练所需张量。"""

    def __init__(self, data_path: Path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        hero_ids = item["rad_hero_ids"] + item["dire_hero_ids"]
        positions = item["rad_positions"] + item["dire_positions"]
        rad_winlabel = float(item["rad_winlabel"])

        return (
            torch.tensor(hero_ids, dtype=torch.long),
            torch.tensor([0] * 5 + [1] * 5, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            torch.tensor(rad_winlabel, dtype=torch.float),
        )


def build_cm_dataloader(
    data_path: Path = Path("data/match_role.json"),
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    val_ratio: float = 0.1,
) -> DataLoader:
    dataset = DotaCMDataset(data_path)
    
    # 计算切分大小
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # 随机切分 (可以加个 generator 固定随机种子保证每次切分一样)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # 构造 Loader：训练集打乱，验证集不需要打乱
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,persistent_workers=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True,pin_memory=True)
    
    print(f"Dataset Split: {train_size} Train | {val_size} Valid")
    return train_loader, val_loader

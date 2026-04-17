import sys
import json
import math
import torch
from pathlib import Path

# Support: python .\train\rolehead_fintune.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_number_of_heroes
from src.model import DotaMultiTaskTransformer
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import build_cm_dataloader
from src.train import CMTrainer


def load_hero_static_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_warm_start_embedding(hero_json, num_heroes, embed_dim, seed=42):
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
        field_stats[field] = (min(vals), max(vals)) if vals else (0.0, 1.0)

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


def init_hero_embedding_warm_start(emb_layer, heroes_path, num_heroes, embed_dim, alpha=0.6, seed=42):
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

def finetune_role_head():
    # 1. 初始化模型（S2 从随机初始化开始，不加载 checkpoint）
    num_heroes = get_number_of_heroes(Path("data/hero_id_to_name.json"))
    embed_dim = 64
    use_warm_start = True
    warm_start_alpha = 0.6
    heroes_path = Path("data/heroes.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=embed_dim).to(device)
    print("S2 starts from random initialization (no checkpoint loaded).")

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

    
    # S2: 不训练 winhead,和win token
    for param in model.win_head.parameters():
        param.requires_grad = False
    model.win_token.requires_grad = False
        
    # 3. 准备 CM DataLoader（match_role.json）
    train_loader, val_loader = build_cm_dataloader(
        data_path=Path("data/match_role.json"),
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        val_ratio=0.1
    )

    # 4. 粗版联合训练配置（你后续可细调 loss/权重）
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    criterion_role = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_mask = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = CMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path="models"
    )

    best_val_role_acc = 0.0
    best_val_total_loss = float("inf")
    patience = 5
    patience_counter = 0
    max_epochs = 50 # 把上限设高，反正有早停拦着

    for epoch in range(max_epochs):
        # 1. 训练阶段
        train_loss, train_role_acc = trainer.train_one_epoch(
            train_loader, criterion_role, criterion_mask
        )
        
        # 2. 验证阶段 (你需要确保 CMTrainer 里有一个 evaluate 方法)
        # 验证阶段不要计算梯度：with torch.no_grad():
        val_loss, val_role_acc, val_mask_loss = trainer.evaluate(
            val_loader, criterion_role, criterion_mask
        )
        
        print(f"Epoch {epoch} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_role_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Val Mask Loss: {val_mask_loss:.4f}, Val Acc: {val_role_acc:.2%}")

        # 3. 分别维护 Role Acc / Total Loss 的最佳模型，并共享一个早停耐心
        improved = False
        if val_role_acc > best_val_role_acc:
            best_val_role_acc = val_role_acc
            improved = True
            torch.save(model.state_dict(), "models/stage2_role_best.pt")
            print("  --> [New Best Role Model Saved: stage2_role_best.pt]")

        if val_loss < best_val_total_loss:
            best_val_total_loss = val_loss
            improved = True
            torch.save(model.state_dict(), "models/stage2_loss_best.pt")
            print("  --> [New Best Loss Model Saved: stage2_loss_best.pt]")

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  --> [Shared Patience: {patience_counter}/{patience}]")
            
        # 4. 共享耐心早停：当两个指标都连续不提升达到耐心阈值时停止
        if patience_counter >= patience:
            print(
                "Early stopping triggered! "
                f"Neither Val Role Acc nor Val Loss improved for {patience} epochs."
            )
            break

    print(
        "Fine-tuning complete. Best checkpoints: "
        "stage2_role_best.pt and stage2_loss_best.pt"
    )

if __name__ == "__main__":
    finetune_role_head()
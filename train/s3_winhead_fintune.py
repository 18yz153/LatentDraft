import sys
import torch
from pathlib import Path

# Support: python .\train\rolehead_fintune.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_number_of_heroes
from src.model import DotaMultiTaskTransformer
import torch.nn as nn
from src.dataset import build_rerank_dataloader
from src.train import S3WinHeadTrainer

def train_value_network():
    # 1. 初始化模型并加载预训练底座
    num_heroes = get_number_of_heroes()
    embed_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=embed_dim).to(device)
    
    # S3 从 S2 的 best role 模型继续训练
    state_dict = torch.load("models/checkpoint/stage2_role_best.pt", map_location=device)
    
    # 3. 再加载到模型中
    keys_to_delete = [k for k in state_dict.keys() if "win_head" in k]
    for k in keys_to_delete:
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print("Pre-trained base loaded (strict=False).")

    for param in model.parameters():
        param.requires_grad = False
    
    # S3: 冻结 role head，打开 winhead/wintoken + backbone 微调
    for param in model.role_head.parameters():
        param.requires_grad = False
    for param in model.win_head.parameters():
        param.requires_grad = True
    model.win_token.requires_grad = True
    for param in model.transformer.parameters():
        param.requires_grad = True
    for param in model.hero_emb.parameters():
        param.requires_grad = True
    for param in model.side_emb.parameters():
        param.requires_grad = True
    for param in model.mask_head.parameters():
        param.requires_grad = True
    factor = 0.25    
    # 3. 准备 CM DataLoader（match_role.json）
    train_loader, val_loader = build_rerank_dataloader(
        batch_size=int(1024*factor),
        shuffle=True,
        num_workers=2
    )

    # 4. S3 优化器：分组学习率
    optimizer = torch.optim.Adam([
        {'params': model.win_head.parameters(), 'lr': 1e-3*factor},
        {'params': [model.win_token], 'lr': 1e-3*factor},
        {'params': model.transformer.parameters(), 'lr': 3e-4*factor},
        {'params': model.hero_emb.parameters(), 'lr': 5e-5*factor},
        {'params': model.side_emb.parameters(), 'lr': 5e-5*factor},
        {'params': model.mask_head.parameters(), 'lr': 1e-4*factor},
    ])
    # 动态降学习率：当验证 AUC 停滞时自动降低各参数组学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=[2e-5, 1e-6, 1e-6, 1e-8, 1e-8, 1e-7],
    )
    criterion_mask = nn.CrossEntropyLoss(ignore_index=-100) 
    criterion_win = nn.BCEWithLogitsLoss()

    trainer = S3WinHeadTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path="models",
    )
    best_auc = 0.0
    best_hit10 = 0.0
    patience = 5        # 如果连续 5 个 Epoch Hit@10 不涨，就停止
    patience_counter = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        # 1. 训练阶段 (S3Trainer 返回 3 个值：总Loss，Win Loss，Mask Loss)
        total_loss, total_win_loss, total_infonce_loss, total_masked_loss = trainer.train_one_epoch(
            train_loader, criterion_mask,alpha=2, beta=0.1
        )
        print(
            f"[Train]   Epoch {epoch} | "
            f"Loss: {total_loss:.4f} (Win: {total_win_loss:.4f}, InfoNCE: {total_infonce_loss:.4f}, Mask: {total_masked_loss:.4f}) | "
        )

        business_metrics = trainer.business_evaluate(
            val_loader
        )

        loss , auc = trainer.evaluate(
            val_loader, criterion_mask, criterion_win
        )

        print(f"Loss: {loss:.4f}, AUC: {auc:.4f}")
        hit10 = business_metrics.get("hit_at_10", 0.0)
        scheduler.step(hit10)

        current_lrs = [group["lr"] for group in optimizer.param_groups]

        mask_hit30 = business_metrics.get("mask_hit_at_30", 0.0)
        mask_hit10 = business_metrics.get("mask_hit_at_10", 0.0)
        mask_hit5 = business_metrics.get("mask_hit_at_5", 0.0)
        hit5 = business_metrics.get("hit_at_5", 0.0)
        hit10 = business_metrics.get("hit_at_10", 0.0)
        query_count = business_metrics.get("queries", 0)

        print(
            f"[Business] Epoch {epoch} | "
            f"Mask Hit@30: {mask_hit30:.2%} ({mask_hit30}) | "
            f"Mask Hit@10: {mask_hit10:.2%} ({mask_hit10}) | "
            f"Mask Hit@5: {mask_hit5:.2%} ({mask_hit5}) | "
            f"Hit@5: {hit5:.2%} ({hit5}) | "
            f"Hit@10: {hit10:.2%} ({hit10})"
            f" | Queries: {query_count}"
            f"LRs: {[f'{lr:.2e}' for lr in current_lrs]}"
        )

        # 3. 早停逻辑：按 Business Hit@10
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0  # 重置计数器
            # 存下最牛逼的价值网络
            torch.save(model.state_dict(), "models/stage3_value_network_best.pt")
            print("  --> [New Best Value Network Saved by AUC!]")
        else:
            patience_counter += 1
            print(f"  --> [Patience: {patience_counter}/{patience}]")

        # 4. 早停触发
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered! Hit@10 hasn't improved for {patience} epochs.")
        #     break
        # if hit10 > best_hit10:
        #     best_hit10 = hit10
        #     patience_counter = 0  # 重置计数器
        #     # 存下最牛逼的价值网络
        #     torch.save(model.state_dict(), "models/stage3_value_network_best.pt")
        #     print("  --> [New Best Value Network Saved by Hit@10!]")
        # else:
        #     patience_counter += 1
        #     print(f"  --> [Patience: {patience_counter}/{patience}]")

        # 4. 早停触发
        if patience_counter >= patience:
            print(f"Early stopping triggered! Hit@10 hasn't improved for {patience} epochs.")
            break

if __name__ == "__main__":
    train_value_network()
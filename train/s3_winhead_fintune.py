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
from src.dataset import 
from src.train import CMTrainer

def finetune_role_head():
    # 1. 初始化模型并加载预训练底座
    num_heroes = get_number_of_heroes(Path("data/hero_id_to_name.json"))
    embed_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=embed_dim).to(device)
    
    # S3 从 S2 的 best role 模型继续训练
    state_dict = torch.load("models/stage2_role_best.pt", map_location=device)
    
    # 3. 再加载到模型中
    model.load_state_dict(state_dict, strict=False)
    print("Pre-trained base loaded (strict=False).")

    # 2. 冻结底座，只练 Role Head
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
        
    # 3. 准备 CM DataLoader（match_role.json）
    train_loader, val_loader = (
        data_path=Path("data/match_role.json"),
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        val_ratio=0.1
    )

    # 4. S3 优化器：分组学习率
    optimizer = torch.optim.Adam([
        {'params': model.win_head.parameters(), 'lr': 1e-3},
        {'params': [model.win_token], 'lr': 1e-3},
        {'params': model.transformer.parameters(), 'lr': 1e-5},
        {'params': model.hero_emb.parameters(), 'lr': 1e-6},
        {'params': model.side_emb.parameters(), 'lr': 1e-6},
        {'params': model.mask_head.parameters(), 'lr': 1e-5},
    ])
    criterion_role = nn.CrossEntropyLoss()
    criterion_win = nn.BCEWithLogitsLoss()

    trainer = CMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_path="models",
    )

    best_val_role_acc = 0.0
    patience = 5  # 如果连续 5 个 Epoch 验证集 Role Acc 不涨，就停止
    patience_counter = 0
    max_epochs = 50 # 把上限设高，反正有早停拦着

    for epoch in range(max_epochs):
        # 1. 训练阶段
        train_loss, train_role_acc = trainer.train_one_epoch(
            train_loader, criterion_role, criterion_win
        )
        
        # 2. 验证阶段 (你需要确保 CMTrainer 里有一个 evaluate 方法)
        # 验证阶段不要计算梯度：with torch.no_grad():
        val_loss, val_role_acc,val_acc= trainer.evaluate(
            val_loader, criterion_role, criterion_win
        )
        
        print(f"Epoch {epoch} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_role_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_role_acc:.2%}")

        # 3. 判断是否是最好的一代，保存模型
        if val_role_acc > best_val_role_acc:
            best_val_role_acc = val_role_acc
            patience_counter = 0 # 重置计数器
            torch.save(model.state_dict(), "models/stage2_role_best.pt")
            print("  --> [New Best Model Saved!]")
        else:
            patience_counter += 1
            print(f"  --> [Patience: {patience_counter}/{patience}]")
            
        # 4. 早停触发
        if patience_counter >= patience:
            print(f"Early stopping triggered! Validation Role Acc hasn't improved for {patience} epochs.")
            break

    print("Fine-tuning complete. Best model is saved as stage2_role_best.pt")

if __name__ == "__main__":
    finetune_role_head()

import random
import torch
from pathlib import Path
from src.utils import get_number_of_heroes
from dota_transformer import DotaMultiTaskTransformer
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DotaRoleDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 英雄 ID [10]
        hero_ids = torch.tensor(item["hero_ids"], dtype=torch.long)
        # 阵营 ID [0,0,0,0,0, 1,1,1,1,1]
        side_ids = torch.tensor([0]*5 + [1]*5, dtype=torch.long)
        # 位置标签 1-5 减去 1 变成 0-4，方便 Loss 计算
        role_labels = torch.tensor([p - 1 for p in item["positions"]], dtype=torch.long)
        win_label = torch.tensor(item["win"], dtype=torch.float) # 0 或 1

        return hero_ids, side_ids, role_labels, win_label

def finetune_role_head():
    # 1. 初始化模型并加载预训练底座
    num_heroes = get_number_of_heroes(Path("data/hero_id_to_name.json"))
    embed_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=embed_dim).to(device)
    
    # 非严格加载，忽略缺失的 role_head 权重
    model.load_state_dict(torch.load("models/dota_bert_warmup_best.pt"), strict=False)
    print("Pre-trained base loaded (strict=False).")

    # 2. 冻结底座，只练 Role Head
    for param in model.parameters():
        param.requires_grad = False
    
    # 开启 Role Head 的梯度
    for param in model.role_head.parameters():
        param.requires_grad = True
        
    # 3. 准备数据
    dataset = DotaRoleDataset("data/role_train_data.json")
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=10)

    # 4. 优化器只针对 role_head
    optimizer = torch.optim.Adam(model.role_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10): # Role Head 收敛很快，通常 5-10 epoch 足够
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for hero_ids, side_ids, role_labels, win_label in loader:
            hero_ids, side_ids, role_labels = hero_ids.to(device), side_ids.to(device), role_labels.to(device)
            win_label = win_label.to(device)

            optimizer.zero_grad()
            _, _, role_logits = model(hero_ids, side_ids) # [Batch, 10, 5]
            
            # 展平进行计算
            loss = criterion(role_logits.view(-1, 5), role_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            preds = torch.argmax(role_logits, dim=-1)
            correct += (preds == role_labels).sum().item()
            total_samples += role_labels.numel()
            
        avg_acc = correct / total_samples
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Role Accuracy: {avg_acc:.2%}")

    # 5. 保存完整模型
    torch.save(model.state_dict(), "models/dota_multitask_full.pt")
    print("Fine-tuning complete. Full model saved.")

if __name__ == "__main__":
    finetune_role_head()
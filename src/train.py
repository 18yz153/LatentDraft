import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class BaseTrainer:
    def __init__(self, model, optimizer, device, save_path):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path

    def save(self, name):
        torch.save(self.model.state_dict(), f"{self.save_path}/{name}.pt")

class WarmupEmbeddingTrainer(BaseTrainer):
    def train_one_epoch(self, loader, criterion):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Warmup Embedding")
        for hero_ids, side_ids, role_labels, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            win_labels = win_labels.to(self.device).float()

            self.optimizer.zero_grad()
            mask_logits, win_logits, _ = self.model(hero_ids, side_ids)

            # 只计算 Mask 任务的 Loss，Win 任务不参与训练
            loss = criterion(mask_logits.view(-1, mask_logits.size(-1)), role_labels.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

class CMTrainer(BaseTrainer):
    def train_one_epoch(self, loader, criterion_role, criterion_win):
        self.model.train() # 开启训练模式 (启用 Dropout 和 BatchNorm)
        total_loss, total_acc = 0, 0
        
        pbar = tqdm(loader, desc="CM Fine-tuning")
        for hero_ids, side_ids, role_labels, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            win_labels = win_labels.to(self.device).float()

            self.optimizer.zero_grad()

            mask_logits, win_logits, role_logits = self.model(hero_ids, side_ids, role_labels)

            # --- 🚨 修正区：将 1-5 映射到 0-4 ---
            role_targets = role_labels - 1 

            # 1. Role Loss
            loss_role = criterion_role(role_logits.view(-1, 5), role_targets.view(-1))
            
            # 2. Win Loss (严谨的 squeeze)
            loss_win = criterion_win(win_logits.squeeze(-1), win_labels)

            # 3. 联合 Loss
            loss = loss_role * 2.0 + loss_win * 1.0 
            
            loss.backward()
            self.optimizer.step()

            # --- 🚨 修正区：Acc 比较也必须用 0-4 ---
            total_loss += loss.item()
            preds = torch.argmax(role_logits, dim=-1)
            acc = (preds == role_targets).float().mean()
            total_acc += acc.item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", role_acc=f"{acc.item():.2%}")

        return total_loss / len(loader), total_acc / len(loader)

    @torch.no_grad() # 绝对不能算梯度
    def evaluate(self, loader, criterion_role, criterion_win):
        self.model.eval() # 极其重要：关闭 Dropout，锁定 BatchNorm 参数
        total_loss, total_acc = 0, 0
        
        # 为了算 AUC，我们需要把验证集里所有的预测概率和真实标签存下来
        all_win_preds = []
        all_win_labels = []
        
        pbar = tqdm(loader, desc="CM Evaluating")
        for hero_ids, side_ids, role_labels, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            win_labels = win_labels.to(self.device).float()

            mask_logits, win_logits, role_logits = self.model(hero_ids, side_ids, role_labels)

            role_targets = role_labels - 1 

            loss_role = criterion_role(role_logits.view(-1, 5), role_targets.view(-1))
            loss_win = criterion_win(win_logits.squeeze(-1), win_labels)
            loss = loss_role * 2.0 + loss_win * 1.0 
            
            total_loss += loss.item()

            preds = torch.argmax(role_logits, dim=-1)
            acc = (preds == role_targets).float().mean()
            total_acc += acc.item()
            
            # 收集 AUC 数据 (将 logits 转为概率)
            win_probs = torch.sigmoid(win_logits.squeeze(-1))
            all_win_preds.extend(win_probs.cpu().numpy())
            all_win_labels.extend(win_labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}", role_acc=f"{acc.item():.2%}")

        # 计算 Validation AUC
        try:
            val_auc = roc_auc_score(all_win_labels, all_win_preds)
        except ValueError:
            # 防止极其罕见的 valid set 只有一种 label 报错
            val_auc = 0.5 
            
        print(f"\n---> Evaluation Results | Val Loss: {total_loss / len(loader):.4f} | Val Role Acc: {total_acc / len(loader):.2%} | Val Win AUC: {val_auc:.4f}\n")

        # 返回时把 AUC 也带上，你外面的 Early Stopping 可以根据 AUC 来判断
        return total_loss / len(loader), total_acc / len(loader), val_auc

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
            
        # 返回时把 AUC 也带上，你外面的 Early Stopping 可以根据 AUC 来判断
        return total_loss / len(loader), total_acc / len(loader), val_auc

class S3WinHeadTrainer(BaseTrainer):
    def train_one_epoch(self, loader, criterion_mask, criterion_win):
        self.model.train()
        # 强制把 role_head 设为 eval，虽然冻结了梯度，但这能关掉它的 Dropout（如果有的话）
        self.model.role_head.eval() 
        
        total_loss, total_win_loss, total_mask_loss = 0, 0, 0
        
        pbar = tqdm(loader, desc="S3 Training (Value Network)")
        # 注意 unpacking 现在的 5 个返回值
        for hero_ids, side_ids, role_labels, target_labels, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            target_labels = target_labels.to(self.device)
            win_labels = win_labels.to(self.device).float()

            self.optimizer.zero_grad()

            # 前向传播：未知的 role_labels(0) 会触发神级 RoleHead 的概率推演
            mask_logits, win_logits, role_logits = self.model(hero_ids, side_ids, role_labels)

            # 1. 核心任务：Win Loss (Value Network)
            known_count = (hero_ids != 0).sum(dim=1).float() 
            # 线性权重：10个英雄满权重1.0，越少权重越低 (缓解残局高方差)
            win_loss_weights = 0.3 + 0.7 * (known_count / 10.0)
            
            # 🚨 关键：使用 reduction='none' 获取每个样本的独立 Loss
            raw_win_loss = F.binary_cross_entropy_with_logits(
                win_logits.squeeze(-1), 
                win_labels.squeeze(-1), 
                reduction='none'
            )
            # 逐个样本乘上权重后，再求整个 Batch 的平均值
            loss_win = (raw_win_loss * win_loss_weights).mean()

            # 2. 辅助任务：Mask Loss
            # criterion_mask 必须是 nn.CrossEntropyLoss(ignore_index=-100)
            # 只有被挖空的位置 target_labels 才不是 -100，才会计算 Loss
            loss_mask = criterion_mask(
                mask_logits.view(-1, mask_logits.size(-1)), 
                target_labels.view(-1)
            )

            # 3. 联合 Loss：Win 占绝对主导，Mask 提供底线防坍塌约束 (权重 0.05 ~ 0.1)
            loss = loss_win + loss_mask * 0.05

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_win_loss += loss_win.item()
            total_mask_loss += loss_mask.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                w_loss=f"{loss_win.item():.4f}", 
                m_loss=f"{loss_mask.item():.4f}"
            )

        steps = len(loader)
        return total_loss / steps, total_win_loss / steps, total_mask_loss / steps

    @torch.no_grad()
    def evaluate(self, loader, criterion_mask, criterion_win):
        self.model.eval()
        total_loss, total_win_loss = 0, 0
        
        all_win_preds = []
        all_win_labels = []
        
        pbar = tqdm(loader, desc="S3 Evaluating")
        for hero_ids, side_ids, role_labels, target_labels, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            target_labels = target_labels.to(self.device)
            win_labels = win_labels.to(self.device).float()

            mask_logits, win_logits, role_logits = self.model(hero_ids, side_ids, role_labels)

            loss_win = criterion_win(win_logits.squeeze(-1), win_labels.squeeze(-1))
            loss_mask = criterion_mask(mask_logits.view(-1, mask_logits.size(-1)), target_labels.view(-1))
            loss = loss_win + loss_mask * 0.05
            
            total_loss += loss.item()
            total_win_loss += loss_win.item()

            # 收集 AUC 数据
            # 注意：如果 win_label 是 0.75，依然可以算 AUC，但 threshold 逻辑需要注意。
            # 通常 ROC-AUC 更倾向于二进制 label。为了算纯净的 AUC，我们把 0.75 当作 1，0.25 当作 0
            binary_labels = (win_labels > 0.5).float()
            win_probs = torch.sigmoid(win_logits.squeeze(-1))
            
            all_win_preds.extend(win_probs.cpu().numpy())
            all_win_labels.extend(binary_labels.squeeze(-1).cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        try:
            val_auc = roc_auc_score(all_win_labels, all_win_preds)
        except ValueError:
            val_auc = 0.5 
            
        return total_loss / len(loader), val_auc
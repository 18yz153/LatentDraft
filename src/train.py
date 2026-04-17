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
    def train_one_epoch(self, loader, criterion_role, criterion_mask, mask_weight=0.2):
        self.model.train() # 开启训练模式 (启用 Dropout 和 BatchNorm)
        total_loss, total_acc = 0, 0
        
        pbar = tqdm(loader, desc="CM Fine-tuning")
        for hero_ids, side_ids, role_labels, hero_ids_full, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            hero_ids_full = hero_ids_full.to(self.device)

            self.optimizer.zero_grad()

            mask_logits, _, role_logits = self.model(hero_ids, side_ids)


            # --- 🚨 修正区：将 1-5 映射到 0-4 ---
            role_targets = role_labels - 1 

            # 1. Role Loss
            loss_role = criterion_role(role_logits.view(-1, 5), role_targets.view(-1))

            # 2. Mask Loss: 仅在被挖空位置计算英雄重建损失
            mask_targets = torch.where(
                hero_ids == 0,
                hero_ids_full,
                torch.full_like(hero_ids_full, -100),
            )
            loss_mask = criterion_mask(
                mask_logits.view(-1, mask_logits.size(-1)),
                mask_targets.view(-1),
            )
            

            # 3. 联合 Loss
            loss = loss_role + loss_mask * mask_weight
            
            loss.backward()
            self.optimizer.step()

            # --- 🚨 修正区：Acc 比较也必须用 0-4 ---
            total_loss += loss.item()
            preds = torch.argmax(role_logits, dim=-1)
            valid_role_mask = role_targets >= 0
            if valid_role_mask.any():
                acc = (preds[valid_role_mask] == role_targets[valid_role_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=preds.device)
            total_acc += acc.item()
                        
            pbar.set_postfix(loss=f"{loss.item():.4f}", role=f"{loss_role.item():.4f}", mask=f"{loss_mask.item():.4f}", role_acc=f"{acc.item():.2%}")

        return total_loss / len(loader), total_acc / len(loader)

    @torch.no_grad() # 绝对不能算梯度
    def evaluate(self, loader, criterion_role, criterion_mask, mask_weight=0.2):
        self.model.eval() # 极其重要：关闭 Dropout，锁定 BatchNorm 参数
        total_loss, total_acc, total_mask_loss = 0, 0, 0
        
        pbar = tqdm(loader, desc="CM Evaluating")
        for hero_ids, side_ids, role_labels, hero_ids_full, win_labels in pbar:
            hero_ids = hero_ids.to(self.device)
            side_ids = side_ids.to(self.device)
            role_labels = role_labels.to(self.device)
            hero_ids_full = hero_ids_full.to(self.device)

            mask_logits, _, role_logits = self.model(hero_ids, side_ids)

            role_targets = role_labels - 1 

            loss_role = criterion_role(role_logits.view(-1, 5), role_targets.view(-1))
            mask_targets = torch.where(
                hero_ids == 0,
                hero_ids_full,
                torch.full_like(hero_ids_full, -100),
            )
            loss_mask = criterion_mask(
                mask_logits.view(-1, mask_logits.size(-1)),
                mask_targets.view(-1),
            )
            loss = loss_role + loss_mask * mask_weight
            
            total_loss += loss.item()
            total_mask_loss += loss_mask.item()

            preds = torch.argmax(role_logits, dim=-1)
            valid_role_mask = role_targets >= 0
            if valid_role_mask.any():
                acc = (preds[valid_role_mask] == role_targets[valid_role_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=preds.device)
            total_acc += acc.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", role=f"{loss_role.item():.4f}", mask=f"{loss_mask.item():.4f}", role_acc=f"{acc.item():.2%}")

        return total_loss / len(loader), total_acc / len(loader), total_mask_loss / len(loader)

class S3WinHeadTrainer(BaseTrainer):
    def __init__(self, model, optimizer, device, save_path):
        super().__init__(model, optimizer, device, save_path)
        self.scaler = torch.amp.GradScaler(device='cuda')
    def train_one_epoch(self, loader, criterion_mask, alpha=1.0, beta=0.1):
        self.model.train()
        # 强制把 role_head 设为 eval，虽然冻结了梯度，但这能关掉它的 Dropout（如果有的话）
        self.model.role_head.eval() 
        
        total_loss, total_win_loss, total_infonce_loss, total_masked_loss = 0, 0, 0, 0
        
        pbar = tqdm(loader, desc="S3 Training (Hybrid Loss)")
        
        for batch in pbar:
            # 1. 搬运数据到设备
            # 这里对应你 Dataset 返回的 dict keys
            masked_seq = batch["masked_seq"].to(self.device)
            side_ids = batch["side_ids"].to(self.device)
            role_labels = batch["role_labels"].to(self.device)
            win_labels = batch["win_label"].to(self.device).float().unsqueeze(-1) # [B, 1]
            full_seq = batch["full_seq"].to(self.device)
            fill_pos = batch["fill_pos"].to(self.device)
            candidates = batch["candidates"].to(self.device) # [B, N_cands]

            
            B, N_cands = candidates.shape
            self.optimizer.zero_grad()

            # ==========================================
            # 第一部分：原生阵容前向传播 (BCE + Mask)
            # ==========================================
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                mask_logits, win_logits, _ = self.model(masked_seq, side_ids, role_labels)

                # 1.1 BCE Win Loss
                loss_win = F.binary_cross_entropy_with_logits(win_logits, win_labels)
                mask_targets = torch.where(
                    masked_seq == 0,
                    full_seq,
                    torch.full_like(full_seq, -100),
                )
                loss_mask = criterion_mask(
                    mask_logits.view(-1, mask_logits.size(-1)),
                    mask_targets.view(-1),
                )


                # ==========================================
                # 第二部分：平行宇宙前向传播 (InfoNCE)
                # ==========================================
                # 动态构建填坑阵容：复制 win_seq 并将 fill_pos 处设为 0 准备填入 candidates
                # 注意：此处我们直接在 win_seq 的副本上操作
                rank_seq_expand = masked_seq.unsqueeze(1).expand(-1, N_cands, -1).clone()
                rank_roles_expand = role_labels.unsqueeze(1).expand(-1, N_cands, -1)
                side_ids_expand = side_ids.unsqueeze(1).expand(-1, N_cands, -1)

                # 生成索引进行批量填坑
                batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, N_cands)
                cand_idx = torch.arange(N_cands, device=self.device).unsqueeze(0).expand(B, -1)
                f_pos_expand = fill_pos.unsqueeze(1).expand(-1, N_cands)

                # 填入候选人
                rank_seq_expand[batch_idx, cand_idx, f_pos_expand] = candidates
                
                # 展平 Mega-Batch
                rank_seq_flat = rank_seq_expand.reshape(-1, 10)
                rank_roles_flat = rank_roles_expand.reshape(-1, 10)
                side_ids_flat = side_ids_expand.reshape(-1, 10)

                # 平行宇宙前向传播 (仅取 win_logits)
                _, cand_logits_flat, _ = self.model(rank_seq_flat, side_ids_flat, rank_roles_flat)
                cand_logits = cand_logits_flat.view(B, N_cands)

                # InfoNCE 视角翻转逻辑：WinHead 永远预测左侧胜率
                # 若左赢(1)，则 logit 越大越好；若右赢(0)，则 -logit 越大越好
                signs = (win_labels * 2 - 1) # [B, 1] -> 1 or -1
                adjusted_logits = cand_logits * signs

                # 真正的 target_hero 永远在 candidates 的 index 0
                target_idx = torch.zeros(B, dtype=torch.long, device=self.device)
                loss_infonce = F.cross_entropy(adjusted_logits, target_idx)

                # ==========================================
                # 第三部分：联合 Loss
                # ==========================================
                # Win 是主任务，InfoNCE 是强化，Mask 是底线
                loss = loss_win + (0 * loss_infonce) + (0 * loss_mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 统计
            total_loss += loss.item()
            total_win_loss += loss_win.item()
            total_infonce_loss += loss_infonce.item()
            total_masked_loss += loss_mask.item()
            pbar.set_postfix(
                TOTAL=f"{loss.item():.3f}", 
                W=f"{loss_win.item():.3f}", 
                Info=f"{loss_infonce.item():.3f}",
                M=f"{loss_mask.item():.3f}"
            )

        steps = len(loader)
        return (total_loss / steps, total_win_loss / steps, 
                total_infonce_loss / steps, total_masked_loss / steps)

    @torch.no_grad()
    def evaluate(self, loader, criterion_mask, criterion_win):
        self.model.eval()
        total_loss, total_win_loss = 0, 0
        
        all_win_preds = []
        all_win_labels = []
        
        pbar = tqdm(loader, desc="S3 Evaluating")
        for batch in pbar:
            masked_seq = batch["masked_seq"].to(self.device)
            side_ids = batch["side_ids"].to(self.device)
            role_labels = batch["role_labels"].to(self.device)
            win_labels = batch["win_label"].to(self.device).float().unsqueeze(-1)
            full_seq = batch["full_seq"].to(self.device)

            mask_logits, win_logits, role_logits = self.model(masked_seq, side_ids, role_labels)

            loss_win = criterion_win(win_logits.squeeze(-1), win_labels.squeeze(-1))
            # loss_mask = criterion_mask(mask_logits.view(-1, mask_logits.size(-1)), target_labels.view(-1))
            loss = loss_win 
            
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
    
    @torch.no_grad()
    def business_evaluate(self, loader):
        self.model.eval()
        
        total_queries = 0
        mask_hit_at_30 = 0
        mask_hit_at_10 = 0
        mask_hit_at_5 = 0
        hit_at_5 = 0
        hit_at_10 = 0

        pbar = tqdm(loader, desc="S3 Business Evaluating (Mega-Batch)")
        
        with torch.no_grad():
            for batch in pbar:
                seq = batch['masked_seq'].to(self.device)      # [B, 10]
                side = batch['side_ids'].to(self.device)       # [B, 10]
                role = batch['role_labels'].to(self.device)    # [B, 10]
                full_seq = batch['full_seq'].to(self.device)   # 🌟 必须拿到完整阵容
                fill_pos = batch['fill_pos'].to(self.device)   # [B]
                win_labels = batch['win_label'].to(self.device).float().unsqueeze(-1)
                
                B = seq.size(0)
                total_queries += B
                batch_idx = torch.arange(B, device=self.device)

                # ==========================================
                # 🌟 全新逻辑：构建“同阵营缺失英雄”的目标集合
                # ==========================================
                # 1. 确定当前在给哪一边选人 (左0-4，右5-9)
                seq[batch_idx, fill_pos] = 0
                role[batch_idx, fill_pos] = 0
                is_left = (fill_pos < 5).unsqueeze(1) # [B, 1]
                slot_idx = torch.arange(10, device=self.device).unsqueeze(0) # [1, 10]
                is_allied_slot = (slot_idx < 5) == is_left # [B, 10] 判定哪些坑位是友军
                
                # 2. 找到所有友军空缺的位置 (seq == 0)
                is_missing = (seq == 0) # [B, 10]
                valid_target_mask = is_allied_slot & is_missing # [B, 10]
                
                # 3. 从 full_seq 里提取这些位置上的真实英雄 ID
                # 把不是目标英雄的位置置为 0
                valid_heroes = torch.where(valid_target_mask, full_seq, torch.zeros_like(full_seq))
                
                # ==========================================
                # Step 1: 绝对防漏的 Mask 召回
                # ==========================================

                mask_logits, _, _ = self.model(seq, side, role) # [B, 10, Vocab]
                pos_logits = mask_logits[batch_idx, fill_pos]   # [B, Vocab]
                vocab_size = pos_logits.size(-1)

                # 🌟 4. 构造目标 Multi-hot 矩阵 [B, Vocab]
                # 这张表里，只要是友军缺失的英雄，对应的 index 就是 True
                target_multihot = torch.zeros(B, vocab_size, device=self.device, dtype=torch.bool)
                target_multihot.scatter_(1, valid_heroes, True)

                # ==========================================
                # Step 2: 极速去重拿 Top 30
                # ==========================================
                banned_indices = seq.clone() 
                pad_tensor = torch.zeros((B, 1), dtype=torch.long, device=self.device)
                banned_indices = torch.cat([banned_indices, pad_tensor], dim=1)
                
                pos_logits.scatter_(1, banned_indices, -float('inf'))
                _, batch_candidates = torch.topk(pos_logits, 30, dim=-1) # [B, 30]
                mask_top10 = batch_candidates[:, :10]
                mask_top5 = batch_candidates[:, :5]

                # 🌟 5. 统计 Mask Hit@30 (只要推荐在目标集合里就算 Hit！)
                # .gather 瞬间查出 batch_candidates 里每个英雄是否在 target_multihot 里
                mask_hits = target_multihot.gather(1, batch_candidates).any(dim=1) # [B]
                mask_hits_10 = target_multihot.gather(1, mask_top10).any(dim=1)
                mask_hits_5  = target_multihot.gather(1, mask_top5).any(dim=1)
                mask_hit_at_30 += mask_hits.sum().item()
                mask_hit_at_10 += mask_hits_10.sum().item() # 在外面记得初始化
                mask_hit_at_5  += mask_hits_5.sum().item()  # 在外面记得初始化

                # ==========================================
                # Step 3: Mega-Batch 批量精排 (WinHead)
                # ==========================================
                N_cands = 30
                seq_expand = seq.unsqueeze(1).expand(-1, N_cands, -1).clone()
                role_expand = role.unsqueeze(1).expand(-1, N_cands, -1)
                side_expand = side.unsqueeze(1).expand(-1, N_cands, -1)
                
                batch_idx_exp = batch_idx.unsqueeze(1).expand(-1, N_cands)
                cand_idx_exp = torch.arange(N_cands, device=self.device).unsqueeze(0).expand(B, -1)
                f_pos_expand = fill_pos.unsqueeze(1).expand(-1, N_cands)
                
                seq_expand[batch_idx_exp, cand_idx_exp, f_pos_expand] = batch_candidates
                
                seq_flat = seq_expand.reshape(-1, 10)
                role_flat = role_expand.reshape(-1, 10)
                side_flat = side_expand.reshape(-1, 10)
                
                _, cand_win_logits_flat, _ = self.model(seq_flat, side_flat, role_flat)
                cand_win_logits = cand_win_logits_flat.view(B, N_cands)

                # ==========================================
                # Step 4: 视角对齐与排序统计
                # ==========================================
                signs = torch.where(fill_pos < 5, 1.0, -1.0).unsqueeze(1)
                rerank_scores = cand_win_logits * signs 
                
                _, sorted_idx = torch.sort(rerank_scores, descending=True, dim=1)
                
                top5_idx = sorted_idx[:, :5]
                top10_idx = sorted_idx[:, :10]
                
                top5_cands = torch.gather(batch_candidates, 1, top5_idx)
                top10_cands = torch.gather(batch_candidates, 1, top10_idx)
                
                # 🌟 6. 统计精排的 Hit@5 和 Hit@10 (同样使用目标集合判定)
                hit_at_5 += target_multihot.gather(1, top5_cands).any(dim=1).sum().item()
                hit_at_10 += target_multihot.gather(1, top10_cands).any(dim=1).sum().item()

                pbar.set_postfix(
                    m30=f"{mask_hit_at_30 / max(total_queries, 1):.2%}", 
                    h5=f"{hit_at_5 / max(total_queries, 1):.2%}", 
                    h10=f"{hit_at_10 / max(total_queries, 1):.2%}"
                )

        total_queries = max(total_queries, 1)
        return {
            "mask_hit_at_30": mask_hit_at_30 / total_queries,
            "mask_hit_at_10": mask_hit_at_10 / total_queries,
            "mask_hit_at_5": mask_hit_at_5 / total_queries,
            "hit_at_5": hit_at_5 / total_queries,
            "hit_at_10": hit_at_10 / total_queries,
            "queries": total_queries,
        }
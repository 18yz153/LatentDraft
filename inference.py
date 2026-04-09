import torch
import numpy as np
import xgboost as xgb
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict

import torch.nn as nn
import torch.nn.functional as F

class TransformerInferenceEngine(nn.Module):
    def __init__(self, model_path, num_heroes, embed_dim=64, nhead=8, num_layers=3):
        super().__init__()
        # 必须与训练时的参数完全一致，否则 load_state_dict 会报错
        self.hero_emb = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)
        self.side_emb = nn.Embedding(2, embed_dim)

        self.win_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 重点：手动定义 layers 列表。PyTorch 会把训练好的 
        # self.transformer.layers.0 ... 自动映射到这里的 self.layers.0
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])

        self.mask_head = nn.Linear(embed_dim, num_heroes + 1)
        self.win_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 加载 4070 炼出的丹
        state_dict = torch.load(model_path, map_location="cpu")
        
        # 核心技巧：如果训练时用了 nn.TransformerEncoder，Key 会带有 "transformer." 前缀
        # 我们需要处理一下 Key 映射到 self.layers
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("transformer.layers."):
                new_state_dict[k.replace("transformer.layers.", "layers.")] = v
            else:
                new_state_dict[k] = v
        
        self.load_state_dict(new_state_dict)
        self.eval()

    def forward_with_attn(self, hero_ids, side_ids, mask_pos=None):
        """特殊的 forward，返回胜率的同时返回最后一层 attention"""
        batch_size = hero_ids.shape[0]
        x = self.hero_emb(hero_ids) + self.side_emb(side_ids)
        w_token = self.win_token.expand(batch_size, -1, -1)
        x = torch.cat([w_token, x], dim=1)
        win_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        hero_mask = (hero_ids == 0)
        full_mask = torch.cat([win_mask, hero_mask], dim=1)
        
        features = x
        last_weights = None
        
        for i, layer in enumerate(self.layers):
            # 手动执行 MultiheadAttention 并强行索要 weights
            # 这一步会跳过训练时的加速路径，但推理 10 个英雄完全没延迟
            attn_output, weights = layer.self_attn(
                features, features, features, 
                key_padding_mask=full_mask,
                need_weights=True 
            )
            
            features = layer.norm1(features + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(F.relu(layer.linear1(features))))
            features = layer.norm2(features + layer.dropout2(ff_output))
            
            if i == len(self.layers) - 1:
                # 此时 weights 的形状是 [Batch, 11, 11]
                # [0, :] 这一行代表了 win_token 对全场 10 个英雄的关注度
                last_weights = weights 

        mask_logits = None
        if mask_pos is not None:
            # mask_pos 是 0-9，对应 features 里的索引是 1-10
            # 我们可以通过 gather 拿到对应的 token 特征
            # indices 形状为 [Batch, 1, Hidden_Dim]
            idx = (mask_pos + 1).view(batch_size, 1, 1).expand(-1, -1, features.size(-1))
            target_feat = features.gather(1, idx).squeeze(1) # [Batch, Hidden_Dim]
            mask_logits = self.mask_head(target_feat) # [Batch, Num_Heroes + 1]
        else:
            # 如果没有指定特定位置，通常返回全场的 mask 预测（用于展示建议）
            # 取除了 win_token 以外的所有位置 [Batch, 10, Hidden_Dim]
            all_hero_feats = features[:, 1:, :]
            mask_logits = self.mask_head(all_hero_feats) # [Batch, 10, Num_Heroes + 1]
        # 4. 【核心修改】：只取第 0 个位置的特征给 win_head
        global_feat = features[:, 0, :] 
        win_logits = self.win_head(global_feat)

        
        return torch.sigmoid(win_logits),mask_logits, last_weights


class BaseInference(ABC):
    @abstractmethod
    def recommend(self, current_ally, current_enemy, valid_hero_ids, mode="pick", topk=10):
        pass
    @abstractmethod
    def get_explanation(self, candidate_id, current_ally, current_enemy, is_ally=True):
        pass
    @abstractmethod
    def get_full_analysis(self, hero_ids, side_ids):
        pass

# --- XGBoost 适配器 ---
class XGBInference(BaseInference):
    def __init__(self, model_path, hero_embeddings):
        self.bst = xgb.Booster()
        self.bst.load_model(model_path)
        self.embeddings = hero_embeddings

    def recommend(self, current_ally, current_enemy, valid_hero_ids, mode="pick", topk=10):
        v_a = np.mean([self.embeddings[h] for h in current_ally], axis=0) if current_ally else np.zeros(64)
        v_e = np.mean([self.embeddings[h] for h in current_enemy], axis=0) if current_enemy else np.zeros(64)
        
        base_feat = np.concatenate([v_a, v_e]) if mode == "pick" else np.concatenate([v_e, v_a])
        used = set(current_ally + current_enemy)
        candidates = [h for h in valid_hero_ids if h not in used]
        
        feats = [np.concatenate([base_feat, self.embeddings[h]]) for h in candidates]
        probs = self.bst.predict(xgb.DMatrix(np.array(feats)))
        return sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)[:topk]
    def get_explanation(self, target_hero_id, current_ally, current_enemy):
        # 1. 计算基准分数
        v_a = np.mean([self.embeddings[h] for h in current_ally], axis=0) if current_ally else np.zeros(64)
        v_e = np.mean([self.embeddings[h] for h in current_enemy], axis=0) if current_enemy else np.zeros(64)
        base_feat = np.concatenate([v_a, v_e, self.embeddings[target_hero_id]])
        base_prob = self.bst.predict(xgb.DMatrix(np.array([base_feat])))[0]

        enemy_deltas = []
        for enemy in current_enemy:
            temp_enemy = [e for e in current_enemy if e != enemy]
            temp_v_e = np.mean([self.embeddings[h] for h in temp_enemy], axis=0) if temp_enemy else np.zeros(64)
            temp_feat = np.concatenate([v_a, temp_v_e, self.embeddings[target_hero_id]])
            temp_prob = self.bst.predict(xgb.DMatrix(np.array([temp_feat])))[0]
            enemy_deltas.append((enemy, float(base_prob - temp_prob), float(base_prob - temp_prob)))

        ally_deltas = []
        for ally in current_ally:
            temp_ally = [a for a in current_ally if a != ally]
            temp_v_a = np.mean([self.embeddings[h] for h in temp_ally], axis=0) if temp_ally else np.zeros(64)
            temp_feat = np.concatenate([temp_v_a, v_e, self.embeddings[target_hero_id]])
            temp_prob = self.bst.predict(xgb.DMatrix(np.array([temp_feat])))[0]
            ally_deltas.append((ally, float(base_prob - temp_prob), float(base_prob - temp_prob)))

        # 排序
        enemy_deltas.sort(key=lambda x: x[1], reverse=True)
        ally_deltas.sort(key=lambda x: x[1], reverse=True)

        return enemy_deltas, ally_deltas

    def get_full_analysis(self, hero_ids, side_ids):
        # XGBoost 模型没有注意力机制，这个接口没法实现
        return None, None
    
# --- Transformer 适配器 ---
class TransformerInference(BaseInference):
    def __init__(self, model_path, num_heroes, device="cpu"):
        self.device = torch.device(device)
        # 使用我们刚定义的拆解版 Engine
        self.engine = TransformerInferenceEngine(model_path, num_heroes=num_heroes)
        self.engine.to(self.device)
        self.engine.eval()

    def recommend(self, current_ally, current_enemy, valid_hero_ids, mode="pick", topk=10):
        used = set(current_ally + current_enemy)
        temp_a = (current_ally + [0]*5)[:5]
        temp_e = (current_enemy + [0]*5)[:5]
        lineup = temp_a + temp_e
        is_ally = (mode == "pick")
        if is_ally:
            mask_idx_in_seq = len(current_ally) 
            if mask_idx_in_seq >= 5:
                return [] # 已经满员了，没位置了
        else:
            mask_idx_in_seq = 5 + len(current_enemy)
            if mask_idx_in_seq >= 10:
                return [] # 已经满员了，没位置了

        
        seq = torch.tensor([lineup]).to(self.device)
        sides = torch.tensor([[0]*5 + [1]*5]).to(self.device)
        with torch.no_grad():
            _, mask_logits, _ = self.engine.forward_with_attn(seq, sides)
            probs = torch.softmax(mask_logits[0, mask_idx_in_seq], dim=-1)
        candidate_probs, candidate_ids = torch.topk(probs, k=50)
        print("Raw candidates:", list(zip(candidate_ids.cpu().numpy(), candidate_probs.cpu().numpy())))
        refined_candidates = []
        for p, h_id in zip(candidate_probs.tolist(), candidate_ids.tolist()):
            if h_id in valid_hero_ids and h_id not in used:
                refined_candidates.append(h_id)
            if len(refined_candidates) >= 30: # 拿 30 个种子选手去精选
                break
        print("Refined candidates:", refined_candidates)
        final_lineups = []
        for h in refined_candidates:
            case_lineup = list(lineup)
            case_lineup[mask_idx_in_seq] = h
            final_lineups.append(case_lineup)
        batch_seq = torch.tensor(final_lineups).to(self.device)
        batch_sides = torch.tensor([[0]*5 + [1]*5] * len(final_lineups)).to(self.device)
        
        with torch.no_grad():
            win_probs, _, _ = self.engine.forward_with_attn(batch_seq, batch_sides)
            win_probs = win_probs.squeeze().cpu().numpy()       

        final_results = []
        # 如果只有一个候选人，win_probs 会变成标量，处理一下
        if len(refined_candidates) == 1:
            final_results.append((refined_candidates[0], float(win_probs)))
        else:
            for h, wp in zip(refined_candidates, win_probs):
                final_results.append((h, float(wp)))         
        if mode == "pick":
            return sorted(final_results, key=lambda x: x[1], reverse=True)[:topk]
        else:
            return sorted(final_results, key=lambda x: x[1], reverse=False)[:topk]

    def get_explanation(self, target_hero_id, current_ally, current_enemy):
        # 1. 基础阵容构造 [Ally x 5, Enemy x 5]
        temp_a = (current_ally + [target_hero_id] + [0]*5)[:5]
        temp_e = (current_enemy + [0]*5)[:5]
        full_lineup = temp_a + temp_e
        target_pos = len(current_ally)
        
        # 2. 构造 Batch 一次性跑完所有可能性
        # Batch[0]: 完整阵容
        # Batch[1-n]: 依次踢掉某个英雄后的阵容
        inference_list = [full_lineup]
        
        # 记录每个位置对应的英雄ID (方便后续对应)
        active_positions = [i for i, h in enumerate(full_lineup) if h != 0 and i != target_pos]
        
        for pos in active_positions:
            modified = list(full_lineup)
            modified[pos] = 0 # 踢掉这个英雄
            inference_list.append(modified)
            
        # 转换为 Tensor (Batch 并行计算)
        batch_seq = torch.tensor(inference_list).to(self.device)
        batch_sides = torch.tensor([[0]*5 + [1]*5] * len(inference_list)).to(self.device)
        
        with torch.no_grad():
            # 这里只需拿第一行的 weights (完整阵容的注意力)
            probs,_, all_weights = self.engine.forward_with_attn(batch_seq, batch_sides)
            
        base_prob = probs[0].item()
        attn_row = all_weights[0, target_pos].cpu().numpy() # [10]
        
        # 3. 结果组装
        ally_results = []
        enemy_results = []
        
        # 遍历所有扰动过的结果
        for idx, pos in enumerate(active_positions):
            h_id = full_lineup[pos]
            attn_val = float(attn_row[pos])
            # 扰动后的胜率结果在 probs[1:] 中
            mutated_prob = probs[idx + 1].item()
            
            # 计算 delta: 
            # 对于盟友: base - mutated > 0 代表他在场更有利 -> 配合
            # 对于敌人: base - mutated > 0 代表他在场反而赢面大 -> 克制 (因为踢掉他胜率反而降了)
            delta = base_prob - mutated_prob
            
            # 自动打标逻辑
            if pos < 5: # 盟友区
                ally_results.append((h_id, attn_val, delta))
            else: # 敌人区
                # 这里的逻辑：如果踢掉敌人胜率降了(delta > 0)，说明他在场我方更有利 -> 克制
                enemy_results.append((h_id, attn_val, delta))
        
        # 4. 排序 (按 Attention 权重，因为注意力代表了关联的强度)
        ally_results.sort(key=lambda x: x[1], reverse=True)
        enemy_results.sort(key=lambda x: x[1], reverse=True)
        
        return enemy_results, ally_results
    
    def get_full_analysis(self, hero_ids_list, side_ids_list):
        """
        hero_ids_list: 长度为 10 的 list
        side_ids_list: 长度为 10 的 list
        """
        # 1. 转换为 Tensor 并移动到正确的设备
        hero_ids = torch.tensor([hero_ids_list], dtype=torch.long).to(self.device)
        side_ids = torch.tensor([side_ids_list], dtype=torch.long).to(self.device)
        
        # 2. 运行模型拿到全量数据 [1, 11, 11]
        # 使用 torch.no_grad() 确保推理时不计算梯度，减少内存占用
        with torch.no_grad():
            win_prob,_, last_weights = self.engine.forward_with_attn(hero_ids, side_ids)
            
            # 【核心修正】：增加 .detach() 以剥离梯度
            attn_matrix = last_weights[0].detach().cpu().numpy() # [11, 11]
            
            # 3. 提取英雄间 10x10 关注度 (跳过位置 0 的 win_token)
            hero_to_hero_attn = attn_matrix[1:11, 1:11] 
            
            # 4. 抖动分析 (Delta)
            deltas = []
            base_prob = win_prob.item()
            
            for i in range(10):
                # 只有当该位置有英雄时才进行抖动 (如果是 0 则跳过)
                if hero_ids_list[i] == 0:
                    deltas.append(0.0)
                    continue
                    
                temp_hero_ids = hero_ids.clone()
                temp_hero_ids[0, i] = 0 # 模拟该位置英雄消失 (置为 0)
                
                # 重新计算胜率
                new_prob, _ ,_ = self.engine.forward_with_attn(temp_hero_ids, side_ids)
                delta = new_prob.item() - base_prob
                deltas.append(delta)

        # 返回百分比矩阵和 Delta 列表
        return hero_to_hero_attn * 100, deltas, base_prob
    
def load_embedding_payload(path: Path) -> Tuple[List[int], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("embedding file must be a dict payload")

    if "hero_pool" in payload and "embedding" in payload:
        hero_pool = [int(x) for x in payload["hero_pool"]]
        embedding = payload["embedding"]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        return hero_pool, embedding

    if "hero_pool" in payload and "state_dict" in payload:
        hero_pool = [int(x) for x in payload["hero_pool"]]
        state_dict = payload["state_dict"]
        if "hero_emb.weight" not in state_dict:
            raise ValueError("state_dict does not contain hero_emb.weight")
        return hero_pool, state_dict["hero_emb.weight"].detach().cpu()

    raise ValueError("unknown embedding payload format")


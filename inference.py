import torch
import numpy as np
import xgboost as xgb
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from src.model import DotaMultiTaskTransformer

class TransformerInferenceEngine(DotaMultiTaskTransformer):
    def __init__(self, model_path, num_heroes, embed_dim=64, nhead=8, num_layers=3):
        # 1. 直接调用父类的 __init__，构建原汁原味的模型结构！
        super().__init__(
            num_heroes=num_heroes, 
            embed_dim=embed_dim, 
            nhead=nhead, 
            num_layers=num_layers
        )

        # 2. 完美加载：因为结构 100% 一致，不需要再魔改 Key 了，直接 load！
        state_dict = torch.load(model_path, map_location="cpu")
        self.load_state_dict(state_dict)
        self.eval()

    def forward_with_attn(self, hero_ids, side_ids, role_labels=None, mask_pos=None):
        """特殊的 forward，返回胜率的同时返回最后一层 attention"""
        batch_size = hero_ids.shape[0]
        x = self.hero_emb(hero_ids) + self.side_emb(side_ids)
        
        # 把单个 win_token 扩展到当前 batch_size
        w_token = self.win_token.expand(batch_size, -1, -1)
        x = torch.cat([w_token, x], dim=1)
        
        win_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        hero_mask = (hero_ids == 0)
        full_mask = torch.cat([win_mask, hero_mask], dim=1)
        
        features = x
        last_weights = None
        
        for i, layer in enumerate(self.transformer.layers):
            attn_output, weights = layer.self_attn(
                features, features, features, 
                key_padding_mask=full_mask,
                need_weights=True 
            )
            features = layer.norm1(features + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(F.relu(layer.linear1(features))))
            features = layer.norm2(features + layer.dropout2(ff_output))
            
            if i == len(self.transformer.layers) - 1:
                last_weights = weights 

        # 剥离特征
        hero_features = features[:, 1:, :]
        global_features = features[:, 0, :]
        
        # 获取基础 Logits
        role_logits = self.role_head(hero_features)
        
        if mask_pos is not None:
            idx = (mask_pos + 1).view(batch_size, 1, 1).expand(-1, -1, features.size(-1))
            target_feat = features.gather(1, idx).squeeze(1)
            mask_logits = self.mask_head(target_feat)
        else:
            mask_logits = self.mask_head(hero_features)

        # 🌟 修复 2：完美复刻 Soft-Slotting (物理入座与冲突报警器)
        if role_labels is None:
            role_labels = torch.zeros((batch_size, 10), dtype=torch.long, device=hero_ids.device)
            
        role_probs = F.softmax(role_logits, dim=-1) 
        
        hard_role_probs = torch.zeros_like(role_probs)
        valid_mask = (role_labels > 0) 
        safe_indices = torch.clamp(role_labels - 1, min=0)
        hard_role_probs.scatter_(2, safe_indices.unsqueeze(-1), 1.0)
        
        valid_mask_expanded = valid_mask.unsqueeze(-1)
        final_role_features = torch.where(valid_mask_expanded, hard_role_probs, role_probs)

        # 阵营分割与矩阵乘法 (bmm)
        rad_features = hero_features[:, :5, :] 
        dire_features = hero_features[:, 5:, :] 
        rad_roles = final_role_features[:, :5, :] 
        dire_roles = final_role_features[:, 5:, :] 
        
        rad_slots = torch.bmm(rad_roles.transpose(1, 2), rad_features) 
        dire_slots = torch.bmm(dire_roles.transpose(1, 2), dire_features)

        structured_features = torch.cat([rad_slots, dire_slots], dim=1).view(batch_size, -1)
        rad_counts = rad_roles.sum(dim=1) 
        dire_counts = dire_roles.sum(dim=1) 
        occupancy = torch.cat([rad_counts, dire_counts], dim=-1) 
        
        # 拼接 714 维终极特征
        combined_features = torch.cat([
            global_features,     # [Batch, 64] 大局观
            structured_features, # [Batch, 640] 1-5号位对阵图
            occupancy            # [Batch, 10] 冲突/空位报警器
        ], dim=-1)
        
        # 喂给 WinHead
        win_logits = self.win_head(combined_features)
        
        return torch.sigmoid(win_logits), mask_logits, role_logits, last_weights


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
        self._recommend_cache: "OrderedDict[Tuple, List[Tuple[int, float]]]" = OrderedDict()
        self._explanation_cache: "OrderedDict[Tuple, Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]]" = OrderedDict()
        self._analysis_cache: "OrderedDict[Tuple, Tuple[np.ndarray, List[float], float]]" = OrderedDict()
        self._cache_max_size = 256

    def _cache_get(self, cache: OrderedDict, key):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _cache_set(self, cache: OrderedDict, key, value) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self._cache_max_size:
            cache.popitem(last=False)

    @staticmethod
    def _lineup_key(current_ally, current_enemy) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return tuple(current_ally), tuple(current_enemy)

    def recommend(self, current_ally, current_enemy, valid_hero_ids, mode="pick", topk=10):
        key = (self._lineup_key(current_ally, current_enemy), mode, int(topk), tuple(valid_hero_ids))
        cached = self._cache_get(self._recommend_cache, key)
        if cached is not None:
            return cached

        used = set(current_ally + current_enemy)
        temp_a = (current_ally + [0]*5)[:5]
        temp_e = (current_enemy + [0]*5)[:5]
        lineup = temp_a + temp_e
        is_ally = (mode == "pick")
        
        if is_ally:
            mask_idx_in_seq = len(current_ally) 
            if mask_idx_in_seq >= 5: return [] 
        else:
            mask_idx_in_seq = 5 + len(current_enemy)
            if mask_idx_in_seq >= 10: return [] 

        seq = torch.tensor([lineup]).to(self.device)
        sides = torch.tensor([[0]*5 + [1]*5]).to(self.device)
        
        with torch.no_grad():
            # ⚡ 极速通道：不需要注意力，直接调原生 forward 拿 Mask
            mask_logits, _, _ = self.engine(seq, sides)
            probs = torch.softmax(mask_logits[0, mask_idx_in_seq], dim=-1)
        
        # 🌟 优化：拿前 60 个候选去筛
        candidate_probs, candidate_ids = torch.topk(probs, k=60)
        refined_candidates = []
        for p, h_id in zip(candidate_probs.tolist(), candidate_ids.tolist()):
            if h_id in valid_hero_ids and h_id not in used:
                refined_candidates.append(h_id)
            if len(refined_candidates) >= 30:  # 🌟 严格截断到前 30
                break

        if not refined_candidates:
            self._cache_set(self._recommend_cache, key, [])
            return []

        final_lineups = []
        for h in refined_candidates:
            case_lineup = list(lineup)
            case_lineup[mask_idx_in_seq] = h
            final_lineups.append(case_lineup)
            
        batch_seq = torch.tensor(final_lineups).to(self.device)
        batch_sides = torch.tensor([[0]*5 + [1]*5] * len(final_lineups)).to(self.device)
        
        with torch.no_grad():
            _, win_logits, _ = self.engine(batch_seq, batch_sides)
            # 🌟 同样拉平转 List
            win_probs = torch.sigmoid(win_logits).view(-1).tolist()       

        final_results = []
        # 不需要再 if else 判断数量了，直接 zip 遍历
        for h, wp in zip(refined_candidates, win_probs):
            final_results.append((h, float(wp)))
        
        if mode == "pick":
            result = sorted(final_results, key=lambda x: x[1], reverse=True)[:topk]
        else:
            result = sorted(final_results, key=lambda x: x[1], reverse=False)[:topk]

        self._cache_set(self._recommend_cache, key, result)
        return result

    def get_full_analysis(self, hero_ids_list, side_ids_list):
        key = (tuple(hero_ids_list), tuple(side_ids_list))
        cached = self._cache_get(self._analysis_cache, key)
        if cached is not None:
            return cached

        hero_ids = torch.tensor([hero_ids_list], dtype=torch.long).to(self.device)
        side_ids = torch.tensor([side_ids_list], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # 🔍 慢速通道：全景分析，必须手抠 Attention，同时拿到所有数据
            win_prob, _, role_logits, last_weights = self.engine.forward_with_attn(hero_ids, side_ids)
            
            attn_matrix = last_weights[0].detach().cpu().numpy() 
            hero_to_hero_attn = attn_matrix[1:11, 1:11] 
            base_prob = win_prob.item()
            
            # ✨ 新增：获取场上 10 个人的详细位置预测概率 [10, 5]
            # 对应每个位置的 (1号位, 2号位, 3号位, 4号位, 5号位) 概率
            role_probs = torch.softmax(role_logits[0], dim=-1).cpu().numpy()

            deltas = []
            valid_positions = [i for i, hid in enumerate(hero_ids_list) if hid != 0]
            if valid_positions:
                mut_lineups = []
                for i in valid_positions:
                    temp = list(hero_ids_list)
                    temp[i] = 0
                    mut_lineups.append(temp)

                mut_hero_ids = torch.tensor(mut_lineups, dtype=torch.long).to(self.device)
                mut_side_ids = torch.tensor([side_ids_list] * len(mut_lineups), dtype=torch.long).to(self.device)
                
                # ⚡ 极速通道：算 Delta 的时候不需要看 Attention，直接调原生 forward 并行加速！
                _, mut_win_logits, _ = self.engine(mut_hero_ids, mut_side_ids)
                new_probs = torch.sigmoid(mut_win_logits).view(-1).cpu().numpy()

                delta_map = {pos: float(prob - base_prob) for pos, prob in zip(valid_positions, new_probs)}
                for i in range(10):
                    deltas.append(delta_map.get(i, 0.0))
            else:
                deltas = [0.0] * 10

        # 返回：注意力矩阵, 抖动列表, 基础胜率, 🌟 位置概率预测
        result = (hero_to_hero_attn * 100, deltas, base_prob, role_probs)
        self._cache_set(self._analysis_cache, key, result)
        return result

    def get_explanation(self, target_hero_id, current_ally, current_enemy):
        key = (self._lineup_key(current_ally, current_enemy), int(target_hero_id))
        cached = self._cache_get(self._explanation_cache, key)
        if cached is not None:
            return cached

        temp_a = (current_ally + [target_hero_id] + [0]*5)[:5]
        temp_e = (current_enemy + [0]*5)[:5]
        full_lineup = temp_a + temp_e
        target_pos = len(current_ally)
        
        seq = torch.tensor([full_lineup]).to(self.device)
        sides = torch.tensor([[0]*5 + [1]*5]).to(self.device)
        
        with torch.no_grad():
            # 🔍 慢速通道：算基础局，抠出 Attention
            win_prob, _, _, last_weights = self.engine.forward_with_attn(seq, sides)
            base_prob = win_prob.item()
            attn_row = last_weights[0, target_pos].detach().cpu().numpy() # [11] 注意 win_token 占位

        active_positions = [i for i, h in enumerate(full_lineup) if h != 0 and i != target_pos]
        if not active_positions:
            return [], []

        inference_list = []
        for pos in active_positions:
            modified = list(full_lineup)
            modified[pos] = 0 
            inference_list.append(modified)
            
        batch_seq = torch.tensor(inference_list).to(self.device)
        batch_sides = torch.tensor([[0]*5 + [1]*5] * len(inference_list)).to(self.device)
        
        with torch.no_grad():
            _, mut_win_logits, _ = self.engine(batch_seq, batch_sides)
            # 🌟 终极修复：强制拉平 (view(-1)) 并直接转原生 List，不管 1 个还是 N 个，永不报错！
            mut_probs = torch.sigmoid(mut_win_logits).view(-1).tolist()

        ally_results = []
        enemy_results = []
        
        for idx, pos in enumerate(active_positions):
            h_id = full_lineup[pos]
            # attn_row 第 0 位是 win_token，所以英雄对应位置要 +1
            attn_val = float(attn_row[pos + 1]) 
            mutated_prob = mut_probs[idx]
            delta = base_prob - mutated_prob
            
            if pos < 5: 
                ally_results.append((h_id, attn_val, delta))
            else: 
                enemy_results.append((h_id, attn_val, delta))
        
        ally_results.sort(key=lambda x: x[1], reverse=True)
        enemy_results.sort(key=lambda x: x[1], reverse=True)

        result = (enemy_results, ally_results)
        self._cache_set(self._explanation_cache, key, result)
        return result
    
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


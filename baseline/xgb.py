import sys
import os
# 把上一级目录（项目根目录）强行加入 Python 视野
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

# 假设这些在你的 src 中
from src.utils import get_all_hero_pools, load_hero_static_json
from src.dataset import build_rerank_dataloader

# ==========================================
# 2. 英雄特征矩阵构建
# ==========================================
def construct_hero_feat_matrix(hero_json, id_to_idx):
    """
    构造静态属性矩阵，维度为 (num_valid_heroes, feat_dim)
    完全消除 Gap，极度紧凑。
    """
    attrs = ["str", "agi", "int", "all"]
    atk_types = ["Melee", "Ranged"]
    roles = sorted({
        role for h in hero_json.values() 
        for role in h.get("roles", []) if isinstance(role, str)
    })
    role_to_idx = {r: i for i, r in enumerate(roles)}

    numeric_fields = [
        "move_speed", "attack_range", "base_armor",
        "base_str", "base_agi", "base_int",
        "str_gain", "agi_gain", "int_gain",
    ]

    field_stats = {}
    for field in numeric_fields:
        vals = [float(h[field]) for h in hero_json.values() if isinstance(h.get(field), (int, float))]
        field_stats[field] = (min(vals), max(vals)) if vals else (0.0, 1.0)

    feat_dim = len(attrs) + len(atk_types) + len(roles) + len(numeric_fields)
    
    # 🌟 矩阵行数现在是真实的英雄数量！
    feat = np.zeros((len(id_to_idx), feat_dim), dtype=np.float32)

    for hid_str, h in hero_json.items():
        hid = int(hid_str)
        if hid not in id_to_idx:
            continue
        
        # 拿到连续的行索引
        idx = id_to_idx[hid]
        offset = 0
        
        attr = h.get("primary_attr")
        if attr in attrs: feat[idx, offset + attrs.index(attr)] = 1.0
        offset += len(attrs)

        attack_type = h.get("attack_type")
        if attack_type in atk_types: feat[idx, offset + atk_types.index(attack_type)] = 1.0
        offset += len(atk_types)

        for role in h.get("roles", []):
            ridx = role_to_idx.get(role)
            if ridx is not None: feat[idx, offset + ridx] = 1.0
        offset += len(roles)

        for i, field in enumerate(numeric_fields):
            v = h.get(field)
            if isinstance(v, (int, float)):
                mn, mx = field_stats[field]
                feat[idx, offset + i] = (float(v) - mn) / (mx - mn + 1e-6)

    return feat

# ==========================================
# 3. XGBoost 特征扁平化 (直接对接 Dataset)
# ==========================================
def prepare_xgb_arrays(dataset, hero_feat_matrix, id_to_idx):
    """
    专门适配 RerankDataset 的 XGBoost 特征转换器
    """
    X_recommend, y_recommend = [], []
    X_win, y_win = [], []
    y_rec_sets=[]
    if torch.is_tensor(hero_feat_matrix):
        hero_feat_np = hero_feat_matrix.cpu().numpy()
    else:
        hero_feat_np = hero_feat_matrix

    num_heroes = len(id_to_idx)
    print(f"Converting {len(dataset)} Rerank samples for XGBoost...")
    
    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        
        m_seq = item['masked_seq'].numpy().copy()
        m_roles = item['role_labels'].numpy().copy()
        fill_pos = item['fill_pos'].item()
        target_hero = item['target_hero'].item()
        w_label = item['win_label'].item()
        w_label = 1 if w_label > 0.5 else 0  # 二分类标签
        full_seq = item['full_seq'].numpy().copy()

        m_seq_rec = m_seq.copy()
        m_roles_rec = m_roles.copy()
        m_seq_rec[fill_pos] = 0
        m_roles_rec[fill_pos] = 0

        is_left = fill_pos < 5
        if is_left:
            # 左边选人：找 0-4 号位里被挖空的真实英雄
            valid_raw_targets = [full_seq[k] for k in range(5) if m_seq_rec[k] == 0]
        else:
            # 右边选人：找 5-9 号位里被挖空的真实英雄
            valid_raw_targets = [full_seq[k] for k in range(5, 10) if m_seq_rec[k] == 0]
            
        # 转换为 XGBoost 的连续 Index
        valid_idx_targets = [id_to_idx[hid] for hid in valid_raw_targets if hid in id_to_idx]
        # ==================================================
        # 🚨 任务 A: Recommend 特征提取 (绝对防漏版)
        # ==================================================

        # 从【严格挖空】的序列里提取 ID
        ally_raw_ids_rec = m_seq_rec[:5][m_seq_rec[:5] > 0]
        enemy_raw_ids_rec = m_seq_rec[5:][m_seq_rec[5:] > 0]

        ally_idx_rec = [id_to_idx[hid] for hid in ally_raw_ids_rec if hid in id_to_idx]
        enemy_idx_rec = [id_to_idx[hid] for hid in enemy_raw_ids_rec if hid in id_to_idx]

        ally_multi_hot_rec = np.zeros(num_heroes)
        enemy_multi_hot_rec = np.zeros(num_heroes)
        ally_multi_hot_rec[ally_idx_rec] = 1
        enemy_multi_hot_rec[enemy_idx_rec] = 1

        ally_attr_sum_rec = hero_feat_np[ally_idx_rec].sum(axis=0) if len(ally_idx_rec) > 0 else np.zeros(hero_feat_np.shape[1])
        enemy_attr_sum_rec = hero_feat_np[enemy_idx_rec].sum(axis=0) if len(enemy_idx_rec) > 0 else np.zeros(hero_feat_np.shape[1])

        # 拼接推荐任务的专属特征
        base_features_rec = np.concatenate([
            ally_multi_hot_rec, 
            enemy_multi_hot_rec, 
            ally_attr_sum_rec, 
            enemy_attr_sum_rec,
            m_roles_rec  # 🌟 修复：挖空后的 Role 真正放进去了！
        ])

        if target_hero in id_to_idx:
            slot_feat = np.zeros(10)
            slot_feat[fill_pos] = 1
            X_recommend.append(np.concatenate([base_features_rec, slot_feat]))
            y_recommend.append(id_to_idx[target_hero])
            y_rec_sets.append(valid_idx_targets) # 🌟 存入集合！

        # ==================================================
        # 🏆 任务 B: WinRate 特征提取 (裁判视角)
        # ==================================================
        # 如果你想让 XGBoost 作为 Baseline 的打分器，用 m_seq 训练是最合理的。
        ally_raw_ids_win = m_seq[:5][m_seq[:5] > 0]
        enemy_raw_ids_win = m_seq[5:][m_seq[5:] > 0]

        ally_idx_win = [id_to_idx[hid] for hid in ally_raw_ids_win if hid in id_to_idx]
        enemy_idx_win = [id_to_idx[hid] for hid in enemy_raw_ids_win if hid in id_to_idx]

        ally_multi_hot_win = np.zeros(num_heroes)
        enemy_multi_hot_win = np.zeros(num_heroes)
        ally_multi_hot_win[ally_idx_win] = 1
        enemy_multi_hot_win[enemy_idx_win] = 1

        ally_attr_sum_win = hero_feat_np[ally_idx_win].sum(axis=0) if len(ally_idx_win) > 0 else np.zeros(hero_feat_np.shape[1])
        enemy_attr_sum_win = hero_feat_np[enemy_idx_win].sum(axis=0) if len(enemy_idx_win) > 0 else np.zeros(hero_feat_np.shape[1])

        base_features_win = np.concatenate([
            ally_multi_hot_win, 
            enemy_multi_hot_win, 
            ally_attr_sum_win, 
            enemy_attr_sum_win,
            m_roles  # WinRate 任务看到的是完整的阵容角色分配
        ])

        X_win.append(base_features_win)
        y_win.append(w_label)

    return (np.array(X_recommend, dtype=np.float32), np.array(y_recommend, dtype=np.int32), y_rec_sets), \
           (np.array(X_win, dtype=np.float32), np.array(y_win, dtype=np.float32))

# ==========================================
# 4. 主执行流
# ==========================================
if __name__ == "__main__":
    # 1. 确定边界
    valid_hero_ids = get_all_hero_pools()
    id_to_idx = {hid: i for i, hid in enumerate(valid_hero_ids)}
    # 0~123 -> raw_id (备用，如果需要输出给人看)
    idx_to_id = {i: hid for i, hid in enumerate(valid_hero_ids)}

    # 2. 构造英雄特征矩阵
    hero_json = load_hero_static_json()
    hero_feat_matrix = construct_hero_feat_matrix(hero_json, id_to_idx=id_to_idx)

    # 3. 加载 Dataset (只用 dataset 即可，不用 Dataloader，因为 XGB 是一次性喂入内存)
    train_dataset, val_dataset = build_rerank_dataloader(batch_size=1024, shuffle=True, num_workers=2, xgb=True)

    # 4. 转换 XGB 特征 (注意：如果是30w数据，内存可能吃紧，建议可以先截断 train_dataset 跑个 demo)
    (X_rec_train, y_rec_train,_), (X_win_train, y_win_train) = prepare_xgb_arrays(train_dataset, hero_feat_matrix, id_to_idx)
    (X_rec_val, y_rec_val, y_rec_val_set), (X_win_val, y_win_val) = prepare_xgb_arrays(val_dataset, hero_feat_matrix, id_to_idx)

    print(f"Win Task Train Shape: {X_win_train.shape}")
    print(f"Rec Task Train Shape: {X_rec_train.shape}")

    # ----------------------------------
    # 训练 1：胜率模型 (WinRate - 二分类)
    # ----------------------------------
    print("\n--- Training WinRate Model ---")
    model_win = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        device='cuda' # 有卡必开！
    )
    
    # 用验证集做 early stopping
    model_win.fit(
        X_win_train, y_win_train,
        eval_set=[(X_win_val, y_win_val)],
        verbose=10
    )

    # 测试 AUC
    y_win_pred = model_win.predict_proba(X_win_val)[:, 1]
    auc_score = roc_auc_score(y_win_val, y_win_pred)
    print(f"🔥 XGBoost WinRate AUC on Validation Set: {auc_score:.4f}")

    # ----------------------------------
    # 训练 2：推荐模型 (Recommend - 多分类)
    # ----------------------------------
    print("\n--- Training Recommend Model ---")
    # 注意：num_class 设为 MAX_HERO_ID + 1 也是合法的，虽然中间有类不会出现，但 XGB hist 不太怕
    model_rec = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        tree_method='hist',
        device='cuda'
    )
    
    model_rec.fit(
        X_rec_train, y_rec_train,
        eval_set=[(X_rec_val, y_rec_val)],
        verbose=10
    )

    model_rec.fit(X_rec_train, y_rec_train, eval_set=[(X_rec_val, y_rec_val)], verbose=10)
    print("\n--- Evaluating Recommend Model (Set-based) ---")
    
    # 概率矩阵 Shape: [N, num_actual_heroes]
    probs = model_rec.predict_proba(X_rec_val) 
    
    # 🌟 去除场上已存在英雄的概率 (防泄露/重推)
    # 既然端水就要端平，我们也把 XGBoost 不能选的英雄置为 -inf
    # 这个可以在特征里搞，但简单起见，我们假设 XGB 会自己学到。
    # 这里我们直接取 Top 30
    top30_idx = np.argsort(probs, axis=1)[:, -30:]
    top10_idx = top30_idx[:, -10:]
    top5_idx  = top30_idx[:, -5:]
    
    hit_5_count = 0
    hit_10_count = 0
    mask_hit_30_count = 0 # 对应 Transformer 的 mask_hit_at_30
    
    # 遍历所有验证集样本
    for i in range(len(y_rec_val)):
        # 将当前样本的“正确答案集合”转为 Python set，查询速度极快
        valid_set = set(y_rec_val_set[i])
        
        # 将预测结果转为 set
        pred_5_set = set(top5_idx[i])
        pred_10_set = set(top10_idx[i])
        pred_30_set = set(top30_idx[i])
        
        # 🌟 集合求交集！只要有一个撞上了，就算 Hit！
        if len(valid_set.intersection(pred_5_set)) > 0:
            hit_5_count += 1
        if len(valid_set.intersection(pred_10_set)) > 0:
            hit_10_count += 1
        if len(valid_set.intersection(pred_30_set)) > 0:
            mask_hit_30_count += 1

    total_val_samples = len(y_rec_val)
    print(f"🎯 XGBoost Mask Hit@30: {mask_hit_30_count / total_val_samples:.4f}")
    print(f"🔥 XGBoost Recommend Hit@10: {hit_10_count / total_val_samples:.4f}")
    print(f"🔥 XGBoost Recommend Hit@5:  {hit_5_count / total_val_samples:.4f}")
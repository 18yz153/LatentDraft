import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import xgboost as xgb


def load_embedding_payload(path: Path) -> Tuple[List[int], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("embedding file must be a dict payload")

    # Preferred format from embedding.py: {hero_pool, embedding}
    if "hero_pool" in payload and "embedding" in payload:
        hero_pool = [int(x) for x in payload["hero_pool"]]
        embedding = payload["embedding"]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        return hero_pool, embedding

    # Fallback checkpoint format: {hero_pool, state_dict}
    if "hero_pool" in payload and "state_dict" in payload:
        hero_pool = [int(x) for x in payload["hero_pool"]]
        state_dict = payload["state_dict"]
        if "hero_emb.weight" not in state_dict:
            raise ValueError("state_dict does not contain hero_emb.weight")
        return hero_pool, state_dict["hero_emb.weight"].detach().cpu()

    raise ValueError("unknown embedding payload format")


def load_hero_id_to_name(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def get_mean_emb(hero_list, hero_embeddings):
    """计算英雄列表的平均向量 (Mean Pooling)，如果为空返回 0 向量"""
    if not hero_list:
        return np.zeros(64)
    return np.mean([hero_embeddings[h] for h in hero_list], axis=0)

# ==========================================
# 2. 构造 XGBoost 训练集 (截断 + 混合负采样)
# ==========================================
def generate_xgb_dataset(matches, num_heroes=150, hero_embeddings=None, all_heroes=None):
    features = []
    labels = []
    if all_heroes is None:
        all_heroes = set(range(1, num_heroes))
    
    for match in matches:
        winner = match['winner']
        loser = match['loser']
        
        k = random.randint(1, 5) 
        
        # 此时己方有 k-1 个英雄
        current_ally = winner[:k-1] 
        
        # 敌方有几个？在合法规则下，只能是 k-1 或 k 个
        # 如果 k=5，敌方最多也只有 5 个，所以用 min(k, 5) 兜底
        num_enemy = random.choice([k-1, min(k, 5)])
        current_enemy = loser[:num_enemy]
        
        # 这时我们要预测的正样本就是 winner 的第 k 个英雄
        pos_candidate = winner[k-1]

        # ---------------- 断言测试（心里踏实） ----------------
        # 验证人数差绝对值是否 <= 1，如果这段代码放到线上，绝对不会报错
        assert abs(len(current_ally) - len(current_enemy)) <= 1
        # ---------------------------------------------------

        # 提取特征：己方均值 (64维) + 敌方均值 (64维)
        vec_ally = get_mean_emb(current_ally, hero_embeddings)
        vec_enemy = get_mean_emb(current_enemy, hero_embeddings)
        base_features = np.concatenate([vec_ally, vec_enemy]) # 128维
        
        # 2. 构造正样本 (Label=1)
        pos_feat = np.concatenate([base_features, hero_embeddings[pos_candidate]]) # 192维
        features.append(pos_feat)
        labels.append(1)
        
        # 3. 构造困难负样本 (Hard Negative) - 败方视角
        # 如果败方在对应轮次有英雄，且没被我们用过，就拿来当负样本
        if k-1 < len(loser):
            hard_neg_candidate = loser[k-1]
            if hard_neg_candidate not in current_ally: 
                hard_feat = np.concatenate([base_features, hero_embeddings[hard_neg_candidate]])
                features.append(hard_feat)
                labels.append(0)
                
        # 4. 构造随机负样本 (Easy Negative) - 1:3 比例
        used_heroes = set(current_ally + current_enemy + [pos_candidate])
        easy_neg_candidates = random.sample(list(all_heroes - used_heroes), 3)
        for neg_cand in easy_neg_candidates:
            neg_feat = np.concatenate([base_features, hero_embeddings[neg_cand]])
            features.append(neg_feat)
            labels.append(0)
            
    return np.array(features), np.array(labels)


def load_matches(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # 已经是 winner/loser 格式就直接返回
    if raw and "winner" in raw[0] and "loser" in raw[0]:
        return raw

    # 兼容 data.json 的 radiant/dire + radiant_win 格式
    matches = []
    for m in raw:
        radiant = m.get("radiant_team", [])
        dire = m.get("dire_team", [])
        if len(radiant) != 5 or len(dire) != 5:
            continue
        if bool(m.get("radiant_win", False)):
            winner, loser = radiant, dire
        else:
            winner, loser = dire, radiant
        matches.append({"winner": winner, "loser": loser})
    return matches


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model from hero embeddings")
    parser.add_argument("--matches", type=Path, default=Path("data/data.json"))
    parser.add_argument("--embedding", type=Path, default=Path("models/hero_embedding.pt"))
    parser.add_argument("--hero-id-to-name", type=Path, default=Path("data/hero_id_to_name.json"))
    parser.add_argument("--model-out", type=Path, default=Path("models/xgb_bp.model"))
    parser.add_argument("--num-heroes", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-boost-round", type=int, default=100)

    return parser.parse_args()


def run_train(args):

    _, emb_tensor = load_embedding_payload(args.embedding)
    hero_embeddings = emb_tensor.cpu().numpy()
    hero_id_to_name = load_hero_id_to_name(args.hero_id_to_name)
    all_heroes = set(hero_id_to_name.keys())

    raw_matches = load_matches(args.matches)
    x_train, y_train = generate_xgb_dataset(
        raw_matches,
        num_heroes=args.num_heroes,
        hero_embeddings=hero_embeddings,
        all_heroes=all_heroes,
    )

    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": args.max_depth,
    }
    bst = xgb.train(params, dtrain, num_boost_round=args.num_boost_round)
    bst.save_model(str(args.model_out))
    print(f"saved model -> {args.model_out}")


def main():
    args = parse_args()
    run_train(args)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import xgboost as xgb


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


def load_hero_id_to_name(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def recommend(current_ally, current_enemy, bst_model, hero_embeddings, valid_hero_ids, mode="pick", topk=10):
    """
    mode='pick': 为我方找胜率最高的候选人
    mode='ban':  为对方找胜率最高的候选人（即我们要 ban 的）
    """
    v_a = np.mean([hero_embeddings[h] for h in current_ally], axis=0) if current_ally else np.zeros(64)
    v_e = np.mean([hero_embeddings[h] for h in current_enemy], axis=0) if current_enemy else np.zeros(64)

    # 如果是 ban 模式，则互换敌我特征位置，实现“极大极小博弈”
    base_feat = np.concatenate([v_a, v_e]) if mode == "pick" else np.concatenate([v_e, v_a])

    used = set(current_ally + current_enemy)
    candidates = [h for h in valid_hero_ids if h not in used]
    feats = [np.concatenate([base_feat, hero_embeddings[h]]) for h in candidates]
    probs = bst_model.predict(xgb.DMatrix(np.array(feats)))

    results = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
    return results[:topk]
    
def get_ablation_explanation(target_hero_id, current_ally, current_enemy, bst_model, hero_embeddings):
    """
    计算推荐英雄的核心克制目标和最佳搭配队友
    返回: 
        enemy_deltas: list of (enemy_id, delta) 按克制贡献度降序
        ally_deltas: list of (ally_id, delta) 按配合贡献度降序
    """
    v_a = np.mean([hero_embeddings[h] for h in current_ally], axis=0) if current_ally else np.zeros(64)
    v_e = np.mean([hero_embeddings[h] for h in current_enemy], axis=0) if current_enemy else np.zeros(64)
    base_feat = np.concatenate([v_a, v_e, hero_embeddings[target_hero_id]])
    base_prob = bst_model.predict(xgb.DMatrix(np.array([base_feat])))[0]

    enemy_deltas = []
    # 1. 找敌方靶点 (所有人)
    for enemy in current_enemy:
        temp_enemy = [e for e in current_enemy if e != enemy]
        temp_v_e = np.mean([hero_embeddings[h] for h in temp_enemy], axis=0) if temp_enemy else np.zeros(64)
        temp_feat = np.concatenate([v_a, temp_v_e, hero_embeddings[target_hero_id]])
        temp_prob = bst_model.predict(xgb.DMatrix(np.array([temp_feat])))[0]
        
        delta = base_prob - temp_prob
        enemy_deltas.append((enemy, float(delta)))

    ally_deltas = []
    # 2. 找己方大腿 (所有人)
    for ally in current_ally:
        temp_ally = [a for a in current_ally if a != ally]
        temp_v_a = np.mean([hero_embeddings[h] for h in temp_ally], axis=0) if temp_ally else np.zeros(64)
        temp_feat = np.concatenate([temp_v_a, v_e, hero_embeddings[target_hero_id]])
        temp_prob = bst_model.predict(xgb.DMatrix(np.array([temp_feat])))[0]
        
        delta = base_prob - temp_prob
        ally_deltas.append((ally, float(delta)))

    # 按贡献值从大到小排序
    enemy_deltas.sort(key=lambda x: x[1], reverse=True)
    ally_deltas.sort(key=lambda x: x[1], reverse=True)

    return enemy_deltas, ally_deltas

def parse_args():
    parser = argparse.ArgumentParser(description="Load saved XGB model and recommend heroes")
    parser.add_argument("--model", type=Path, default=Path("xgb_bp.model"))
    parser.add_argument("--embedding", type=Path, default=Path("hero_embedding.pt"))
    parser.add_argument("--hero-id-to-name", type=Path, default=Path("hero_id_to_name.json"))
    parser.add_argument("--ally", type=int, nargs="+", required=True)
    parser.add_argument("--enemy", type=int, nargs="+", required=True)
    parser.add_argument("--mode", type=str, choices=["pick", "ban"], default="pick")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    hero_pool, emb_tensor = load_embedding_payload(args.embedding)
    hero_embeddings = emb_tensor.cpu().numpy()
    hero_id_to_name = load_hero_id_to_name(args.hero_id_to_name)

    bst = xgb.Booster()
    bst.load_model(str(args.model))

    valid_hero_ids = sorted(set(hero_pool) & set(hero_id_to_name.keys()))
    results = recommend(
        current_ally=args.ally,
        current_enemy=args.enemy,
        bst_model=bst,
        hero_embeddings=hero_embeddings,
        valid_hero_ids=valid_hero_ids,
        mode=args.mode,
        topk=args.topk,
    )

    ally_text = [f"{hid}:{hero_id_to_name.get(int(hid), 'Unknown')}" for hid in args.ally]
    enemy_text = [f"{hid}:{hero_id_to_name.get(int(hid), 'Unknown')}" for hid in args.enemy]
    print(f"mode={args.mode}")
    print("ally=" + ", ".join(ally_text))
    print("enemy=" + ", ".join(enemy_text))
    print(f"top-{len(results)} results:")
    for i, (hero_id, prob) in enumerate(results, start=1):
        hero_name = hero_id_to_name.get(int(hero_id), "Unknown")
        print(f"{i:2d}. hero_id={hero_id:<3d} hero_name={hero_name:<24s} score={float(prob):.4f}")


if __name__ == "__main__":
    main()
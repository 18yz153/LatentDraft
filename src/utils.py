from typing import List, Tuple, Dict
from pathlib import Path
import torch
import numpy as np
import json

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

def get_all_hero_pools():
    all_hero_pools = set()
    with open('data/heroes.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)
        for heroid in raw.keys():
            hero_id = int(heroid)
            all_hero_pools.add(hero_id)

    return all_hero_pools

def load_hero_id_to_name() -> Dict[int, str]:
    with open('data/hero_id_to_name.json', "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}

def load_hero_id_to_url_name(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}
def get_number_of_heroes() -> int:
    return max(load_hero_id_to_name().keys())

def load_hero_static_json():
    with open('data/heroes.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def load_id_to_name(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


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


def topk_similar(
    hero_pool: List[int],
    embedding: torch.Tensor,
    query_hero_id: int,
    topk: int,
) -> List[Tuple[int, float]]:
    if query_hero_id not in hero_pool:
        raise ValueError(f"hero_id {query_hero_id} not found in hero_pool")

    valid_ids = torch.tensor(hero_pool, dtype=torch.long)
    valid_vecs = embedding[valid_ids]
    valid_vecs = F.normalize(valid_vecs, p=2, dim=1)

    query_pos = hero_pool.index(query_hero_id)
    query_vec = valid_vecs[query_pos : query_pos + 1]

    sims = (valid_vecs @ query_vec.t()).squeeze(1)
    sims[query_pos] = -1.0

    topk = min(topk, len(hero_pool) - 1)
    vals, idxs = torch.topk(sims, k=topk, largest=True)

    result: List[Tuple[int, float]] = []
    for i in range(topk):
        hid = int(valid_ids[idxs[i]].item())
        score = float(vals[i].item())
        result.append((hid, score))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect hero embedding similarity by hero id")
    parser.add_argument("--embedding", type=Path, default=Path("hero_embedding.pt"))
    parser.add_argument("--hero-id-to-name", type=Path, default=Path("hero_id_to_name.json"))
    parser.add_argument("--hero-id", type=int, required=True)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hero_pool, embedding = load_embedding_payload(args.embedding)
    id_to_name = load_id_to_name(args.hero_id_to_name)

    query_name = id_to_name.get(args.hero_id, "Unknown")
    recs = topk_similar(hero_pool, embedding, args.hero_id, args.topk)

    print(f"query hero: id={args.hero_id} name={query_name}")
    print(f"top-{len(recs)} similar heroes:")
    for rank, (hid, sim) in enumerate(recs, start=1):
        name = id_to_name.get(hid, "Unknown")
        print(f"{rank:2d}. hero_id={hid:<3d} hero_name={name:<24s} cosine={sim:.4f}")


if __name__ == "__main__":
    main()

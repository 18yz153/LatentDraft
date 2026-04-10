import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path

    candidates = [
        PROJECT_ROOT / path,
        PROJECT_ROOT / "models" / path.name,
        PROJECT_ROOT / "data" / path.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    tried = "\n".join(str(p) for p in [path, *candidates])
    raise FileNotFoundError(f"file not found: {path}\ntried:\n{tried}")


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


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")

    candidates = [payload]
    for key in ("state_dict", "model_state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for sd in candidates:
        if "hero_emb.weight" in sd or "module.hero_emb.weight" in sd:
            return sd

    raise ValueError("checkpoint does not contain hero_emb.weight")


def load_dota_bert_embeddings(
    path: Path,
    num_heroes: Optional[int] = None,
    max_known_hero_id: Optional[int] = None,
) -> Tuple[List[int], torch.Tensor]:
    payload = _safe_torch_load(path)
    state_dict = _extract_state_dict(payload)

    if "hero_emb.weight" in state_dict:
        emb_weight = state_dict["hero_emb.weight"].detach().cpu()
    else:
        emb_weight = state_dict["module.hero_emb.weight"].detach().cpu()

    if emb_weight.ndim != 2:
        raise ValueError("hero_emb.weight must be a 2D tensor")

    rows, dim = emb_weight.shape

    # Prefer metadata-driven inference; fallback to checkpoint-driven inference.
    if max_known_hero_id is not None and max_known_hero_id <= rows - 1:
        inferred_num_heroes = max_known_hero_id
    elif num_heroes is not None and num_heroes <= rows - 1:
        inferred_num_heroes = num_heroes
    else:
        inferred_num_heroes = rows - 1

    if inferred_num_heroes <= 0:
        raise ValueError(f"invalid inferred hero count: {inferred_num_heroes} from shape {rows}x{dim}")

    # Keep id=0 as reserved token. If checkpoint has extra tail tokens
    # (e.g. mask token at num_heroes+1), they are ignored here.
    end = 1 + inferred_num_heroes
    if end > rows:
        raise ValueError(
            f"inferred hero range [1:{end}) exceeds embedding rows={rows}; "
            f"try passing a smaller --num-heroes"
        )
    all_hero_embeddings = emb_weight[1:end]

    padded = torch.zeros((inferred_num_heroes + 1, all_hero_embeddings.shape[1]), dtype=all_hero_embeddings.dtype)
    padded[1:] = all_hero_embeddings

    hero_pool = list(range(1, inferred_num_heroes + 1))
    return hero_pool, padded


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
    parser.add_argument("--embedding", type=Path, default=Path("models/hero_embedding.pt"))
    parser.add_argument("--dota-bert", type=Path, default=None, help="optional dota_bert.pt path")
    parser.add_argument("--num-heroes", type=int, default=None, help="optional override for hero count")
    parser.add_argument("--hero-id-to-name", type=Path, default=Path("data/hero_id_to_name.json"))
    parser.add_argument("--hero-id", type=int, required=True)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_to_name = load_id_to_name(resolve_input_path(args.hero_id_to_name))
    max_known_hero_id = max(id_to_name.keys()) if id_to_name else None

    if args.dota_bert is not None:
        ckpt_path = resolve_input_path(args.dota_bert)
        hero_pool, embedding = load_dota_bert_embeddings(
            path=ckpt_path,
            num_heroes=args.num_heroes,
            max_known_hero_id=max_known_hero_id,
        )
    else:
        emb_path = resolve_input_path(args.embedding)
        hero_pool, embedding = load_embedding_payload(emb_path)

    query_name = id_to_name.get(args.hero_id, "Unknown")
    recs = topk_similar(hero_pool, embedding, args.hero_id, 100)

    print(f"query hero: id={args.hero_id} name={query_name}")
    print(f"top-{len(recs)} similar heroes:")
    for rank, (hid, sim) in enumerate(recs, start=1):
        name = id_to_name.get(hid, "Unknown")
        print(f"{rank:2d}. hero_id={hid:<3d} hero_name={name:<24s} cosine={sim:.4f}")


if __name__ == "__main__":
    main()

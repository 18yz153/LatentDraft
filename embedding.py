import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_matches(data_path: Path) -> List[Dict]:
    with data_path.open("r", encoding="utf-8") as f:
        raw_matches = json.load(f)

    matches: List[Dict] = []
    dropped = 0
    for m in raw_matches:
        radiant = m.get("radiant_team", [])
        dire = m.get("dire_team", [])
        if len(radiant) == 5 and len(dire) == 5:
            matches.append(m)
        else:
            dropped += 1

    if dropped > 0:
        print(f"dropped {dropped} invalid matches (non-5v5)")
    return matches


def build_hero_pool(matches: Sequence[Dict]) -> List[int]:
    pool = sorted(
        {
            int(h)
            for m in matches
            for h in (m["radiant_team"] + m["dire_team"])
        }
    )
    return pool


def load_hero_id_to_name(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


class LatentDraftBPR(nn.Module):
    def __init__(self, num_heroes: int, embed_dim: int = 64, enemy_weight: float = 0.8):
        super().__init__()
        self.hero_emb = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)
        self.hero_bias = nn.Embedding(num_heroes + 1, 1, padding_idx=0)
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.enemy_weight = enemy_weight

        nn.init.xavier_uniform_(self.hero_emb.weight)
        nn.init.zeros_(self.hero_bias.weight)

    def get_context_vector(self, ally_ids: torch.Tensor, enemy_ids: torch.Tensor) -> torch.Tensor:
        ally_emb = self.hero_emb(ally_ids).mean(dim=1)
        enemy_emb = self.hero_emb(enemy_ids).mean(dim=1)
        context_cat = torch.cat([ally_emb, self.enemy_weight * enemy_emb], dim=-1)
        return self.context_proj(context_cat)

    def score(self, context_vec: torch.Tensor, hero_ids: torch.Tensor) -> torch.Tensor:
        hero_vec = self.hero_emb(hero_ids)
        hero_b = self.hero_bias(hero_ids).squeeze(-1)
        return (context_vec * hero_vec).sum(dim=-1) + hero_b

    def forward(
        self,
        ally_ids: torch.Tensor,
        enemy_ids: torch.Tensor,
        pos_hero_id: torch.Tensor,
        neg_hero_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_vec = self.get_context_vector(ally_ids, enemy_ids)
        pos_score = self.score(context_vec, pos_hero_id)
        neg_score = self.score(context_vec, neg_hero_id)
        return pos_score, neg_score


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_score - neg_score).mean()


class Dota2LineupDataset(Dataset):
    def __init__(
        self,
        matches: Sequence[Dict],
        hero_pool: Sequence[int],
        mask_min: int = 1,
        mask_max: int = 1,
        samples_per_match: int = 1,
    ):
        self.matches = list(matches)
        self.hero_pool = list(hero_pool)
        self.hero_set = set(hero_pool)
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.samples_per_match = samples_per_match

    def __len__(self) -> int:
        return len(self.matches) * self.samples_per_match * 2

    def _build_one(self, match: Dict, use_radiant: bool) -> Dict[str, torch.Tensor]:
        if use_radiant:
            ally_team = [int(x) for x in match["radiant_team"]]
            enemy_team = [int(x) for x in match["dire_team"]]
        else:
            ally_team = [int(x) for x in match["dire_team"]]
            enemy_team = [int(x) for x in match["radiant_team"]]

        # Context masking: 从己方挖空 k 个英雄，逐个当正样本。
        k = random.randint(self.mask_min, self.mask_max)
        k = max(1, min(k, len(ally_team) - 1))
        masked_idx = random.sample(range(len(ally_team)), k)
        pos_idx = random.choice(masked_idx)
        pos_hero_id = ally_team[pos_idx]

        ally_context = [h for i, h in enumerate(ally_team) if i != pos_idx]
        used_heroes = set(ally_team + enemy_team)
        available_neg = list(self.hero_set - used_heroes)
        neg_hero_id = random.choice(available_neg)

        return {
            "ally_ids": torch.tensor(ally_context, dtype=torch.long),
            "enemy_ids": torch.tensor(enemy_team, dtype=torch.long),
            "pos_hero_id": torch.tensor(pos_hero_id, dtype=torch.long),
            "neg_hero_id": torch.tensor(neg_hero_id, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        match_idx = idx // (self.samples_per_match * 2)
        side_idx = idx % 2
        match = self.matches[match_idx]
        use_radiant = side_idx == 0
        return self._build_one(match, use_radiant=use_radiant)


def recommend_similar(
    emb_weight: torch.Tensor,
    hero_pool: Sequence[int],
    hero_id: int,
    topk: int = 10,
) -> List[Tuple[int, float]]:
    if hero_id not in hero_pool:
        raise ValueError(f"hero_id {hero_id} is not in hero_pool")

    valid_ids = torch.tensor(hero_pool, dtype=torch.long)
    valid_vecs = emb_weight[valid_ids]
    valid_vecs = F.normalize(valid_vecs, p=2, dim=1)

    target_pos = hero_pool.index(hero_id)
    target_vec = valid_vecs[target_pos : target_pos + 1]

    sims = (valid_vecs @ target_vec.t()).squeeze(1)
    sims[target_pos] = -1.0

    topk = min(topk, len(hero_pool) - 1)
    vals, idxs = torch.topk(sims, k=topk, largest=True)

    out: List[Tuple[int, float]] = []
    for i in range(topk):
        hid = int(valid_ids[idxs[i]].item())
        score = float(vals[i].item())
        out.append((hid, score))
    return out


def train(args: argparse.Namespace) -> None:
    set_seed(42)

    matches = load_matches(args.data)
    hero_pool = build_hero_pool(matches)
    num_heroes = max(hero_pool)

    dataset = Dota2LineupDataset(
        matches=matches,
        hero_pool=hero_pool,
        mask_min=args.mask_min,
        mask_max=args.mask_max,
        samples_per_match=args.samples_per_match,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        print("cuda is not available, fallback to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(requested)

    model = LatentDraftBPR(
        num_heroes=num_heroes,
        embed_dim=args.embed_dim,
        enemy_weight=args.enemy_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"device={device} matches={len(matches)} samples={len(dataset)} "
        f"heroes={len(hero_pool)} embed_dim={args.embed_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in loader:
            ally_ids = batch["ally_ids"].to(device)
            enemy_ids = batch["enemy_ids"].to(device)
            pos_id = batch["pos_hero_id"].to(device)
            neg_id = batch["neg_hero_id"].to(device)

            optimizer.zero_grad()
            pos_score, neg_score = model(ally_ids, enemy_ids, pos_id, neg_id)
            loss = bpr_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"epoch={epoch} avg_bpr_loss={avg_loss:.6f}")

    emb_weight = model.hero_emb.weight.detach().cpu()
    hero_id_to_name = load_hero_id_to_name(args.hero_id_to_name)

    ckpt = {
        "state_dict": model.state_dict(),
        "hero_pool": hero_pool,
        "num_heroes": num_heroes,
        "embed_dim": args.embed_dim,
        "enemy_weight": args.enemy_weight,
    }
    torch.save(ckpt, args.out_ckpt)

    emb_payload = {
        "hero_pool": hero_pool,
        "embedding": emb_weight,
    }
    torch.save(emb_payload, args.out_embedding)
    print(f"saved checkpoint -> {args.out_ckpt}")
    print(f"saved embedding -> {args.out_embedding}")

    if args.query_hero_id is not None:
        recs = recommend_similar(
            emb_weight=emb_weight,
            hero_pool=hero_pool,
            hero_id=args.query_hero_id,
            topk=args.topk,
        )
        query_name = hero_id_to_name.get(args.query_hero_id, "Unknown")
        print(f"top-{len(recs)} similar heroes for hero_id={args.query_hero_id} ({query_name}):")
        for hid, sim in recs:
            hero_name = hero_id_to_name.get(hid, "Unknown")
            print(f"hero_id={hid} hero_name={hero_name} cosine={sim:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPR hero embedding with context masking")
    parser.add_argument("--data", type=Path, default=Path("data.json"))
    parser.add_argument("--hero-id-to-name", type=Path, default=Path("hero_id_to_name.json"))
    parser.add_argument("--out-ckpt", type=Path, default=Path("bpr_context_model.pt"))
    parser.add_argument("--out-embedding", type=Path, default=Path("hero_embedding.pt"))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument("--mask-min", type=int, default=1, help="min masked ally heroes per sample")
    parser.add_argument("--mask-max", type=int, default=2, help="max masked ally heroes per sample")

    parser.add_argument("--samples-per-match", type=int, default=1, help="samples drawn per match per side")

    parser.add_argument("--query-hero-id", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weight_decay = 1e-4
    args.enemy_weight = 0.8
    train(args)


if __name__ == "__main__":
    main()
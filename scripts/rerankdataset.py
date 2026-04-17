import argparse
import json
import random
import sys
from pathlib import Path
from xml.parsers.expat import model
import torch
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.dataset import build_s3_dataloader
from src.model import DotaMultiTaskTransformer
from src.utils import get_number_of_heroes
from src.utils import get_all_hero_pools, load_hero_id_to_name


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Build rerank dataset with mask top30 candidates")
	parser.add_argument("--model-path", type=Path, default=Path("models/stage3_value_network_best.pt"))
	parser.add_argument("--output-path", type=Path, default=Path("data/rerank_s3_val.jsonl"))
	parser.add_argument("--match-role-path", type=Path, default=Path("data/match_role.json"))
	parser.add_argument("--allmatch-path", type=Path, default=Path("data/allmatch.json"))
	parser.add_argument("--hero-name-path", type=Path, default=Path("data/hero_id_to_name.json"))
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--val-ratio", type=float, default=0.1)
	parser.add_argument("--split", choices=["train", "val"], default="val")
	parser.add_argument("--topk-mask", type=int, default=60)
	parser.add_argument("--topk-refined", type=int, default=30)
	parser.add_argument("--hard-neg", type=int, default=10)
	parser.add_argument("--easy-neg", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cuda")
	return parser



@torch.no_grad()
def main() -> None:
	args = build_parser().parse_args()
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	num_heroes = get_number_of_heroes(args.hero_name_path)
	if args.device == "cuda" and torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=64).to(device)
	state_dict = torch.load(args.model_path, map_location=device)
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	train_loader, val_loader = build_s3_dataloader(
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		val_ratio=args.val_ratio,
	)
	for split_name, loader, out_path in [
		("train", train_loader, Path(str(args.output_path).replace("val", "train"))),
		("val", val_loader, args.output_path)
	]:
		all_hero_pool = get_all_hero_pools()
		side_template = [0] * 5 + [1] * 5
		rows = []
		query_id = 0
		pbar = tqdm(loader, desc=f"Building rerank-{split_name}")
		for masked_seq, side_ids, role_labels, target_labels, full_seq, win_labels in pbar:
			batch_size = masked_seq.size(0)
			winloss_masked_seq = [masked_seq[b].tolist() for b in range(batch_size)]
			winloss_masked_roles = [role_labels[b].tolist() for b in range(batch_size)]
			fill_pos_list = []
			target_hero_list = []
			masked_seq_new_list = []
			masked_roles_new_list = []
			used_ids_list = []
			valid_indices = []
			for b in range(batch_size):
				is_winner_left = float(win_labels[b].item()) > 0.5
				if is_winner_left:
					winner_indices = [i for i in range(5) if masked_seq[b, i] > 0]
				else:
					winner_indices = [i for i in range(5, 10) if masked_seq[b, i] > 0]
				if not winner_indices:
					continue
				fill_pos = random.choice(winner_indices)
				target_hero = int(masked_seq[b, fill_pos].item())
				masked_seq_new = masked_seq[b].clone()
				masked_roles_new = role_labels[b].clone()
				masked_seq_new[fill_pos] = 0
				masked_roles_new[fill_pos] = 0
				fill_pos_list.append(fill_pos)
				target_hero_list.append(target_hero)
				masked_seq_new_list.append(masked_seq_new)
				masked_roles_new_list.append(masked_roles_new)
				used_ids_list.append({int(x) for x in full_seq[b].tolist() if int(x) > 0})
				valid_indices.append(b)
			if not masked_seq_new_list:
				continue
			input_seq = torch.stack(masked_seq_new_list, dim=0).to(device)
			input_roles = torch.stack(masked_roles_new_list, dim=0).to(device)
			input_sides = torch.tensor([side_template] * len(masked_seq_new_list), device=device, dtype=torch.long)
			mask_logits, _, _ = model(input_seq, input_sides, input_roles)
			for i, b in enumerate(valid_indices):
				pos_probs = torch.softmax(mask_logits[i, fill_pos_list[i], :], dim=-1)
				_, candidate_ids = torch.topk(pos_probs, k=args.topk_mask)
				used_ids = used_ids_list[i]
				refined_candidates = []
				for h_id in candidate_ids.tolist():
					hid = int(h_id)
					if hid <= 0 or hid in used_ids:
						continue
					if hid not in refined_candidates:
						refined_candidates.append(hid)
					if len(refined_candidates) >= args.topk_refined:
						break
				mask_top30 = refined_candidates
				hard_pool = [hid for hid in mask_top30 if hid != target_hero_list[i]]
				hard_negs = random.sample(hard_pool, min(args.hard_neg, len(hard_pool)))
				easy_pool = list(all_hero_pool - set(mask_top30) - used_ids)
				easy_negs = random.sample(easy_pool, min(args.easy_neg, len(easy_pool)))
				row = {
					"win_label": win_labels[b].item(),
					"masked_seq": winloss_masked_seq[b],
					"role_labels": winloss_masked_roles[b],
					"side_ids": side_template,
					"full_seq": full_seq[b].tolist(),
					"fill_pos": fill_pos_list[i],
					"target_hero": target_hero_list[i],
					"mask_top30": mask_top30,
					"hard_negs": hard_negs,
					"easy_negs": easy_negs,
					"query_id": query_id,
				}
				rows.append(row)
				query_id += 1
			pbar.set_postfix(samples=len(rows))
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with out_path.open("w", encoding="utf-8") as f:
			for row in rows:
				f.write(json.dumps(row, ensure_ascii=False) + '\n')
		print(f"Saved {len(rows)} rerank samples to {out_path}")


		
        

@torch.no_grad()
def demo_mask_prediction():
	args = build_parser().parse_args()
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	num_heroes = get_number_of_heroes()
	if args.device == "cuda" and torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=64).to(device)
	state_dict = torch.load(args.model_path, map_location=device)
	model.load_state_dict(state_dict, strict=False)
	model.eval()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	seq = [78,
      52,
      50,
      71,
      67,
	  129,
      54,
      119,
      35,
      68]
	fill_pos=4
	seq[fill_pos] = 0
	target = 67
	heroidtoname = load_hero_id_to_name()
	side = [0,0,0,0,0,1,1,1,1,1]
	role = [0,0,0,0,0,0,0,0,0,0]
	seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
	side_tensor = torch.tensor([side], dtype=torch.long).to(device)
	role_tensor = torch.tensor([role], dtype=torch.long).to(device)
	mask_logits, _, role_logits = model(seq_tensor, side_tensor, role_tensor)
	mask_probs = torch.softmax(mask_logits, dim=-1)
	role_probs = torch.softmax(role_logits, dim=-1)
	print("seq =", [heroidtoname.get(h, f"Unknown Hero {h}") for h in seq])
	print('target =', heroidtoname.get(target, f"Unknown Hero {target}"))
	for pos, val in enumerate(seq):
		if val == 0:
			probs = mask_probs[0, pos, :].detach().cpu().numpy()
			topk = min(30, len(probs))
			topk_ids = probs.argsort()[-topk:][::-1]
			topk_name = [heroidtoname.get(int(h_id), f"Unknown Hero {h_id}") for h_id in topk_ids]
			topk_scores = [probs[i] for i in topk_ids]
			print(f"Position {pos}: top-{topk} hero ids: {topk_name} probs: {[round(float(s),4) for s in topk_scores]}")
	

    
if __name__ == "__main__":
	# main()
	demo_mask_prediction()

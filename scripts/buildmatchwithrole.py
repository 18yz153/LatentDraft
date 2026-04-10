import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build per-match role-ordered hero ids from matchrole data"
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/matchrole.json"),
		help="Input JSON with player rows: match_id, player_slot, hero_id, lane_role, gold_per_min, tower_damage",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/match_train.json"),
		help="Output JSON path",
	)
	parser.add_argument(
		"--no-shuffle-order",
		action="store_true",
		help="Keep fixed [1,2,3,4,5] order per side (default is shuffled)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used when shuffling per-side output order",
	)
	return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
	with path.open("r", encoding="utf-8") as f:
		rows = json.load(f)
	if not isinstance(rows, list):
		raise ValueError("input file must be a JSON array")
	return rows


def is_radiant(player_slot: int) -> bool:
	return int(player_slot) < 128


def assign_team_positions(team_rows: List[Dict]) -> Tuple[Dict[int, int], bool]:
	"""
	Return role->hero_id mapping for one team.

	Rules from user:
	- Role 1/5 are split from lane_role==1 by GPM:
	  highest GPM -> pos1, lowest GPM -> pos5
	- Role 3/4 are split from lane_role==3 by GPM:
	  highest GPM -> pos3, lowest GPM -> pos4
	- Role 2 comes from lane_role==2
	"""
	if len(team_rows) != 5:
		return {}, False

	lane1 = [r for r in team_rows if int(r.get("lane_role", -1)) == 1]
	lane2 = [r for r in team_rows if int(r.get("lane_role", -1)) == 2]
	lane3 = [r for r in team_rows if int(r.get("lane_role", -1)) == 3]

	lane1.sort(key=lambda x: float(x.get("gold_per_min", 0.0)), reverse=True)
	lane2.sort(key=lambda x: float(x.get("gold_per_min", 0.0)), reverse=True)
	lane3.sort(key=lambda x: float(x.get("gold_per_min", 0.0)), reverse=True)

	assigned: Dict[int, int] = {}
	used_slots = set()

	# lane_role 1 -> position 1 (highest), position 5 (lowest)
	if lane1:
		assigned[1] = int(lane1[0]["hero_id"])
		used_slots.add(int(lane1[0]["player_slot"]))
	if len(lane1) >= 2:
		assigned[5] = int(lane1[-1]["hero_id"])
		used_slots.add(int(lane1[-1]["player_slot"]))

	# lane_role 2 -> position 2
	if lane2:
		assigned[2] = int(lane2[0]["hero_id"])
		used_slots.add(int(lane2[0]["player_slot"]))

	# lane_role 3 -> position 3 (highest), position 4 (lowest)
	if lane3:
		assigned[3] = int(lane3[0]["hero_id"])
		used_slots.add(int(lane3[0]["player_slot"]))
	if len(lane3) >= 2:
		assigned[4] = int(lane3[-1]["hero_id"])
		used_slots.add(int(lane3[-1]["player_slot"]))

	# Fallback: fill any missing positions by remaining players' GPM.
	missing_positions = [p for p in [1, 2, 3, 4, 5] if p not in assigned]
	if missing_positions:
		leftovers = [r for r in team_rows if int(r.get("player_slot", -1)) not in used_slots]
		leftovers.sort(key=lambda x: float(x.get("gold_per_min", 0.0)), reverse=True)
		for p, r in zip(missing_positions, leftovers):
			assigned[p] = int(r["hero_id"])

	is_complete = all(p in assigned for p in [1, 2, 3, 4, 5])
	return assigned, is_complete


def build_matches(rows: List[Dict], shuffle_order: bool, seed: int) -> Tuple[List[Dict], Dict[str, int], List[Dict]]:
	grouped: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: {"radiant": [], "dire": []})
	rng = random.Random(seed)

	for r in rows:
		try:
			match_id = int(r["match_id"])
			player_slot = int(r["player_slot"])
			_ = int(r["hero_id"])
		except (KeyError, TypeError, ValueError):
			continue

		side = "radiant" if is_radiant(player_slot) else "dire"
		grouped[match_id][side].append(r)

	out: List[Dict] = []
	stats = {
		"total_matches_seen": 0,
		"kept_matches": 0,
		"dropped_incomplete_team": 0,
		"dropped_tower_damage_tie": 0,
	}
	tower_damage_ties: List[Dict] = []

	for match_id, sides in grouped.items():
		stats["total_matches_seen"] += 1

		rad_map, rad_ok = assign_team_positions(sides["radiant"])
		dire_map, dire_ok = assign_team_positions(sides["dire"])
		if not rad_ok or not dire_ok:
			stats["dropped_incomplete_team"] += 1
			continue

		rad_tower_damage = sum(float(r.get("tower_damage", 0.0)) for r in sides["radiant"])
		dire_tower_damage = sum(float(r.get("tower_damage", 0.0)) for r in sides["dire"])
		if rad_tower_damage == dire_tower_damage:
			stats["dropped_tower_damage_tie"] += 1
			tower_damage_ties.append(
				{
					"match_id": match_id,
					"radiant_tower_damage": rad_tower_damage,
					"dire_tower_damage": dire_tower_damage,
				}
			)
			continue
		rad_winlabel = 1 if rad_tower_damage > dire_tower_damage else 0

		rad_order = [1, 2, 3, 4, 5]
		dire_order = [1, 2, 3, 4, 5]
		if shuffle_order:
			rng.shuffle(rad_order)
			rng.shuffle(dire_order)

		rad_hero_ids = [rad_map[i] for i in rad_order]
		dire_hero_ids = [dire_map[i] for i in dire_order]

		out.append(
			{
				"match_id": match_id,
				"rad_hero_ids": rad_hero_ids,
				"dire_hero_ids": dire_hero_ids,
				"rad_positions": rad_order,
				"dire_positions": dire_order,
				"rad_winlabel": rad_winlabel,
			}
		)
		stats["kept_matches"] += 1

	return out, stats, tower_damage_ties


def main() -> None:
	args = parse_args()
	rows = load_rows(args.input)
	matches, stats, tower_damage_ties = build_matches(
		rows,
		shuffle_order=not args.no_shuffle_order,
		seed=args.seed,
	)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w", encoding="utf-8") as f:
		json.dump(matches, f, ensure_ascii=False, indent=2)

	print(f"saved -> {args.output}")
	print(json.dumps(stats, ensure_ascii=False))
	if tower_damage_ties:
		print("tower_damage_ties:")
		print(json.dumps(tower_damage_ties, ensure_ascii=False))
	if matches:
		print("example:")
		print(json.dumps(matches[0], ensure_ascii=False))


if __name__ == "__main__":
	main()


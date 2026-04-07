import argparse
import json
from pathlib import Path


def extract_url_name(raw_name: str) -> str:
	prefix = "npc_dota_hero_"
	if raw_name.startswith(prefix):
		return raw_name[len(prefix):]
	return raw_name


def convert(input_path: Path, output_path: Path) -> None:
	with input_path.open("r", encoding="utf-8") as f:
		heroes = json.load(f)

	id_to_url_name = {}
	for key, hero in heroes.items():
		hero_id = int(hero.get("id", key))
		raw_name = str(hero.get("name", ""))
		id_to_url_name[str(hero_id)] = extract_url_name(raw_name)

	id_to_url_name = dict(sorted(id_to_url_name.items(), key=lambda kv: int(kv[0])))

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(id_to_url_name, f, ensure_ascii=False, indent=2)

	print(f"saved -> {output_path}")
	print(f"count -> {len(id_to_url_name)}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert heroes.json to id->url_name json")
	parser.add_argument("--input", type=Path, default=Path("heroes.json"))
	parser.add_argument("--output", type=Path, default=Path("hero_id_to_url_name.json"))
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	convert(args.input, args.output)


if __name__ == "__main__":
	main()

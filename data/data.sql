SELECT 
    plm.match_id,
    plm.player_slot,
    plm.hero_id,
    plm.lane_role,
    plm.gold_per_min
FROM player_matches plm
where lane_role IS NOT NULL
ORDER BY plm.match_id DESC -- 加上倒序，确保抽出来的是较新的比赛
LIMIT 500000;


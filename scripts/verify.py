import json
from collections import defaultdict
from itertools import combinations
import numpy as np

def test_data_entropy(filepath="data/allmatch.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        matches = json.load(f)
        
    # 用来统计每个 3 人组合赢了多少局，输了多少局
    # Key: (英雄A, 英雄B, 英雄C) 升序排列
    # Value: [赢的次数, 输的次数]
    lineup_stats = defaultdict(lambda: [0, 0])
    
    for m in matches:
        try:
            # 必须排序！因为选人顺序不影响最终阵容
            rad_team = tuple(sorted([int(x) for x in m["radiant_team"]]))
            dire_team = tuple(sorted([int(x) for x in m["dire_team"]]))
            rad_win = bool(m.get("radiant_win", False))
            
            # 规则改为：5个英雄里任意3个英雄组合都计入
            # 若队伍里有重复英雄，先去重后再取3人组合
            rad_unique = sorted(set(rad_team))
            dire_unique = sorted(set(dire_team))
            if len(rad_unique) >= 3 and len(dire_unique) >= 3:
                rad_combos = list(combinations(rad_unique, 3))
                dire_combos = list(combinations(dire_unique, 3))

                if rad_win:
                    for c in rad_combos:
                        lineup_stats[c][0] += 1
                    for c in dire_combos:
                        lineup_stats[c][1] += 1
                else:
                    for c in rad_combos:
                        lineup_stats[c][1] += 1
                    for c in dire_combos:
                        lineup_stats[c][0] += 1
        except Exception:
            continue

    # 过滤出出现次数超过 20 次的热门阵容
    threshold = 20
    popular_lineups = {k: v for k, v in lineup_stats.items() if sum(v) >= threshold}
    
    win_rates = []
    for lineup, stats in popular_lineups.items():
        total = sum(stats)
        wr = stats[0] / total
        win_rates.append(wr)
        
    # 计算这些极度公式化阵容的平均胜率偏离度
    if win_rates:
        win_rates = np.array(win_rates)
        print(f"找到了 {len(popular_lineups)} 个出现 >={threshold} 次的 3 人组合。")
        print(f"它们的胜率分布:")
        print(f"  - 最低胜率: {win_rates.min():.2%}")
        print(f"  - 最高胜率: {win_rates.max():.2%}")
        print(f"  - 绝对偏离 50% 的平均值: {np.abs(win_rates - 0.5).mean():.2%}")
    else:
        print("没有找到高频重复阵容。")

if __name__ == "__main__":
    test_data_entropy()
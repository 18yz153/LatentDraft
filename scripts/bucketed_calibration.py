import numpy as np
import pandas as pd
from src.model import DotaMultiTaskTransformer
from src.dataset import build_rerank_dataloader
from src.utils import get_number_of_heroes
import torch
from tqdm import tqdm


def calculate_bucketed_accuracy(y_true, y_pred_prob):
    """
    计算分桶校准准确率
    y_true: 真实的胜负标签 (1 = Radiant赢, 0 = Dire赢)
    y_pred_prob: 模型预测 Radiant 赢的概率 (Sigmoid 后的输出，0~1)
    """
    
    # 1. 将概率转化为“预测谁赢”和“置信度”
    # 如果 prob >= 0.5，预测 1 (Radiant)；如果 prob < 0.5，预测 0 (Dire)
    predicted_winner = (y_pred_prob >= 0.5).astype(int)
    
    # 置信度：离 0.5 越远，置信度越高 (永远在 0.5 ~ 1.0 之间)
    confidence = np.maximum(y_pred_prob, 1 - y_pred_prob)
    
    # 判断模型是否预测正确
    is_correct = (predicted_winner == y_true).astype(int)
    
    # 2. 构建 DataFrame
    df = pd.DataFrame({
        'Confidence': confidence,
        'Is_Correct': is_correct
    })
    
    # 3. 定义你想要的桶 (Bins) 的边界 (和 LoLDraftAI 保持一致)
    bins = [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 1.0]
    labels = ['50-52%', '52-55%', '55-57%', '57-60%', '60-62%', 
              '62-65%', '65-68%', '68-70%', '70-75%', '75%+']
    
    # 把每个预测分到对应的桶里
    df['Bucket'] = pd.cut(df['Confidence'], bins=bins, labels=labels, include_lowest=True)
    
    # 4. 分组统计每个桶的数据
    result = df.groupby('Bucket', observed=False).agg(
        Number_of_Games=('Is_Correct', 'count'),          # 桶里的总场数
        Model_Accuracy=('Is_Correct', 'mean'),            # 实际预测准确率
        Expected_Accuracy=('Confidence', 'mean')          # 平均预测置信度
    ).reset_index()
    
    # 过滤掉没有样本的桶
    result = result[result['Number_of_Games'] > 0]
    
    return result

# ==========================================
# 🧪 模拟测试你的数据
# ==========================================
# 假设这是你用 Full Seq 模型跑测试集得到的结果
# 你只需要替换这两列真实的 numpy array 即可
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heroes = get_number_of_heroes()
    model = DotaMultiTaskTransformer(num_heroes=num_heroes, embed_dim=64)

    # ⚠️ 极其重要：你需要加载你训练好的、能跑到 0.558 AUC 的模型权重！
    model.load_state_dict(torch.load('models/stage3_value_network_best.pt')) 

    model.to(device)
    model.eval() # 开启评估模式，关闭 Dropout 等

    # 2. 准备 DataLoader 
    # 注意：跑评估的时候用的是测试集，别用训练集，不然是作弊
    _, val_loader = build_rerank_dataloader(batch_size=1024, shuffle=False, num_workers=2, xgb=False)

    # 用于收集结果的列表
    all_labels = []
    all_probs = []

    print("🚀 开始跑测试集，收集分桶预测数据...")
    # 3. 推理循环
    with torch.no_grad(): # 节省显存，不计算梯度
        for batch in tqdm(val_loader):
            # 【注意】这里需要根据你 dataloader 实际 yield 的格式来解包
            # 假设你的 batch 返回的是: hero_ids, side_ids, role_labels, win_labels
            # 如果你没传 role_labels，就把对应的地方设为 None
            
            # 解析数据并放到 GPU/CPU 上
            masked_seq = batch['masked_seq'].to(device)      # [B, 10]
            side = batch['side_ids'].to(device)       # [B, 10]
            role = batch['role_labels'].to(device)    # [B, 10]
            full_seq = batch['full_seq'].to(device)   # 🌟 必须拿到完整阵容
            fill_pos = batch['fill_pos'].to(device)   # [B]
            soft_labels = batch['win_label'].to(device).float().unsqueeze(-1)
                
            win_labels = (soft_labels > 0.5).float()
            # 4. 模型前向传播
            _, win_logits, _ = model(masked_seq, side, role)
            
            # 5. 将 Logits 转换为概率 (Sigmoid)
            # win_logits 出来是 [-inf, +inf] 的数，用 sigmoid 压到 [0, 1]
            win_probs = torch.sigmoid(win_logits).squeeze() 
            
            # 6. 收集这一个 batch 的结果
            # 注意把 tensor 移回 cpu 并转成 numpy
            all_labels.extend(win_labels.cpu().numpy())
            all_probs.extend(win_probs.cpu().numpy())

    # 7. 转化为 numpy array，喂给你刚刚写的统计函数
    
    y_true = np.array(all_labels).flatten()
    y_pred_prob = np.array(all_probs).flatten()

    print("✅ 数据收集完毕！开始生成 Calibration Table...")

    calibration_table = calculate_bucketed_accuracy(y_true, y_pred_prob)

    # 打印出可以放在论文里的结果
    print("\n=== Model Accuracy Results ===")
    print(calibration_table.to_string(index=False))

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotaMultiTaskTransformer(nn.Module):
    def __init__(self, num_heroes, embed_dim=64, nhead=8, num_layers=3):
        super().__init__()
        self.pad_token_id = 0
        # 1. 共享 Embedding 层 (这就是你之后用来查“谁和 Puck 近”的)
        # vocab: [PAD=0] + [real heroes: 1..num_heroes]
        self.hero_emb = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=self.pad_token_id)
        self.side_emb = nn.Embedding(2, embed_dim) # 0: 盟友, 1: 敌人
        self.win_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 2. 共享 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 头 A: Mask 预测 (输出 130 维，看哪个英雄概率最高)
        self.mask_head = nn.Linear(embed_dim, num_heroes+1)

        # 4. 头 B: 胜率预测 (输出 1 维)
        concat_dim = embed_dim +10*embed_dim +10
        self.win_head = nn.Sequential(
            # 第一层：宽一点，捕捉原始对位特征
            nn.Linear(concat_dim, embed_dim*4),
            nn.LayerNorm(embed_dim*4),
            nn.ReLU(),
            
            # 第二层：压缩并提取高级战术语义（如阵容控制链、后期能力）
            nn.Linear(embed_dim*4, embed_dim*2),
            nn.LayerNorm(embed_dim*2),
            nn.ReLU(),
            
            # 输出层
            nn.Linear(embed_dim*2, 1)
        )
        # role head
        self.role_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5) # 输出 5 个位置的 logits
        )

    def forward(self, hero_ids, side_ids, role_labels=None):
        x = self.hero_emb(hero_ids) + self.side_emb(side_ids)

        batch_size = x.shape[0]
        # 把单个 win_token 扩展到当前 batch_size
        w_token = self.win_token.expand(batch_size, -1, -1)
        x = torch.cat([w_token, x], dim=1)

        features = self.transformer(x) 
        
        hero_features = features[:, 1:, :]
        global_features = features[:, 0, :]
        
        role_logits = self.role_head(hero_features)
        mask_logits = self.mask_head(hero_features)

        if role_labels is None:
            # 如果没传 role_labels，就生成全 0 的张量，表示“全靠系统自动分析”
            role_labels = torch.zeros((batch_size, 10), dtype=torch.long, device=hero_ids.device)
        role_probs = F.softmax(role_logits, dim=-1) 
        
        # 2. 将输入的真实 Role (1-5) 转化为 One-Hot 向量
        hard_role_probs = torch.zeros_like(role_probs)
        # 找出哪些位置是已知的 (大于0)
        valid_mask = (role_labels > 0) # [Batch, 10]
        
        # 把 1~5 映射到索引 0~4，并用 scatter_ 填入 1.0
        safe_indices = torch.clamp(role_labels - 1, min=0)
        hard_role_probs.scatter_(2, safe_indices.unsqueeze(-1), 1.0)
        
        # 3. 完美融合 (Where): 已知的用 One-Hot，未知的 (0) 用模型预测的 Probs
        valid_mask_expanded = valid_mask.unsqueeze(-1) # [Batch, 10, 1]
        final_role_features = torch.where(valid_mask_expanded, hard_role_probs, role_probs)

        # ==========================================
        rad_features = hero_features[:, :5, :] # [Batch, 5, 64]
        dire_features = hero_features[:, 5:, :] # [Batch, 5, 64]
        rad_roles = final_role_features[:, :5, :] # [Batch, 5, 5]
        dire_roles = final_role_features[:, 5:, :] # [Batch, 5, 5]
        rad_slots = torch.bmm(rad_roles.transpose(1, 2), rad_features) 
        dire_slots = torch.bmm(dire_roles.transpose(1, 2), dire_features)

        # 3. 压平 10 个槽位特征 [Batch, 640]
        # 这里就是你说的：固定的 1,2,3,4,5, 1,2,3,4,5 英雄特征排列
        structured_features = torch.cat([rad_slots, dire_slots], dim=1).view(batch_size, -1)
        rad_counts = rad_roles.sum(dim=1) # [Batch, 5]
        dire_counts = dire_roles.sum(dim=1) # [Batch, 5]
        occupancy = torch.cat([rad_counts, dire_counts], dim=-1) # [Batch, 10]
        combined_features = torch.cat([
            global_features,     # 大局观
            structured_features, # 1-5号位对阵图
            occupancy            # 冲突/空位报警器
        ], dim=-1)
        
        win_logits = self.win_head(combined_features)

        
        return mask_logits, win_logits, role_logits
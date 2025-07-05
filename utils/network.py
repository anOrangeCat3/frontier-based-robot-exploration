import torch
import torch.nn as nn
import math

from utils.parameter import *


class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim=EMBEDDING_DIM):
        super().__init__()
        self.x_embed = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        traj: [B, 64, 3] -> [B, 64, 64]
        frontier: [B, 16, 2] -> [B, 16, 64]
        """
        x_feat = self.x_embed(x)          # [B, 64, 64]
        return x_feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=EMBEDDING_DIM, max_len=MAX_EPISODE_STEP):
        super().__init__()
        # 生成固定位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 保存为buffer，不更新梯度

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    

class CrossMultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim=EMBEDDING_DIM,
                 num_heads=NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads, 
                                          batch_first=True) 

    def forward(self, traj_embed, traj_mask, frontier_embed):
        '''
        traj_embed: [B, 64, 64]
        traj_mask: [B, 64]
        frontier_embed: [B, 16, 64]
        '''

        # Attention: Q=traj, K=V=frontier
        out, weights = self.attn(
            query=traj_embed,
            key=frontier_embed,
            value=frontier_embed,
            key_padding_mask=None  # 可添加前沿点 mask（如果有）
        )  # out: [B, 64, 64]

        # 处理 traj_mask: 将 padding 的输出置为 0
        if traj_mask is not None:
            traj_mask = traj_mask.unsqueeze(-1)  # [B, 64, 1]
            out = out.masked_fill(traj_mask, 0.0)

        return out  # [B, 64, 64]


class Normalization(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = CrossMultiheadAttention()
        self.norm1 = Normalization()
        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, EMBEDDING_DIM)
        )
        self.norm2 = Normalization()


    def forward(self, traj_embed, traj_mask, frontier_embed):
        """
        traj_embed: [B, 64, 64]
        traj_mask: [B, 64] (bool)
        frontier_embed: [B, 16, 64]
        """
        attn_out = self.cross_attn(traj_embed, traj_mask, frontier_embed)  # [B, 64, 64]
        x = self.norm1(attn_out + traj_embed)  # residual connection

        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x  # [B, 64, E]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改：enbedding_layer -> embedding_layer
        self.traj_embedding_layer = Embedding(input_dim=3)      
        self.frontier_embedding_layer = Embedding(input_dim=2)
        self.positional_encoding = PositionalEncoding()
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(NUM_LAYERS)])

    def forward(self, frontier, traj, traj_mask):
        # 修改：self.enbedding_layer -> self.embedding_layer
        traj_embed = self.traj_embedding_layer(traj)     
        traj_embed = self.positional_encoding(traj_embed)
        frontier_embed = self.frontier_embedding_layer(frontier)     
        
        # 修复：逐层更新轨迹特征
        traj_enc = traj_embed
        for layer in self.encoder_layers:
            traj_enc = layer(traj_enc, traj_mask, frontier_embed)  # 逐层更新
        
        return traj_enc, frontier_embed


class PolicyNet(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, frontier_num=N_CLUSTERS):
        super().__init__()
        self.embed_dim = embed_dim
        self.frontier_num = frontier_num
        self.encoder = Encoder()
        
        # Actor的MLP头，输入轨迹全局向量 + 单个前沿点特征拼接
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, frontier, traj, traj_mask):
        """
        frontier_embed: [B, N, 64] 前沿点特征
        traj_enc: [B, 64, 64] 来自Encoder的轨迹编码
        traj_mask: [B, 64] 轨迹掩码, True表示valid位置
        """
        traj_enc, frontier_embed = self.encoder(frontier, traj, traj_mask)
        # 1. 对前沿点进行embedding
        N = frontier_embed.size(1)

        # 2. 使用traj_mask对轨迹编码进行加权pooling，得到全局轨迹状态
        # 假设traj_mask中True表示valid位置
        traj_mask_f = traj_mask.unsqueeze(-1).float()  # [B, 64, 1]
        # 计算有效位置的加权平均
        h_state = (traj_enc * traj_mask_f).sum(dim=1) / traj_mask_f.sum(dim=1).clamp(min=1e-6)  # [B, 64]

        # 3. 扩展轨迹全局状态以匹配前沿点数量
        h_state_exp = h_state.unsqueeze(1).expand(-1, N, -1)  # [B, N, 64]

        # 4. 拼接轨迹全局状态和前沿点特征
        combined = torch.cat([h_state_exp, frontier_embed], dim=-1)  # [B, N, 128]

        # 5. 输出每个前沿点的选择logits
        logits = self.mlp(combined).squeeze(-1)  # [B, N]

        return logits


class QNet(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, frontier_num=N_CLUSTERS):
        super().__init__()
        self.embed_dim = embed_dim
        self.frontier_num = frontier_num
        self.encoder = Encoder()
        
        # Critic的MLP头
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, frontier, traj, traj_mask):
        """
        frontier_embed: [B, N, 64] 前沿点特征
        traj_enc: [B, 64, 64] 来自Encoder的轨迹编码
        traj_mask: [B, 64] 轨迹掩码, True表示valid位置
        """
        traj_enc, frontier_embed = self.encoder(frontier, traj, traj_mask)
        # 1. 对前沿点进行embedding
        N = frontier_embed.size(1)

        # 2. 使用traj_mask对轨迹编码进行加权pooling，得到全局轨迹状态
        if traj_mask is not None:
            # 假设traj_mask中True表示valid位置
            traj_mask_f = traj_mask.unsqueeze(-1).float()  # [B, 64, 1]
            # 计算有效位置的加权平均
            h_state = (traj_enc * traj_mask_f).sum(dim=1) / traj_mask_f.sum(dim=1).clamp(min=1e-6)  # [B, 64]
        else:
            # 如果没有mask，直接平均
            h_state = traj_enc.mean(dim=1)  # [B, 64]

        # 3. 扩展轨迹全局状态以匹配前沿点数量
        h_state_exp = h_state.unsqueeze(1).expand(-1, N, -1)  # [B, N, 64]

        # 4. 拼接轨迹全局状态和前沿点特征
        combined = torch.cat([h_state_exp, frontier_embed], dim=-1)  # [B, N, 128]

        # 5. 输出每个前沿点的Q值
        q_values = self.mlp(combined).squeeze(-1)  # [B, N]

        return q_values

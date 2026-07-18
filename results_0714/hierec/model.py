"""C-HMCNN 모델: ESM-3B feature -> base MLP -> node 확률 -> MCM(계층 일관 강제)."""
import torch
import torch.nn as nn


class MLPBase(nn.Module):
    """C-HMCNN 의 base network h. frozen ESM-3B feature 위의 가벼운 head."""
    def __init__(self, in_dim, n_nodes, hidden=1024, layers=2, dropout=0.3):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout)]
            d = hidden
        mods += [nn.Linear(d, n_nodes)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return torch.sigmoid(self.net(x))          # 노드별 확률 (0~1)


def get_constr_out(x, dst, src, N):
    """MCM: 각 노드 = 자기+자손 확률의 max -> 부모>=자식 (계층 일관) by construction.

    x:(B,N) 확률. dst(조상)/src(자손) 는 build_hierarchy 의 scatter 쌍.
    out[:,i] = max_{j in subtree(i)} x[:,j]  를 (B,N,N) 없이 scatter_reduce 로 계산.
    """
    B = x.shape[0]
    out = torch.zeros(B, N, device=x.device, dtype=x.dtype)
    vals = x[:, src]                                # (B, P) 자손 확률
    idx = dst.unsqueeze(0).expand(B, -1)            # (B, P) 조상 위치
    out = out.scatter_reduce(1, idx, vals, reduce='amax', include_self=True)
    return out

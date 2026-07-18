"""C-HMCNN Max-Constraint Loss (MCLoss).

핵심 트릭: 음성 라벨은 MCM(전체 자손 max)으로, 양성 라벨은 '양성 자손만'의 max 로
확률을 만든 뒤 BCE. 이렇게 하면 하위(자손) 예측이 상위(조상) 예측을 돕도록 학습됨.
"""
import torch
import torch.nn.functional as F
from model import get_constr_out


def mc_loss(h, y, dst, src, N, node_mask=None, posw=None):
    """h:(B,N) base 확률, y:(B,N) 정답, node_mask:(B,N) 손실에 반영할 노드(불완전 EC 제외).
    posw:(N,) 노드별 pos_weight(neg/pos, 클리핑). 있으면 양성 칸을 가중 → 불균형 상쇄
    (하위 레벨 all-zero-F1 함정 방지, ECPICK §S4 방식과 동일)."""
    constr = get_constr_out(h, dst, src, N)        # 전체 자손 max
    pos = get_constr_out(y * h, dst, src, N)       # 양성 자손만의 max
    out = (1.0 - y) * constr + y * pos
    bce = F.binary_cross_entropy(out.clamp(1e-7, 1 - 1e-7), y, reduction='none')
    if posw is not None:
        bce = bce * (y * posw + (1.0 - y))         # 양성=pos_weight배, 음성=1
    if node_mask is not None:
        bce = bce * node_mask
        return bce.sum() / node_mask.sum().clamp(min=1.0)
    return bce.mean()

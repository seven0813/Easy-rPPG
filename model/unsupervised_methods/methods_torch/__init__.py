"""可微的传统 rPPG 方法（torch 实现）。

与 ``methods/`` 下的 numpy/scipy 版本数学一致，但全程可微、支持 batch 与
GPU，可直接作为训练损失中的"可微回读器"：把 edited video 读成 BVP 波形，
再与目标波形比较，从而在没有可训练参数、无合谋风险的前提下，为视频编辑
提供指向生理信号的梯度监督。

已实现（均可微）：

- :func:`pos_wang_torch` —— POS，滑窗平面正交投影 + 去趋势 + 带通；
- :func:`chrom_dehaan_torch` —— CHROM，分窗色度投影 + Hann 重叠相加；
- :func:`green_torch` —— GREEN，绿通道均值基线；
- :func:`lgi_torch` —— LGI，SVD 主方向投影；
- :func:`omit_torch` —— OMIT，QR 主方向投影；
- :func:`pbv_torch` —— PBV，血容签名向量投影。

未实现可微版本：**ICA（Poh 2010 / JADE）**。JADE 依赖数据依赖的迭代联合
对角化（Givens 旋转 + 收敛阈值判据）、复数特征分解，以及用频谱峰值 argmax
选源，这些是不可微的控制流与离散选择，强行反传会得到不稳定/无意义的梯度；
且其盲源分离能力已被上述投影族方法覆盖。因此 ICA 仅保留 numpy 版
（``methods/ICA_POH.py``）用于 held-out 评估，不作为可微回读监督。
"""

from model.unsupervised_methods.methods_torch.pos_wang import pos_wang_torch
from model.unsupervised_methods.methods_torch.chrom_dehaan import (
    chrom_dehaan_torch,
)
from model.unsupervised_methods.methods_torch.green import green_torch
from model.unsupervised_methods.methods_torch.lgi import lgi_torch
from model.unsupervised_methods.methods_torch.omit import omit_torch
from model.unsupervised_methods.methods_torch.pbv import pbv_torch

__all__ = [
    "pos_wang_torch",
    "chrom_dehaan_torch",
    "green_torch",
    "lgi_torch",
    "omit_torch",
    "pbv_torch",
]

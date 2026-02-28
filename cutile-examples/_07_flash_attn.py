import cuda.tile as ct
import torch
import math

INV_LOG_2 = math.log(2)

@ct.kernel
def flash_attn(
    Q: ct.Array, K: ct.Array, V: ct.Array, O: ct.Array,
    tileS: ct.Constant, tileD: ct.Constant
):
    block_idx_B = ct.bid(0)
    block_idx_H = ct.bid(1)
    block_idx_S = ct.bid(2)

    tileQ = ct.load(Q, (block_idx_B, block_idx_S, block_idx_H, 0), (1, tileS, 1, tileD)).reshape((tileS, tileD))
    tileQ *= INV_LOG_2
    acc = ct.full((tileS, tileD), 0.0, dtype=ct.float32)
    row_sum = ct.full((tileS, 1), 0.0, dtype=ct.float32)
    row_max = ct.full((tileS, 1), 0.0, dtype=ct.float32)


    for iterS in range(ct.cdiv(K.shape[1], tileS)):
        tileK = ct.load(K, (block_idx_B, iterS, block_idx_H, 0), (1, tileD, 1, tileS)).reshape((tileS, tileD))
        tileK_T = tileK.transpose(-1, -2)
        tileV = ct.load(V, (block_idx_B, iterS, block_idx_H, 0), (1, tileS, 1, tileD)).reshape((tileS, tileD))
        
        # [tileS, tileS]
        tileQK = ct.matmul(tileQ, tileK_T)
        M = ct.max(tileQK, axis=-1).reshape((tileS, 1))
        M = ct.maximum(M, row_max)
        deltaM = row_max - M

        # Softmax
        tileQK = ct.exp2(tileQK - M)
        row_sum = row_sum * ct.exp2(deltaM) + ct.sum(tileQK, axis=-1).reshape((tileS, 1))
        acc = ct.mma(tileQK, tileV, acc * ct.exp2(deltaM))
        row_max = M
    
    acc = acc / row_sum.reshape((tileS, 1))
    acc = acc.reshape((1, tileS, 1, tileD))
    ct.store(O, (block_idx_B, block_idx_S, block_idx_H, 0), acc.astype(O.dtype))
        

B, S, H, D = 1, 128, 8, 32
Q = torch.randn(size=[B, S, H, D], dtype=torch.float32, device="cuda")
K = torch.randn(size=[B, S, H, D], dtype=torch.float32, device="cuda")
V = torch.randn(size=[B, S, H, D], dtype=torch.float32, device="cuda")
O = torch.empty_like(Q)

Q_ = Q.permute(0, 2, 1, 3)
K_ = K.permute(0, 2, 1, 3)
V_ = V.permute(0, 2, 1, 3)
real = torch.softmax(Q_ @ K_.transpose(-1, -2), dim=-1) @ V_
real = real.permute(0, 2, 1, 3)

ct.launch(
    torch.cuda.current_stream(),
    (B, H, ct.cdiv(S, 32)),
    flash_attn,
    (Q, K, V, O, 32, D),
)

print(O - real)
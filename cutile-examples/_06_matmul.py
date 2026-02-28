import cuda.tile as ct
import torch

@ct.kernel
def matmul(
    A: ct.Array, B: ct.Array, O: ct.Array,
    tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant,
    transposeA: ct.Constant, transposeB: ct.Constant
):
    """
    O = A * B

    A: [M, K]
    B: [K, N]
    O: [M, N]

    """
    block_idx_M, block_idx_N = ct.bid(0), ct.bid(1)
    K = A.shape[0] if transposeA else A.shape[-1]
    num_iter_k = ct.cdiv(K, tileK)

    acc = ct.full((tileM, tileN), 0.0, dtype=ct.float32)
    for iter_k in range(num_iter_k):
        # [tileM, tileK]
        A_tile = ct.load(A, index=(block_idx_M, iter_k), shape=(tileM, tileK), order="F" if transposeA else "C")
        # [tileK, tileN]
        B_tile = ct.load(B, index=(iter_k, block_idx_N), shape=(tileK, tileN), order="F" if transposeB else "C")

        # [tileM, tileN]
        acc = ct.mma(A_tile, B_tile, acc=acc)
    
    ct.store(O, (block_idx_M, block_idx_N), acc.astype(O.dtype)) 

M, N, K = 4096, 4096, 4096
tileM, tileN, tileK = 128, 64, 32
A = torch.randn(size=[M, K], dtype=torch.float16, device="cuda")
B = torch.randn(size=[K, N], dtype=torch.float16, device="cuda")
O = torch.empty(size=[M, N], dtype=torch.float16, device="cuda")

grid = (ct.cdiv(M, tileM), ct.cdiv(N, tileN))
ct.launch(torch.cuda.current_stream(), grid, matmul, (A, B, O, tileM, tileN, tileK, False, False))

real = A @ B
print(real - O)
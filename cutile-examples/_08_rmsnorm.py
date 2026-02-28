import torch
import cuda.tile as ct

@ct.kernel
def rms_norm(x: ct.Array, w: ct.Array, o: ct.Array, tile_size: ct.Constant, eps: float, dim: int):
    block_idx = ct.bid(0)
    tile_x = ct.load(x, (block_idx, 0), (1, tile_size), padding_mode=ct.PaddingMode.ZERO)
    tile_w = ct.load(w, (0, ), (tile_size, ), padding_mode=ct.PaddingMode.ZERO)

    tile_var = ct.sum(tile_x * tile_x * (1 / dim))
    tile_rsqrt = ct.rsqrt(tile_var + eps)

    tile_x = tile_w * tile_x * tile_rsqrt
    ct.store(o, (block_idx, 0), tile_x.astype(o.dtype))

M, N = 128, 1024
X = torch.randn(size=[M, N], dtype=torch.bfloat16, device="cuda")
W = torch.randn(size=[N], dtype=torch.bfloat16, device="cuda")
O = torch.empty_like(X)
real = torch.nn.functional.rms_norm(X, normalized_shape=[N], weight=W, eps=1e-7)
ct.launch(torch.cuda.current_stream(), (M, ), rms_norm, (X, W, O, 1024, 1e-7, 1024))
print(real - O)
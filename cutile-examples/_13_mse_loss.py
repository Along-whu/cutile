import torch
import cuda.tile as ct

@ct.kernel
def mse_loss(x: ct.Array, y:ct.Array, o: ct.Array, dodx: ct.Array, tile_size: ct.Constant, allow_tma: ct.Constant=True):
    
    M, N = x.shape[0], x.shape[1]
    inv_term = 1.0 / (M * N)
    
    # block_x -> N, block_y -> M
    block_x, block_y = ct.bid(0), ct.bid(1)
    
    tile_x = ct.load(x, (block_y, block_x), (1, tile_size), padding_mode=ct.PaddingMode.ZERO, allow_tma=allow_tma).astype(ct.float32)
    tile_y = ct.load(y, (block_y, block_x), (1, tile_size), padding_mode=ct.PaddingMode.ZERO, allow_tma=allow_tma).astype(ct.float32)
    
    loss = ct.sum(ct.pow(tile_x - tile_y, 2)) * inv_term
    
    tile_dodx = (tile_x - tile_y) * (2.0 * inv_term)
    ct.store(dodx, (block_y, block_x), tile_dodx.astype(dodx.dtype))
    ct.atomic_add(o, (0, ), loss.astype(o.dtype))
    
M, N = 16384, 1024
tile_size = 256
x = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
y = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
o = torch.zeros([1, ], device="cuda", dtype=torch.float32)
dodx = torch.empty_like(x)
real_loss = torch.nn.functional.mse_loss(x, y, reduction="mean")
ct.launch(
    torch.cuda.current_stream(), (ct.cdiv(N, tile_size), M),
    mse_loss, (x, y, o, dodx, tile_size, True)
)

print(real_loss - o)
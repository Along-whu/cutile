import torch
import cuda.tile as ct
import math

INV_LOG_2 = math.log(2)

@ct.kernel
def ct_softmax(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    # x, y: [M, N]
    block_id = ct.bid(0)

    tile_x = ct.load(x, (block_id, 0), (1, tile_size))
    tile_x.astype(ct.float32)
    tile_x *= INV_LOG_2
    # softmax: exp(x) / sum(exp(x))
    # exp(x) = exp2(x / ln2)
    tile_x = ct.exp2(tile_x - ct.max(tile_x))

    expsum = ct.sum(tile_x)
    tile_x = tile_x / expsum
    ct.store(y, (block_id, 0), tile_x.astype(y.dtype))

m, n = 15, 256
num_blocks, tile_size = m, n

x = torch.randn([m, n], device="cuda", dtype=torch.float16)
y = torch.empty_like(x)

ct.launch(
    torch.cuda.current_stream(),
    (num_blocks, ),
    ct_softmax,
    (x, y, tile_size),
)

real = torch.softmax(x, dim=-1)

print(real - y)
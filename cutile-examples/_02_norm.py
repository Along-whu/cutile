import torch
import cuda.tile as ct

@ct.function
def tile_norm(tile_x: ct.Tile, tile_size: ct.Constant, eps: float=1e-6) -> ct.Tile:
    mean = ct.sum(tile_x) / tile_size
    tile_x = tile_x - mean
    tile_x =  tile_x * ct.rsqrt(ct.sum(tile_x * tile_x) / tile_size + eps)
    return tile_x

@ct.kernel
def ct_norm(x: ct.Array, y: ct.Array, tile_size: ct.Constant):
    block_id = ct.bid(0)

    # x, y: [M, N]
    # tile_x: [N]
    tile_x = ct.load(x, index=(block_id, 0), shape=(1, tile_size))
    tile_x = tile_norm(tile_x, tile_size)

    ct.store(y, (block_id, 0), tile_x.astype(y.dtype))

x = torch.randn(size=[1024, 1024], device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
tile_size: int = 1024

num_blocks = x.shape[0]
ct.launch(
    torch.cuda.current_stream(), (num_blocks, ), 
    ct_norm, (x, y, tile_size)
)
real = torch.nn.functional.layer_norm(x, (1024, ))
print(real - y)
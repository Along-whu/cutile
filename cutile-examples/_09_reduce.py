import cuda.tile as ct
import torch

@ct.kernel
def sum_v1(A: ct.Array, O: ct.Array, tile_size: ct.Constant):
    # A [M(block_x 1:1), N]
    block_x = ct.bid(0)
    tile_A = ct.load(
        A, (block_x, 0), (1, tile_size), padding_mode=ct.PaddingMode.ZERO,
    ).reshape((tile_size, ))
    tile_A = ct.sum(tile_A)
    ct.store(O, (block_x, ), tile_A)

@ct.kernel
def sum_v2(A: ct.Array, O: ct.Array, tile_size: ct.Constant):
    # A [M(block_x 1:1), N(block_y 1:tile_size)]
    block_x, block_y = ct.bid(0), ct.bid(1)
    tile_A = ct.load(
        A, (block_x, block_y), (1, tile_size), padding_mode=ct.PaddingMode.ZERO,
    ).reshape((tile_size, ))
    tile_A = ct.sum(tile_A)
    ct.atomic_add(O, (block_x, ), tile_A)

@ct.kernel
def sum_v3(A: ct.Array, O: ct.Array, tileM: ct.Constant, tileN: ct.Constant):
    # A [M(block_x 1:tileM), N(block_y 1:tileN)]
    block_x = ct.bid(0)
    # [tileM, tileN]
    tile_A = ct.load(
        A, (block_x, 0), (tileM, tileN), padding_mode=ct.PaddingMode.ZERO,
    )
    tile_B = ct.full(shape=(tileN, tileN), fill_value=1.0, dtype=A.dtype)
    """
    tile_A: 
    [[1, 2],
     [3,4]]
    tile_B:
    [[1, 1],
     [1, 1]]
    tile_A * tile_B:
    [[3, 3],
     [7, 7]]
    """
    tile_sum = ct.matmul(tile_A, tile_B)
    tile_sum = ct.extract(tile_sum, (0, 0), (tileM, 1)).reshape((tileM, ))
    ct.store(O, (block_x, ), tile_sum)

@ct.kernel
def sum_v4(A: ct.Array, O: ct.Array, tileM: ct.Constant, tileN: ct.Constant):
    # A [M(block_x 1:tileM), N(block_y 1:tileN)]
    block_x, block_y = ct.bid(0), ct.bid(1)
    # [tileM, tileN]
    tile_A = ct.load(
        A, (block_x, block_y), (tileM, tileN), padding_mode=ct.PaddingMode.ZERO,
    )
    tile_sum = ct.sum(tile_A, axis=-1)
    ct.store(O, (block_x, ), tile_sum)


M, N = 1048576, 16000
A = torch.randn(size=[M, N], device="cuda", dtype=torch.float32)
real = torch.sum(A, dim=-1)

O = torch.zeros(size=[M], device="cuda", dtype=torch.float32)
ct.launch(torch.cuda.current_stream(), (M, ), sum_v1, (A, O, 32))

O = torch.zeros(size=[M], device="cuda", dtype=torch.float32)
ct.launch(torch.cuda.current_stream(), (M, ct.cdiv(N, 32)), sum_v2, (A, O, 32))

O = torch.zeros(size=[M], device="cuda", dtype=torch.float32)
ct.launch(torch.cuda.current_stream(), (ct.cdiv(M, 32), ), sum_v3, (A, O, 32, 32))

O = torch.zeros(size=[M], device="cuda", dtype=torch.float32)
ct.launch(torch.cuda.current_stream(), (ct.cdiv(M, 32), ct.cdiv(N, 32)), sum_v4, (A, O, 32, 32))

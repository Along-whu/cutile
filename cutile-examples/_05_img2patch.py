import torch
import cuda.tile as ct

@ct.kernel
def img2patch(
    x: ct.Array,
    y: ct.Array,
    coord: ct.Array,
    patch_size_x: ct.Constant,
    patch_size_y: ct.Constant,
):
    # x: [C, H, W] -> y: [N, C * patch_sie_x * patch_size_y]
    block_x = ct.bid(0) # num_patch_h
    block_y = ct.bid(1) # num_patch_w
    block_z = ct.bid(2) # C

    num_block_y = ct.num_blocks(1)
    tile_x = ct.load(
        x,
        (block_z, block_x, block_y),
        (1, patch_size_x, patch_size_y),
        padding_mode=ct.PaddingMode.ZERO
    )

    tile_x = ct.reshape(tile_x, (1, patch_size_x * patch_size_y))
    ct.store(y, (block_x * num_block_y + block_y, block_z), tile_x)
    if block_z == 0:
        local_coord_x = ct.full((1, 1), block_x, dtype=ct.int32)
        local_coord_y = ct.full((1, 1), block_y, dtype=ct.int32)
        ct.store(
            coord, (block_x * num_block_y + block_y, 0), ct.cat((local_coord_x, local_coord_y), axis=1), 
        )

@ct.kernel
def patch2img(
    x: ct.Array,
    y: ct.Array,
    patch_size_x: ct.Constant,
    patch_size_y: ct.Constant,
):
    block_x = ct.bid(0) # num_patch_h
    block_y = ct.bid(1) # num_patch_w
    block_z = ct.bid(2) # C
    num_block_y = ct.num_blocks(1)

    tile_x = ct.load(x, (block_x * num_block_y + block_y, block_z), (1, patch_size_x * patch_size_y))
    tile_x = ct.reshape(tile_x, (1, patch_size_x, patch_size_y))
    ct.store(y, (block_z, block_x, block_y), tile_x)
    

if __name__ == "__main__":
    C, H, W, = 2, 8, 8
    patch_size_h, patch_size_w = 4, 4
    num_patch_h, num_patch_w = ct.cdiv(H, patch_size_h), ct.cdiv(W, patch_size_w)
    x = torch.rand(size=[C, H, W], device="cuda")
    y = torch.empty(size=[num_patch_h * num_patch_w, C * patch_size_h * patch_size_w], device="cuda")
    z = torch.empty(size=[C, H, W], device="cuda")
    coord = torch.empty(size=[num_patch_h * num_patch_w, 2], device="cuda", dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (num_patch_h, num_patch_w, C), img2patch, (x, y, coord, patch_size_h, patch_size_w))
    ct.launch(torch.cuda.current_stream(), (num_patch_h, num_patch_w, C), patch2img, (y, z, patch_size_h, patch_size_w))

    print(x - z)
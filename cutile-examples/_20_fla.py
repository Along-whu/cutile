import torch
import cuda.tile as ct

@ct.kernel
def chunk_fwd_h_kernel(
    k: ct.Array,
    v: ct.Array,
    h: ct.Array,
    g: ct.Array,
    g_gamma: ct.Array,
    gk: ct.Array,
    gv: ct.Array,
    h0: ct.Array,
    ht: ct.Array,
    cu_seqlens: ct.Array,
    split_offsets: ct.Array,
    NUM_SEQS: ct.Constant,
    NUM_HEADS: ct.Constant,
    HEAD_DIM_K: ct.Constant,
    HEAD_DIM_V: ct.Constant,
    TILE_T: ct.Constant,
    TILE_S: ct.Constant,
    TILE_K: ct.Constant,
    TILE_V: ct.Constant,
    USE_G: ct.Constant,
    USE_G_GAMMA: ct.Constant,
    USE_GK: ct.Constant,
    USE_GV: ct.Constant,
    USE_INITIAL_STATE: ct.Constant,
    STORE_FINAL_STATE: ct.Constant,
    IS_VARLEN: ct.Constant,
):
    block_k_idx = ct.bid(0)
    block_v_idx = ct.bid(1)
    block_bh_idx = ct.bid(2)
    
    block_b_idx = block_bh_idx // NUM_HEADS
    block_h_idx = block_bh_idx % NUM_HEADS
    
    if IS_VARLEN:
        seq_start = ct.load(cu_seqlens, (block_b_idx, ), (1, ))
        seq_end = ct.load(cu_seqlens, (block_b_idx + 1, ), (1, ))
        
        seq_len = seq_end - seq_start
        num_seq_blocks = ct.cdiv(seq_len, TILE_T) # number of sequence blocks for this sequence
        num_states = ct.cdiv(seq_len, TILE_S) # number of states for this sequence
        num_seq_per_state = ct.cdiv(num_seq_blocks, num_states) # number of sequence blocks that contribute to each state
    else:
        num_seq_blocks = ct.cdiv(NUM_SEQS, TILE_T)
        num_states = ct.cdiv(NUM_SEQS, TILE_S)
        num_seq_per_state = ct.cdiv(num_seq_blocks, num_states)
        
    acc = ct.zeros((TILE_K, TILE_V), dtype=ct.float32)
    
    if USE_INITIAL_STATE:
        tile_h0 = ct.load(h0, (block_b_idx, block_h_idx, 0, 0), (1, 1, HEAD_DIM_K, HEAD_DIM_V), padding_mode=ct.PADDING_MODE.ZERO)
        acc += tile_h0.reshape((TILE_K, TILE_V))
    
    for block_seq_idx in range(num_seq_blocks):
        states_idx = block_seq_idx // num_seq_per_state
        
        tileK = ct.load(k, (block_b_idx, block_seq_idx, block_h_idx, block_k_idx), (1, TILE_T, 1, TILE_K), padding_mode=ct.PADDING_MODE.ZERO)
        tileV = ct.load(v, (block_b_idx, block_seq_idx, block_h_idx, block_v_idx), (1, TILE_T, 1, TILE_V), padding_mode=ct.PADDING_MODE.ZERO)
        
        tileK = tileK.reshape((TILE_T, TILE_K)).astype(ct.float32)
        tileV = tileV.reshape((TILE_T, TILE_V)).astype(ct.float32)
        
        # Assume we have 8 tokens, and TILE_T = 2, TILE_S = 4, then 
        if block_seq_idx % num_seq_per_state == 0:
            ct.store(h, (block_b_idx, states_idx, block_h_idx, block_k_idx, block_v_idx), acc.reshape((1, 1, 1, TILE_K, TILE_V)).astype(h.dtype))
        
        if USE_G:
            last_token_idx = ct.min((block_seq_idx + 1) * TILE_T, seq_len) - 1
            g_end = ct.load(g, (block_b_idx, last_token_idx, block_h_idx), (1, 1, 1), padding_mode=ct.PADDING_MODE.ZERO)
            g_end = g_end.reshape((1, 1)).astype(ct.float32)
            
            g_chunk = ct.load(g, (block_b_idx, block_seq_idx, block_h_idx), (1, TILE_T, 1), padding_mode=ct.PADDING_MODE.ZERO)
            g_chunk = g_chunk.reshape((TILE_T, 1)).astype(ct.float32)
            
            g_chunk = ct.exp(g_chunk - g_end) # [TILE_T, 1]
            tileV = tileV * g_chunk # [TILE_T, TILE_V]
            acc *= ct.exp(g_end) # [TILE_K, TILE_V]
        
        if USE_G_GAMMA:
            
            
        
        
        
    
    
    

def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
):
    """
    k: [bs, seq_len, num_heads, head_dim_k]
    v: [bs, seq_len, num_heads, head_dim_v]
    g: [bs, seq_len, num_heads] or None
    g_gamma: [num_heads] or None
    gk: [bs, seq_len, num_heads, head_dim_k] or None
    gv: [bs, seq_len, num_heads, head_dim_v] or None
    h0: [bs, num_heads, head_dim_k, head_dim_v] or None
    cu_seqlens: [bs + 1] or None
    """
    
    bs, seq_len, num_heads, head_dim_k, head_dim_v = *k.shape, v.shape[-1]
    tileT = chunk_size
    tileS = tileT if split_size is None else split_size
    assert tileS % tileT == 0, f"The `split_size` (got {tileS}) must be a multiple of `chunk_size` {tileT}"
    
    if cu_seqlens is not None:
        cu_seqlens = torch.diff(cu_seqlens) # [0, 16, 32, 64] -> [16, 16, 32]
        cu_seqlens = ct.cdiv(cu_seqlens, tileS) # [16, 16, 32] -> [1, 1, 2]
        split_offsets = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).cumsum(dim=0) # [0, 1, 2, 4]
        num_seqs, num_states = len(cu_seqlens), split_offsets[-1].item() # we have 3 sequences, and 4 states (2 for the last sequence)
        h = k.new_empty([num_states, num_heads, head_dim_k, head_dim_v], dtype=k.dtype if not states_in_fp32 else torch.float32)
    else:
        num_seqs = bs
        num_states = ct.cdiv(seq_len, tileS),
        split_offsets = None
        h = k.new_empty([bs, num_states, num_heads, head_dim_k, head_dim_v], dtype=k.dtype if not states_in_fp32 else torch.float32)
    
    ht = k.new_empty([num_seqs, num_heads, head_dim_k, head_dim_v], dtype=k.dtype if not states_in_fp32 else torch.float32) if output_final_state else None
    
    tileK = 16
    tileV = 16
    
    grid = (ct.cdiv(head_dim_k, tileK), ct.cdiv(head_dim_v, tileV), num_seqs * num_heads)
    
    ct.launch(torch.cuda.current_stream(), grid, )
        

def FlashLinearAttention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
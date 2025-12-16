# Part 1. Define Functions
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional
import numpy as np
from einops import repeat, einsum

# the X should be (B, T, C), B is batch size, T is the max trajectory length, C is the embedding channel
# adj should be (V,V)

class NormalizedEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd

    def forward(self, x):
        x = self.embedding(x)
        return x/torch.norm(x, dim=-1, keepdim=True)


def get_1d_sincos_geo_embed(d_cross, pos):
    """
    d_cross: output dimension for each position
    pos: a list of positions to be encoded: size (V) or (B, V)
    out: (V, D) or (B, V, D)
    """
    assert d_cross % 2 == 0
    omega = torch.arange(d_cross // 2, dtype=torch.float64)
    omega /= d_cross / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if len(pos.shape) == 1:
        pos = pos.reshape(-1)  # (V,)
        out = torch.einsum('v,d->vd', pos, omega)  # (V, D/2), outer product
    elif len(pos.shape) == 2:
        # (B, V)
        out = torch.einsum('bv,d->bvd', pos, omega)  # (B, V, D/2), outer product

    emb_sin = torch.sin(out)  # (V, D/2)
    emb_cos = torch.cos(out)  # (V, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (V, D)
    emb[..., 0::2] = emb_sin
    emb[..., 1::2] = emb_cos

    return emb

def get_rope_qk_embed(q, k, pos):
    # qw, kw: (B, V, D)
    # pos: (B, blcok_size) for example, 60 seconds: [[0, 60, 120, ...],[0,1,2,...]...]
    d_cross = q.shape[-1]
    geo_emb = get_1d_sincos_geo_embed(d_cross, pos) # (B, V, D)
    if len(q.shape) == 2:
        geo_emb = repeat(geo_emb, 'v d -> b v d', b=q.shape[0]) # (B, V, D)
    cos_pos = repeat(geo_emb[...,1::2], 'b v d -> b v (d 2)') # (B, V, D), d = D/2
    sin_pos = repeat(geo_emb[...,0::2], 'b v d -> b v (d 2)') # (B, V, D)
    q_ = torch.stack([-q[...,1::2], q[...,0::2]], dim=-1) # (B, V, D/2, 2)
    q_ = q_.reshape(q.shape) # (B, V, D)
    q = q*cos_pos + q_*sin_pos # (B, V, D)
    k_ = torch.stack([-k[...,1::2], k[...,0::2]], dim=-1) # (B, V, D/2, 2)
    k_ = k_.reshape(k.shape[0],k.shape[1], -1) # (B, V, D)
    k = k*cos_pos + k_*sin_pos # (B, V, D)

    return q, k

def get_2d_sincos_geo_embed(emb_1d):
    """
    1d_emb: (V, D)
    out: (V, V, D)
    """
    emb_1d = emb_1d.reshape(-1, emb_1d.shape[-1])  # (V, D)
    emb_2d = torch.einsum('hd,wd->hwd', emb_1d, emb_1d)  # (V, V, D)
    # print(emb_2d)
    return emb_2d


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, block_size, n_embd, dropout=0.1, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=in_proj_bias)
        self.query = nn.Linear(n_embd, head_size, bias=in_proj_bias)
        self.value = nn.Linear(n_embd, head_size, bias=in_proj_bias)
        self.out_proj = nn.Linear(head_size, head_size, bias=out_proj_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos = None, mask=None):
        input_shape = x.shape
        batch_size, sequence_length, head_size = input_shape

        # (batch_size, sequence_length, head_size)
        k = self.key(x)
        # (batch_size, sequence_length, head_size)
        q = self.query(x)
        # (batch_size, sequence_length, head_size)
        v = self.value(x)

        if pos is not None:
            q, k = get_rope_qk_embed(q, k, pos)

        # (B*N, T, T)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(head_size)
        weight = torch.masked_fill(weight, mask, value=-1e7)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        output = weight @ v
        output = self.out_proj(output)

        return output


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embd, dropout=dropout
                                         ) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None,mask=None):
        out = torch.cat([h(x, pos, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.SiLU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1, norm_position='prenorm'):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.norm_position = norm_position
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, pos=None, mask=None):
        if self.norm_position == 'prenorm':
            x = x + self.sa(self.ln1(x), pos, mask)
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.ln1(x + self.sa(x, pos, mask))
            x = self.ln2(x + self.ffwd(x))
        
        return x


class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_embd, n_hidden, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_embd, n_hidden, bias=in_proj_bias)

        self.out_proj = nn.Linear(n_hidden, n_hidden, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = n_hidden // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pos=None, mask=None):
        # x: (Traj) (B, T, H)
        # if use matrix adj: (Road) (B, V, V, C)
        # if use table adj: (B, V, E, C), same process as the matrix adj

        input_shape = x.shape
        batch_size = x.shape[0]

        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)  # (B, T, H) -> (B, T, H)
        k = self.k_proj(adj)  # (B, V, V, C) -> (B, V, V, H)
        v = self.v_proj(adj)  # (B, V, V, C) -> (B, V, V, H)

        # (B, T, H) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        # (B, V, V, H) -> (B, V*V, n_heads, d_head) -> (B, n_heads, V*V, d_head)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # (B, n_heads, T, V*V) @ (B, n_heads, V*V, d_head) -> (B, n_heads, T, d_head)
        output = weight @ v

        # (B, n_heads, T, d_head) -> (B, T, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()

        # (B, T, n_heads, d_head) -> (B, T, H)
        output = output.view(input_shape)

        output = self.out_proj(output)

        # (B, N, L, H)

        return output


class CrossAttentionBlock(nn.Module):

    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True, norm_position='prenorm'):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        self.att = CrossAttention(
            n_heads, n_hidden, n_embd, dropout, in_proj_bias, out_proj_bias)
        self.ffd = FeedFoward(n_hidden, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.norm_position = norm_position

    def forward(self, x, adj):
        # x: (B, N, L, H)
        # adj: (B, V, V, C)
        if self.norm_position == 'prenorm':
            x = x + self.ln1(self.att(x, adj))
            x = x + self.ln2(self.ffd(x))
        else:
            x = self.ln1(x + self.att(x, adj))
            x = self.ln2(x + self.ffd(x))
        return x


class no_diffusion_model_addembed(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int, n_hidden: int, n_layer: int, n_head: int, block_size: int,
                 dropout=0.1,
                 weight_quantization_scale: Optional[int] = None,
                 use_adj_table=True,
                 use_ne=True,
                 use_ge=True,
                 use_agent_mask=False,
                 norm_position='prenorm',
                 device='cuda'):
        super().__init__()

        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(
                vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.block_size = block_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        if weight_quantization_scale:
            if use_ne:
                self.weight_embedding_table = NormalizedEmbedding(
                    weight_quantization_scale+1, n_embd)
            else:
                self.weight_embedding_table = nn.Embedding(
                    weight_quantization_scale+1, n_embd)
            # +1 because 0 means no edge
        else:
            self.adj_embed = nn.Sequential(
                nn.Linear(1, 2*vocab_size),
                nn.LayerNorm(2*vocab_size),
                nn.SiLU(),
                nn.Linear(2*vocab_size, n_embd),
            )

        # Geolocation embedding
        # (B, V, V, n_embd)
        # to demonstrate the end, we add 0 to the rest of trajectory, so the vocab_size = V + 1
        if use_adj_table:
            if use_ge:
                self.geolocation_embedding = get_1d_sincos_geo_embed(n_embd, torch.arange(1, vocab_size)).to(device).float().unsqueeze(0).unsqueeze(2) # (1, V, 1, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, 1, n_embd), device=device)
        else:
            if use_ge:
                self.geolocation_embedding = get_2d_sincos_geo_embed(get_1d_sincos_geo_embed(n_embd, np.arange(1, vocab_size))).to(device).float().unsqueeze(0) # (1, V, V, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, vocab_size-1, n_embd), device=device)

        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([CrossAttentionBlock(
            n_head, n_hidden, n_embd, dropout, norm_position=norm_position) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_hidden)  # final layer norm
        self.lm_head = nn.Linear(n_hidden, vocab_size)

        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 2),
        )
        # self.condition_proj_list = nn.ModuleList([nn.Sequential(
        #     nn.Linear(n_embd*2, n_hidden),
        #     nn.LayerNorm(n_hidden),
        #     nn.SiLU(),
        #     nn.Linear(n_hidden, 2),
        # ) for _ in range(block_size)])

        self.device = device
        self.block_size = block_size
        self.use_agent_mask = use_agent_mask
        self.use_adj_table = use_adj_table
        self.weight_quantization_scale = weight_quantization_scale

    def forward(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: torch.Tensor, y: Optional[torch.Tensor] = None, time_step: Optional[torch.Tensor] = None,
                agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        # Input:
        # x: (B, T)
        # condition: (B, (N-1), T), still denote as N
        # weighted_adj: (B, V, V) or (B, V, E, 2)
        # y: (B, T)
        # adjmask: (B, V, V)
        # special mask: (B, T)
        # Output: (B T V)

        B, N, T = condition.shape

        if not self.use_agent_mask:
            agent_mask = None

        # x and y are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(x)  # (B, T ,C)

        pos_emb = self.position_embedding_table(torch.arange(
            T, device=self.device)).view(1, T, -1)  # (1,T,C)

        if self.use_adj_table:
            if self.weight_quantization_scale:
                adj = self.token_embedding_table(weighted_adj[:, :, :, 0].int()) + self.weight_embedding_table(weighted_adj[:, :, :, 1].int()) + self.geolocation_embedding
            else:
                adj = self.token_embedding_table(weighted_adj[:, :, :, 0].int())+ self.adj_embed(weighted_adj[:, :, :, 1].unsqueeze(-1)) + self.geolocation_embedding
        else:
            if self.weight_quantization_scale:
                adj = self.weight_embedding_table(weighted_adj.int()) + self.geolocation_embedding
            else:
                adj = self.adj_embed(weighted_adj.unsqueeze(-1)) + self.geolocation_embedding

        if condition is not None:
            # TODO find an effiective way
            condition_emb = 0
            for i in range(N):
                # add the condition to the embedding one by one
                condition_s = condition[:,i,:]  # (B, T)
                condition_s_emb = self.token_embedding_table(condition_s.int())  # (B, T, C)
                condition_s_emb = torch.cat((tok_emb, condition_s_emb), dim=-1)  # (B, T, 2C)
                condition_s_score = torch.softmax(self.condition_proj(condition_s_emb), dim=-1)  # (B, T, 2)
                condition_s_emb = torch.einsum('btd,btdc->btc', condition_s_score, condition_s_emb.view(B, T, 2, -1))# (B, T, C)

                condition_emb = condition_emb + condition_s_emb
            condition_emb = condition_emb/N

            # for i in range(N):
            #     for j in range(T):
            #         condition_s = condition[:, i, j] # (B)
            #         condition_s_emb = self.token_embedding_table(condition_s.int()) # (B, C)
            #         condition_s_emb = condition_s_emb.unsqueeze(1).expand(-1, T, -1) # (B, T, C)
            #         condition_s_emb = torch.cat((tok_emb, condition_s_emb), dim=-1) # (B, T, 2C)
            #         condition_s_score = torch.softmax(self.condition_proj(condition_s_emb), dim=-1) # (B, T, 2)
            #         condition_s_emb = torch.einsum('btd,btdc->btc', condition_s_score, condition_s_emb.view(B, T, 2, -1)) # (B, T, C)

            #         condition_emb = condition_emb + condition_s_emb
            # condition_emb = condition_emb/(N*T)

        else:
            condition_emb = 0

        x = tok_emb + pos_emb + condition_emb  # (B,T,C)
        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x, adj)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,V)

        if y is None:
            loss = None
        else:
            if special_mask is None:
                special_mask = torch.ones_like(y).float() # (B, T)
            B, T, V = logits.shape
            logits_ = logits.view(B*T, V)
            y = y.reshape(B*T)
            special_mask = special_mask.view(B*T)
            if agent_mask is not None:
                mask_weight = agent_mask.view(B*T)
                loss = (F.cross_entropy(logits_, y, reduction='none')
                        * special_mask*mask_weight).sum()/mask_weight.sum()/special_mask.sum()
            else:
                loss = (F.cross_entropy(logits_, y, reduction='none')
                        * special_mask).sum()/special_mask.sum()

        return logits, loss

    def test(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: torch.Tensor, y: Optional[torch.Tensor] = None,
             agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        logits, _ = self.forward(
            x, condition, weighted_adj, agent_mask, special_mask)
        # (B, T) + (B, N, T) -> (B, T, V)
        return logits
    

class corss_attention_parallel_block(nn.Module):
    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True, norm_position='prenorm'):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        self.att_condition = CrossAttention(
            n_heads, n_hidden, n_embd, dropout, in_proj_bias, out_proj_bias)
        self. att_adj = CrossAttention(
            n_heads, n_hidden, n_embd, dropout, in_proj_bias, out_proj_bias)
        self.self_att = MultiHeadAttention(
            n_heads, n_hidden//n_heads, n_hidden, n_hidden, dropout=dropout)
        self.ffd = FeedFoward(n_hidden, dropout=dropout)
        self.ln1_1 = nn.LayerNorm(n_hidden)
        self.ln1_2 = nn.LayerNorm(n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.ln3 = nn.LayerNorm(n_hidden)
        self.norm_position = norm_position

    def forward(self, x, cond, adj, pos=None, mask=None):
        # x: (B, N, L, H)
        # adj: (B, V, V, C)
        if self.norm_position == 'prenorm':
            x = x + self.ln1_1(self.att_condition(x, cond)) + self.ln1_2(self.att_adj(x, adj))
            x = x + self.ln2(self.self_att(x, pos, mask))
            x = x + self.ln3(self.ffd(x))
        else:
            x = self.ln1_1(x + self.att_condition(x, cond)) + self.ln1_2(x + self.att_adj(x, adj))
            x = self.ln2(x + self.self_att(x, pos, mask))
            x = self.ln3(x + self.ffd(x))
        return x
    

class no_diffusion_model_cross_attention_parallel(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int, n_hidden: int, n_layer: int, n_head: int, block_size: int,
                 dropout=0.1,
                 weight_quantization_scale: Optional[int] = None,
                 use_adj_table=True,
                 use_ne=True,
                 use_ge=True,
                 use_agent_mask=False,
                 norm_position='prenorm',
                 device='cuda'):
        super().__init__()

        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(
                vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.block_size = block_size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        if weight_quantization_scale:
            if use_ne:
                self.weight_embedding_table = NormalizedEmbedding(
                    weight_quantization_scale+1, n_embd)
            else:
                self.weight_embedding_table = nn.Embedding(
                    weight_quantization_scale+1, n_embd)
            # +1 because 0 means no edge
        else:
            self.adj_embed = nn.Sequential(
                nn.Linear(1, 2*vocab_size),
                nn.LayerNorm(2*vocab_size),
                nn.SiLU(),
                nn.Linear(2*vocab_size, n_embd),
            )

        # Geolocation embedding
        # (B, V, V, n_embd)
        # to demonstrate the end, we add 0 to the rest of trajectory, so the vocab_size = V + 1
        if use_adj_table:
            if use_ge:
                self.geolocation_embedding = get_1d_sincos_geo_embed(n_embd, np.arange(1, vocab_size)).to(device).float().unsqueeze(0).unsqueeze(2) # (1, V, 1, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, 1, n_embd), device=device)
        else:
            if use_ge:
                self.geolocation_embedding = get_2d_sincos_geo_embed(get_1d_sincos_geo_embed(n_embd, np.arange(1, vocab_size))).to(device).float().unsqueeze(0) # (1, V, V, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, vocab_size-1, n_embd), device=device)

        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([corss_attention_parallel_block(
            n_head, n_hidden, n_embd, dropout, norm_position=norm_position) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_hidden)  # final layer norm
        self.lm_head = nn.Linear(n_hidden, vocab_size)

        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 2),
        )
        # self.condition_proj_list = nn.ModuleList([nn.Sequential(
        #     nn.Linear(n_embd*2, n_hidden),
        #     nn.LayerNorm(n_hidden),
        #     nn.SiLU(),
        #     nn.Linear(n_hidden, 2),
        # ) for _ in range(block_size)])

        self.device = device
        self.block_size = block_size
        self.use_agent_mask = use_agent_mask
        self.use_adj_table = use_adj_table
        self.weight_quantization_scale = weight_quantization_scale

    def forward(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: torch.Tensor, y: Optional[torch.Tensor] = None, time_step: Optional[torch.Tensor] = None,
                agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        # Input:
        # x: (B, T)
        # condition: (B, (N-1), T), still denote as N
        # weighted_adj: (B, V, V) or (B, V, E, 2)
        # y: (B, T)
        # adjmask: (B, V, V)
        # special mask: (B, T)
        # Output: (B T V)

        B, T = x.shape

        if not self.use_agent_mask:
            agent_mask = None

        # x and y are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(x)  # (B, T ,C)

        pos_emb = self.position_embedding_table(torch.arange(
            T, device=self.device)).view(1, T, -1)  # (1,T,C)

        if self.use_adj_table:
            if self.weight_quantization_scale:
                adj = self.token_embedding_table(weighted_adj[:, :, :, 0].int()) + self.weight_embedding_table(weighted_adj[:, :, :, 1].int()) + self.geolocation_embedding
            else:
                adj = self.token_embedding_table(weighted_adj[:, :, :, 0].int())+ self.adj_embed(weighted_adj[:, :, :, 1].unsqueeze(-1)) + self.geolocation_embedding
        else:
            if self.weight_quantization_scale:
                adj = self.weight_embedding_table(weighted_adj.int()) + self.geolocation_embedding
            else:
                adj = self.adj_embed(weighted_adj.unsqueeze(-1)) + self.geolocation_embedding

        if condition is not None:
            condition = self.token_embedding_table(condition.int())  # (B, N, T, C)
            condition = condition + pos_emb  # (B, N, T, C)
        else:
            condition = torch.zeros_like(tok_emb)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.in_proj(x)

        if time_step is not None:
            # time_step = (B)
            time_pos = torch.arange(T, device=self.device)
            time_pos = torch.einsum('t,b->bt', time_pos, time_step)

        for block in self.blocks:
            x = block(x, condition ,adj, time_pos)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,V)

        if y is None:
            loss = None
        else:
            if special_mask is None:
                special_mask = torch.ones_like(y).float() # (B, T)
            B, T, V = logits.shape
            logits_ = logits.view(B*T, V)
            y = y.reshape(B*T)
            special_mask = special_mask.view(B*T)
            if agent_mask is not None:
                mask_weight = agent_mask.view(B*T)
                loss = (F.cross_entropy(logits_, y, reduction='none')
                        * special_mask*mask_weight).sum()/mask_weight.sum()/special_mask.sum()
            else:
                loss = (F.cross_entropy(logits_, y, reduction='none')
                        * special_mask).sum()/special_mask.sum()

        return logits, loss

    def test(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: torch.Tensor, y: Optional[torch.Tensor] = None,
             agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            logits, _ = self.forward(
                x, condition, weighted_adj, agent_mask, special_mask)
        # (B, T) + (B, N, T) -> (B, T, V)
        return logits
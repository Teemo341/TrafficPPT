# Part 1. Define Functions
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional
import numpy as np
from einops import repeat

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
    omega = torch.arange(d_cross // 2, dtype=torch.float32, device=pos.device)  # (D/2,)
    omega /= d_cross / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if isinstance(pos, np.ndarray):
        pos = torch.tensor(pos, dtype=torch.float32)

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
        if mask is not None:
            weight = torch.masked_fill(weight, mask, value=0)
            weight.to_sparse()
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

    def forward(self, x, pos=None, mask=None):
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

    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True, group = None):
        super().__init__()

        self.n_heads = n_heads
        if group is None:
            self.group = n_heads
        else:
            self.group = group
        self.d_head = n_hidden // n_heads

        self.q_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_embd, self.d_head*self.group, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_embd, self.d_head*self.group, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_hidden, n_hidden, bias=out_proj_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pos=None, mask=None):
        # x: (Traj) (B, T, H)
        # if use matrix adj: (Road) (B, V, V, C)
        # if use table adj: (B, V, E, C), same process as the matrix adj

        input_shape = x.shape
        batch_size = x.shape[0]

        interm_shape = (batch_size, -1, self.n_heads, self.d_head)
        group_shape = (batch_size, -1, self.group, self.d_head)

        q = self.q_proj(x)  # (B, T, H) -> (B, T, H)
        k = self.k_proj(adj)  # (B, V, E, C) -> (B, V, E, H_group)
        v = self.v_proj(adj)  # (B, V, E, C) -> (B, V, E, H_group)

        # (B, T, H) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        # (B, V, E, H) -> (B, V*E, group, d_head) -> (B, group, V*E, d_head)
        k = k.view(group_shape).transpose(1, 2)
        v = v.view(group_shape).transpose(1, 2)

        # (B, n_heads, T, d_head) @ (B, group, V*E, d_head) -> (B, n_heads, T, V*E)
        weight = torch.einsum('bhtd,bgvd->bhtv', q, k)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # (B, n_heads, T, V*E) @ (B, group, V*E, d_head) -> (B, n_heads, T, d_head)
        output = torch.einsum('bhtv,bgvd->bhtd', weight, v)

        # (B, n_heads, T, d_head) -> (B, T, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()

        # (B, T, n_heads, d_head) -> (B, T, H)
        output = output.view(input_shape)

        output = self.out_proj(output)

        # (B, N, L, H)

        return output


class CrossAttention_series(nn.Module):

    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True, group = None):
        super().__init__()

        self.n_heads = n_heads
        if group is None:
            self.group = n_heads
        else:
            self.group = group
        self.d_head = n_hidden // n_heads
        self.head_per_group = n_heads // self.group

        self.q_proj = nn.ModuleList([nn.Linear(n_hidden, self.d_head, bias=in_proj_bias) for _ in range(n_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(n_embd, self.d_head, bias=in_proj_bias) for _ in range(self.group)])
        self.v_proj = nn.ModuleList([nn.Linear(n_embd, self.d_head, bias=in_proj_bias) for _ in range(self.group)])
        self.out_proj = nn.Linear(n_hidden, n_hidden, bias=out_proj_bias)
        self.dropout = nn.Dropout(dropout)
    
    def head_forward(self, x , adj, i, j):
        q = self.q_proj[i](x) # (B, T, C)
        k = self.k_proj[j](adj) # (B, V, E, C)
        v = self.v_proj[j](adj) # (B, V, E, C)

        weight = torch.einsum('btc,bvec->bt(ve)', q, k)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        
        output = torch.einsum('bt(ve),bvec->btc', weight, v)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pos=None, mask=None):
        # x: (Traj) (B, T, H)
        # if use matrix adj: (Road) (B, V, V, C)
        # if use table adj: (B, V, E, C), same process as the matrix adj

        output = []
        for i in range(self.head_per_group):
            for j in range(self.group):
                output.append(self.head_forward(x, adj, i+j*self.group*self.head_per_group, j))
        output = torch.cat(output, dim=-1)
        output = self.dropout(self.out_proj(output))

        return output


class CrossAttentionBlock(nn.Module):

    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True, norm_position='prenorm', group = None):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        self.att = CrossAttention(
            n_heads, n_hidden, n_embd, dropout, in_proj_bias, out_proj_bias, group)
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
                 norm_position='prenorm'):
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
                self.geolocation_embedding = get_1d_sincos_geo_embed(n_embd, torch.arange(1, vocab_size)).float().unsqueeze(0).unsqueeze(2) # (1, V, 1, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, 1, n_embd))
        else:
            if use_ge:
                self.geolocation_embedding = get_2d_sincos_geo_embed(get_1d_sincos_geo_embed(n_embd, torch.arange(1, vocab_size))).float().unsqueeze(0) # (1, V, V, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, vocab_size-1, n_embd))

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

        self.block_size = block_size
        self.use_agent_mask = use_agent_mask
        self.use_adj_table = use_adj_table
        self.weight_quantization_scale = weight_quantization_scale

    def forward(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: list[torch.Tensor], y: Optional[torch.Tensor] = None, time_step: Optional[torch.Tensor] = None,
                agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        '''
        # Input:
            x: (B, T)
            condition: (B, (N-1), T), still denote as N
            weighted_adj: [(B, V, V)] or [(B, V, E), (B, V, E), ...]
            y: (B, T)
            adjmask: (B, V, V)
            special mask: (B, T)
        # Output:
            (B T V)
        '''

        B, N, T = condition.shape

        if not self.use_agent_mask:
            agent_mask = None

        # x and y are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(x)  # (B, T ,C)

        pos_emb = self.position_embedding_table(torch.arange(
            T, device=x.device)).view(1, T, -1)  # (1,T,C)

        if self.use_adj_table:
            if self.weight_quantization_scale:
                adj = self.token_embedding_table(weighted_adj[...,0].int()) + self.weight_embedding_table(weighted_adj[...,1].int()) + self.geolocation_embedding
            else:
                adj = self.token_embedding_table(weighted_adj[...,0].int())+ self.adj_embed(weighted_adj[...,1].unsqueeze(-1)) + self.geolocation_embedding
        else:
            if self.weight_quantization_scale:
                adj = self.weight_embedding_table(weighted_adj[...,0].int()) + self.geolocation_embedding
            else:
                adj = self.adj_embed(weighted_adj[...,0].unsqueeze(-1)) + self.geolocation_embedding

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

    def test(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: list[torch.Tensor], y: Optional[torch.Tensor] = None,
             agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        logits, _ = self.forward(
            x, condition, weighted_adj, agent_mask, special_mask)
        # (B, T) + (B, N, T) -> (B, T, V)
        return logits
    

# version 2.1 change rotation position embedding into adaption layer norm, only do on self attention
# version 2.2 change B N T pos embed into B N
# version 2.3 change B N T pos embed into B N C
class corss_attention_parallel_block(nn.Module):
    def __init__(self, n_heads: int, n_hidden: int, n_embd: int, dropout=0.1, in_proj_bias=True, out_proj_bias=True, norm_position='prenorm'):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        self.att_condition = CrossAttention(
            n_heads, n_hidden, n_embd, dropout, in_proj_bias, out_proj_bias)
        self. att_adj = CrossAttention(
            n_heads, n_hidden, n_embd, dropout, in_proj_bias, out_proj_bias, group=1)
        self.self_att = MultiHeadAttention(
            n_heads, n_hidden//n_heads, n_hidden, n_hidden, dropout=dropout)
        self.ffd = FeedFoward(n_hidden, dropout=dropout)
        self.ln1_1 = nn.LayerNorm(n_hidden)
        self.ln1_2 = nn.LayerNorm(n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.ln3 = nn.LayerNorm(n_hidden)
        self.adaln_x = nn.Linear(n_embd, n_hidden*6)
        self.adaln_condition = nn.Linear(n_embd, n_hidden*3)
        self.norm_position = norm_position

    def forward(self, x, cond, adj, pos=None, mask=None):
        # x: (B, T, H)
        # cond: (B, N-1, T, H)
        # adj: (B, V, E, C)
        # pos: (B, N, C)
        
        if pos is not None:
        # if False:
            adaln_x = self.adaln_x(pos[:,0,:]) # (B, 6H)
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_x.chunk(6, dim=-1) # (B, H)
            gamma1 = gamma1.unsqueeze(1).expand(-1, x.shape[1], -1) # (B, T, H)
            beta1 = beta1.unsqueeze(1).expand(-1, x.shape[1], -1) # (B, T, H)
            alpha1 = alpha1.unsqueeze(1).expand(-1, x.shape[1], -1) # (B, T, H)
            gamma2 = gamma2.unsqueeze(1).expand(-1, x.shape[1], -1) # (B, T, H)
            beta2 = beta2.unsqueeze(1).expand(-1, x.shape[1], -1) # (B, T, H)
            alpha2 = alpha2.unsqueeze(1).expand(-1, x.shape[1], -1) # (B, T, H)

            adaln_cond = self.adaln_condition(pos) # (B, N, 3H)
            gamma_cond, beta_cond, alpha_cond = adaln_cond.chunk(3, dim=-1) # (B, N, H)
            gamma_cond = gamma_cond.unsqueeze(2).expand(-1, -1, x.shape[1], -1) # (B, N, T, H)
            beta_cond = beta_cond.unsqueeze(2).expand(-1, -1, x.shape[1], -1) # (B, N, T, H)
            alpha_cond = alpha_cond.unsqueeze(2).expand(-1, -1, x.shape[1], -1) # (B, N, T, H)

            if self.norm_position == 'prenorm':
                x = x + self.att_condition(self.ln1_1(x)*(gamma_cond[:,0,:,:]+1)+beta_cond[:,0,:,:], self.ln1_1(cond)*(gamma_cond[:,1:,:,:]+1)+beta_cond[:,1:,:,:])*alpha_cond[:,0,:,:] + self.att_adj(self.ln1_2(x), adj)
                x = x + self.self_att(self.ln2(x)*(gamma1+1)+beta1, None, mask)*alpha1
                x = x + self.ffd(self.ln3(x)*(gamma2+1)+beta2)*alpha2
            else:
                x = self.ln1_1(x + self.att_condition(x, cond))*(gamma_cond[:,0,:,:]+1)+beta_cond[:,0,:,:] + self.ln1_2(x + self.att_adj(x, adj))
                x = alpha_cond[:,0,:,:]*x
                x = self.ln2(x + self.self_att(x, None, mask))*(gamma1+1)+beta1
                x = x*alpha1
                x = self.ln3(x + self.ffd(x))*(gamma2+1)+beta2
                x = x*alpha2
        else:
            if self.norm_position == 'prenorm':
                x = x + self.att_condition(self.ln1_1(x), cond) + self.att_adj(self.ln1_2(x), adj)
                x = x + self.self_att((self.ln2(x)), pos, mask)
                x = x + self.ffd(self.ln3(x))
            else:
                x = self.ln1_1(x + self.att_condition(x, cond)) + self.ln1_2(x + self.att_adj(x, adj))
                x = self.ln2(x + self.self_att(x, pos, mask))
                x = self.ln3(x + self.ffd(x))
        return x
    

class no_diffusion_model_cross_attention_parallel(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int, n_hidden: int, n_layer: int, n_head: int, block_size: int,
                 dropout=0.1,
                 weight_quantization_scale: Optional[int] = None,
                 use_condition=True,
                 use_adj_table=True,
                 use_timestep=True,
                 use_ne=True,
                 use_ge=True,
                 use_agent_mask=False,
                 norm_position='prenorm',
                 adj_type = "b11h"):
        super().__init__()

        self.block_size = block_size
        self.use_agent_mask = use_agent_mask
        self.use_condition = use_condition
        self.use_adj_table = use_adj_table
        self.use_timestep = use_timestep
        self.weight_quantization_scale = weight_quantization_scale

        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(
                vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

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
                nn.Linear(1, vocab_size),
                nn.LayerNorm(vocab_size),
                nn.SiLU(),
                nn.Linear(vocab_size, n_embd),
            )

        # Geolocation embedding
        # (B, V, E, n_embd)
        # to demonstrate the end, we add 0 to the rest of trajectory, so the vocab_size = V + 1
        if use_adj_table:
            if use_ge:
                self.geolocation_embedding = get_1d_sincos_geo_embed(n_embd, torch.arange(1, vocab_size)).float().unsqueeze(0).unsqueeze(2) # (1, V, 1, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, 1, n_embd))
        else:
            if use_ge:
                self.geolocation_embedding = get_2d_sincos_geo_embed(get_1d_sincos_geo_embed(n_embd, torch.arange(1, vocab_size))).float().unsqueeze(0) # (1, V, E, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, vocab_size-1, n_embd))
        #! control bvec bv1c b11c
        if adj_type == 'bveh':
            self.adj_pooling = nn.AdaptiveAvgPool3d((None, None, None)) # (B, V, E, n_embd) -> (B, V, E, n_embd)
        elif adj_type == 'bv1h':
            self.adj_pooling = nn.AdaptiveAvgPool3d((None, 1, None)) # (B, V, E, n_embd) -> (B, V, 1, n_embd)
        elif adj_type == 'b11h':
            self.adj_pooling = nn.AdaptiveAvgPool3d((1, 1, None)) # (B, V, E, n_embd) -> (B, 1, 1, n_embd)
        elif adj_type == 'bk1h':
            self.adj_pooling = nn.AdaptiveAvgPool3d((block_size, 1, None)) # (B, V, E, n_embd) -> (B, K, 1, n_embd)
        else:
            raise ValueError(f'adj_type {adj_type} not supported')
        
        if use_timestep:
            self.time_embedding_table = nn.Embedding(3, n_embd)

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

    def forward(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: list[torch.Tensor], y: Optional[torch.Tensor] = None, time_step: Optional[torch.Tensor] = None,
                agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        # Input:
        # x: (B, T)
        # condition: (B, (N-1), T), still denote as N
        # weighted_adj: [B x V x V] or [B x V x 4 x 2]
        # y: (B, T)
        # adjmask: (B, V, V)
        # special mask: (B, T)
        # Output: (B T V)

        B, T = x.shape
        #print(B,T)
        if not self.use_agent_mask:
            agent_mask = None

        # x and y are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(x)  # (B, T ,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device)).view(1, T, -1)  # (1,T,C)

        if self.use_adj_table:
            if self.weight_quantization_scale:
                adj = self.token_embedding_table(weighted_adj[...,0].int()) + self.weight_embedding_table(weighted_adj[...,1].int()) + self.geolocation_embedding.to(x.device)
            else:
                adj = self.token_embedding_table(weighted_adj[...,0].int())+ self.adj_embed(weighted_adj[...,1].unsqueeze(-1)) + self.geolocation_embedding.to(x.device)
        else:
            if self.weight_quantization_scale:
                adj = self.weight_embedding_table(weighted_adj[...,0].int()) + self.geolocation_embedding.to(x.device)
            else:
                adj = self.adj_embed(weighted_adj[...,0].unsqueeze(-1)) + self.geolocation_embedding.to(x.device)

        #! control bvec bv1c b11c
        if 'musa' not in str(x.device):
            adj = self.adj_pooling(adj) # (B, 1, 1, C) or others
        else:
            # manual adaptive avg pooling fallback (supports output_size with Nones)
            out_size = getattr(self.adj_pooling, 'output_size', (None, None, None))

            if isinstance(out_size, int):
                d_out, e_out, _ = (out_size, out_size, None)
            elif isinstance(out_size, tuple) and len(out_size) == 3:
                d_out, e_out, _ = out_size
            else:
                d_out, e_out = None, None

            B, V, E, C = adj.shape

            def adaptive_avg_along(x: torch.Tensor, L_out: int | None, dim: int) -> torch.Tensor:
                if L_out is None or L_out == x.shape[dim]:
                    return x
                L_in = x.shape[dim]
                # PyTorch adaptive pool binning rule
                chunks = []
                for i in range(L_out):
                    start = math.floor(i * L_in / L_out)
                    end = math.ceil((i + 1) * L_in / L_out)
                    size = max(end - start, 1)
                    chunks.append(x.narrow(dim, start, size).mean(dim=dim, keepdim=True))
                return torch.cat(chunks, dim=dim)

            # pool along V then E (C stays untouched)
            adj = adaptive_avg_along(adj, d_out, dim=1)
            adj = adaptive_avg_along(adj, e_out, dim=2)

            # if the pooling module has affine params, apply them to include in gradient
            weight = getattr(self.adj_pooling, 'weight', None)
            bias = getattr(self.adj_pooling, 'bias', None)
            if weight is not None:
                adj = adj * weight
            if bias is not None:
                adj = adj + bias

        if not self.use_condition:
            condition = None
        if condition is not None:
            condition = self.token_embedding_table(condition.int())  # (B, N, T, C)
            condition = condition + pos_emb  # (B, N, T, C)
        else:
            condition = torch.zeros_like(tok_emb)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.in_proj(x)
        # condition = self.in_proj(condition)

        if not self.use_timestep:
            time_step = None
        if time_step is not None:
            # time_step = (B,N)

            time_pos = torch.where(time_step == 1, 0, torch.where(time_step == 60, 1, 2))
            time_pos = self.time_embedding_table(time_pos) # (B, N, C)

            # time_step = time_step[:,0] # (B)
            # time_pos = torch.arange(1, T+1, device=x.device) # (T)
            # time_pos = torch.einsum('b,t->bt', time_step, time_pos) # (B, T)
        else:
            time_pos = None
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

    def test(self, x: torch.Tensor, condition: torch.Tensor, weighted_adj: list[torch.Tensor], y: Optional[torch.Tensor] = None,
             agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            logits, _ = self.forward(
                x, condition, weighted_adj, agent_mask, special_mask)
        # (B, T) + (B, N, T) -> (B, T, V)
        return logits
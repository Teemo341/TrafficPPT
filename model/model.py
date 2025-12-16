# Part 1. Define Functions
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional
import numpy as np


# the X should be (B, N, L, C), B is batch size, N is the number of car, L is the max trajectory length, C is the embedding channel
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
    pos: a list of positions to be encoded: size (V,)
    out: (M, D)
    """
    assert d_cross % 2 == 0
    omega = np.arange(d_cross // 2, dtype=np.float64)
    omega /= d_cross / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (V,)
    out = np.einsum('v,d->vd', pos, omega)  # (V, D/2), outer product

    emb_sin = np.sin(out)  # (V, D/2)
    emb_cos = np.cos(out)  # (V, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (V, D)
    return emb


def get_2d_sincos_geo_embed(emb_1d):
    """
    1d_emb: (V, D)
    out: (V, V, D)
    """
    emb_1d = emb_1d.reshape(-1, emb_1d.shape[-1])  # (V, D)
    emb_2d = np.einsum('hd,wd->hwd', emb_1d, emb_1d)  # (V, V, D)
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

    def forward(self, x, mask=None):
        input_shape = x.shape
        batch_size, trajectory_num, sequence_length, head_size = input_shape

        # (batch_size, trajectory_num, sequence_length, head_size)
        k = self.key(x)
        # (batch_size, trajectory_num, sequence_length, head_size)
        q = self.query(x)
        # (batch_size, trajectory_num, sequence_length, head_size)
        v = self.value(x)

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

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
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

    def forward(self, x):
        if self.norm_position == 'prenorm':
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.ln1(x + self.sa(x))
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

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: (Traj) (B, N, L, H)
        # if use matrix adj: (Road) (B, V, V, C)
        # if use table adj: (B, V, E, C), same process as the matrix adj

        input_shape = x.shape
        batch_size, trajectory_num, sequence_length, n_embd = input_shape

        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)  # (B, N, L, H) -> (B, N, L, H)
        k = self.k_proj(adj)  # (B, V, V, C) -> (B, V, V, H)
        v = self.v_proj(adj)  # (B, V, V, C) -> (B, V, V, H)

        # (B, N, L, H) -> (B, N*L, n_heads, d_head) -> (B, n_heads, N*L, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        # (B, V, V, H) -> (B, V*V, n_heads, d_head) -> (B, n_heads, V*V, d_head)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # (B, n_heads, N*L, V*V) @ (B, n_heads, V*V, d_head) -> (B, n_heads, N*L, d_head)
        output = weight @ v

        # (B, n_heads, N*L, d_head) -> (B, N*L, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()

        # (B, N*L, n_heads, d_head) -> (B, N, L, H)
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


class no_diffusion_model(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int, n_hidden: int, n_layer: int, n_head: int, block_size: int,
                 dropout=0.1,
                 weight_quantization_scale: Optional[int] = None,
                 use_adj_table=True,
                 use_ne=True,
                 use_ge=False,
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
                self.geolocation_embedding = torch.from_numpy(get_1d_sincos_geo_embed(
                    n_embd, np.arange(1, vocab_size)),
                ).to(device).float().unsqueeze(0).unsqueeze(2) # (1, V, 1, n_embd)
            else:
                self.geolocation_embedding = torch.zeros(
                    (1, vocab_size-1, 1, n_embd), device=device)
        else:
            if use_ge:
                self.geolocation_embedding = torch.from_numpy(get_2d_sincos_geo_embed(get_1d_sincos_geo_embed(
                    n_embd, np.arange(1, vocab_size)),
                )).to(device).float().unsqueeze(0) # (1, V, V, n_embd)
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
            nn.Linear(n_embd*3, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 3),
        )

        self.device = device
        self.block_size = block_size
        self.use_agent_mask = use_agent_mask
        self.use_adj_table = use_adj_table
        self.weight_quantization_scale = weight_quantization_scale

    def forward(self, condition: torch.Tensor, weighted_adj: torch.Tensor, y: Optional[torch.Tensor] = None,
                agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        # Input:
        # condition: (B, N, 2)
        # weighted_adj: (B, V, V) or (B, V, E, 2)
        # y: (B, N, L)
        # adjmask: (B, V, V)
        # special mask: (B, N, L)
        # Output: (B, N, L)

        B, N, _ = condition.shape
        L = self.block_size
        x = torch.zeros((B, N, L), device=self.device).int()
        # put the origin into the first position
        x[:, :, 0] = condition[:, :, 0]

        if not self.use_agent_mask:
            agent_mask = None

        # x and y are both (B, N, L) tensor of integers
        tok_emb = self.token_embedding_table(x)  # (B, N, L ,C)

        pos_emb = self.position_embedding_table(torch.arange(
            L, device=self.device)).view(1, 1, L, -1)  # (1,1,L,C)

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
            # broadcastTensor = torch.zeros((B,T,N,2)).to(self.device).long()
            # condition = condition + broadcastTensor
            # TODO how to add condition to the input
            condition_s = condition[:, :, 0]  # (B, N)
            condition_e = condition[:, :, 1]  # (B, N)
            # (B, N, L)
            condition_s = condition_s.unsqueeze(-1).expand(B, N, L)
            # (B, N, L)
            condition_e = condition_e.unsqueeze(-1).expand(B, N, L)
            condition_s_emb = self.token_embedding_table(
                condition_s.int())  # (B, N, L, C)
            condition_e_emb = self.token_embedding_table(
                condition_e.int())  # (B, N, L, C)
            condition_emb = torch.cat(
                (tok_emb, condition_s_emb, condition_e_emb), dim=-1)  # (B, N, L, 3C)
            condition_score = torch.softmax(self.condition_proj(
                condition_emb), dim=-1)  # (B, N, L, 3)
            condition_emb = torch.einsum(
                # (B, N, L, C)
                'bnld,bnldc->bnlc', condition_score, condition_emb.view(B, N, L, 3, -1))
        else:
            condition_emb = 0

        x = tok_emb + pos_emb + condition_emb  # (B,N,L,C)
        x = self.in_proj(x)
        # x = self.blocks(x) # (B,T,N,C)

        for block in self.blocks:
            x = block(x, adj)

        x = self.ln_f(x)  # (B,N,L,C)
        logits = self.lm_head(x)  # (B,N,L,V)

        if y is None:
            loss = None
        else:
            if special_mask is None:
                special_mask = torch.ones_like(y).float()
            B, N, L, V = logits.shape
            logits_ = logits.view(B*N*L, V)
            y = y.view(B*N*L)
            special_mask = special_mask.view(B*N*L)
            if agent_mask is not None:
                mask_weight = agent_mask.view(B*L*N)
                loss = (F.cross_entropy(logits_, y, reduction='none')
                        * special_mask*mask_weight).sum()/mask_weight.sum()/special_mask.sum()
            else:
                loss = (F.cross_entropy(logits_, y, reduction='none')
                        * special_mask).sum()/special_mask.sum()

        return logits, loss

    def test(self, condition: torch.Tensor, weighted_adj: torch.Tensor, y: Optional[torch.Tensor] = None,
             agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        logits, _ = self.forward(
            condition, weighted_adj, y, agent_mask, special_mask)
        # (B, N, L, V) -> (B, N, L)
        logits = torch.argmax(logits, dim=-1)
        return logits
    
    def test_with_adj_matrix(self, condition: torch.Tensor, weighted_adj: torch.Tensor, connection_filter = None, y: Optional[torch.Tensor] = None, agent_mask: Optional[torch.Tensor] = None, special_mask: Optional[torch.Tensor] = None):
        logits, _ = self.forward(
            condition, weighted_adj, y, agent_mask, special_mask)
        # (B, N, L, V) -> (B, N, L)
        if connection_filter is not None:
            logits_ = torch.zeros((logits.shape[0],logits.shape[1],logits.shape[2])).to(logits.device) # (B, N, L)
            # conection_filter: (B, V, V)
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    logits_[i,j,0] = torch.argmax(logits[i,j,0])
                    for k in range(1,logits.shape[2]):
                        if logits_[i,j,k-1] == condition[i,j,1]:
                            break
                        # 0 is special token add filter
                        connection_filter_ = torch.cat((torch.tensor([0]).to(logits.device),connection_filter[i,int(logits_[i,j,k-1])-1]))
                        logits_[i,j,k] = torch.argmax(logits[i,j,k] * connection_filter_)
        else:
            logits_ = torch.argmax(logits, dim=-1)
        return logits_
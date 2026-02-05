import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA = torch.cuda.is_available()


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x
    

class RotatELayer(nn.Module):
    def __init__(self, e_dim: int, r_dim: int, p: int = 2):
        super().__init__()
        if e_dim % 2 != 0:
            raise ValueError("RotatE requires even embedding dim (real+imag).")
        self.p = p

        self.rel_proj = (
            nn.Linear(r_dim, e_dim // 2, bias=False) if r_dim != e_dim // 2 else None
        )
        self.input_drop = nn.Dropout(0.3)

    def forward(self, h, r):
        d = h.size(1) // 2
        h_re, h_im = h[:, :d], h[:, d:]
        if self.rel_proj is not None:
            phase = self.rel_proj(r) 
        else:
            phase = r 
        phase = phase / (phase.norm(p=2, dim=-1, keepdim=True) + 1e-9)
        phase = phase * math.pi

        cos_r, sin_r = torch.cos(phase), torch.sin(phase)   # (B, d/2)

        h_re_rot =  h_re * cos_r - h_im * sin_r
        h_im_rot =  h_re * sin_r + h_im * cos_r
        out = torch.cat([h_re_rot, h_im_rot], dim=-1)       # (B, d)
        out = self.input_drop(out)
        return out

class DistMultLayer(nn.Module):
    def __init__(self, e_dim: int, r_dim: int):
        super().__init__()
        self.rel_proj = (
            nn.Linear(r_dim, e_dim, bias=False) if e_dim != r_dim else None
        )
        self.input_drop = nn.Dropout(0.3)

    def forward(self, e_embed, r_embed):
        if self.rel_proj is not None:
            r_embed = self.rel_proj(r_embed)
        x = self.input_drop(e_embed)
        return x * r_embed  
    

class TransELayer(nn.Module):
    def __init__(self, e_dim: int, r_dim: int, p: int = 2):
        super().__init__()
        self.p = p
        self.rel_proj = (
            nn.Linear(r_dim, e_dim, bias=False) if e_dim != r_dim else None
        )
        self.input_drop = nn.Dropout(0.3)

    def forward(self, e_embed, r_embed):
        if self.rel_proj is not None:
            r_embed = self.rel_proj(r_embed)
        h = self.input_drop(e_embed)
        return h + r_embed

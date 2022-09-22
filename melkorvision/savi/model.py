import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np


class Encoder(nn.Module):
    def __init__(self,input_nc=3,z_dim=64,bottom=False):
        super().__init__()
        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential([
                nn.Conv2d(input_nc + 4,z_dim,3,stride=1,padding=1),
                nn.ReLU(True)])
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1, padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self,x):
        """
        input:
            x: input image, [B,3,H,W]
        output:
            feature_map: [B,C,H,W]
        """
        W,H = x.shape[3], x.shape[2]
        X = torch.linspace(-1,1,W)
        Y = torch.linspace(-1,1,H)
        y1_m,x1_m = torch.meshgrid([Y,X])
        x2_m,y2_m = 2 - x1_m,2 - y1_m # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m,x2_m,y1_m,y2_m]).to(x.device).unsqueeze(0) # [1,4,H,W]
        pixel_emb = pixel_emb.repeat([x.size(0),1,1,1])
        inputs = torch.cat([x,pixel_emb],dim=1)

        if self.bottom:
            x_down_0 = self.enc_down_0(inputs)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(inputs)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map

class SlotAttention(nn.Module):
    def __init__(self,num_slots,in_dim=64,slot_dim=64,iters=3,eps=1e-8,hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn([1,1,slot_dim]))
        self.slots_logvars = nn.Parameter(torch.randn([1,1,slot_dim]))
        nn.init.xavier_uniform_(self.slots_logvars)
        self.slots_mu_bg = nn.Parameter(torch.randn([1,1,slot_dim]))
        self.slots_logvars_bg = nn.Parameter(torch.randn([1,1,slot_dim]))
        nn.init.xavier_uniform_(self.slots_logvars_bg)

        self.to_k = nn.Linear(in_dim,slot_dim,bias=False)
        self.to_v = nn.Linear(in_dim,slot_dim,bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        
        self.gru = nn.GRUCell(slot_dim,slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim,slot_dim)

        hidden_dim = max(slot_dim,hidden_dim)
        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim,hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim,slot_dim)
        )
        self.to_res_bg = nn.Sequential([
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,slot_dim)
        ])
        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self,feat,num_slots = None):
        """
        input:
            feat: visual feature with position information, [B,N,C]
        output:
            slots: [B,K,C], attn: [B,K,N]
        """
        B,_,_ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(B,K-1,-1)
        sigma = self.slots_logvars.exp().expand(B,K-1,-1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand([B,1,-1])
        sigma_bg = self.slots_logvars_bg.exp().expand([B,1,-1])
        slot_bg = mu + sigma_bg * torch.randn(mu_bg)

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg)

            dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
            dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
            dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
            attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
            attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
            updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

            slot_bg = self.gru_bg(
                updates_bg.reshape(-1, self.slot_dim),
                slot_prev_bg.reshape(-1, self.slot_dim)
            )
            slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
            slot_bg = slot_bg + self.to_res_bg(slot_bg)

            slot_fg = self.gru(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )
            slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
            slot_fg = slot_fg + self.to_res(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=1)
        return slots, attn

inputs = torch.randn([3,3,128,128])
encoder = Encoder(3)
outputs = encoder(inputs)

print(outputs.shape)
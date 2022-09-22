import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma_bg)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self,feat,num_slots = None):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        B, _, _ = feat.shape
        K = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(B, K-1, -1)
        sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

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

class Decoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False):
        """
        freq: raised frequency
        input_dim: pos emb dim + slot dim
        z_dim: network latent dim
        n_layers: #layers before/after skip connection.
        locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
        locality_ratio: if locality, what value is the boundary to clamp?
        fixed_locality: if True, compute locality in world space instead of in transformed view space
        """
        super().__init__()
        self.n_freq = n_freq
        self.locality = locality
        self.locality_ratio = locality_ratio
        self.fixed_locality = fixed_locality
        self.out_ch = 4
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim+input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers-1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        self.f_before = nn.Sequential(*before_skip)
        self.f_after = nn.Sequential(*after_skip)
        self.f_after_latent = nn.Linear(z_dim, z_dim)
        self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
        self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim//4),
                                     nn.ReLU(True),
                                     nn.Linear(z_dim//4, 3))
        before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
        after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
        for i in range(n_layers - 1):
            before_skip.append(nn.Linear(z_dim, z_dim))
            before_skip.append(nn.ReLU(True))
            after_skip.append(nn.Linear(z_dim, z_dim))
            after_skip.append(nn.ReLU(True))
        after_skip.append(nn.Linear(z_dim, self.out_ch))
        self.b_before = nn.Sequential(*before_skip)
        self.b_after = nn.Sequential(*after_skip)

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, dens_noise=0.):
        """
        1. pos emb by Fourier
        2. for each slot, decode all points from coord and slot feature
        input:
            sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
            sampling_coor_fg: (K-1)xPx3
            z_slots: KxC, K: #slots, C: #feat_dim
            fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
            dens_noise: Noise added to density
        """
        K, C = z_slots.shape
        P = sampling_coor_bg.shape[0]

        if self.fixed_locality:
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
        else:
            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)  # (K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

        z_bg = z_slots[0:1, :]  # 1xC
        z_fg = z_slots[1:, :]  # (K-1)xC
        query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim

        input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)

        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3

        query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
        z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
        input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)

        tmp = self.b_before(input_bg)
        bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.out_ch])  # Px5 -> 1xPx5
        tmp = self.f_before(input_fg)
        tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
        latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
        fg_raw_rgb = self.f_color(latent_fg).view([K-1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
        fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
        if self.locality:
            fg_raw_shape[outsider_idx] *= 0
        fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

        all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
        raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
        masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
        raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
        raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
        masked_raws = unmasked_raws * masks
        raws = masked_raws.sum(dim=0)

        return raws, masked_raws, unmasked_raws, masks

def sin_emb(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d coordinates
    input:
        x: Px3
        n_freq: number of raised frequency
    """
    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded_ = torch.cat(embedded, dim=1)
    return embedded_


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(3)
        self.decoder = Decoder()
        self.slotAttention = SlotAttention(8)

    def forward(self,x):
        # encoder model
        encoder_outputs = self.encoder(x)
        features = encoder_outputs.flatten(start_dim=2).permute([0,2,1]) # shape of [B,N,C]

        # slot attention model
        z_slots,attn = self.slotAttention(features)
        z_slots,attn = z_slots.squeeze(0),attn.squeeze(0) # in the shape of [K,C] , [K,N]

        K = attn.shape[0]
        # decode some thing balbalblba

        P = 64 * 64
        sample_coor_bg = torch.randn([P,3])
        sample_coor_fg = torch.randn([7,P,3])
        cam = torch.eye(3)
        raw,masked_raw,unmask_raw,masks = self.decoder(sample_coor_bg,sample_coor_fg,z_slots,cam)
        print("raw        : ",raw.shape)
        print("masked_raw : ",masked_raw.shape)
        print("unmask_raw : ",unmask_raw.shape)
        print("masks      : ",masks.shape)

        return {"recons":raw}


inputs  = torch.randn([3,3,128,128])
encoder = Encoder(3)
outputs = encoder(inputs)

print(outputs.shape)

inputs     = torch.randn([3,100,64])
slat       = SlotAttention(8)
slots,attn = slat(inputs)

print(slots.shape,attn.shape)


print("start the decoder part:")

inputs  = torch.randn([1,3,128,128])
model   = AutoEncoder()
outputs   = model(inputs)

print("### start the training session")
import time
time.sleep(100)

### start the training session
import datetime
import random

class PARTNET(torch.utils.data.Dataset):
    def __init__(self,name):
        super().__init__()
        self.name = "PartNet"
    def __getitem__(self,index):return index

    def __len__(self):return 6

criterion = nn.MSELoss()
params  = [{"params":model.parameters()}]

train_set = PARTNET("train")
train_loader = torch.utils.data.Dataloader(train_set,batch_size = 2,shuffle=True)

optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)

device = "cpu"

import time
start = time.time()
warmup_step = 5000
model_dir = "model.ckpt"

num_slots = 7
fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))

i = 0
for epoch in range(100):
    model.train()

    total_loss = 0
    
    for sample in train_loader:
        i += 1
        
        if i < warmup_step:
            learning_rate = optimizer.learning_rate * ((i+1)/warmup_step)
        else:
            learning_rate = 2e-4
        
        optimizer.param_groups[0]["lr"] = learning_rate

        image = sample["image"].to(device)
        recon_combined,recons,masks,slots = model(image)
        loss = criterion(recon_combined,image)
        total_loss += loss.item()

        del recons,masks,slots

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    total_loss /= len(train_loader)

    if (not epoch %10):
        print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))
        image = train_set[int(random.random() * 4)]["image"].to(device).unsqueeze(0)

        recon_combined, recons, masks, slots = model(image)
        num_slots = 7
    
        image = image.squeeze(0)
        recon_combined = recon_combined.squeeze(0)
        recons = recons.squeeze(0)
        masks = masks.squeeze(0)
        image = image.permute(1,2,0).cpu().numpy()
        
        recon_combined = recon_combined.permute(1,2,0)
        recon_combined = recon_combined.cpu().detach().numpy()
        
        recons = recons.cpu().detach().numpy()
        masks = masks.cpu().detach().numpy()

        ax[0].imshow(image)
        ax[0].set_title('Image{}'.format(epoch))
        ax[1].imshow(recon_combined)
        ax[1].set_title('Recon.')
        for i in range(7):
            picture = recons[i] * masks[i] + (1 - masks[i])
            ax[i + 2].imshow(picture)
            ax[i + 2].set_title('Slot %s' % str(i + 1))
        for i in range(len(ax)):
            ax[i].grid(False)
            ax[i].axis('off')
        plt.pause(0.001)
    if not epoch % 10:
        torch.save(model, model_dir)

plt.ioff()
plt.show()
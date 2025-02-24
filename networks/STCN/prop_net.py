"""
Modifed from the original STM code https://github.com/seoungwugoh/STM
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.STCN.modules import *


# class Decoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.compress = ResBlock(1024, 512, bn=True)
#         # self.up_16_8 = UpsampleBlock(512+512, 512, 256) # 1/16 -> 1/8
#         self.up_16_8 = UpsampleBlock(512, 512, 256)
#         self.up_8_4 = UpsampleBlock(256+256, 256, 128) # 1/8 -> 1/4
#         self.up_4_2 = UpsampleBlock(128, 128, 64)
#         self.up_2_1 = UpsampleBlock(64, 64, 32)

#         self.pred = nn.Conv2d(32, 1, kernel_size=(3,3), padding=(1,1), stride=1)

#     def forward(self, f16, f8, f4, f_3d):
#         x = self.compress(f16)
#         # x = self.up_16_8(torch.cat([f8, f_3d["f8"]], dim=1), x)
#         x = self.up_16_8(f8, x)
#         x = self.up_8_4(torch.cat([f4, f_3d["f4"]], dim=1), x)
#         x = self.up_4_2(f_3d["f2"], x)
#         x = self.up_2_1(f_3d["f1"], x)

#         x = self.pred(F.relu(x))

#         return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512, bn=True)
        # self.up_16_8 = UpsampleBlock(512+512, 512, 256) # 1/16 -> 1/8
        self.up_16_8 = UpsampleBlock(512, 512, 256)
        self.up_8_4 = UpsampleBlock(256+256+128+64, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4, f_3d):
        x = self.compress(f16)
        # x = self.up_16_8(torch.cat([f8, f_3d["f8"]], dim=1), x)
        x = self.up_16_8(f8, x)
        f_3d_2 = F.interpolate(f_3d["f2"], scale_factor=0.5, mode='bilinear', align_corners=False)
        f_3d_1 = F.interpolate(f_3d["f1"], scale_factor=0.25, mode='bilinear', align_corners=False)
        x = self.up_8_4(torch.cat([f4, f_3d["f4"], f_3d_2, f_3d_1], dim=1), x)

        x = self.pred(F.relu(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g

def softmax_w_g_top(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)*gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1) # b*topk*hw
            x_exp = torch.exp(values - values[:,0:1])

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp = x_exp / x_exp_sum
        x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

        output = x # 只有topk的位置有值，其余位置为0，topk的值经过了softmax
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x-maxes)*gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output

class EvalMemoryReader(nn.Module):
    def __init__(self, top_k, km):
        super().__init__()
        self.top_k = top_k
        self.km = km

    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape

        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        c = qk.pow(2).sum(1).unsqueeze(1)

        affinity = (-a+b-c) / math.sqrt(CK)   # B, THW, HW

        if self.km is not None:
            # Make a bunch of Gaussian distributions
            argmax_idx = affinity.max(2)[1]
            y_idx, x_idx = argmax_idx//W, argmax_idx%W
            g = make_gaussian(y_idx, x_idx, H, W, sigma=self.km)
            g = g.view(B, T*H*W, H*W)

            affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=g)  # B, THW, HW
        else:
            if self.top_k is not None:
                affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=None)  # B, THW, HW
            else:
                affinity = F.softmax(affinity, dim=1)

        return affinity

    def readout(self, affinity, mv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        return mem

class AttentionMemory(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
 
    def forward(self, mk, qk): 
        """
        T=1 only. Only needs to obtain W
        """
        B, CK, _, H, W = mk.shape

        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        c = qk.pow(2).sum(1).unsqueeze(1)

        affinity = (-a+b-c) / math.sqrt(CK)   # B, THW, HW
        affinity = F.softmax(affinity, dim=1)

        return affinity

class PropagationNetwork(nn.Module):
    def __init__(self, top_k=20):
        super().__init__()
        self.value_encoder = ValueEncoder() 
        self.key_encoder = KeyEncoder() 

        self.key_proj = KeyProjection(1024, keydim=64)
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = EvalMemoryReader(top_k, km=None)
        self.attn_memory = AttentionMemory(top_k)
        self.decoder = Decoder()

    def encode_value(self, frame, kf16, mask):
        # frame, mask: b*1*h*w 
        f16 = self.value_encoder(frame, kf16, mask)
        return f16.unsqueeze(2) # B*512*1*H*W

    def encode_key(self, frame): 
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16) # b*64*h/16*w/16
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4 # f16_thin = qv16

    def segment_with_query(self, mk16, mv16, qf8, qf4, qk16, qv16, f_3d):  # f_3d: b*c*h*w，由三维特征转换得到的某一帧的特征
        affinity = self.memory.get_affinity(mk16, qk16) # b*thw*hw

        k = mv16.shape[0]
        # Do it batch by batch to reduce memory usage
        batched = 1
        m4 = torch.cat([
            self.memory.readout(affinity[i:i+1], mv16[i:i+1]) for i in range(0, k, batched)
        ], 0)

        qv16 = qv16.expand(k, -1, -1, -1)
        m4 = torch.cat([m4, qv16], 1)

        return torch.sigmoid(self.decoder(m4, qf8, qf4, f_3d)), affinity

    def get_W(self, mk16, qk16):
        W = self.attn_memory(mk16, qk16)
        return W

    def get_attention(self, mk16, pos_mask, neg_mask, qk16):
        b, _, h, w = pos_mask.shape
        nh = h//16
        nw = w//16

        W = self.get_W(mk16, qk16)

        pos_map = (F.interpolate(pos_mask, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
        neg_map = (F.interpolate(neg_mask, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
        attn_map = torch.cat([pos_map, neg_map], 1)
        attn_map = attn_map.reshape(b, 2, nh, nw)
        attn_map = F.interpolate(attn_map, mode='bilinear', size=(h,w), align_corners=False)

        return attn_map

class Converter(nn.Module): # convert 3d feature to 2d feature
    def __init__(self, f_levels=[4,2], channels=[256, 128], c_last_conv=128):
        super().__init__()

        self.convert_f_3d = nn.ModuleDict({})
        for i in range(len(f_levels)):
            self.convert_f_3d['f'+str(f_levels[i])] = nn.Sequential(
                nn.Conv2d(channels[i]+f_levels[i], channels[i], kernel_size=1),
                nn.ReLU()
            )

        self.f_levels = f_levels
        self.channels = channels + [c_last_conv]
        
        self.pred_blocks = nn.ModuleList([ResBlock(self.channels[i],self.channels[i+1]) for i in range(len(f_levels))])
        self.pred_out = nn.Sequential(
            nn.Conv2d(c_last_conv, 1, kernel_size=(3,3), padding=(1,1), stride=1),
            nn.Sigmoid()
        )

    def convert_3d_feature(self, f_3d, pos, pred_mask=False):
        '''
        f_3d: dict of 3d features with size b*c*h*w, slice from original 3d feature map with size b*c*d*h*w

        pos: dict of position embedding, one-hot vectors with dim 4/8/16, \
            for example, a feature map slice with size h*w and level step 4 corresponds to 4*4h*4w in the original image, \
                if the target 2d slice is the second one in original image, then the position vector is (0,1,0,0).
        '''
        # for k in f_3d.keys():
        #     print("1", k, f_3d[k].shape)
        preds = {}
        for r in self.f_levels:
            k = "f"+str(r)
            b, c, h, w = f_3d[k].shape
            pos_map = pos[k].repeat(b,h,w,1).permute(0,3,1,2) # b*r*h*w
            f_3d[k] = self.convert_f_3d[k](torch.cat([f_3d[k], pos_map], dim=1))

            if pred_mask:
                # print("========  {}  ========".format(r))
                # print("feature shape", f_3d[k].shape)
                # print("index", self.f_levels.index(r))
                preds[k] = self.pred_f_i(f_3d[k], self.f_levels.index(r))

        # for k in f_3d.keys():
        #     print("2", k, f_3d[k].shape)
        # for k in preds.keys():
        #     print("3", k, preds[k].shape)

        return f_3d, preds
    
    def pred_f_i(self, f, i):
        f = self.pred_blocks[i](f)
        if i < len(self.f_levels)-1:
            f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
            return self.pred_f_i(f, i+1)
        else:
            f = self.pred_out(f)
            f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
            return f

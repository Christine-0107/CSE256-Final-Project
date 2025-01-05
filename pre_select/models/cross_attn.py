import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.distributed as dist
from torch.nn.parameter import Parameter

import math

# 计算OCR score
class Verify(nn.Module):
    def __init__(self):
        super().__init__()

        img2img_attn_args = {'d_model': 256, 'h': 8, 'dropout': 0.1, 'pos_x_range': [-20, 20], 'pos_y_range': [-20, 20], 'pos_index_offset': 20}
        args = {'embed_dim': 256, 'num_heads': 8, 'dropout': 0.1}
        
        # q: question_0   k,v: img
        #self.MH_Attn_0 = nn.MultiheadAttention(**args,add_zero_attn=True)
        self.MH_Attn_0 = nn.MultiheadAttention(**args)
        
        # q, k: question_0 + question_1    v: question_0
        #self.MH_Attn_1 = nn.MultiheadAttention(**args,add_zero_attn=True)
        self.MH_Attn_1 = nn.MultiheadAttention(**args)

        # q: question_2   k,v: ocr
        #self.MH_Attn_2 = nn.MultiheadAttention(**args, add_zero_attn=True)
        self.MH_Attn_2 = nn.MultiheadAttention(**args)



    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, ques_feat, ques_mask, ocr_feat, ocr_mask, img_feat, img_mask, pos_embed):
        question_0 = ques_feat
        #print("MH 0")
        #print("question0: ",question_0.size())
        #print("img: ", img_feat.size())
        # MH Attn 0
        question_1 = self.MH_Attn_0(
            query=question_0, key=self.with_pos_embed(img_feat, None),
            value=img_feat, key_padding_mask=img_mask)[0]
        #print("question1: ",question_1.size())
        

        # MH Attn 1
        #print("MH 1")
        q = k = question_0 + question_1
        #print(q.size())
        question_2 = self.MH_Attn_1(
            query=q, key=self.with_pos_embed(k, None),
            value=question_0, key_padding_mask=ques_mask)[0]
        #print("question2: ",question_2.size())
        

        # MH Attn 2
        attn_weights = self.MH_Attn_2(
            query=question_2, key=self.with_pos_embed(ocr_feat, None),
            value=ocr_feat, key_padding_mask=ocr_mask)[1]
        tmp = self.MH_Attn_2(
            query=question_2, key=self.with_pos_embed(ocr_feat, None),
            value=ocr_feat, key_padding_mask=ocr_mask)[0]
        #print("att weights")
        #print(attn_weights)     
        #attn_weights = attn_weights[:, :, :-1]
        results_list = []
        for attn_weight in attn_weights:
            #print("attention 遍历")
            #print(attn_weight)
            #print(attn_weight.size())
            results = torch.sum(attn_weight, dim = 0)
            results_list.append(results)
            #print("results")
            #print(results)
            #print(results.size())
        results_list = torch.stack([results for results in results_list])
        #print("result_list ")
        #print(results_list)
        #print(results_list.size())
        return results_list

class MHAttentionRPE(nn.Module):
    ''' With relative position embedding '''
    def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
                 pos_x_range=[-20, 20], pos_y_range=[-20, 20], pos_index_offset=20,
                 learnable_pos_embed=False):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.scaling = float(self.d_k) ** -0.5
        self.return_raw_attention = return_raw_attention

        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None
        # self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout
        self._reset_parameters()

        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            self.pos_x = nn.Embedding(pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)
            self.pos_y = nn.Embedding(pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
        else:
            pos_x, pos_y = position_embedding_sine(d_model // 2, normalize=True,
                                                   x_range=pos_x_range, y_range=pos_y_range)
            self.register_buffer('pos_x', pos_x) # [x_range, C]
            self.register_buffer('pos_y', pos_y) # [y_range, C]

        self.pos_index_offset = pos_index_offset

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)


    def forward(self, query, key, value, key_padding_mask=None):
        tgt_len, bs, dim = query.size()
        src_len, _, dim = key.size()

        weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
        weight_k, bias_k = self.in_proj_weight[dim:dim*2], self.in_proj_bias[dim:dim*2]
        weight_v, bias_v = self.in_proj_weight[dim*2:], self.in_proj_bias[dim*2:]

        q = query.matmul(weight_q.t()) + bias_q
        k = key.matmul(weight_k.t()) + bias_k
        v = value.matmul(weight_v.t()) + bias_v

        q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)  # [bs*h, tgt_len, dim//h]
        k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)  # [bs*h, dim//h, src_len], To calculate qTk (bmm)
        v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

        q = q * self.scaling
        attn_weights = torch.bmm(q, k)  # [bs*h, tgt_len, src_len]

        ### compute the relative positions
        bs, HW = key_padding_mask.size()
        assert (HW == 400) and (HW == tgt_len)
        img_mask = ~key_padding_mask.view(bs, 20, 20)
        yy = img_mask.cumsum(1, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
        xx = img_mask.cumsum(2, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
        diff_yy = yy[:, :, None] - yy[:, None, :]  # [bs, HW, HW]
        diff_xx = xx[:, :, None] - xx[:, None, :]  # [bs, HW, HW]
        if self.learnable_pos_embed:
            k_posy = self.pos_y.weight.matmul(weight_k.t()[:dim//2])  # [x_range, dim]
            k_posx = self.pos_x.weight.matmul(weight_k.t()[dim//2:])  # [y_range, dim]
        else:
            k_posy = self.pos_y.matmul(weight_k.t()[:dim//2])  # [x_range, dim]
            k_posx = self.pos_x.matmul(weight_k.t()[dim//2:])  # [y_range, dim]
        k_posy = k_posy.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
                        reshape(-1, bs * self.h, dim//self.h).permute(1, 2, 0)  # [bs*h, dim//h, y_range]
        k_posx = k_posx.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
                        reshape(-1, bs * self.h, dim//self.h).permute(1, 2, 0)  # [bs*h, dim//h, x_range]
        posy_attn_weights = torch.bmm(q, k_posy).view(bs, self.h, HW, -1)  # [bs, h, HW, y_range]
        posx_attn_weights = torch.bmm(q, k_posx).view(bs, self.h, HW, -1) # [bs, h, HW, x_range]
        diff_yy_idx = diff_yy[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset
        diff_xx_idx = diff_xx[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset

        posy_attn_weights = torch.gather(posy_attn_weights, -1, diff_yy_idx.long()) # [bs, h, HW, HW]
        posx_attn_weights = torch.gather(posx_attn_weights, -1, diff_xx_idx.long())  # [bs, h, HW, HW]
        pos_attn_weights = (posy_attn_weights + posx_attn_weights).view(bs*self.h, HW, -1)
        attn_weights = attn_weights + pos_attn_weights


        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [bs, 1, 1, src_len]
                float('-inf')
            )
            attn_weights = attn_weights.view(-1, tgt_len, src_len)
        raw_attn_weights = attn_weights
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights


def position_embedding_sine(num_pos_feats=64, temperature=10000, normalize=False, scale=None,
             x_range=[-20, 20], y_range=[-20, 20], device=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device) #
    y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1] + eps) * scale
        x_embed = x_embed / (x_embed[-1] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
    return pos_x, pos_y
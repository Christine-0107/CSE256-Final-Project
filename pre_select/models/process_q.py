import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.distributed as dist
from torch.nn.parameter import Parameter

import math



class Process_q(nn.Module):
    def __init__(self):
        super().__init__()

        args = {'embed_dim': 256, 'num_heads': 8, 'dropout': 0.1}
        
        # q: self attention
        #self.MH_Attn_0 = nn.MultiheadAttention(**args,add_zero_attn=True)
        self.MH_Attn_0 = nn.MultiheadAttention(**args)
        
        # q: question     k,v: ocr
        #self.MH_Attn_1 = nn.MultiheadAttention(**args,add_zero_attn=True)
        self.MH_Attn_1 = nn.MultiheadAttention(**args)



    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, ques_feat, ques_mask, img_feat, img_mask, ocr_feat, ocr_mask):
        question_0 = ques_feat
        
        # MH Attn 0
        question_attn_weights = self.MH_Attn_0(
            query=question_0, key=self.with_pos_embed(ques_feat, None),
            value=ques_feat, key_padding_mask=ques_mask)[1]
        # print(question_attn_weights.size()) # 8 40 40
        #question_attn_weights = question_attn_weights[:, :, :-1]
        
        # MH Attn 1
        ocr_attn_weights = self.MH_Attn_1(
            query=question_0, key=self.with_pos_embed(ocr_feat, None),
            value=ocr_feat, key_padding_mask=ocr_mask)[1]       
        # print(ocr_attn_weights.size()) # 8 40 500
        #ocr_attn_weights = ocr_attn_weights[:, :, :-1]

        results_list = []
        for attn_weight in question_attn_weights:
            # print(attn_weight.size())
            results = torch.sum(attn_weight, dim = 0)
            results_list.append(results)
            # print(results.size())
        results_list = torch.stack([results for results in results_list])
        # print(results_list)
        # print(results_list.size()) # 8 40
        question_weights = results_list

        results_list = []
        for attn_weight in ocr_attn_weights:
            # print(attn_weight.size())
            results = torch.sum(attn_weight, dim = 0)
            results_list.append(results)
            # print(results.size())
        results_list = torch.stack([results for results in results_list])
        ocr_weights = results_list
        # print(results_list)
        # print(results_list.size()) # 8 500
        return question_weights, ocr_weights
    
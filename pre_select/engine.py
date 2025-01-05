import math
import os
import sys
from typing import Iterable

import torch
import torch.nn as nn

import misc as utils
import box_ops

import logging
import torch.distributed as dist
import time
import datetime
from tqdm import tqdm

import numpy as np
import json

from PIL import Image
from transformers import AutoTokenizer

import torch.optim as optim

from torch import cuda
#device = torch.device("cpu")
device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.set_device(2)

import heapq


class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            #image, mask, targets, deepsolo_ocr, answers, item, t5_dict = next(self.loader)
            #self.next_img = image
            #self.next_mask = mask
            samples, targets, deepsolo_ocr, answers, item, t5_dict = next(self.loader)
            self.next_img, self.next_mask = samples.decompose()
            self.next_target = targets
            self.next_deepsolo_ocr = deepsolo_ocr
            self.next_answers = answers
            self.next_item = item
            self.next_t5_dict = t5_dict
        except StopIteration:
            self.next_img = self.next_mask = self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.to(self.device, non_blocking=True)
            self.next_mask = self.next_mask.to(self.device, non_blocking=True)
            tensor_dict = self.next_target.tensor_dict
            self.next_target.tensor_dict = {k: tensor_dict[k].to(self.device, non_blocking=True) for k in tensor_dict}
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, mask, target, deepsolo_ocr, answers, item, t5_dict = self.next_img, self.next_mask, self.next_target, self.next_deepsolo_ocr, self.next_answers, self.next_item, self.next_t5_dict
        self.preload()
        return img, mask, target, deepsolo_ocr, answers, item, t5_dict

    def __next__(self):
        img, mask, target, deepsolo_ocr, answers, item, t5_dict = self.next()
        if img == None:
            raise StopIteration
        return img, mask, target, deepsolo_ocr, answers, item, t5_dict

    def __iter__(self):
        return self

    def __len__(self):
        return self.length



def create_features(
    tokenizer,
    target_size=(1000, 1000),
    max_seq_length=-1,
    use_ocr=False,
    bounding_box=None,
    words=None,
    pad_token_box=[0, 0, 0, 0]
):
    # Rescaling the bounding box as per the image size
    if (use_ocr == False) and (bounding_box == None or words == None):
        raise Exception(
            'Please provide the bounding box and words or pass the argument "use_ocr" = True')

    tokenized_words, boxes, attention_mask = get_tokens_with_boxes(bounding_box, words, tokenizer,
                                                                   pad_token_box, max_seq_len=max_seq_length)
    return boxes, tokenized_words, attention_mask


def get_tokens_with_boxes(unnormalized_word_boxes, list_of_words, tokenizer, pad_token_box=[0, 0, 0, 0], max_seq_len=-1, eos_token_box=[0, 0, 1000, 1000]):
    # 2. Performing the semantic pre-processing
    encoding = tokenizer(list_of_words, is_split_into_words=True,
                         add_special_tokens=False)

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Note that, there is no need for bboxes, since the model does not use bbox as feature, so no pre-processing of that
    bbox_according_to_tokenizer = [unnormalized_word_boxes[i]
                                   for i in encoding.word_ids()]

    # Truncation of token_boxes + token_labels
    special_tokens_count = 1
    if max_seq_len != -1 and len(input_ids) > max_seq_len - special_tokens_count:
        bbox_according_to_tokenizer = bbox_according_to_tokenizer[: (
            max_seq_len - special_tokens_count)]
        input_ids = input_ids[: (max_seq_len - special_tokens_count)]
        attention_mask = attention_mask[: (max_seq_len - special_tokens_count)]

    # Padding
    input_ids = input_ids + [tokenizer.eos_token_id]
    bbox_according_to_tokenizer = bbox_according_to_tokenizer + [eos_token_box]
    attention_mask = attention_mask + [1]

    if max_seq_len != -1:
        pad_length = max_seq_len - len(input_ids)

        input_ids = input_ids + [tokenizer.pad_token_id] * (pad_length)
        bbox_according_to_tokenizer = bbox_according_to_tokenizer + \
            [pad_token_box] * (pad_length)
        attention_mask = attention_mask + [0] * (pad_length)

    return input_ids, bbox_according_to_tokenizer, attention_mask


def order_ocr_tokens(boxes, words):
    order_list = [[] for i in range(100)]
    # print(words)
    # print(boxes)
    
    for i in range(len(boxes)):
        boxes[i] = boxes[i][:4]
        boxes[i].append(i)
        # print(boxes[i][1])
        # print(int(boxes[i][1] / 100))
        order_list[int(boxes[i][1] / 10)].append(boxes[i])
   
    index = 100
    for i in range(len(order_list)):
        temp_list = []
        while order_list[i]:
            min_left = 1000
            for j in range(len(order_list[i])):
                if order_list[i][j][0] < min_left:
                    min_left = order_list[i][j][0]
                    index = j
            temp_list.append(order_list[i][index])
            del order_list[i][index]
        order_list[i] = temp_list
    
    new_boxes = []
    for i in range(len(order_list)):
        if order_list[i]:
            for box in order_list[i]:
                new_boxes.append(box)
    
    new_words = []
    last_boxes = []
    # print(new_boxes)
    for new_box in new_boxes:
        new_words.append(words[new_box[4]])
        last_boxes.append(new_box[:-1])
    # print(last_boxes)
    return last_boxes, new_words


class SpatialModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_left_x = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_x = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.top_left_y = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_y = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.width_emb = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        
    def forward(self, coordinates):
        top_left_x_feat =     self.top_left_x(coordinates[:,:, 0])
        top_left_y_feat =     self.top_left_y(coordinates[:,:, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:,:, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:,:, 3])
        width_feat =          self.width_emb(coordinates[:,:, 4])
        height_feat =         self.height_emb(coordinates[:,:, 5])
        
        layout_feature = top_left_x_feat + top_left_y_feat + bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat
        return layout_feature



from transformers import T5ForConditionalGeneration, ViTModel
from transformers import AutoTokenizer, AutoConfig, AutoProcessor

class mymodel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config = config)
        self.t5model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        # self.img_feat_extractor = AutoModel.from_pretrained('"google/vit-base-patch16-224-in21k"')
        self.img_feat_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        max_2d_position_embeddings = 1024
        vit_model = "google/vit-base-patch16-224-in21k"
        config.update({"max_2d_position_embeddings" : max_2d_position_embeddings,
                    "vit_model" : vit_model})
        self.spatial_feat_extractor = SpatialModule(config)
    
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def forward(self, 
                input_ids, 
                img = None, 
                bbox = None,
                attention_mask = None, 
                labels = None,
                is_generate = True):
        # print(input_ids.size())
        lang_feat = self.t5model.shared(input_ids)
        spatial_feat = self.spatial_feat_extractor(bbox)
        layout_feat = lang_feat + spatial_feat
        
        img_feat = self.img_feat_extractor(img).last_hidden_state
        inputs_embeds = torch.cat([img_feat, layout_feat], axis = 1)
        # print(attention_mask.size())
        attention_mask = torch.cat([torch.ones(img_feat.shape[:2]).to(img_feat.device), attention_mask], axis = 1)
        # print(lang_feat.size())
        # print(spatial_feat.size())
        # print(img_feat.size())
        # print(attention_mask.size())
        # print(attention_mask.size())
        
        if is_generate == True:
            return self.t5model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels), self.t5model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, num_beams = 2)
        else:
            return self.t5model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels)
        # print(img)
        '''
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = self.img_feat_extractor(**inputs)
        img_feat = outputs.last_hidden_state
        
        lang_feat = self.t5model.shared(input_ids)
        
        print(lang_feat.size())
        print(img_feat.size())
        print(input_ids.size())
        
        inputs_embeds = torch.cat([lang_feat, img_feat], axis = 1)
        attention_mask = torch.ones(inputs_embeds.shape[:2]).to(device)
        # inputs_embeds = lang_feat
        # print(inputs_embeds.size())
        # print(attention_mask.size())
        if if_generate == True:
            return self.t5model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels), self.t5model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, num_beams=2, min_length=0, max_length=20)
        else:
            return self.t5model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels)
        '''
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logging.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

def pad_sequence(sequence, pad_value):
    '''
    A function to pad a sequence of tensors to the maximum length tensor, currently it supports 1d and 2d tensors
    Arguments:
        sequence: A list of tensors
        pad_value: The value to pad the tensors with
    Returns:
        A tensor with the padded tensors
    '''
    max_len = 0
    for i in sequence:
        max_len = max(max_len, len(i))

    for i, _ in enumerate(sequence):
        pad_length = max_len - len(_)
        if pad_length != 0:
            pad_entry = torch.stack([pad_value for j in range(pad_length)])
            sequence[i] = torch.cat([sequence[i], pad_entry])

    return torch.stack(sequence)

# latr_model = LaTrForConditionalGeneration(model_config)

tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_config = AutoConfig.from_pretrained("t5-base")
latr_model = mymodel(t5_config)

latr_model = latr_model.to(device)
#latr_model.load_state_dict(torch.load("/home/yanruxue/latr-main/src/new_latr/checkpoints/select_singleloss/model_8000.pth"))
t5_optimizer = torch.optim.AdamW(latr_model.parameters(), lr = 1e-5)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.hidden_dim = 100
        self.relu = nn.ReLU() #relu激活函数
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim//2) #全连接层1，输入100，输出50
        self.fc2 = nn.Linear(self.hidden_dim//2, 1) #全连接层2，输入50，输出1（最终输出一个标量）

    def forward(self, x): #将输入x映射成一个实数输出
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out






def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, data_loader_val: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    value_network = ValueNetwork().to(device)
    value_optimizer = optim.Adam(value_network.parameters(), lr=0.0001)
    model.train() #这个任务中传入的model是OCR得分器
    criterion.train() #这个任务中传入的criterion是计算model的损失
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast = True)
    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target, deepsolo_ocr, answers, item, t5_dict = prefetcher.next()
    iteration = 0
    cnt_1 = 0
    total_1 = 0
    cnt_2 = 0
    total_2 = 0
    new_imdb = []
    while img is not None:

        print("train iter:", iteration)
        # iteration = iteration + 1
        if iteration!=0 and iteration % 1000 == 0: #训练1000iter评估一次
            loss, total_3, cnt_3, acc = evaluate(
            model, criterion, data_loader_val, device
            )
            logging.info("iter:{}, loss:{}, total_2:{}, cnt_2:{}, acc:{}".format(iteration, loss, total_3, cnt_3, acc))
        '''
        save_path = "/home/yanruxue/latr-main/src/VLTVG/checkpoints/rl_0.7/epoch_" + str(epoch) + '.pth'
        torch.save(model.state_dict(), save_path)
        '''
        iteration = iteration + 1

        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        ocr_id, ocr_mask, cnt_mask, ocr_label = target_dict['ocr_id'], target_dict['ocr_mask'], target_dict['cnt_mask'], target_dict['ocr_label']
        ocr_str_id, ocr_str_mask = target_dict['ocr_str_id'], target_dict['ocr_str_mask']
        binary_label = target_dict['binary_label']

        
        binary_output, verify_output = model(img, mask, word_id, word_mask, ocr_id, ocr_mask, ocr_label, ocr_str_id, ocr_str_mask, binary_label)
        #verify_output = model(img, mask, word_id, word_mask, ocr_id, ocr_mask)

        '''
        for output, binary, label, mask_1, ocr, answer, imdb, i in zip(verify_output, binary_output, ocr_label, cnt_mask, deepsolo_ocr, answers, item, range(8)):
            total_2 += 1
            max_attn = output.max()
            max_index = (output==max_attn).nonzero()
            word = ocr[max_index.item()][0]
            if word in answer:
                cnt_2 += 1

        losses = criterion(verify_output, ocr_label)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(total_2, cnt_2)
        '''
        add_list = []
        # print(outputs)
        #for output, mask_1, ocr, answer, imdb, i in zip(verify_output, cnt_mask, deepsolo_ocr, answers, item, range(8)):
        for output, binary, label, mask_1, ocr, answer, imdb, i in zip(verify_output, binary_output, ocr_label, cnt_mask, deepsolo_ocr, answers, item, range(2)):
            # print(mask_1)
            cnt_index = (mask_1==1).nonzero()
            output = output[:len(cnt_index)]
            
            if len(output):

                index = output.argmax()
                # print(index.item())

            
                if binary[0].item() > 0.5:
                #if True:
                    total_2 += 1
                    word = ocr[index.item()][0]
                    add_list.append([i, index.item(), word])
                    if word in answer:
                        cnt_2 += 1

        
        # add word and box 
        #print("add_list:")
        #print(add_list)
        #print("verify_output:")
        #print(verify_output)
           
        for info in add_list:
            # print(t5_dict['origin_boxes'][info[0]])
            t5_dict['origin_words'][info[0]].append(t5_dict['origin_words'][info[0]][info[1]])
            t5_dict['origin_boxes'][info[0]].append(t5_dict['origin_boxes'][info[0]][info[1]])
            # print(t5_dict['origin_boxes'][info[0]])

        # print(t5_dict['input_ids'])
        bbox_list = []
        input_ids_list = []
        attention_mask_list = []
        for words, boxes, question_id, question_mask, box_pretext in zip(t5_dict['origin_words'], t5_dict['origin_boxes'], t5_dict['question_id'], t5_dict['question_mask'], t5_dict['box_pretext']):
            if words:
                boxes, words = order_ocr_tokens(boxes, words)
            boxes, tokenized_words, attention_mask = create_features(tokenizer=t5_tokenizer, use_ocr=False, 
                                                                     words=words, bounding_box=boxes,
                                                                    target_size=(1000, 1000))
            
            boxes = box_pretext + boxes  
            tokenized_words = question_id + tokenized_words
            attention_mask = question_mask + attention_mask

            boxes = torch.as_tensor(boxes, dtype=torch.int32)

            ## Clamping the values of boxes, since there are some entries, which makes width | height negative
            width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1)
            height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1)
            boxes = torch.cat([boxes, width, height], axis=-1)
            boxes = torch.clamp(boxes, min=0, max=1000)
            boxes = boxes.numpy().tolist()

            
            bbox_list.append(torch.tensor(boxes))
            input_ids_list.append(torch.tensor(tokenized_words))
            attention_mask_list.append(torch.tensor(attention_mask))

        bbox_list = pad_sequence(
                bbox_list, torch.as_tensor([0, 0, 0, 0, 0, 0]))
        bbox_list = torch.stack([bbox for bbox in bbox_list])
        t5_dict['bbox'] = bbox_list

        input_ids_list = pad_sequence(
                input_ids_list, torch.as_tensor(0))
        input_ids_list = torch.stack([input_ids for input_ids in input_ids_list])
        t5_dict['input_ids'] = input_ids_list

        attention_mask_list = pad_sequence(
                attention_mask_list, torch.as_tensor(0))
        attention_mask_list = torch.stack([attention_mask for attention_mask in attention_mask_list])
        t5_dict['attention_mask'] = attention_mask_list
        # print(t5_dict['input_ids'])
        
        # t5_dict finished
        for key in t5_dict:
            if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key != 'origin_words' and key != 'origin_boxes' and key != 'question_id' and key != 'question_mask' and key != 'box_pretext':
                # print(key)
                t5_dict[key] = t5_dict[key].to(device)

        outputs = latr_model(input_ids = t5_dict['input_ids'],
                            labels = t5_dict['labels'], 
                            bbox = t5_dict['bbox'],
                            attention_mask = t5_dict['attention_mask'],
                            img = t5_dict['img'],
                            is_generate = False)


        t5_loss = outputs['loss'][0]
        print("t5_loss:", t5_loss)
        t5_optimizer.zero_grad()
        t5_loss.backward()
        t5_optimizer.step()
        
        



        """
        # reinforce 
        #out_logits = verify_output  # 获取模型的最后100个分数 <b, 100>
        out_logits = verify_output[:, -100:]

        # 检查out_logits中的每个元素是否为NaN
        #isnan_out_logits = torch.isnan(out_logits)

        # 检查是否所有元素都是NaN
        #all_nan = torch.all(isnan_out_logits)

        

        
        indices = torch.zeros_like(out_logits)  # 获取<b, 100>的one hot值，直接argmax之后的结果
        for i in range(out_logits.shape[0]):
            out_logit = out_logits[i]
            index = out_logit.argmax()
            indices[i][index] = 1

        
        baseline = value_network(out_logits)  # <b, 100> -> <b, 1> 模型输出->reward维度的网络
        
        # standardize rewards
        #print('loss-1:')
        #print(outputs['loss'][1])
        #print(outputs['loss'][1].shape)
        rewards = get_reward(outputs['loss'][1])  # 计算T5的loss <b, 1>
        # reward = torch.Tensor(rewards).to(baselines[0].device)
        
        if(rewards.dim()==1):
            rewards = rewards - rewards.mean()
        else:
            rewards = (rewards - rewards.mean()) / (
                rewards.std() + float(np.finfo(np.float32).eps))  # !!!问题
        
        # baseline = torch.cat(baselines).squeeze()

        losses = []
        #print("out_logits:")
        #print(out_logits)
        for action, p, r, b in zip(indices, out_logits, rewards, baseline):
            advantage = (-r) - b
            #torch.softmax(p, dim=-1)
            #print(p)
            #print(action)
            #j=p
            #torch.softmax(j, dim=-1)
            #print(j)
            tmp = criterion(p, action)
            tmp1 = advantage.detach()/len(indices)  #问题！
            #print("advantage.detach():")
            #print(advantage.detach())
            losses.append(tmp
                    * tmp1) 
        #print("losses:")
        #print(len(losses))
        #print(losses)
        

        baseline = baseline.squeeze().to(device) 
        
        
        # rewards = rewards.to(device) 
        MSELoss = nn.MSELoss().to(device)
        rewards = rewards.to(device)
        critic_loss = MSELoss(baseline, rewards)
        

        critic_loss = critic_loss.unsqueeze(-1)
        #print("critic_loss:")
        #print(critic_loss)
        m2_loss = sum(losses)/ len(losses)
        #print("m2_loss:")
        #print(m2_loss)
        # print(critic_loss + m2_loss)
        r_loss = critic_loss + m2_loss
        print("r_loss:", r_loss)
        r_loss.backward()
        '''
        torch.autograd.backward(
            critic_loss + losses,
            torch.ones_like(critic_loss + losses)
            # [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
        )'''
        optimizer.step()
        value_optimizer.step()
        """
       
        img, mask, target, deepsolo_ocr, answers, item, t5_dict = prefetcher.next()
        
    
    return total_1, cnt_1, total_2, cnt_2

def get_reward(loss_list):
    update_loss_list = []
    if loss_list.dim() == 0:
        update_loss_list.append(loss_list.item())
    else:
        for loss in loss_list:
            update_loss_list.append(loss.item())
    return torch.tensor(update_loss_list)

def discount_reward_update(rewards, GAMMA=0.9):
    bz = rewards.shape[0]
    length = rewards.shape[1]

    Qvals = torch.zeros(rewards.size())
    for b in range(len(rewards)):
        Qvals[b][length - 1] = rewards[b][length - 1]
        for l in reversed(range(1, length)):
            Qvals[b][l-1]  = rewards[b][l-1] + GAMMA * Qvals[b][l]
    return Qvals




class Linear_to_100(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(251, 100)

    def forward(self, x):
        out = self.Linear(x)
        return out


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast = True)
    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target, deepsolo_ocr, answers, item, t5_dict = prefetcher.next()
    iteration = 0
    cnt_1 = 0
    total_1 = 0
    cnt_2 = 0
    total_2 = 0
    losses_list = []
    new_imdb = []
    loss = 0
    acc = 0
    predict_results = []
    while img is not None:
        single_result = {}
        single_result['question_id'] = item[0]['question_id']

        print("val iter:", iteration)
        iteration = iteration + 1


        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        ocr_id, ocr_mask, cnt_mask, ocr_label = target_dict['ocr_id'], target_dict['ocr_mask'], target_dict['cnt_mask'], target_dict['ocr_label']
        ocr_str_id, ocr_str_mask = target_dict['ocr_str_id'], target_dict['ocr_str_mask']
        binary_label = target_dict['binary_label']

        

        binary_output, verify_output = model(img, mask, word_id, word_mask, ocr_id, ocr_mask, ocr_label, ocr_str_id, ocr_str_mask, binary_label)
        #verify_output = model(img, mask, word_id, word_mask, ocr_id, ocr_mask)
        #print("answers:")
        #print(answers)
        
        #for i in range(verify_output.size(0)):  # 行
            #for j in range(verify_output.size(1)):  # 列
                #print(verify_output[i,j])
        
        
        '''
        for output, binary, label, mask_1, ocr, answer, imdb, i in zip(verify_output, binary_output, ocr_label, cnt_mask, deepsolo_ocr, answers, item, range(8)):
            total_2 += 1
            max_attn = output.max()
            max_index = (output==max_attn).nonzero()
            word = ocr[max_index.item()][0]
            if word in answer:
                cnt_2 += 1

        losses = criterion(verify_output, ocr_label)
        losses_list.append(losses)
        print(total_2, cnt_2)
        '''

        
        add_list = []
        # print(outputs)
        #for output, mask_1, ocr, answer, imdb, i in zip(verify_output, cnt_mask, deepsolo_ocr, answers, item, range(6)):
        for output, binary, label, mask_1, ocr, answer, imdb, i in zip(verify_output, binary_output, ocr_label, cnt_mask, deepsolo_ocr, answers, item, range(8)):
            #print("check一下")
            #print(mask_1)
            
            cnt_index = (mask_1==1).nonzero()
            output = output[:len(cnt_index)]
            
            if len(output):

                index = output.argmax()
                #print("check")
                #print(index.item())

                #print("answer:")
                #print(answer)
                if binary[0].item() > 0.5:
                #if True:
                    total_2 += 1
                    word = ocr[index.item()][0]
                    #print("word: ",word)
                    #print("answer: ",answer)
                    answer_lower = [ans.lower() for ans in answer]
                    add_list.append([i, index.item(), word])
                    if word.lower() in answer_lower:
                        cnt_2 += 1

                
                
        single_result['add_list'] = add_list
        # add word and box 
        print("add_list:")
        print(add_list)
        
        for info in add_list:
            t5_dict['origin_words'][info[0]].append(t5_dict['origin_words'][info[0]][info[1]])
            t5_dict['origin_boxes'][info[0]].append(t5_dict['origin_boxes'][info[0]][info[1]])

        # print(t5_dict['input_ids'])
        bbox_list = []
        input_ids_list = []
        attention_mask_list = []
        for words, boxes, question_id, question_mask, box_pretext in zip(t5_dict['origin_words'], t5_dict['origin_boxes'], t5_dict['question_id'], t5_dict['question_mask'], t5_dict['box_pretext']):
            if words:
                boxes, words = order_ocr_tokens(boxes, words)
            boxes, tokenized_words, attention_mask = create_features(tokenizer=t5_tokenizer, use_ocr=False, 
                                                                     words=words, bounding_box=boxes,
                                                                    target_size=(1000, 1000))
            
            boxes = box_pretext + boxes  
            tokenized_words = question_id + tokenized_words
            attention_mask = question_mask + attention_mask

            boxes = torch.as_tensor(boxes, dtype=torch.int32)

            ## Clamping the values of boxes, since there are some entries, which makes width | height negative
            width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1)
            height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1)
            boxes = torch.cat([boxes, width, height], axis=-1)
            boxes = torch.clamp(boxes, min=0, max=1000)
            boxes = boxes.numpy().tolist()

            
            bbox_list.append(torch.tensor(boxes))
            input_ids_list.append(torch.tensor(tokenized_words))
            attention_mask_list.append(torch.tensor(attention_mask))

        bbox_list = pad_sequence(
                bbox_list, torch.as_tensor([0, 0, 0, 0, 0, 0]))
        bbox_list = torch.stack([bbox for bbox in bbox_list])
        t5_dict['bbox'] = bbox_list

        input_ids_list = pad_sequence(
                input_ids_list, torch.as_tensor(0))
        input_ids_list = torch.stack([input_ids for input_ids in input_ids_list])
        t5_dict['input_ids'] = input_ids_list

        attention_mask_list = pad_sequence(
                attention_mask_list, torch.as_tensor(0))
        attention_mask_list = torch.stack([attention_mask for attention_mask in attention_mask_list])
        t5_dict['attention_mask'] = attention_mask_list
        # print(t5_dict['input_ids'])
        
        # t5_dict finished
        for key in t5_dict:
            if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key != 'origin_words' and key != 'origin_boxes' and key != 'question_id' and key != 'question_mask' and key != 'box_pretext':
                # print(key)
                t5_dict[key] = t5_dict[key].to(device)

        outputs, generate_ids = latr_model(input_ids = t5_dict['input_ids'],
                            labels = t5_dict['labels'], 
                            bbox = t5_dict['bbox'],
                            attention_mask = t5_dict['attention_mask'],
                            img = t5_dict['img'],
                            is_generate = True)
        predict_answer = t5_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        single_result['predict_answer'] = predict_answer
        #print(single_result)
        
        """
        for word in t5_dict['answers'][0]:
            if word.lower() == predict_answer.lower():
                acc += 1
                print(acc)
                # print("question:", sample['question'])
                # print("ocr_tokens:", sample['ocr_tokens'])
                print(tokenizer.batch_decode(t5_dict['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                print("answers:", t5_dict['answers'])
                print("answers0:", t5_dict['answers'][0])
                print("predict:", predict_answer)
                break
        """
        
        """
        print("answers:", t5_dict['answers'])
        for answers in t5_dict['answers']:
            flag = 0
            for word in answers:
                if word.lower() == predict_answer.lower():
                    acc += 1
                    print(acc)
                    # print("question:", sample['question'])
                    # print("ocr_tokens:", sample['ocr_tokens'])
                    print(tokenizer.batch_decode(t5_dict['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                    
                    print("answersi:", answers)
                    print("!!!! predict:", predict_answer)
                    flag=1
                    break
            if flag == 1:
                break
        """
        logging.info("question_id:{}, add_list:{}".format(single_result['question_id'], single_result['add_list']))
        for pred, ans_list, i in zip(predict_answer, t5_dict['answers'],range(8)):
            ans_list_lower = [ans.lower() for ans in ans_list]
            if pred.lower() in ans_list_lower:
                acc += 1
                print("acc: ",acc)
                print("pred: ", pred)
                logging.info("i:{}, predict_answer:{}".format(i, pred))
        
        

        
        loss = outputs['loss'][0]
        losses_list.append(loss)  
        predict_results.append(single_result) 
        img, mask, target, deepsolo_ocr, answers, item, t5_dict = prefetcher.next()
    #save_path = '/home/yanruxue/latr-main/src/new_latr/predictions/ours_0.5' + '.npy'
    #save_path = '/home/yanruxue/latr-main/src/new_latr/predictions/std/ours_0.5' + '.npy'
    save_path = '/home/yanruxue/lsf/predictions/binary_imdb/ours_0.5' + '.npy'
    np.save(save_path, predict_results)
    loss = sum(losses_list)/ len(losses_list)
    
    return loss, total_2, cnt_2, acc
        
        
    

    
    
    '''
    val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/deepsolo_textvqa_train.npy", allow_pickle=True)
    total = len(val_imdb)

    new_imdb_dict = {}
    new_imdb = []
    for i in range(total):
        item = val_imdb[i]
        if item['question_id'] in index_list:
            new_imdb.append(item)
    print(len(new_imdb))
    '''
    # np.save("deepsolo_13413_train.npy", new_imdb)
    

    
    
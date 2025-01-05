from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from transformers import T5ForConditionalGeneration, ViTModel
import os
import json
import pandas as pd
from dataset import TextVQA
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import collate, draw_bounding_box_on_pil_image
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.set_device(3)
import numpy as np
import logging
from tqdm import tqdm

# 配置日志
log_format = '%(levelname)s %(asctime)s %(message)s'
logging.basicConfig(filename='/home/yanruxue/lsf/logs/origin_deepsolo_case.log', level=logging.INFO, format=log_format)
logging.info("sss")

PAD_TOKEN_BOX = [0, 0, 0, 0] # 用作填充的占位符
QUESTION_BOX = [0, 0, 0, 0] # 用作问题的占位符
EOS_BOX = [0, 0, 0, 0] # 用作结束的占位符

batch_size = 4
target_size = (224,224)
t5_model = "t5-base" # 采用T5-base模型，相当于Encoder和Decoder都用BERT-base

model_name = 't5-base'
model_config = AutoConfig.from_pretrained(model_name) # 使用Hugging Face的AutoConfig类，根据模型名称加载配置信息

max_2d_position_embeddings = 1024 # 最大的二维位置嵌入向量长度为1024
vit_model = "google/vit-base-patch16-224-in21k" # 定义了一个Vision Transformer模型。谷歌提供的ViT基础版本，输入尺寸为224x224，使用了21k个标签的预训练模型。
model_config.update({"max_2d_position_embeddings" : max_2d_position_embeddings,
                    "vit_model" : vit_model})  # 更新模型配置，将上面两者加入模型配置中

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True) # 根据模型名称加载分词器，使用快速分词器
processor = AutoProcessor.from_pretrained(vit_model) # 根据给定的Vit模型加载处理器，用于准备输入数据

# base_path = '/home/yanruxue/latr-main/src/new_latr/'
base_path = '/home/yanruxue/lsf/TextVQA'
ocr_json_path = os.path.join(base_path, 'TextVQA_Rosetta_OCR_v0.2_train.json')
train_json_path = os.path.join(base_path, 'TextVQA_0.5.1_train.json')

val_ocr_json_path = os.path.join(base_path, 'TextVQA_Rosetta_OCR_v0.2_val.json')
val_json_path = os.path.join(base_path, 'TextVQA_0.5.1_val.json')

with open(ocr_json_path) as f:
    train_ocr_json = json.load(f)['data']
with open(train_json_path) as f:
    train_json = json.load(f)['data']
    
## Validation
with open(val_ocr_json_path) as f:
    val_ocr_json = json.load(f)['data']
with open(val_json_path) as f:
    val_json = json.load(f)['data']


train_json_df = pd.DataFrame(train_json)
train_ocr_json_df = pd.DataFrame(train_ocr_json)


val_json_df = pd.DataFrame(val_json)
val_ocr_json_df = pd.DataFrame(val_ocr_json)


# print(val_json_df.keys())
train_json_df.drop(columns = ['flickr_original_url', 'flickr_300k_url','image_classes', 'question_tokens',# 'path_exists'
                              ], axis = 1, inplace = True)

val_json_df.drop(columns = ['flickr_original_url', 'flickr_300k_url','image_classes', 'question_tokens',# 'path_exists'
                              ], axis = 1, inplace = True)

base_img_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/textocr/test_images"
#base_img_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/totaltext1/test_images/"
max_seq_len = -1

# print(val_json_df.keys())


# 自定义数据集，TextVQA和STVQA
train_ds = TextVQA(base_img_path = base_img_path,
                   json_df = train_json_df,
                   ocr_json_df = train_ocr_json_df,
                   tokenizer = tokenizer,
                   transform = processor, 
                   max_seq_length = max_seq_len, 
                   )

val_ds = TextVQA(base_img_path = base_img_path,
                   json_df = val_json_df,
                   ocr_json_df = val_ocr_json_df,
                   tokenizer = tokenizer,
                   transform = processor, 
                   max_seq_length = max_seq_len, 
                   )

stvqa_train_ds = TextVQA(base_img_path = base_img_path,
                   json_df = train_json_df,
                   ocr_json_df = train_ocr_json_df,
                   tokenizer = tokenizer,
                   transform = processor, 
                   max_seq_length = max_seq_len,
                   STVQA = True,
                   train_ds=True 
                   )

stvqa_val_ds = TextVQA(base_img_path = base_img_path,
                   json_df = val_json_df,
                   ocr_json_df = val_ocr_json_df,
                   tokenizer = tokenizer,
                   transform = processor, 
                   max_seq_length = max_seq_len, 
                   STVQA=True,
                   val_ds=True
                   )

# 加载数据
class DataModule(pl.LightningDataModule):

  def __init__(self, train_dataset, val_dataset,  batch_size = 128):

    super(DataModule, self).__init__()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.batch_size = batch_size

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                      collate_fn = collate, shuffle = True)
  
  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = 1,
                      collate_fn = collate, shuffle = False)

#print(len(stvqa_train_ds)) 
#print(len(stvqa_val_ds))

dl = DataModule(stvqa_train_ds, stvqa_val_ds, 4) #跑STVQA
#dl = DataModule(train_ds, val_ds, 8) #跑TextVQA
sample = next(iter(dl.train_dataloader())) # 从训练数据加载器中获取一个批次的数据
print(sample['bbox'])
for key in sample:
    if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key != 'imdb':
    #if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key != 'question_id':
        print(f"Key : {key}, has shape : {sample[key].shape}")
        sample[key] = sample[key].to(device) # 将当前键所对应的值转移到设备上，即将数据转移到GPU上加速训练

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import torch.nn as nn
from torch.nn import CrossEntropyLoss

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
        lang_feat = self.t5model.shared(input_ids) # 使用T5模型的shared层对输入进行编码，得到语言特征lang_feat，shared 层是T5模型中的共享编码器层。
        spatial_feat = self.spatial_feat_extractor(bbox) # 调用SpatialModule的前向传播方法，得到传入的bbox的空间特征
        layout_feat = lang_feat + spatial_feat # 布局特征=语言特征+空间特征
        
        img_feat = self.img_feat_extractor(img).last_hidden_state # 使用ViT提取图像信息，last_hidden_state是最后一个隐藏状态的张量
        inputs_embeds = torch.cat([img_feat, layout_feat], axis = 1)
        # print(attention_mask.size())
        attention_mask = torch.cat([torch.ones(img_feat.shape[:2]).to(img_feat.device), attention_mask], axis = 1)
        # print(lang_feat.size())
        # print(spatial_feat.size())
        # print(img_feat.size())
        # print(attention_mask.size())
        # print(attention_mask.size())
        
        if is_generate == True:
            return self.t5model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels), self.t5model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, num_beams = 2) # 返回loss和输出的文本序列
        else:
            return self.t5model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = labels) # 只调用t5的forward方法，计算模型损失
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


# latr_model = LaTrForConditionalGeneration(model_config)

tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_config = AutoConfig.from_pretrained("t5-base")
latr_model = mymodel(t5_config)

latr_model = latr_model.to(device)
# 使用load_state_dict加载预训练模型的参数到latr_model里
#latr_model.load_state_dict(torch.load('/home/yanruxue/latr-main/src/new_latr/checkpoints/select_singleloss/model_12000.pth', map_location='cpu'))
latr_model.load_state_dict(torch.load('/home/yanruxue/lsf/checkpoints/textvqa_4/model_25000.pth', map_location='cpu'))
T5_Model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)


def evaluate(iter):
    results = []
    acc = 0
    val_loss_list = []
    with torch.no_grad(): #确保评估阶段不进行梯度计算
        for sample in tqdm(dl.val_dataloader()): #使用tqdm可以在循环中显示进度条
            single_result = {}
            question_id = sample['imdb'][0]['question_id'] #跑stvqa
            #print("check一下")
            #for key in sample.keys():
                #print(f"Keys: {key}")
            #question_id = sample['question_id']
            
            for key in sample:
                if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key != 'imdb':
                #if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key!='question_id':
                    sample[key] = sample[key].to(device)
            outputs, generate_ids = latr_model(input_ids = sample['input_ids'], 
                                                attention_mask = sample['attention_mask'], 
                                                labels = sample['labels'],
                                                bbox = sample['bbox'],
                                                img = sample['img']
                                                )
            # print(".... loss:", outputs['loss'][0].item())
            loss = outputs['loss'][0].item()
            val_loss_list.append(loss)      
            # outputs = T5_Model.generate(input_ids = sample['input_ids'], num_beams = 2)
            # print(outputs)
            predict_answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] #解码成文本
            single_result['question_id'] = question_id
            single_result['predict_answer'] = predict_answer
            results.append(single_result)
            logging.info("question_id:{}, predict_answer:{}".format(question_id, predict_answer))
            for word in sample['answers'][0]:
                if word.lower() == predict_answer.lower(): #迭代真实答案列表，并比较是否和预测答案匹配
                    acc += 1
                    print(acc)
                    # print("question:", sample['question'])
                    # print("ocr_tokens:", sample['ocr_tokens'])
                    print(tokenizer.batch_decode(sample['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                    print("answers:", sample['answers'])
                    print("answers0:",sample['answers'][0])
                    print("predict:", predict_answer)
                    logging.info("question_id:{}, yes".format(question_id))
                    break
        folder_path = "/home/yanruxue/lsf/predictions/origin_deepsolo_textvqa/"
        results_path = folder_path + "origin_" + str(iter) + ".npy"
        np.save(results_path, results)
        val_loss = sum(val_loss_list)/len(val_loss_list)
        logging.info("iteration:{}, val_loss:{}, val_acc:{}".format(iter, val_loss, acc))

optimizer = torch.optim.AdamW(latr_model.parameters(), lr = 1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #学习率调度器，每10步学习率乘0.1          
iter = 0
loss_list = []
for epoch in range(10):
    print("epoch: ", epoch)
    for sample in (tqdm)(dl.train_dataloader()):
        # iter += 1
        
        if iter % 1000 == 0:
            #save_path = '/home/yanruxue/latr-main/src/new_latr/checkpoints/stvqa/model_' + str(iter) + '.pth'
            save_path = '/home/yanruxue/lsf/checkpoints/textvqa_4/model_' + str(iter) + '.pth'
            
            if True:
                torch.save(latr_model.state_dict(), save_path) 
            
            evaluate(iter)
            # lr_scheduler.step()
        iter += 1
        
        for key in sample:
            if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key != 'imdb':
            #if key != 'question' and key != 'answers' and key != 'ocr_tokens' and key!='question_id':
                sample[key] = sample[key].to(device)
        
        outputs = latr_model(input_ids = sample['input_ids'],
                            labels = sample['labels'], 
                            bbox = sample['bbox'],
                            attention_mask = sample['attention_mask'],
                            img = sample['img'],
                            is_generate = False)
        
        loss = outputs['loss'][0]
        
        print("iter:", iter, "batch_loss:", loss, "current_lr:", lr_scheduler.get_lr()) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        





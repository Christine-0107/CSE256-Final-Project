import argparse
import datetime
import json
import random
import time
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import misc as utils
from misc import collate_fn_with_mask as collate_fn
from engine import train_one_epoch,evaluate
from models import build_model

from datasets1 import build_dataset, train_transforms, test_transforms

from logger import get_logger
from config import Config

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.set_device(2)




import logging

from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from transformers import T5ForConditionalGeneration, ViTModel
import os
import json
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('Transformer-based visual grounding', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_vis_enc', default=1e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--checkpoint_step', default=1, type=int)
    parser.add_argument('--checkpoint_latest', action='store_true')
    parser.add_argument('--checkpoint_best', action='store_true')

    # Model parameters
    parser.add_argument('--load_weights_path', type=str, default=None,
                        help="Path to the pretrained model.")
    parser.add_argument('--freeze_modules', type=list, default=[])
    parser.add_argument('--freeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_epochs', type=int, default=1)
    parser.add_argument('--freeze_losses', type=list, default=[])

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Bert
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='Bert model')
    parser.add_argument('--bert_token_mode', default='bert-base-uncased', type=str, help='Bert tokenizer mode')
    parser.add_argument('--bert_output_dim', default=768, type=int,
                        help='Size of the output of Bert')
    parser.add_argument('--bert_output_layers', default=4, type=int,
                        help='the output layers of Bert')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='The maximum total input sequence length after WordPiece tokenization.')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_loc', default='loss_boxes', type=str,
                        help="The loss function for the predicted boxes")
    parser.add_argument('--box_xyxy', action='store_true',
                        help='Use xyxy format to encode bounding boxes')

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--split_root', default='./split/data/')
    parser.add_argument('--dataset', default='gref')
    parser.add_argument('--test_split', default='val')
    parser.add_argument('--img_size', default=640)
    parser.add_argument('--cache_images', action='store_true')
    parser.add_argument('--output_dir', default='work_dirs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_pred_path', default='output/textVQA/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default='VLTVG_R50_gref.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--collate_fn', default='collate_fn')
    parser.add_argument('--batch_size_val', default=8, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    parser.add_argument('--train_transforms', default=train_transforms)
    parser.add_argument('--test_transforms', default=test_transforms)
    parser.add_argument('--enable_batch_accum', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # configure file
    parser.add_argument('--config', type=str, help='Path to the configure file.')
    parser.add_argument('--model_config')
    return parser


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class ParamFreezer(object):
    def __init__(self, module_names, param_names=[]):
        self.module_names = module_names
        self.freeze_params = dict()
        self.global_param_names = param_names

    def freeze(self, model):
        for name in self.module_names:
            module = getattr(model, name)
            self.freeze_params[name] = list()
            for k, v in module.named_parameters():
                if v.requires_grad:
                    v.requires_grad_(False)
                    self.freeze_params[name].append(k)

        if len(self.global_param_names) == 0:
            return
        for k, v in model.named_parameters():
            if k in self.global_param_names and v.requires_grad:
                v.requires_grad_(False)

    def unfreeze(self, model):
        for name in self.module_names:
            module = getattr(model, name)
            keys = self.freeze_params[name]
            for k, v in module.named_parameters():
                if k in keys:
                    v.requires_grad_(True)

        if len(self.global_param_names) == 0:
            return
        for k, v in model.named_parameters():
            if k in self.global_param_names:
                v.requires_grad_(True)


def main(args):
    utils.init_distributed_mode(args)

    log_format = '%(levelname)s %(asctime)s %(message)s'
    #logging.basicConfig(filename='/home/yanruxue/latr-main/src/VLTVG/logs/rl_0.7.log', level=logging.INFO, format=log_format)
    logging.basicConfig(filename='/home/yanruxue/lsf/logs/pre_select_case.log', level=logging.INFO, format=log_format)
    logging.info("sss")

    device = torch.device(args.device)
    
    model, criterion, postprocessor = build_model(args)
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()

   
    model.to(device)
    criterion = criterion.to(device)

    model_without_ddp = model
    model_without_ddp.load_state_dict(torch.load('/home/yanruxue/latr-main/src/VLTVG/checkpoints/select/epoch_10.pth'), strict=False)

    
    backbone_param = [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]
    vis_enc_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                     (n.startswith('trans_encoder') or n.startswith('input_proj'))]
    bert_param = [p for p in model_without_ddp.bert.parameters() if p.requires_grad]
    freeze_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                     (n.startswith('bert_proj') or n.startswith('Binary_model'))]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                     n.startswith('Verify_model')]

    cnt_backbone = sum([p.numel() for p in backbone_param])
    cnt_vis_enc = sum([p.numel() for p in vis_enc_param])
    cnt_bert = sum([p.numel() for p in bert_param])
    cnt_freeze = sum([p.numel() for p in freeze_param])
    cnt_rest = sum([p.numel() for p in rest_param])
    cnt_whole = sum([p.numel() for p in model_without_ddp.parameters() if p.requires_grad])

    logging.info(f'The num of learnable parameters: backbone({cnt_backbone}), vis_enc({cnt_vis_enc}), '
                f'bert({cnt_bert}), freeze({cnt_freeze}, rest({cnt_rest})')
    logging.info(f'Check the whole parameters: {cnt_whole} = {cnt_backbone + cnt_vis_enc + cnt_bert + cnt_freeze + cnt_rest}')

    param_dicts = [{'params': rest_param}, # base_lr
                   {'params': freeze_param}, 
                   {'params': backbone_param, 'lr': args.lr_backbone}, # base_lr/10.
                   {'params': vis_enc_param, 'lr': args.lr_vis_enc},
                   {'params': bert_param, 'lr': args.lr_bert},] # base_lr/10.

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay, eps=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_path = '/home/yanruxue/latr-main/src/new_latr/'
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
    

    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast = True)
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    base_img_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/textocr/test_images"
    max_seq_len = -1

    dataset_train = build_dataset(test=False, 
                                args=args,
                                base_img_path = base_img_path,
                                json_df = train_json_df,
                                ocr_json_df = train_ocr_json_df,
                                tokenizer = tokenizer,
                                transform = processor, 
                                max_seq_length = max_seq_len,
                                STVQA = True,
                                )
    
    dataset_val = build_dataset(test=True, 
                                args=args,
                                base_img_path = base_img_path,
                                json_df = val_json_df,
                                ocr_json_df = val_ocr_json_df,
                                tokenizer = tokenizer,
                                transform = processor, 
                                max_seq_length = max_seq_len, 
                                STVQA=True,
                                )

    logging.info(f'The size of dataset: train({len(dataset_train)}), test({len(dataset_val)})')
    # print(dataset_val[3])
    # print(dataset_train[3])
    
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=args.pin_memory, collate_fn=collate_fn,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size_val, sampler=sampler_val,
                                 pin_memory=args.pin_memory, drop_last=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)

    epoch_trainer = train_one_epoch
    epoch_eval = evaluate

    output_dir = Path(args.output_dir)
    

    if args.start_epoch < args.freeze_epochs and args.freeze_modules:
        logging.info(f'Freeze weights: {args.freeze_modules} and {args.freeze_param_names}')
        param_freezer = ParamFreezer(args.freeze_modules, args.freeze_param_names)
        param_freezer.freeze(model_without_ddp)        


    logging.info("Start training")
    start_time = time.time()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        '''
        loss, total_2, cnt_2, acc = epoch_eval(
            model, criterion, postprocessor, data_loader_val, device, args.save_pred_path, epoch
        )
        logging.info("epoch:{}, loss:{}, total_2:{}, cnt_2:{}, acc:{}".format(epoch, loss, total_2, cnt_2, acc))
        '''

        total_1, cnt_1, total_2, cnt_2 = epoch_trainer(
            model, criterion, data_loader_train, data_loader_val, optimizer, device, epoch, args.epochs, args.clip_max_norm
        )
        logging.info("train_epoch:{}, total_1:{}, cnt_1:{}, total_2:{}, cnt_2:{}".format(epoch, total_1, cnt_1, total_2, cnt_2))
        if (epoch + 1) == args.freeze_epochs and args.freeze_modules:
            logging.info(f'Unfreeze weights: {args.freeze_modules}')
            param_freezer.unfreeze(model_without_ddp)
            if args.distributed: # re-wrap the model to ensure the same gradients for unfrozen weights
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu])
                model_without_ddp = model.module

        lr_scheduler.step()
        
        '''
        loss, total_2, cnt_2, acc = epoch_eval(
            model, criterion, postprocessor, data_loader_val, device, args.save_pred_path, epoch
        )
        logging.info("epoch:{}, loss:{}, total_2:{}, cnt_2:{}, acc:{}".format(epoch, loss, total_2, cnt_2, acc))
        
        save_path = "/home/yanruxue/latr-main/src/VLTVG/checkpoints/rl_0.7/epoch_" + str(epoch) + '.pth'
        torch.save(model_without_ddp.state_dict(), save_path)
        '''






if __name__ == '__main__':
    parser = argparse.ArgumentParser('VLTVG test script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.config:
        cfg = Config(args.config)
        cfg.merge_to_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

import torch
import torch.nn.functional as F
from torch import nn

#from util import box_ops
import box_ops
from misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone

from .transformer import build_visual_encoder
from .decoder import build_vg_decoder
from pytorch_pretrained_bert.modeling import BertModel
from .cross_attn import Verify
from .bclass import Binary_classi

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.set_device(2)

def sliding_window_tokenize(input_ids, attention_mask, max_len=512, stride=256):
    
    batch_size, seq_length = input_ids.shape[0],input_ids.shape[1]
    
    chunks = []
    chunk_masks = []
    
    for i in range(0, seq_length, stride):
        end = min(i + max_len, seq_length)
        chunk = input_ids[:, i:end]
        chunk_mask = attention_mask[:, i:end]
        if chunk.size(1) < max_len:
            padding_length = max_len - chunk.size(1)
            padding = torch.zeros((batch_size, padding_length), dtype=torch.long, device=chunk.device)
            chunk = torch.cat([chunk, padding], dim=1)
            chunk_mask = torch.cat([chunk_mask, padding], dim=1)
        chunks.append(chunk)
        chunk_masks.append(chunk_mask)
    
    return chunks, chunk_masks

def combine_outputs(outputs, stride, original_length):
    batch_size, feature_dim = outputs[0].shape[0], outputs[0].shape[-1]
    combined = torch.zeros((batch_size, original_length, feature_dim), device=outputs[0].device)
    count = torch.zeros((batch_size, original_length), device=outputs[0].device)
    
    position = 0
    for output in outputs:
        length = output.shape[1]
        end_position = min(position + length, original_length)
        combined[:, position:end_position, :] += output[:, :end_position-position, :]
        #combined[:, position:position+length, :] += output
        count[:, position:position+length] += 1
        position += stride
    
    count[count == 0] = 1
    combined /= count.unsqueeze(-1)
    return combined


class VLTVG(nn.Module):
    def __init__(self, pretrained_weights, args=None):
        """ Initializes the model."""
        super().__init__()

        # Image feature encoder (CNN + Transformer encoder)
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.trans_encoder.d_model, kernel_size=1)

        # Text feature encoder (BERT)
        print("args:", args)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)  # 768 256
        self.bert_output_layers = args.bert_output_layers  # 4
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        # verify
        Verify_model = Verify()
        self.Verify_model = Verify_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # binary
        Binary_model = Binary_classi(512, 10, 1)
        self.Binary_model = Binary_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # if pretrained_weights:
        #     self.load_pretrained_weights(pretrained_weights)

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = dict()
            for k in module_keys:
                prefix_k = prefix+'.'+k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)


    


    def forward(self, image, image_mask, word_id, word_mask, ocr_id, ocr_mask, ocr_label, ocr_str_id, ocr_str_mask, binary_label):

        N = image.size(0)

        # Image features
        features, pos = self.backbone(NestedTensor(image, image_mask))
        src, mask = features[-1].decompose()
        assert mask is not None
        img_feat, mask, pos_embed = self.trans_encoder(self.input_proj(src), mask, pos[-1])
        print("=====img========")
        print(img_feat.size(), mask.size(), pos_embed.size())

        
        # Text features
        # print(word_mask)
        word_feat, _ = self.bert(word_id, token_type_ids=None, attention_mask=word_mask)
        # print(word_feat.size())
        word_feat = torch.stack(word_feat[-self.bert_output_layers:], 1).mean(1)
        word_feat = self.bert_proj(word_feat)
        print("word_feat:",word_feat.size())
        word_feat = word_feat.permute(1, 0, 2) # NxLxC -> LxNxC
        word_mask = ~word_mask
        print("word:", word_feat.size(), word_mask.size())


        # ocr features
        
        
        ocr_id = ocr_id.flatten(1)
        ocr_mask = ocr_mask.flatten(1)

        
        chunks, chunk_masks = sliding_window_tokenize(ocr_id, ocr_mask)
        outputs_ocr = []
        for chunk, o_mask in zip(chunks, chunk_masks):
            output, _ = self.bert(chunk, token_type_ids=None, attention_mask=o_mask)
            output = torch.stack(output[-self.bert_output_layers:], 1).mean(1)
            output = self.bert_proj(output)
            outputs_ocr.append(output)
        ocr_feat = combine_outputs(outputs_ocr, 256, 1500)
        print("ocr_feat:",ocr_feat.size())
        """
        ocr_id_1 = ocr_id[:, :500]
        ocr_mask_1 = ocr_mask[:, :500]
        ocr_id_2 = ocr_id[:, 500:1000]
        ocr_mask_2 = ocr_mask[:, 500:1000]
        ocr_id_3 = ocr_id[:, 1000:]
        ocr_mask_3 = ocr_id[:, 1000:]

        ocr_feat_1, _ = self.bert(ocr_id_1, token_type_ids=None, attention_mask=ocr_mask_1)
        ocr_feat_2, _ = self.bert(ocr_id_2, token_type_ids=None, attention_mask=ocr_mask_2)
        ocr_feat_3, _ = self.bert(ocr_id_3, token_type_ids=None, attention_mask=ocr_mask_3)
        #print("ocr_feat_1: ",len(ocr_feat_1))
        
        ocr_feat_1 = torch.stack(ocr_feat_1[-self.bert_output_layers:], 1).mean(1)
        ocr_feat_2 = torch.stack(ocr_feat_2[-self.bert_output_layers:], 1).mean(1)
        ocr_feat_3 = torch.stack(ocr_feat_3[-self.bert_output_layers:], 1).mean(1)
        #print("ocr_feat_1: ",ocr_feat_1.size())
        ocr_feat = torch.cat((ocr_feat_1, ocr_feat_2, ocr_feat_3), dim=1)
        #print("ocr_feat: ",ocr_feat.size())
        
        
        #ocr_feat, _ = self.bert(ocr_id, token_type_ids=None, attention_mask=ocr_mask)
        #ocr_feat = torch.stack(ocr_feat[-self.bert_output_layers:], 1).mean(1)
        ocr_feat = self.bert_proj(ocr_feat)
        """

        ocr_feat = ocr_feat.permute(1, 0, 2) # NxLxC -> LxNxC
        ocr_mask = ~ocr_mask
        # print("=========ocr=======")
        print("ocr_feat:", ocr_feat.size(), ocr_mask.size())
        

        #修改
        # ocr_str features
        
        ocr_str_feat, _ = self.bert(ocr_str_id, token_type_ids=None, attention_mask=ocr_str_mask)
        # print(word_feat.size())
        ocr_str_feat = torch.stack(ocr_str_feat[-self.bert_output_layers:], 1).mean(1)
        ocr_str_feat = self.bert_proj(ocr_str_feat)
        # ocr_str_feat = ocr_str_feat.permute(1, 0, 2) # NxLxC -> LxNxC
        ocr_str_mask = ~ocr_str_mask

        # print("======= ocr_str =====")
        # print(ocr_str_feat.size(), ocr_str_mask.size())
        



        
        scores = self.Verify_model(word_feat, word_mask, ocr_feat, ocr_mask, img_feat, mask, pos_embed)
        print("score1:")
        print(scores.size())
        
        m_inf = torch.tensor([float('-inf')], device=device)
        m_min = torch.tensor([-1e9], device=device)
        
        softmax = nn.Softmax()
        # scores = softmax(scores)
        # print(scores[0])
        scores = torch.reshape(scores, (scores.shape[0], 300, 5))
        
        scores = torch.sum(scores, -1)
        scores = torch.where(scores == 0, m_inf, scores)

        #scores = scores - torch.max(scores, dim=-1, keepdim=True).values
        
        scores = softmax(scores)

        #print("score2:")
        #print(scores)
        
        

        binary_results = self.Binary_model(ocr_str_feat, word_feat, word_mask, img_feat, mask, ocr_feat, ocr_mask)
        print("binary_results:")
        print(binary_results)
        return binary_results, scores
        





class VGCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes}

        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def build_vgmodel(args):
    device = torch.device(args.device)

    model = VLTVG(pretrained_weights=args.load_weights_path, args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)

    return model, criterion, postprocessor

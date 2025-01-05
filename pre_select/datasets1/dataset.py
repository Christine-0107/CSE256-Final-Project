import os
import os.path as osp
import sys
import random
import math
import numpy as np
import torch
import pickle
import PIL
from PIL import Image
import io

from torch.utils.data import Dataset

from .utils import convert_examples_to_features, read_examples
from box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .transforms import PIL_TRANSFORMS, ToTensor



# Meta Information
SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')}
}

def get_tokens_with_boxes(unnormalized_word_boxes, list_of_words, tokenizer, pad_token_box=[0, 0, 0, 0], max_seq_len=-1, eos_token_box=[0, 0, 1000, 1000]):

    '''
    A function to get the tokens with the bounding boxes
    Arguments:
        unnormalized_word_boxes: A list of bounding boxes
        list_of_words: A list of words
        tokenizer: The tokenizer to use
        pad_token_box: The padding token box
        max_seq_len: The maximum sequence length, not padded if max_seq_len is -1
        eos_token_box: The end of sequence token box
    Returns:
        A list of input_ids, bbox_according_to_tokenizer, attention_mask
    '''

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

def apply_ocr(tif_path):
    '''
    A function to apply OCR on the tif image
    Arguments:
        tif_path: The path to the tif image
    Returns:
        A dictionary containing the words and the bounding boxes
    '''
    img = Image.open(tif_path).convert("RGB")

    ocr_df = pytesseract.image_to_data(img, output_type="data.frame")
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text.apply(lambda x: str(x).strip()))
    actual_bboxes = ocr_df.apply(
        get_topleft_bottomright_coordinates, axis=1).values.tolist()

    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}

def get_topleft_bottomright_coordinates(df_row):
    '''
    A function to get the top left and bottom right coordinates of the bounding box
    Arguments:
        df_row: A row of the dataframe
    Returns:
        A list of the top left and bottom right coordinates
    '''
    left, top, width, height = df_row["left"], df_row["top"], df_row["width"], df_row["height"]
    return [left, top, left + width, top + height]


def order_ocr_tokens(boxes, words):
    order_list = [[] for i in range(100)]
    # print(words)
    # print(boxes)
    for i in range(len(boxes)):
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
    for new_box in new_boxes:
        new_words.append(words[new_box[4]])
        last_boxes.append(new_box[:-1])
    
    return last_boxes, new_words


def normalize_box(box, width, height, size=1000):   
    return [
        int(size * (box[0] / width)),
        int(size * (box[1] / height)),
        int(size * (box[2] / width)),
        int(size * (box[3] / height)),
    ]


def create_features(
    img_path,
    tokenizer,
    target_size=(1000, 1000),
    max_seq_length=-1,
    use_ocr=False,
    bounding_box=None,
    words=None,
    pad_token_box=[0, 0, 0, 0]
):
    '''
    Arguments:
        img_path: Path to the image
        tokenizer: The tokenizer used for tokenizing the words
        target_size: The size to which the image is to be resized
        max_seq_length: The maximum sequence length of the tokens
        use_ocr: Whether to use OCR or not
        bounding_box: The bounding box of the words
        words: The words in the image
        pad_token_box: The padding token for the bounding box
    Returns:
        A list of the image, the bounding box, the tokenized words and the attention mask
    '''

    img = Image.open(img_path).convert("RGB")
    width_old, height_old = img.size
    img = img.resize(target_size)
    width, height = img.size

    # Rescaling the bounding box as per the image size
    if (use_ocr == False) and (bounding_box == None or words == None):
        raise Exception(
            'Please provide the bounding box and words or pass the argument "use_ocr" = True')

    if use_ocr == True:
        entries = apply_ocr(img_path)
        bounding_box = entries["bbox"]
        words = entries["words"]
        bounding_box = list(map(lambda x: normalize_box(
            x, width_old, height_old), bounding_box))

    tokenized_words, boxes, attention_mask = get_tokens_with_boxes(bounding_box, words, tokenizer,
                                                                   pad_token_box, max_seq_len=max_seq_length)
    return img, boxes, tokenized_words, attention_mask


class VGDataset(Dataset):
    def __init__(self, data_root, split_root='data', dataset='referit', transforms=[],
                 debug=False, test=False, split='train', max_query_len=128,
                 bert_mode='bert-base-uncased', cache_images=False,
                 base_img_path = None, json_df = None, ocr_json_df = None, tokenizer = None, transform=None, max_seq_length=-1, target_size=(1000, 1000),
                 pad_token_box=[0, 0, 0, 0], qa_box=[0, 0, 0, 0], STVQA = False, train_ds = False, val_ds = False):
        super(VGDataset, self).__init__()
        self.test = test
        self.STVQA = STVQA
        if self.STVQA:
            print("init STVQA dataset...")
            if self.test:
                #val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/deepsolo_textvqa_val.npy", allow_pickle=True)
                #val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/rm_stvqa_val.npy", allow_pickle=True)
                val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/origin_deepsolo/deepsolo_origin_val.npy", allow_pickle=True) #textvqa
                #val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/deepsolo_st_val.npy", allow_pickle=True)
            else:
                #val_imdb = np.load("/home/yanruxue/latr-main/src/VLTVG/imdbs_binary/OnlyHaveLabels_train0.npy", allow_pickle=True)
                #val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/rm_stvqa_train.npy", allow_pickle=True)
                val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/origin_deepsolo/deepsolo_origin_train.npy", allow_pickle=True) #textvqa
                #val_imdb = np.load("/home/yanruxue/latr-main/src/new_latr/OnlyLabels_stvqa.npy", allow_pickle=True)
            self.val_imdb = val_imdb
            
            #self.base_img_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/totaltext1/test_images/"
            self.base_img_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/textocr/test_images"
        # else:
        # self.base_img_path = base_img_path
        self.json_df = json_df
        self.ocr_json_df = ocr_json_df
        self.t5_tokenizer = tokenizer
        self.target_size = target_size
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.pad_token_box = pad_token_box
        self.qa_box = qa_box
        self.test = test
        
        self.transforms = []
        
        self.getitem = self.getitem__PIL
        self.read_image = self.read_image_from_path_PIL
        
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        
        self.debug = debug

        self.query_len = max_query_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)
        


    def __len__(self):
        return len(self.val_imdb)

    def image_path(self, idx):  # notice: db index is the actual index of data.
        return osp.join(self.im_dir, self.img_names[idx])

    def annotation_box(self, idx):
        return self.covert_bbox[idx].copy()

    def phrase(self, idx):
        return self.phrases[idx]

    def cache(self, idx):
        self.images_cached[idx] = self.read_image_orig_func(idx)

    def read_image_from_path_PIL(self, idx):
        #image_path = '/home/yanruxue/latr-main/src/VLTVG/data/train_val_images/train_images/' + self.val_imdb[idx]['image_id'] + '.jpg'
        #image_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/totaltext1/test_images/" + self.val_imdb[idx]['image_path']
        image_path = "/home/yanruxue/latr-main/src/deepsolo/datasets/textocr/test_images/" + self.val_imdb[idx]['image_id'] + '.jpg'
        pil_image = Image.open(image_path).convert('RGB')
        return pil_image

    def read_image_from_cache(self, idx):
        image = self.images_cached[idx]
        return image

    def __getitem__(self, idx):
        return self.getitem(idx)


    def getitem__PIL(self, idx):
        # t5 item
        sample_entry = self.val_imdb[idx]
        sample_ocr_entry = sample_entry['deepsolo_ocr']

        width, height = sample_entry['image_width'], sample_entry['image_height']

        boxes = []
        words = []
        
        # Getting the ocr and the corresponding bounding boxes
        for entry in sample_ocr_entry:
            xmin, ymin, w, h = entry[1][0], entry[1][1], entry[1][2], entry[1][3]
            xmin = xmin / width
            ymin = ymin / height
            w = w / width
            h = h / width
            xmin, ymin, w, h = normalize_box([xmin, ymin, w, h], 1, 1, size=1000)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            w = max(0, w)
            h = max(0, h)

            xmax = xmin + w
            ymax = ymin + h

            # Bounding boxes are normalized
            curr_bbox = [xmin, ymin, xmax, ymax]

            boxes.append(curr_bbox)
            words.append(entry[0])
        img_path = os.path.join(self.base_img_path, sample_entry['image_id']) + '.jpg'
        #img_path = os.path.join(self.base_img_path, sample_entry['image_path'])

        assert os.path.exists(img_path) == True, f'Make sure that the image exists at {img_path}!!'


        origin_words = words
        origin_boxes = boxes
        # print("111111111111111")
        # print(boxes)
        # print(origin_boxes)
        
        t5_dict = {'origin_boxes': origin_boxes}
        # print(t5_dict)
        if words:
            # print(words)
            boxes, words = order_ocr_tokens(boxes, words = words)
        
        img, boxes, tokenized_words, attention_mask = create_features(img_path=img_path,
                                                                 tokenizer=self.t5_tokenizer, use_ocr=False, words=words, bounding_box=boxes,
                                                                      target_size=self.target_size)

        
        
        if self.transform is not None:
            try:
                img = self.transform(img, return_tensors='pt')['pixel_values'][0]
            except:
                img = self.transform(img)
        else:
            img = ToTensor()(img)

        # Getting the Question
        question = sample_entry['question']
        question_pretext = self.t5_tokenizer(
            "question: {:s}  context: ".format(question), add_special_tokens=False)
        question_id = question_pretext.input_ids
        question_attn_mask = question_pretext.attention_mask
        length_pretext = len(question_id)
        box_pretext = [self.qa_box] * length_pretext

        # Combining all the stuffs
        boxes = box_pretext + boxes
        
        tokenized_words = question_id + tokenized_words
        attention_mask = question_attn_mask + attention_mask

        # Converting the boxes as per the format required for model input
        boxes = torch.as_tensor(boxes, dtype=torch.int32)

        ## Clamping the values of boxes, since there are some entries, which makes width | height negative
        width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1)
        height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1)
        boxes = torch.cat([boxes, width, height], axis=-1)
        boxes = torch.clamp(boxes, min=0, max=1000)
        boxes = boxes.numpy().tolist()

        answer = self.t5_tokenizer(random.choice(sample_entry['valid_answers']))['input_ids']
        
        
        t5_dict['pixel_values'] = img 
        t5_dict['bbox'] = torch.tensor(boxes) 
        t5_dict['input_ids'] = torch.tensor(tokenized_words)
        t5_dict['labels'] = torch.tensor(answer)
        t5_dict["attention_mask"] = torch.tensor(attention_mask)
        t5_dict['id'] = torch.as_tensor([idx])
        t5_dict['question'] = sample_entry['question']
        t5_dict['ocr_tokens'] = words
        t5_dict['answers'] =sample_entry['valid_answers']
        t5_dict['origin_words'] = origin_words
        t5_dict['origin_boxes'] = origin_boxes
        t5_dict['question_id'] = question_id
        t5_dict['question_mask'] = question_attn_mask
        t5_dict['box_pretext'] = box_pretext                                                        
        
        # print(t5_dict['origin_boxes'])

        # select item
        # reading images
        image = self.read_image(idx)
        orig_image = image

        # read bbox annotation
        # bbox = self.annotation_box(idx)
        # bbox = torch.tensor(bbox)
        bbox = np.array([0, 0, 0, 0], dtype=np.float32)
        bbox = torch.tensor(bbox)

        # read phrase
        phrase = self.val_imdb[idx]['question']
        phrase = phrase.lower()
        orig_phrase = phrase

        ocr_str = ""
        # read deepsolo_ocr
        ocr_label = [0 for i in range(300)]  
        mask_list_1 = [False for i in range(300)]
        word_list = ['###' for i in range(300)]    # mask for each word
        flag = 0
        flag_str = 0
        for i in range(len(self.val_imdb[idx]['deepsolo_ocr'])):
            ocr = self.val_imdb[idx]['deepsolo_ocr'][i]
            # print(i, ocr)
            word_list[i] = ocr[0]
            mask_list_1[i] = True
            if ocr[0] in self.val_imdb[idx]['valid_answers']:
                if flag == 0:
                    ocr_label[i] = 1
                    flag = 1
            if flag_str == 0:
                ocr_str += ocr[0]
                flag_str = 1
            else:
                ocr_str += (" " + ocr[0]) 
        # print(ocr_str)
        '''
        if ocr_label == [0 for i in range(101)]:
            ocr_label[len(self.val_imdb[idx]['deepsolo_ocr'])] = 1
        mask_list_1[len(self.val_imdb[idx]['deepsolo_ocr'])] = True
        '''
        # print("word_list:", word_list)
        # print(mask_list_1)
        # print(ocr_label)
        
        id_list = []
        mask_list_2 = []
        for i in range(len(word_list)):
            word = word_list[i]
            word = word.lower()
            examples = read_examples(word, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=5, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
            word_id = torch.tensor(word_id, dtype=torch.long)
            word_mask = torch.tensor(word_mask, dtype=torch.bool)
            id_list.append(word_id)
            mask_1 = mask_list_1[i]
            mask_list_2.append([(mask_1 and mask_2) for mask_2 in word_mask])
        # print(mask_list_2)
        final_mask_list = []
        for mask in mask_list_2:
            final_mask_list.append(torch.tensor(mask))
        ocr_id = torch.stack([id for id in id_list])
        ocr_mask = torch.stack([mask for mask in final_mask_list])
        ocr_label = torch.tensor(ocr_label, dtype= torch.float)
        # print(ocr_mask)
        
        target = {}

        target['ocr_id'] = ocr_id  # 100 20
        target['ocr_mask'] = ocr_mask # 100 20
        target['ocr_label'] = ocr_label # 100

        target['phrase'] = phrase
        # target['ocr'] = deepsolo_ocr
        target['bbox'] = bbox
        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        # print("=======self.transformer======")
        # print(self.transforms)
        for transform in self.transforms:
            image, target = transform(image, target)
        # print(image, target)

        # For BERT
        # print("phrase:", target['phrase'])
        # print(target['ocr'])

        examples = read_examples(target['phrase'], idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        
        target['cnt_mask'] = torch.tensor(mask_list_1, dtype = torch.long)
        deepsolo_ocr = self.val_imdb[idx]['deepsolo_ocr']
        
        answers = self.val_imdb[idx]['valid_answers']

        item = self.val_imdb[idx]
        
        
        # ocr_str
        examples = read_examples(ocr_str, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=512, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        target['ocr_str_id'] = torch.tensor(word_id, dtype=torch.long)
        target['ocr_str_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        
        # binary labe
        # print(ocr_label)
        if True in ocr_label:
            binary_label = [1]   # select
        else:
            binary_label = [0]   # not select
        # print(binary_label)
        target['binary_label'] = torch.tensor(binary_label, dtype = torch.float)
        
        
        
        if 'mask' in target:
            mask = target.pop('mask')
            # print("111111111111111111111")
            return image, mask, target, deepsolo_ocr, answers, item, t5_dict
        # print("return return return ")
        return image, target

    
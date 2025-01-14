import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
# import pytesseract
from PIL import Image
import os
import numpy as np
import random


def normalize_box(box, width, height, size=1000):
    
    return [
        int(size * (box[0] / width)),
        int(size * (box[1] / height)),
        int(size * (box[2] / width)),
        int(size * (box[3] / height)),
    ]


# Reference: https://github.com/uakarsh/TiLT-Implementation/blob/main/src/dataset.py

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


# Defining the pytorch dataset
class TextVQA(Dataset):
    def __init__(self, base_img_path, json_df, ocr_json_df, tokenizer, transform=None, max_seq_length=-1, target_size=(1000, 1000),
                 pad_token_box=[0, 0, 0, 0], qa_box=[0, 0, 0, 0], STVQA = False, train_ds = False, val_ds = False):
        '''
        Arguments:
            base_img_path: The path to the images
            json_df: The dataframe containing the questions and answers
            ocr_json_df: The dataframe containing the words and bounding boxes
            tokenizer: The tokenizer used for tokenizing the words
            transform: The transforms to be applied to the images
            max_seq_length: The maximum sequence length of the tokens
            target_size: The size to which the image is to be resized
            pad_token_box: The padding token for the bounding box
            qa_box: The bounding box for the question
        Returns:
            A dataset object
        '''
        self.STVQA = STVQA
        if self.STVQA:
            print("init STVQA dataset...")
            if train_ds:
                self.imdb_file = np.load("/home/yanruxue/latr-main/src/new_latr/deepsolo_st_train", allow_pickle=True)
            if val_ds:
                self.imdb_file = np.load("/home/yanruxue/latr-main/src/new_latr/deepsolo_st_val", allow_pickle=True)
            # self.imdb_file = self.imdb_file[1:]
            print(len(self.imdb_file))
            # self.base_img_path = "/home/yanruxue/latr-main/src/new_latr/stvqa_images"
        # else:
        # self.base_img_path = base_img_path
        self.json_df = json_df
        self.ocr_json_df = ocr_json_df
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.pad_token_box = pad_token_box
        self.qa_box = qa_box
        

    def __len__(self):
        if self.STVQA:
            return len(self.imdb_file)
        return len(self.json_df)

    def order_ocr_tokens(self, boxes, words):
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

    def __getitem__(self, index):
        if self.STVQA:
            # print("Load STVQA dataset...")
            sample_entry = self.imdb_file[index]
            
            # Amazon
            # sample_ocr_entry = sample_entry['ocr_info']
            # deepsolo
            sample_ocr_entry = sample_entry['deepsolo_ocr']

            width, height = sample_entry['image_width'], sample_entry['image_height']

            boxes = []
            words = []
        
            # Getting the ocr and the corresponding bounding boxes
            for entry in sample_ocr_entry:
                # Amazon
                '''
                xmin, ymin, w, h = entry['Geometry']['BoundingBox']['Left'], entry['Geometry']['BoundingBox']['Top'], entry['Geometry']['BoundingBox']['Width'], entry['Geometry']['BoundingBox']['Height']
                xmin, ymin, w, h = normalize_box([xmin, ymin, w, h], 1, 1, size=1000)

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                w = max(0, w)
                h = max(0, h)

                xmax = xmin + w
                ymax = ymin + h

                # Bounding boxes are normalized
                curr_bbox = [xmin, ymin, xmax, ymax]
                '''
                # deepsolo
                # print(entry)
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

                # Amazon
                #words.append(entry['DetectedText'])
                # deepsolo
                words.append(entry[0])
            '''
            s_0 = [x for x in range(len(words))]
            new_words = []
            new_boxes = []
            random.shuffle(s_0)
            for j in s_0:
                new_words.append(words[j])
                new_boxes.append(boxes[j])
            words = new_words
            boxes = new_boxes
            '''
            # Adding .jpg at end of the image, as the grouped key does not have the extension format
            # img_path = os.path.join(self.base_img_path, sample_entry['image_path'])
            img_path = os.path.join(self.base_img_path, sample_entry['image_id']) + '.jpg'

            assert os.path.exists(img_path) == True, f'Make sure that the image exists at {img_path}!!'
            # Extracting the feature
            
            origin_boxes = boxes
            origin_words = words

            if words:
                boxes, words = self.order_ocr_tokens(boxes, words)
            img, boxes, tokenized_words, attention_mask = create_features(img_path=img_path,
                                                                      tokenizer=self.tokenizer, use_ocr=False, words=words, bounding_box=boxes,
                                                                      target_size=self.target_size)

            # Tensor tokenized words
            # tokenized_words = torch.as_tensor(tokenized_words, dtype=torch.int32)

            if self.transform is not None:
                try:
                    img = self.transform(img, return_tensors='pt')['pixel_values'][0]
                except:
                    img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            # Getting the Question
            question = sample_entry['question']
            question_pretext = self.tokenizer(
                "question: {:s}  context: ".format(question), add_special_tokens=False)
            question_id = question_pretext.input_ids
            question_attn_mask = question_pretext.attention_mask
            length_pretext = len(question_id)
            box_pretext = [self.qa_box] * length_pretext

            '''
            ## For fine-tuning, the authors use boxes[:, 0] = 0, boxes[:, 1] = 0, boxes[:, 2] = 1000, boxes[:, 3] = 1000
            for i in range(len(boxes)):
                boxes[i][0] = 0
                boxes[i][1] = 0
                boxes[i][2] = 1000
                boxes[i][3] = 1000
            '''
            
            
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

            # Getting the Answer
            # print("answers:", sample_entry['answers'])
            answer = self.tokenizer(random.choice(sample_entry['valid_answers']))['input_ids']
            # answer = self.tokenizer((sample_entry['answers']))['input_ids']
            # answer = self.tokenizer(answer)['input_ids']
            # print(answer)
            '''
            return {'pixel_values': img, 'bbox': torch.tensor(boxes), 'input_ids': torch.tensor(tokenized_words), 'labels': torch.tensor(answer),
                    "attention_mask" : torch.tensor(attention_mask),'id': torch.as_tensor([index]), 'question': sample_entry['question'], 'ocr_tokens': words, 'answers':sample_entry['answers']}
                    '''
            return {'pixel_values': img, 'bbox': torch.tensor(boxes), 'input_ids': torch.tensor(tokenized_words), 'labels': torch.tensor(answer),
                    "attention_mask" : torch.tensor(attention_mask),'id': torch.as_tensor([index]), 'question': sample_entry['question'], 'ocr_tokens': words, 'answers':sample_entry['answers'], 'origin_words': origin_words, 'origin_boxes': origin_boxes}

        else:
            # print("Load TextVQA dataset...")
            sample_entry = self.json_df.iloc[index]
            sample_ocr_entry = self.ocr_json_df[self.ocr_json_df['image_id'] == sample_entry['image_id']].values.tolist()[
                0][-1]

            width, height = sample_entry['image_width'], sample_entry['image_height']

            boxes = []
            words = []
        
            # Getting the ocr and the corresponding bounding boxes
            for entry in sample_ocr_entry:
                xmin, ymin, w, h, angle = entry['bounding_box']['top_left_x'], entry['bounding_box']['top_left_y'], entry[
                    'bounding_box']['width'], entry['bounding_box']['height'], entry['bounding_box']['rotation']
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
                words.append(entry['word'])
        
            s_0 = [x for x in range(len(words))]
            new_words = []
            new_boxes = []
            random.shuffle(s_0)
            for j in s_0:
                new_words.append(words[j])
                new_boxes.append(boxes[j])
            words = new_words
            boxes = new_boxes
            # Adding .jpg at end of the image, as the grouped key does not have the extension format
            img_path = os.path.join(self.base_img_path, sample_entry['image_id']) + '.jpg'

            assert os.path.exists(img_path) == True, f'Make sure that the image exists at {img_path}!!'
            # Extracting the feature
            img, boxes, tokenized_words, attention_mask = create_features(img_path=img_path,
                                                                      tokenizer=self.tokenizer, use_ocr=False, words=words, bounding_box=boxes,
                                                                      target_size=self.target_size)

            # Tensor tokenized words
            # tokenized_words = torch.as_tensor(tokenized_words, dtype=torch.int32)

            if self.transform is not None:
                try:
                    img = self.transform(img, return_tensors='pt')['pixel_values'][0]
                except:
                    img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            # Getting the Question
            question = sample_entry['question']
            question_pretext = self.tokenizer(
                "question: {:s}  context: ".format(question), add_special_tokens=False)
            question_id = question_pretext.input_ids
            question_attn_mask = question_pretext.attention_mask
            length_pretext = len(question_id)
            box_pretext = [self.qa_box] * length_pretext

            
            ## For fine-tuning, the authors use boxes[:, 0] = 0, boxes[:, 1] = 0, boxes[:, 2] = 1000, boxes[:, 3] = 1000
            for i in range(len(boxes)):
                boxes[i][0] = 0
                boxes[i][1] = 0
                boxes[i][2] = 1000
                boxes[i][3] = 1000
            
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

            # Getting the Answer
            # print("answers:", sample_entry['answers'])
            answer = self.tokenizer(random.choice(sample_entry['answers']))['input_ids']
            # answer = self.tokenizer((sample_entry['answers']))['input_ids']
            # answer = self.tokenizer(answer)['input_ids']
            # print(answer)
            '''
            return {'pixel_values': img, 'bbox': torch.tensor(boxes), 'input_ids': torch.tensor(tokenized_words), 'labels': torch.tensor(answer),
                    "attention_mask" : torch.tensor(attention_mask),'id': torch.as_tensor([index]), 'question': sample_entry['question'], 'ocr_tokens': words, 'answers':sample_entry['answers']}
                    '''
            return {'bbox': torch.tensor(boxes), 'input_ids': torch.tensor(tokenized_words), 'labels': torch.tensor(answer),
                    "attention_mask" : torch.tensor(attention_mask),'id': torch.as_tensor([index]), 'question': sample_entry['question'], 'ocr_tokens': words, 'answers':sample_entry['answers']}

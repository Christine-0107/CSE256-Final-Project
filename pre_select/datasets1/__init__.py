from .dataset import VGDataset


def build_dataset(test, args, base_img_path=None,
                                json_df=None,
                                ocr_json_df=None,
                                tokenizer=None,
                                transform=None, 
                                max_seq_length=-1, 
                                STVQA=False,):
    if test:
        return VGDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split=args.test_split,
                         test=True,
                         transforms=args.test_transforms,
                         max_query_len=args.max_query_len,
                         bert_mode=args.bert_token_mode,
                         base_img_path = base_img_path, 
                         json_df = json_df, 
                         ocr_json_df = ocr_json_df, 
                         tokenizer = tokenizer, 
                         transform=transform, 
                         max_seq_length=-1, 
                         target_size=(1000, 1000),
                         pad_token_box=[0, 0, 0, 0], 
                         qa_box=[0, 0, 0, 0], 
                         STVQA = STVQA, 
                         )
    else:
        return VGDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split='train',
                          transforms=args.train_transforms,
                          max_query_len=args.max_query_len,
                          bert_mode=args.bert_token_mode,
                          base_img_path = base_img_path, 
                         json_df = json_df, 
                         ocr_json_df = ocr_json_df, 
                         tokenizer = tokenizer, 
                         transform=transform, 
                         max_seq_length=-1, 
                         target_size=(1000, 1000),
                         pad_token_box=[0, 0, 0, 0], 
                         qa_box=[0, 0, 0, 0], 
                         STVQA = STVQA, 
                         )


train_transforms = [
    dict(
        type='RandomSelect',
        transforms1=dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640]),
        transforms2=dict(
            type='Compose',
            transforms=[
                dict(type='RandomResize', sizes=[400, 500, 600], resize_long_side=False),
                dict(type='RandomSizeCrop', min_size=384, max_size=600, check_method=dict(func='iou', iou_thres=0.5)),
                dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640])
            ],
        ),
        p=0.5
    ),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, aug_translate=True)
]

test_transforms = [
    dict(type='RandomResize', sizes=[640], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, center_place=True)
]
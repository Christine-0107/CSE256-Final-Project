import numpy as np
from tqdm import tqdm

imdb_1 = np.load("/home/yanruxue/latr-main/src/new_latr/deepsolo_textvqa_train.npy", allow_pickle=True)
imdb_2 = np.load('/home/yanruxue/latr-main/src/VLTVG/checkpoints/binary_imdbs_06/train_6.npy', allow_pickle=True)

print(len(imdb_2))

new_imdb = []
for item in tqdm(imdb_1):
    question_id = item['question_id']
    flag = 0
    for info in imdb_2:
        if question_id == info['question_id']:
            flag = 1
            new_imdb.append(info)
            break
    if flag == 0:
        new_imdb.append(item)
print(len(new_imdb))

np.save('/home/yanruxue/latr-main/src/VLTVG/checkpoints/binary_imdbs_06/subtrain_6.npy', new_imdb)

# np.save("/home/yanruxue/latr-main/src/VLTVG/imdbs/labels_train.npy", new_imdb)
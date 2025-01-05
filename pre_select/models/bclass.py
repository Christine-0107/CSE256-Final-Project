import torch.nn as nn
import torch

from .process_q import Process_q

class Binary_classi(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.process_q = Process_q()
        self.input_layer = nn.Linear(n_input, n_hidden)
        self.hidden_layer = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ocr_str_feat, ques_feat, ques_mask, img_feat, img_mask, ocr_feat, ocr_mask):
        question_weights, ocr_weights = self.process_q(ques_feat, ques_mask, img_feat, img_mask, ocr_feat, ocr_mask)
        # 8 40, 8 500
        
        '''
        ocr_weights = torch.reshape(ocr_weights, (ocr_weights.shape[0], 100, 5))
        # 8 100 5
        ocr_weights = torch.sum(ocr_weights, -1)
        # 8 100 
        '''
        
        # softmax: get the weights
        #m_inf = torch.tensor([float('-inf')]).to('cuda')
        m_inf = torch.tensor([-1e9]).to('cuda')
        ocr_weights = torch.where(ocr_weights == 0, m_inf, ocr_weights)  
        
        #ocr_weights = ocr_weights - torch.max(ocr_weights, dim=-1, keepdim=True).values
        #print("bclass1-1:")
        #print(ocr_weights)
        ocr_weights = self.softmax(ocr_weights)  # 8 500
        #print("bclass1-2:")
        #print(ocr_weights)

        question_weights = torch.where(question_weights == 0, m_inf, question_weights)      
        #question_weights = question_weights - torch.max(question_weights, dim=-1, keepdim=True).values
        question_weights = self.softmax(question_weights)  # 8 40
        

        # mul weights
        ques_feat = ques_feat.permute(1, 0, 2)  # 8 40 256
        ocr_feat = ocr_feat.permute(1, 0, 2)   # 8 500 256

        question_weights = torch.unsqueeze(question_weights, -1)
        question_weights = question_weights.expand(question_weights.size(0), 40, 256) # 第一维是batch_size
        # print(question_weights)
        # print(question_weights.size())

        ocr_weights = torch.unsqueeze(ocr_weights, -1)
        ocr_weights = ocr_weights.expand(ocr_weights.size(0), 1500, 256) # 第一维是batch_size，第二维是序列长度
        # print(ocr_weights)
        # print(ocr_weights.size())

        ques_feat = ques_feat * question_weights  # 8 40 256
        ocr_feat = ocr_feat * ocr_weights  # 8 500 256
        # print(question_weights.size())
        # print(ocr_weights.size())

        ques_emb = torch.sum(ques_feat, 1)
        ocr_emb = torch.sum(ocr_feat, 1)
        # print(ques_emb.size(), ocr_emb.size())


        # classify
        input = torch.cat((ques_emb, ocr_emb), 1)
        # print(input.size())  # 8 140  / 8 2
        
        output = self.input_layer(input)
        output = self.tanh(output)
        
        output = self.hidden_layer(output)
        output = self.tanh(output)   
        
        output = self.output_layer(output)
        output = self.sigmoid(output)
        # output = self.softmax(output)
        return output
